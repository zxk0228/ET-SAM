import os
import random
import argparse

import numpy as np
import torch
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader, ConcatDataset, BatchSampler
from torch.utils.data.distributed import DistributedSampler
from torchvision import transforms

from et_sam.data import *
from et_sam.model.build import build_et_sam
from et_sam.model.loss import loss_hi_masks, loss_hi_iou_mse, loss_heatmap
from utils import misc
from utils.utils import sample_train_points


def initial(args):
    misc.init_distributed_mode(args)
    seed = args.seed + misc.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def train(args, load_data_func, train_epoch_func, is_joint_train=True):
    initial(args)
    
    # 构建模型
    model = build_et_sam(args=args)
    if torch.cuda.is_available():
        model.cuda()
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu],
                                                      find_unused_parameters=args.find_unused_params)

    train_dataloader = load_data_func(args)
    epoch_start = args.start_epoch
    epoch_num = args.epoch_num

    # 设置优化器
    params_to_train = [param for param in model.module.parameters() if param.requires_grad]
    optimizer = torch.optim.AdamW(
        params_to_train, lr=args.lr, betas=(args.beta1, args.beta2), weight_decay=args.weight_decay
    )
    del params_to_train

    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, args.lr_drop_epoch)

    # 加载断点
    if args.checkpoint is not None and args.continue_training:
        with open(args.checkpoint, "rb") as f:
            state_dict = torch.load(f, weights_only=True)
        optimizer.load_state_dict(state_dict['optimizer'])
        lr_scheduler.load_state_dict(state_dict['lr_scheduler'])
        epoch_start = state_dict['epoch'] + 1
        epoch_num = state_dict['epoch_num']
        del state_dict

    model.train()
    device = model.device
    grad_scaler = GradScaler()

    for epoch in range(epoch_start, epoch_num):
        train_dataloader.batch_sampler.sampler.set_epoch(epoch)
        print(f"epoch:{epoch}")
        misc.synchronize()
        metric_logger = misc.MetricLogger(delimiter="  ")
        for data in metric_logger.log_every(train_dataloader, print_freq=args.visualize_freq):
            with (autocast()):
                loss, loss_value, loss_dict_reduced = train_epoch_func(model, data, device)
            # 反向传播、更新梯度
            optimizer.zero_grad()
            grad_scaler.scale(loss).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0, norm_type=2)
            grad_scaler.step(optimizer)
            grad_scaler.update()
            metric_logger.update(training_loss=loss_value, **loss_dict_reduced)
        metric_logger.synchronize_between_processes()
        lr_scheduler.step()

        # 保存
        if misc.is_main_process() and (epoch + 1) % args.save_freq == 0:
            params_to_save = {name: param for name, param in model.module.named_parameters()
                              if (name.startswith("image_encoder") and "Adapter" in name)
                              or not(name.startswith("image_encoder") or name.startswith("prompt_encoder"))}
            buffer_to_save = {name: buffer for name, buffer in model.module.named_buffers()
                              if name.startswith("point_decoder.pe_layer")}
            params_to_save.update(buffer_to_save)
            checkpoint = {
                'epoch': epoch,
                'epoch_num': epoch_num,
                'optimizer': optimizer.state_dict(),
                'lr_scheduler': lr_scheduler.state_dict(),
                'trained_params': params_to_save
            }
            torch.save(checkpoint, args.output_path)

    if misc.is_main_process():
        params_to_save = {name: param for name, param in model.module.named_parameters()
                          if (name.startswith("image_encoder") and "Adapter" in name)
                          or not (name.startswith("image_encoder") or name.startswith("prompt_encoder"))}
        buffer_to_save = {name: buffer for name, buffer in model.module.named_buffers()
                          if name.startswith("point_decoder.pe_layer")}
        params_to_save.update(buffer_to_save)
        checkpoint = {'trained_params': params_to_save}
        torch.save(checkpoint, args.output_path)

    print("============Training finished.============")
    torch.distributed.destroy_process_group()


def load_data_joint(args):
    heatmap_scale, min_sigma, hier_det = args.heatmap_scale, args.min_sigma, args.hier_det
    transform = transforms.Compose([ColorJitter(), RandomRotate(), LargeScaleJitter(), ToTensor()])
    hier_dataset_list, word_dataset_list, line_dataset_list = [], [], []
    collate_fn = unified_collate_fn
    for dataset_name in args.used_datasets:
        root_dir = os.path.join(args.dataset_dir, dataset_name)
        if dataset_name in args.hier_datasets:
            hier_dataset_list.append(
                HierTextDataset(root_dir, transform, hier_det=hier_det,
                                heatmap_scale=heatmap_scale, min_sigma=min_sigma))
        elif dataset_name in args.word_datasets:
            word_dataset_list.append(
                WordLevelDataset(root_dir, transform, hier_det=hier_det,
                                 heatmap_scale=heatmap_scale, min_sigma=min_sigma))
        elif dataset_name in args.line_datasets:
            line_dataset_list.append(
                LineLevelDataset(root_dir, transform, hier_det=hier_det,
                                 heatmap_scale=heatmap_scale, min_sigma=min_sigma))

    hier_dataset = ConcatDataset(hier_dataset_list)
    word_dataset = ConcatDataset(word_dataset_list)
    line_dataset = ConcatDataset(line_dataset_list)
    final_dataset = UnifiedDataset(hier_dataset, word_dataset, line_dataset)

    sampler = DistributedSampler(final_dataset)
    batch_sampler = BatchSampler(sampler, 1, drop_last=True)
    dataloader = DataLoader(final_dataset, batch_sampler=batch_sampler, pin_memory=True,
                            num_workers=args.num_workers, collate_fn=collate_fn, persistent_workers=True)
    return dataloader


def train_epoch_joint(model, data, device):
    images, heatmaps, heatmaps_shape = data['image'].to(device), data['heatmap'].to(device), data['heatmap_shape'].to(device)
    para_masks, line_masks, line_word_masks, word_masks = data['paragraph_masks'], data['line_masks'], data['line_word_masks'], data['word_masks']
    line2para_idx = data['line2paragraph_index']
    points, para_masks, line_masks, line_word_masks, word_masks = sample_train_points(
        heatmaps, para_masks, line_masks, line_word_masks, word_masks, line2para_idx
    )
    hier_length, word_length, line_length = len(points[0]), len(points[1]), len(points[2])

    pred_heatmaps, word_masks_logits, hi_masks_logits, hi_iou_output = model(images, points, tasks=data['tasks'])
    pred_word_mask = word_masks_logits[:hier_length+word_length,0:1]
    pred_word_iou = hi_iou_output[:hier_length+word_length,0:1]
    pred_line_word_mask = word_masks_logits[:hier_length, 1:2]
    pred_line_word_iou = hi_iou_output[:hier_length, 1:2]
    pred_line_mask = torch.concat([hi_masks_logits[:hier_length, 0:1], hi_masks_logits[-line_length:, 0:1]], dim=0)
    pred_line_iou = torch.concat([hi_iou_output[:hier_length,2:3], hi_iou_output[-line_length:,2:3]], dim=0)
    pred_para_mask = hi_masks_logits[:hier_length, 1:2]
    pred_para_iou = hi_iou_output[:hier_length, 3:]

    # 热图损失
    loss_point = loss_heatmap(pred_heatmaps, heatmaps, heatmaps_shape)

    # 计算para_mask的损失
    loss_focal_para, loss_dice_para = loss_hi_masks(pred_para_mask, para_masks, hier_length)
    loss_mse_para = loss_hi_iou_mse(pred_para_iou, pred_para_mask, model.module.mask_threshold, para_masks)
    loss_para = loss_focal_para + loss_dice_para + loss_mse_para

    # 计算line_mask的损失
    loss_focal_line, loss_dice_line = loss_hi_masks(pred_line_mask, line_masks, hier_length + line_length)
    loss_mse_line = loss_hi_iou_mse(pred_line_iou, pred_line_mask, model.module.mask_threshold, line_masks)
    loss_line = loss_focal_line + loss_dice_line + loss_mse_line

    # 计算line_word_mask的损失
    loss_focal_line_word, loss_dice_line_word = loss_hi_masks(pred_line_word_mask, line_word_masks, hier_length)
    loss_mse_line_word = loss_hi_iou_mse(pred_line_word_iou, pred_line_word_mask, model.module.mask_threshold, line_word_masks)
    loss_line_word = loss_focal_line_word + loss_dice_line_word + loss_mse_line_word

    # 计算word_mask的损失
    loss_focal_word, loss_dice_word = loss_hi_masks(pred_word_mask, word_masks, hier_length + word_length)
    loss_mse_word = loss_hi_iou_mse(pred_word_iou, pred_word_mask, model.module.mask_threshold, word_masks)
    loss_word = loss_focal_word + loss_dice_word + loss_mse_word

    loss = loss_point * 50 + loss_word + loss_line_word + loss_line + loss_para * 0.5
    loss_dict = {
        "loss_point": loss_point * 50,
        "loss_word": loss_word,
        "loss_line_word": loss_line_word,
        "loss_line": loss_line,
        "loss_para": loss_para * 0.5
    }

    loss_dict_reduced = misc.reduce_dict(loss_dict)
    losses_reduced_scaled = sum(loss_dict_reduced.values())
    loss_value = losses_reduced_scaled.item()
    return loss, loss_value, loss_dict_reduced


def get_args():
    parser = argparse.ArgumentParser()

    # optimizer
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--beta1', type=float, default=0.9)
    parser.add_argument('--beta2', type=float, default=0.999)
    parser.add_argument('--weight_decay', type=float, default=0.05)

    # training
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--start_epoch', type=int, default=0)
    parser.add_argument('--epoch_num', type=int, default=120)
    parser.add_argument('--lr_drop_epoch', type=int, default=100)
    parser.add_argument('--save_freq', type=int, default=5)
    parser.add_argument('--visualize_freq', type=int, default=100)
    parser.add_argument('--num_workers', type=int, default=8)

    # model
    parser.add_argument('--model_type', type=str, default="vit_l")
    parser.add_argument('--hier_det', action='store_true', default=False)
    parser.add_argument('--use_task_prompt', action='store_true', default=False)

    parser.add_argument('--heatmap_scale', type=int, default=4)
    parser.add_argument('--min_sigma', type=float, default=1)
    parser.add_argument('--nms_kernel_size', type=int, default=3)
    parser.add_argument('--point_threshold', type=float, default=0.5)

    # checkpoint & paths
    parser.add_argument('--checkpoint', type=str)
    parser.add_argument('--continue_training', action='store_true', default=False)
    parser.add_argument('--output_path', type=str, default="checkpoints/joint_train.pth")
    parser.add_argument('--dataset_dir', type=str, default="../datasets")
    parser.add_argument(
        '--used_datasets', nargs='+',
        default=["HierText", "TotalText", "ICDAR2013", "ICDAR2015", "TextSeg", "CTW1500"]
    )
    parser.add_argument('--hier_datasets', nargs='+', default=["HierText"])
    parser.add_argument('--word_datasets', nargs='+', default=["TotalText", "ICDAR2013", "ICDAR2015", "TextSeg"])
    parser.add_argument('--line_datasets', nargs='+', default=["CTW1500"])

    # distributed training
    parser.add_argument('--find_unused_params', action='store_true', default=False)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--dist_url', type=str, default='env://')
    parser.add_argument('--world_size', type=int, default=0)
    parser.add_argument('--rank', type=int, default=0)
    parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--distributed', action='store_true', default=False)
    parser.add_argument('--dist_backend', type=str, default='nccl')
    return parser.parse_args()


if __name__ == '__main__':
    args = get_args()
    train(args, load_data_joint, train_epoch_joint)
