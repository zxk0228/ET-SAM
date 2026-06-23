import os

from torch.utils.data import DataLoader, BatchSampler
from torch.utils.data.distributed import DistributedSampler
from torchvision import transforms

from et_sam.data import *
from et_sam.model.loss import loss_hi_masks, loss_hi_iou_mse, loss_heatmap
from joint_train import train, get_args
from utils import misc
from utils.utils import sample_train_points


def load_data_ft(args):
    heatmap_scale, min_sigma, hier_det = args.heatmap_scale, args.min_sigma, args.hier_det
    transform = transforms.Compose([ColorJitter(), RandomRotate(), LargeScaleJitter(), ToTensor()])
    assert len(args.used_datasets) == 1, "Must use 1 dataset."
    dataset_name = args.used_datasets[0]
    root_dir = os.path.join(args.dataset_dir, dataset_name)
    if dataset_name in args.hier_datasets:
        final_dataset = HierTextDataset(root_dir, transform, hier_det=hier_det, heatmap_scale=heatmap_scale, min_sigma=min_sigma)
    elif dataset_name in args.word_datasets:
        final_dataset = WordLevelDataset(root_dir, transform, hier_det=hier_det, heatmap_scale=heatmap_scale, min_sigma=min_sigma)
    elif dataset_name in args.line_datasets:
        final_dataset = LineLevelDataset(root_dir, transform, hier_det=hier_det, heatmap_scale=heatmap_scale, min_sigma=min_sigma)
    sampler = DistributedSampler(final_dataset)
    batch_sampler = BatchSampler(sampler, args.batch_size, drop_last=True)
    dataloader = DataLoader(final_dataset, batch_sampler=batch_sampler, pin_memory=True,
                            num_workers=args.num_workers, collate_fn=concat_collate_fn, persistent_workers=True)
    return dataloader


def train_epoch_ft(model, data, device):
    images, heatmaps, heatmaps_shape = data['image'].to(device), data['heatmap'].to(device), data['heatmap_shape'].to(device)
    para_masks, line_masks, line_word_masks, word_masks = data['paragraph_masks'], data['line_masks'], data['line_word_masks'], data['word_masks']
    line2para_idx = data['line2paragraph_index']
    points, para_masks, line_masks, line_word_masks, word_masks = sample_train_points(
        heatmaps, para_masks, line_masks, line_word_masks, word_masks, line2para_idx
    )
    hier_level, word_level, line_level = line_word_masks is not None, word_masks is not None, line_masks is not None
    pred_heatmaps, word_masks_logits, hi_masks_logits, hi_iou_output = model(images, points, tasks=data["tasks"])

    loss_word, loss_line_word, loss_line, loss_para = 0,0,0,0
    # 热图损失
    loss_point = loss_heatmap(pred_heatmaps, heatmaps, heatmaps_shape)

    if word_level:
        # 计算word_mask的损失
        loss_focal_word, loss_dice_word = loss_hi_masks(word_masks_logits[:, 0:1], word_masks, len(word_masks))
        loss_mse_word = loss_hi_iou_mse(hi_iou_output[:, 0:1], word_masks_logits[:, 0:1], model.module.mask_threshold, word_masks)
        loss_word = loss_focal_word + loss_dice_word + loss_mse_word

    if line_level:
        # 计算line_mask的损失
        loss_focal_line, loss_dice_line = loss_hi_masks(hi_masks_logits[:, 0:1], line_masks, len(line_masks))
        loss_mse_line = loss_hi_iou_mse(hi_iou_output[:, 2:3], hi_masks_logits[:, 0:1], model.module.mask_threshold, line_masks)
        loss_line = loss_focal_line + loss_dice_line + loss_mse_line

    if hier_level:
        # 计算line_word_mask的损失
        loss_focal_line_word, loss_dice_line_word = loss_hi_masks(word_masks_logits[:, 1:], line_word_masks, len(line_word_masks))
        loss_mse_line_word = loss_hi_iou_mse(hi_iou_output[:, 1:2], word_masks_logits[:, 1:], model.module.mask_threshold,line_word_masks)
        loss_line_word = loss_focal_line_word + loss_dice_line_word + loss_mse_line_word

        # 计算para_mask的损失
        loss_focal_para, loss_dice_para = loss_hi_masks(hi_masks_logits[:, 1:], para_masks, len(para_masks))
        loss_mse_para = loss_hi_iou_mse(hi_iou_output[:, 3:], hi_masks_logits[:, 1:], model.module.mask_threshold, para_masks)
        loss_para = loss_focal_para + loss_dice_para + loss_mse_para
    loss_mask = loss_word + loss_line_word + loss_line + loss_para * 0.5
    loss = 50 * loss_point + loss_mask

    loss_dict = {
        "loss_point": loss_point * 50,
        "loss_mask": loss_mask
    }

    loss_dict_reduced = misc.reduce_dict(loss_dict)
    losses_reduced_scaled = sum(loss_dict_reduced.values())
    loss_value = losses_reduced_scaled.item()
    return loss, loss_value, loss_dict_reduced


if __name__ == "__main__":
    args = get_args()
    train(args, load_data_ft, train_epoch_ft)

