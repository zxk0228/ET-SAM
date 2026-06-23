import time
from typing import List

import torch
from torch import nn
from torch.nn import functional as F

from utils.utils import GLOBAL_TIMER
from .image_encoder import ImageEncoderViT
from .mask_decoder import HiDecoder
from .point_decoder import PointDecoder
from .prompt_encoder import PromptEncoder


class ETSam(nn.Module):

    def __init__(
            self,
            image_encoder: ImageEncoderViT,
            point_decoder: PointDecoder,
            prompt_encoder: PromptEncoder,
            hi_decoder: HiDecoder,
            pixel_mean: List[float] = [123.675, 116.28, 103.53],
            pixel_std: List[float] = [58.395, 57.12, 57.375],
            hier_det: bool = False
    ):
        super(ETSam, self).__init__()
        self.image_encoder = image_encoder
        self.point_decoder = point_decoder
        self.hier_det = hier_det  # 是否需要多层次检测
        if hier_det:
            self.prompt_encoder = prompt_encoder
            self.hi_decoder = hi_decoder
        self.set_params()

        # 缓冲区，保存不参与更新的参数
        self.register_buffer("pixel_mean", torch.Tensor(pixel_mean).view(1, -1, 1, 1), False)
        self.register_buffer("pixel_std", torch.Tensor(pixel_std).view(1, -1, 1, 1), False)

        self.mask_threshold = 0.0

    def set_params(self):
        for n, p in self.image_encoder.named_parameters():
            if "Adapter" not in n:
                p.requires_grad = False
        if self.hier_det:
            for p in self.prompt_encoder.parameters():
                p.requires_grad = False

    def forward(self, images_input, points_input=None, tasks=None):
        image_embeddings = self.image_encoder(self.preprocess(images_input))
        heatmaps = []
        assert len(image_embeddings) == len(tasks)
        for image_embedding, task in zip(image_embeddings, tasks):
            heatmap = self.point_decoder(image_embedding.unsqueeze(0), task=task)['pred_heatmaps']
            heatmaps.append(heatmap)
        heatmaps = torch.cat(heatmaps, dim=0)
        if self.hier_det:
            hi_masks_logits, hi_iou_preds, word_masks_logits = [], [], []
            assert len(image_embeddings) == len(tasks)
            for image_embedding, points_coord, task in zip(image_embeddings, points_input, tasks):
                points_coord = points_coord[:, None, :]
                points_label = torch.ones(points_coord.shape[:2], device=points_coord.device)
                points = (points_coord, points_label)
                word_mask_logits, hi_mask_logits, hi_iou_pred = self.forward_hi_decoder(points, image_embedding.unsqueeze(0), task=task)

                hi_masks_logits.append(hi_mask_logits)
                hi_iou_preds.append(hi_iou_pred)
                word_masks_logits.append(word_mask_logits)
            word_masks_logits = torch.cat(word_masks_logits, dim=0)
            hi_masks_logits = torch.cat(hi_masks_logits, dim=0)
            hi_iou_preds = torch.cat(hi_iou_preds, dim=0)
            return heatmaps, word_masks_logits, hi_masks_logits, hi_iou_preds
        else:
            return heatmaps

    def forward_hi_decoder(self, points, image_embedding, task: int = 0):
        point_embeddings, dense_embeddings = self.prompt_encoder(points=points, boxes=None, masks=None)
        return self.hi_decoder(
            image_embeddings=image_embedding,
            image_pe=self.prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=point_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            task=task,
        )

    @torch.no_grad()
    def predict(self, image_input, point_batch_size=100, task=0):
        last_time = time.time()
        image_input = self.preprocess(image_input)
        image_embedding = self.image_encoder(image_input)
        GLOBAL_TIMER["image_encoder"] += time.time() - last_time
        last_time = time.time()
        point_decoder_output = self.point_decoder(image_embedding, task=task)
        pred_points = point_decoder_output["pred_points"][0]
        GLOBAL_TIMER["point_decoder"] += time.time() - last_time
        if self.hier_det:
            last_time = time.time()
            pred_points = pred_points[:, None, :]  # (point_num,1,2)
            point_labels = torch.ones(pred_points.shape[:2])  # (point_num,1)
            point_num = len(pred_points)
            # 对点进行批处理，全部处理容易造成显存溢出
            hi_masks, hi_iou, word_masks = [], [], []
            for start_idx in range(0, point_num, point_batch_size):
                end_idx = min(start_idx + point_batch_size, point_num)
                points = (pred_points[start_idx:end_idx], point_labels[start_idx:end_idx])
                word_mask_logits, hi_mask_logits, hi_iou_pred = self.forward_hi_decoder(points, image_embedding, task=task)

                hi_masks.append(hi_mask_logits)  # 排除掉低分辨率word_mask
                hi_iou.append(hi_iou_pred)
                word_masks.append(word_mask_logits)
            if len(hi_masks) > 0:
                del hi_mask_logits, hi_iou_pred, word_mask_logits
                hi_masks = torch.cat(hi_masks, dim=0)  # (points_num,2,256,256)
                hi_iou = torch.cat(hi_iou, dim=0)  # (points_num,3)
                word_masks = torch.cat(word_masks, dim=0)  # (points_num,1,384,384)
            else:
                hi_masks = torch.zeros((0, 2, 256, 256))
                hi_iou = torch.zeros((0, 4))
                word_masks = torch.zeros((0, 2, 384, 384))
            GLOBAL_TIMER["hier_seg"] += time.time() - last_time
            return pred_points[:, 0], word_masks, hi_masks, hi_iou
        else:
            pred_heatmap = point_decoder_output["pred_heatmaps"][0]
            return pred_points, pred_heatmap

    def preprocess(self, x: torch.Tensor) -> torch.Tensor:
        """Normalize pixel values and pad to a square input."""
        # Normalize colors
        x = (x - self.pixel_mean) / self.pixel_std

        # Pad
        h, w = x.shape[-2:]
        padh = self.image_encoder.img_size - h
        padw = self.image_encoder.img_size - w
        x = F.pad(x, (0, padw, 0, padh))
        return x
