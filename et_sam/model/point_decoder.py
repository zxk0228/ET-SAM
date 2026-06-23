from typing import Type, Tuple

import torch
import torch.nn as nn

from .common import LayerNorm2d
from .mask_decoder import MLP
from .prompt_encoder import PositionEmbeddingRandom


class PointDecoder(nn.Module):
    def __init__(
            self,
            transformer: nn.Module,
            transformer_dim: int = 256,
            activation: Type[nn.Module] = nn.GELU,
            embed_dim: int = 256,
            image_embedding_size: Tuple[int, int] = (64, 64),
            nms_kernel_size: int = 3,
            point_threshold: float = 0.3,
            max_points: int = 1000,
            use_task_prompt: bool = True,
    ) -> None:
        super().__init__()
        self.transformer_dim = transformer_dim
        self.transformer = transformer

        self.mask_tokens = nn.Embedding(1, transformer_dim)
        self.output_upscaling = nn.Sequential(
            nn.ConvTranspose2d(transformer_dim, transformer_dim // 4, kernel_size=2, stride=2),
            LayerNorm2d(transformer_dim // 4),
            activation(),
            nn.ConvTranspose2d(transformer_dim // 4, transformer_dim // 8, kernel_size=2, stride=2),
            activation(),
        )
        self.output_hypernetworks_mlp = MLP(transformer_dim, transformer_dim, transformer_dim // 8, 3)

        self.image_embedding_size = image_embedding_size
        self.pe_layer = PositionEmbeddingRandom(embed_dim // 2)
        self.nms_kernel_size = nms_kernel_size
        self.point_threshold = point_threshold
        self.max_points = max_points

        self.use_task_prompt = use_task_prompt
        task_prompt_tokens = [nn.Embedding(1, transformer_dim) for i in range(3)]
        self.task_prompt_tokens = nn.ModuleList(task_prompt_tokens)
        for i in range(3):
            nn.init.zeros_(self.task_prompt_tokens[i].weight)

    def get_dense_pe(self) -> torch.Tensor:
        return self.pe_layer(self.image_embedding_size).unsqueeze(0)

    def forward(self, image_embeddings, masks=None, task=0):
        output_tokens = self.mask_tokens.weight
        if self.use_task_prompt:
            output_tokens = output_tokens + self.task_prompt_tokens[task].weight
        sparse_embeddings = output_tokens.unsqueeze(0).expand(image_embeddings.size(0), -1, -1)
        image_pe = self.get_dense_pe()
        src = image_embeddings
        pos_src = image_pe
        b, c, h, w = src.shape
        hs, src = self.transformer(src, pos_src, sparse_embeddings)
        src = src.transpose(1, 2).view(b, c, h, w)
        mask_tokens_out = hs[:, 0, :]
        upscaled_embedding = self.output_upscaling(src)
        hyper_in = self.output_hypernetworks_mlp(mask_tokens_out).unsqueeze(1)
        b, c, h, w = upscaled_embedding.shape
        pred_heatmaps = (hyper_in @ upscaled_embedding.view(b, c, h * w)).view(b, -1, h, w)
        if self.training:
            return {'pred_heatmaps': pred_heatmaps}  # (bs,1,256,256)

        if masks is not None:
            pred_heatmaps *= masks

        with torch.no_grad():
            from utils.utils import nms
            device = image_embeddings.device
            pred_heatmaps_nms = nms(pred_heatmaps.detach().clone(), self.nms_kernel_size)
            pred_points = []
            for i in range(b):
                points = torch.nonzero((pred_heatmaps_nms[i] > self.point_threshold).squeeze())
                points.to(device)
                points = torch.flip(points, dims=(-1,))
                pred_points_score_ = pred_heatmaps_nms[i, 0, points[:, 1], points[:, 0]].flatten(0)

                idx = torch.argsort(pred_points_score_, dim=0, descending=True)[
                      :min(self.max_points, pred_points_score_.size(0))]

                points = points[idx]
                # pred_points_score_ = pred_points_score_[idx]
                pred_points.append(points*4)
                # pred_points_score[i, :points.size(0)] = pred_points_score_
            # pred_points_score = pred_points_score[:, :m]

        return {'pred_heatmaps': pred_heatmaps,  # (bs,1,256,256)
                'pred_points': pred_points
                # 'pred_points_score': pred_points_score,
                # 'pred_heatmaps_nms': pred_heatmaps_nms
                }
