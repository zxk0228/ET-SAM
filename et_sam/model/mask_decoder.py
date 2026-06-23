# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
from torch import nn
from torch.nn import functional as F
from typing import Type, List

from .common import LayerNorm2d


class HiDecoder(nn.Module):
    def __init__(
        self,
        *,
        transformer_dim: int,
        transformer: nn.Module,
        activation: Type[nn.Module] = nn.GELU,
        iou_head_depth: int = 3,
        iou_head_hidden_dim: int = 256,
        use_task_prompt: bool = True,
    ) -> None:
        """
        Predicts masks given an image and prompt embeddings, using a
        tranformer architecture.

        Arguments:
          transformer_dim (int): the channel dimension of the transformer
          transformer (nn.Module): the transformer used to predict masks
          activation (nn.Module): the type of activation to use when
            upscaling masks
          iou_head_depth (int): the depth of the MLP used to predict
            mask quality
          iou_head_hidden_dim (int): the hidden dimension of the MLP
            used to predict mask quality
        """
        super().__init__()
        self.transformer = transformer
        self.iou_token = nn.Embedding(1, transformer_dim)
        self.num_mask_tokens = 4

        self.output_upscaling = nn.Sequential(
            nn.ConvTranspose2d(transformer_dim, transformer_dim // 4, kernel_size=2, stride=2),
            LayerNorm2d(transformer_dim // 4),
            activation(),
            nn.ConvTranspose2d(transformer_dim // 4, transformer_dim // 8, kernel_size=2, stride=2),
            activation(),
        )

        self.iou_prediction_head = MLP(
            transformer_dim, iou_head_hidden_dim, self.num_mask_tokens, iou_head_depth
        )

        self.word_mask_dc = nn.Sequential(
            nn.Conv2d(transformer_dim // 8, transformer_dim // 16, kernel_size=1),
            LayerNorm2d(transformer_dim // 16),
            activation(),
        )
        self.word_mask_refine = nn.Sequential(
            nn.Conv2d(transformer_dim // 16, transformer_dim // 16, kernel_size=3, padding=1),
            LayerNorm2d(transformer_dim // 16),
            activation(),
            nn.Conv2d(transformer_dim // 16, transformer_dim // 16, kernel_size=3, padding=1),
            LayerNorm2d(transformer_dim // 16),
            activation(),
            nn.Conv2d(transformer_dim // 16, transformer_dim // 16, kernel_size=3, padding=1),
            LayerNorm2d(transformer_dim // 16),
            activation(),
            nn.Conv2d(transformer_dim // 16, transformer_dim // 16, kernel_size=3, padding=1),
            activation(),
        )

        self.mask_tokens = nn.Embedding(self.num_mask_tokens, transformer_dim)

        self.output_hypernetworks_mlps = nn.ModuleList(
            [
                MLP(transformer_dim, transformer_dim, transformer_dim // 8, 3)
                for _ in range(2)
            ]
        )
        self.output_word_mlps = nn.ModuleList(
            [
                MLP(transformer_dim, transformer_dim, transformer_dim // 16, 3)
                for _ in range(2)
            ]
        )

        self.use_task_prompt = use_task_prompt
        task_prompt_embeddings = [nn.Embedding(1, transformer_dim) for i in range(3)]
        self.task_prompt_embeddings = nn.ModuleList(task_prompt_embeddings)
        for i in range(3):
            nn.init.zeros_(self.task_prompt_embeddings[i].weight)

        task_prompt_tokens = [nn.Embedding(self.num_mask_tokens, transformer_dim) for i in range(3)]
        self.task_prompt_tokens = nn.ModuleList(task_prompt_tokens)
        for i in range(3):
            nn.init.zeros_(self.task_prompt_tokens[i].weight)



    def forward(
        self,
        image_embeddings: torch.Tensor,
        image_pe: torch.Tensor,
        sparse_prompt_embeddings: torch.Tensor,
        dense_prompt_embeddings: torch.Tensor,
        task=0,
    ): # -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Predict masks given image and prompt embeddings.
        """
        if self.use_task_prompt:
            dense_prompt_embeddings_tempt = dense_prompt_embeddings + self.task_prompt_embeddings[
                task].weight.reshape(1, -1, 1, 1).expand(
                dense_prompt_embeddings.shape[0], -1, dense_prompt_embeddings.shape[-2], dense_prompt_embeddings.shape[-1]
            )
        else:
            dense_prompt_embeddings_tempt = None
        return self.predict_masks(
            image_embeddings=image_embeddings,
            image_pe=image_pe,
            sparse_prompt_embeddings=sparse_prompt_embeddings,
            dense_prompt_embeddings=dense_prompt_embeddings_tempt,
            task=task,
        )

    def predict_masks(
        self,
        image_embeddings: torch.Tensor,
        image_pe: torch.Tensor,
        sparse_prompt_embeddings: torch.Tensor, # (point_num, 2 ,transformer_dim)
        dense_prompt_embeddings: torch.Tensor,
        task=0,
    ):  # -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Predicts masks. See 'forward' for more details."""
        if self.use_task_prompt:
            mask_tokens = self.mask_tokens.weight + self.task_prompt_tokens[task].weight
        else:
            mask_tokens = self.mask_tokens.weight
        tokens = torch.cat([self.iou_token.weight, mask_tokens], dim=0) # (5, transformer_dim)
        tokens = tokens.unsqueeze(0).expand(sparse_prompt_embeddings.size(0), -1, -1) # (point_num, 5, transformer_dim)
        tokens = torch.cat((tokens, sparse_prompt_embeddings), dim=1) #　(point_num, 5+2, transformer_dim)

        # Expand per-image data in batch direction to be per-mask
        src = torch.repeat_interleave(image_embeddings, tokens.shape[0], dim=0)   # (point_num, 256, 64, 64)

        if dense_prompt_embeddings is not None:
            src = src + dense_prompt_embeddings
        pos_src = torch.repeat_interleave(image_pe, tokens.shape[0], dim=0)
        b, c, h, w = src.shape
        # del image_embeddings, sparse_prompt_embeddings, image_pe
        # torch.cuda.empty_cache()

        # Run the transformer
        hs, src = self.transformer(src, pos_src, tokens) # hs(point_num, 5+2, transformer_dim)
        # del pos_src, tokens
        # torch.cuda.empty_cache()
        iou_token_out = hs[:, 0, :]  # (point_num, 256)
        mask_tokens_out = hs[:, 1: (1 + self.num_mask_tokens), :] # (point_num, 4, 256)

        # Upscale mask embeddings and predict masks using the mask tokens
        src = src.transpose(1, 2).view(b, c, h, w)
        upscaled_embedding = self.output_upscaling(src)
        hyper_in: List[torch.Tensor] = []
        for i in range(len(self.output_hypernetworks_mlps)):
            hyper_in.append(self.output_hypernetworks_mlps[i](mask_tokens_out[:, i+2]))
        hyper_in = torch.stack(hyper_in, dim=1)
        b, c, h, w = upscaled_embedding.shape
        hi_masks = (hyper_in @ upscaled_embedding.view(b, c, h * w)).view(b, -1, h, w)
        # del src, hs
        # torch.cuda.empty_cache()
        # 生成高分辨率掩码
        upscaled_embedding = self.word_mask_dc(upscaled_embedding)
        upscaled_embedding = F.interpolate(upscaled_embedding, (384, 384), mode="bilinear", align_corners=False)
        upscaled_embedding = self.word_mask_refine(upscaled_embedding)
        hyper_in: List[torch.Tensor] = []
        for i in range(len(self.output_word_mlps)):
            hyper_in.append(self.output_word_mlps[i](mask_tokens_out[:, i]))
        hyper_in = torch.stack(hyper_in, dim=1)
        b, c, h, w = upscaled_embedding.shape
        word_masks = (hyper_in @ upscaled_embedding.view(b, c, h * w)).view(b, -1, h, w)
        iou_pred = self.iou_prediction_head(iou_token_out)

        return word_masks, hi_masks, iou_pred

# Lightly adapted from
# https://github.com/facebookresearch/MaskFormer/blob/main/mask_former/modeling/transformer/transformer_predictor.py # noqa
class MLP(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        num_layers: int,
        sigmoid_output: bool = False,
    ) -> None:
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(
            nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim])
        )
        self.sigmoid_output = sigmoid_output

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        if self.sigmoid_output:
            x = F.sigmoid(x)
        return x
