# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
from functools import partial

import torch

from .et_sam import ETSam
from .image_encoder import ImageEncoderViT
from .mask_decoder import HiDecoder
from .point_decoder import PointDecoder
from .prompt_encoder import PromptEncoder
from .transformer import TwoWayTransformer


def build_et_sam(args):
    model_type = args.model_type

    model_type_dict = {
        'vit_b': (768, 12, 12, [2, 5, 8, 11]),
        'vit_l': (1024, 24, 16, [5, 11, 17, 23]),
        'vit_h': (1280, 32, 16, [7, 15, 23, 31])
    }
    encoder_embed_dim, encoder_depth, encoder_num_heads, encoder_global_attn_indexes = model_type_dict[model_type]

    prompt_embed_dim = 256
    image_size = 1024
    vit_patch_size = 16
    image_embedding_size = image_size // vit_patch_size  # 64

    model = ETSam(
        image_encoder=ImageEncoderViT(
            depth=encoder_depth,
            embed_dim=encoder_embed_dim,
            img_size=image_size,
            mlp_ratio=4,
            norm_layer=partial(torch.nn.LayerNorm, eps=1e-6),
            num_heads=encoder_num_heads,
            patch_size=vit_patch_size,
            qkv_bias=True,
            use_rel_pos=True,
            global_attn_indexes=encoder_global_attn_indexes,
            window_size=14,
            out_chans=prompt_embed_dim,
        ),

        point_decoder=PointDecoder(
            transformer=TwoWayTransformer(
                depth=2,
                embedding_dim=prompt_embed_dim,
                mlp_dim=2048,
                num_heads=8
            ),
            transformer_dim=prompt_embed_dim,
            nms_kernel_size=args.nms_kernel_size,
            point_threshold=args.point_threshold,
            use_task_prompt=args.use_task_prompt,
        ),

        prompt_encoder=PromptEncoder(
            embed_dim=prompt_embed_dim,
            image_embedding_size=(image_embedding_size, image_embedding_size),
            input_image_size=(image_size, image_size),
            mask_in_chans=16,
        ),

        hi_decoder=HiDecoder(
            transformer=TwoWayTransformer(
                depth=2,
                embedding_dim=prompt_embed_dim,
                mlp_dim=2048,
                num_heads=8,
            ),
            transformer_dim=prompt_embed_dim,
            use_task_prompt=args.use_task_prompt,
        ),

        pixel_mean=[123.675, 116.28, 103.53],
        pixel_std=[58.395, 57.12, 57.375],
        hier_det=args.hier_det
    )

    root_dir = 'checkpoints'
    if model_type == "vit_b":
        sam_name = 'sam_vit_b_01ec64.pth'
    elif model_type == "vit_l":
        sam_name = 'sam_vit_l_0b3195.pth'
    elif model_type == "vit_h":
        sam_name = 'sam_vit_h_4b8939.pth'
    sam_path = os.path.join(root_dir, sam_name)

    with open(sam_path, "rb") as f:
        sam_dict = torch.load(f, weights_only=True)

    et_sam_dict = dict()

    # 导入image_encoder和prompt_encoder参数
    for key, value in sam_dict.items():
        if key.startswith("image_encoder") or (key.startswith("prompt_encoder") and args.hier_det):
            et_sam_dict[key] = value
    del sam_dict
    print("Loaded image encoder and prompt encoder.")

    if args.checkpoint is None:
        with open(os.path.join(root_dir, args.model_type + '_maskdecoder.pth'), "rb") as f:
            mask_decoder_dict = torch.load(f, weights_only=True)
        # 将sam的mask_decoder参数变为point_decoder并导入
        for key, value in mask_decoder_dict.items():
            if (key.startswith('mask_decoder.transformer')
                    or key.startswith('mask_decoder.output_upscaling')):
                new_key = key.replace('mask_decoder', 'point_decoder')
                et_sam_dict[new_key] = value
            elif key.startswith('mask_decoder.output_hypernetworks_mlps.0'):
                new_key = key.replace('mask_decoder.output_hypernetworks_mlps.0',
                                      'point_decoder.output_hypernetworks_mlp')
                et_sam_dict[new_key] = value
        # 初始化hi_decoder
        for key, value in mask_decoder_dict.items():
            # 将sam的mask_decoder参数变为hi_decoder并导入
            if key.startswith("mask_decoder"):
                new_key = key.replace('mask_decoder', 'hi_decoder')
                et_sam_dict[new_key] = value
    else:
        with open(args.checkpoint, "rb") as f:
            checkpoint_dict = torch.load(f, weights_only=True)
        if "trained_params" in checkpoint_dict.keys():
            checkpoint_dict = checkpoint_dict["trained_params"]
        et_sam_dict.update(checkpoint_dict)
        del checkpoint_dict
        print("Loaded checkpoint.")

    info = model.load_state_dict(et_sam_dict, strict=False)
    # print("combine_pretrained_params:", info)

    return model