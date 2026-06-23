import time

import cv2
import numpy as np
import torch

from et_sam.model.build import build_et_sam
from utils.utils import matrix_nms, get_iou_matrix, postprocess_masks, GLOBAL_TIMER, DisjointSet


class Predictor:
    def __init__(self, args):
        self.model = build_et_sam(args)
        self.args = args
        self.model = self.model.to(self.args.device).eval()
        self.mask_threshold: float = self.model.mask_threshold

    def set_image(self, image):
        self.original_image_size = image.shape[:2]
        self.scale = 1024 / np.max(np.array(self.original_image_size))
        new_shape = (self.scale * np.array(self.original_image_size)).round().astype(np.int32)
        input_image = cv2.resize(image, dsize=new_shape[::-1], interpolation=cv2.INTER_LINEAR)
        input_image = torch.as_tensor(input_image, dtype=torch.float32, device=self.args.device)
        input_image = input_image.permute(2, 0, 1).contiguous().unsqueeze(0)
        self.input_size = tuple(input_image.shape[-2:])
        return input_image

    @torch.no_grad()
    def predict(self, image, task=0):
        torch.cuda.empty_cache()
        input_image = self.set_image(image)
        if self.args.hier_det:
            points, word_masks, hi_masks, hi_iou = self.model.predict(input_image, self.args.point_batch_size, task=task)
            last_time = time.time()
            mask_outputs = self.predict_postprocess(word_masks, hi_masks, hi_iou, task=task)
            points = points.cpu().numpy()
            points = points // self.scale
            GLOBAL_TIMER["postprocess"] += time.time() - last_time
            return points, mask_outputs
        else:
            points, heatmap = self.model.predict(input_image)
            points = points.cpu().numpy()
            points = points // self.scale
            heatmap = postprocess_masks(heatmap, self.input_size, self.original_image_size)
            heatmap = heatmap[0, 0].cpu().numpy()
            return points, heatmap

    def predict_postprocess(self, word_masks, hi_masks, scores, task=0, score_thresh=0.5, nms_thresh=0.5):
        input_size = self.input_size
        original_size = self.original_image_size

        if task == 0:
            line_word_masks = word_masks[:, 1:]
            # filter low quality lines
            keep = scores[:, -2] > score_thresh
            if keep.sum() == 0:
                return None, None, None, None, None
            hi_masks = hi_masks[keep]
            scores = scores[keep]
            line_word_masks = line_word_masks[keep]

            # conduct mask nms, use 256x256 mask for nms to save memory and time
            updated_scores = matrix_nms(
                seg_masks=(hi_masks[:, 0, :, :] > self.mask_threshold),
                scores=scores[:, -2]
            )
            keep = updated_scores > nms_thresh
            if keep.sum() == 0:
                return None, None, None, None, None,
            hi_masks = hi_masks[keep]  # line and paragraph masks
            scores = scores[keep]
            line_word_masks = line_word_masks[keep]

            line_masks = hi_masks[:, 0:1]
            para_masks = hi_masks[:, 1:]

            affinity = get_iou_matrix(masks=(hi_masks[:, 1, :, :] > self.mask_threshold))
            line_grouping = DisjointSet(len(affinity))
            for i1, i2 in zip(*np.where(affinity.cpu().numpy() > score_thresh)):
                line_grouping.union(i1, i2)
            line_groups = line_grouping.to_group()
            para_indexes = []
            for line_group in line_groups:
                index = line_group[torch.argmax(scores[line_group, -1])]
                para_indexes.append(index)
            para_masks = para_masks[para_indexes]

            line_word_masks = postprocess_masks(line_word_masks, input_size, original_size) > self.mask_threshold
            line_masks = postprocess_masks(line_masks, input_size, original_size) > self.mask_threshold
            para_masks = postprocess_masks(para_masks, input_size, original_size) > self.mask_threshold if para_masks is not None else None

            line_word_masks = line_word_masks[:, 0].cpu().numpy()
            line_masks = line_masks[:, 0].cpu().numpy()
            para_masks = para_masks[:, 0].cpu().numpy() if para_masks is not None else None
            hi_scores = scores.cpu().numpy()
            affinity = affinity.cpu().numpy()
            return line_word_masks, line_masks, para_masks, hi_scores, affinity

        if task == 1:
            word_masks = word_masks[:, 0:1]
            word_scores = scores[:, 0]
            keep = word_scores > score_thresh
            if keep.sum() == 0:
                return None, None
            word_masks = word_masks[keep]
            word_scores = word_scores[keep]
            updated_scores = matrix_nms(
                seg_masks=(word_masks[:, 0, :, :] > self.mask_threshold),
                scores=word_scores
            )
            keep = updated_scores > score_thresh
            if keep.sum() == 0:
                return None, None
            word_masks = word_masks[keep]
            word_scores = word_scores[keep]
            word_masks = postprocess_masks(word_masks, input_size, original_size) > self.mask_threshold

            word_masks = word_masks[:, 0].cpu().numpy()
            word_scores = word_scores.cpu().numpy()
            return word_masks, word_scores

        if task == 2:
            line_masks = hi_masks[:, 0:1]
            line_scores = scores[:, -2]
            # filter low quality lines
            keep = line_scores > score_thresh
            if keep.sum() == 0:
                return None, None
            line_masks = line_masks[keep]
            line_scores = line_scores[keep]

            updated_scores = matrix_nms(
                seg_masks=(line_masks[:, 0, :, :] > self.mask_threshold),
                scores=line_scores
            )
            keep = updated_scores > nms_thresh
            if keep.sum() == 0:
                return None, None
            line_masks = line_masks[keep]  # line and paragraph masks
            line_scores = line_scores[keep]
            line_masks = postprocess_masks(line_masks, input_size, original_size) > self.mask_threshold

            line_masks = line_masks[:, 0].cpu().numpy()
            line_scores = line_scores.cpu().numpy()
            return line_masks, line_scores