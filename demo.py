import os

import cv2
import numpy as np
import torch
from tqdm import tqdm

from et_sam.model.predictor import Predictor
from test import process_contour, visualize_points, visualize_masks, visualize_hi_masks, visualize_heatmap, log_output
from utils.utils import DisjointSet
from test import get_args

def save_hierText_masks(masks, affinity, image_shape, layout_threshold=0.5):
    if masks is None:
        lines = [{'words': [{'text': '', 'vertices': [[0, 0], [1, 0], [1, 1], [0, 1]]}], 'text': ''}]
        paragraphs = [{'lines': lines}]
        result = {
            "paragraphs": paragraphs
        }
    else:
        masks = masks.astype(np.uint8)  # word masks, (n, h, w)
        lines = []
        line_indices = []
        for index, mask in enumerate(masks):
            line = {'words': [], 'text': ''}
            contours, _ = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
            for cont in contours:
                pts = process_contour(cont, image_shape)
                if pts is None:
                    continue
                cnt_list = pts.tolist()
                line['words'].append({'text': '', 'vertices': cnt_list})
            if line['words']:
                lines.append(line)
                line_indices.append(index)

        line_grouping = DisjointSet(len(line_indices))
        affinity = affinity[line_indices][:, line_indices]
        for i1, i2 in zip(*np.where(affinity > layout_threshold)):
            line_grouping.union(i1, i2)
        line_groups = line_grouping.to_group()
        paragraphs = []
        for line_group in line_groups:
            paragraph = {'lines': []}
            for id_ in line_group:
                paragraph['lines'].append(lines[id_])
            if paragraph:
                paragraphs.append(paragraph)
        result = {
            "paragraphs": paragraphs
        }
    return result

def main(args):
    torch.cuda.set_device(args.device)
    total_time, pred_point_num = 0, 0
    predictor = Predictor(args)
    output_dir = os.path.join(args.output_dir, os.path.basename(args.checkpoint).split('.')[0])
    os.makedirs(output_dir, exist_ok=True)
    assert args.task_type in [0,1,2], f"Invalid task_type: {args.task_type}, must be one of [0, 1, 2]."

    if os.path.isdir(args.test_image_dir):
        images_name = os.listdir(args.test_image_dir)
    else:
        images_name = [os.path.basename(args.test_image_dir)]
        args.test_image_dir = os.path.dirname(args.test_image_dir)

    for image_name in tqdm(images_name):
        image_path = os.path.join(args.test_image_dir, image_name)
        image_name = image_name.split('.')[0]
        output_image_path = os.path.join(output_dir, image_name)

        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_shape = image.shape[:2]
        outputs = predictor.predict(image, task=args.task_type)

        if not args.hier_det:
            points, heatmap = outputs
            visualize_points(image, points, output_image_path)
            visualize_heatmap(image, heatmap, output_image_path)
        else:
            points, mask_outputs = outputs
            if args.task_type == 0:
                line_word_masks, line_masks, para_masks, hi_scores, affinity = mask_outputs
                layout = save_hierText_masks(line_word_masks, affinity, image_shape, args.layout_threshold)
                visualize_hi_masks(image, line_word_masks, line_masks, para_masks, layout, output_image_path)
            else:
                masks, scores = mask_outputs
                visualize_masks(image, masks, output_image_path, args.task_type)
        pred_point_num += len(points)

    log_output(len(images_name), pred_point_num, args.point_threshold)


if __name__ == '__main__':
    args = get_args()
    main(args)
