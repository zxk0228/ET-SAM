import json
import os
import argparse

import cv2
import numpy as np
import pyclipper
import torch
from shapely.geometry import Polygon
from tqdm import tqdm

from et_sam.model.predictor import Predictor
from utils.utils import DisjointSet, GLOBAL_TIMER


def unclip(p, unclip_ratio=2.0):
    poly = Polygon(p)
    distance = poly.area * unclip_ratio / poly.length
    offset = pyclipper.PyclipperOffset()
    offset.AddPath(p, pyclipper.JT_ROUND, pyclipper.ET_CLOSEDPOLYGON)
    results = offset.Execute(distance)
    return results

def process_contour(contour, image_shape, min_area=32, epsilon_ratio=0.002):
    epsilon = epsilon_ratio * cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, epsilon, True)
    pts = approx.reshape((-1, 2))
    if pts.shape[0] < 4:
        return None
    pts = unclip(pts)
    if len(pts) != 1:
        return None
    pts = np.array(pts[0], dtype=np.int32)
    if Polygon(pts).area < min_area:
        return None
    pts[:, 0] = np.clip(pts[:, 0], 0, image_shape[1] - 1)
    pts[:, 1] = np.clip(pts[:, 1], 0, image_shape[0] - 1)
    return pts


def visualize_points(image, points, output_path):
    image = cv2.cvtColor((image * 0.7).astype(np.uint8), cv2.COLOR_RGB2BGR)
    for point in points:
        cv2.circle(image, (int(point[0]), int(point[1])), 5, (0, 0, 255), thickness=-1)
    output_path = output_path + "_point.jpg"
    cv2.imwrite(output_path, image)

def visualize_heatmap(image, heatmap, output_path):
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    heatmap = (np.clip(heatmap, 0, 1) * 255).astype(np.uint8)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    image = cv2.addWeighted(image, 0.7, heatmap, 0.3, 0)
    output_path = output_path + "_heatmap.jpg"
    cv2.imwrite(output_path, image)

def masked_add(image, mask, alpha=0.5):
    new_image = image.copy()
    new_image[mask > 0] = alpha * image[mask>0] + (1-alpha) * mask[mask>0]
    return new_image

def visualize_hi_masks(image, word_masks, line_masks, para_masks, layout, output_path, alpha=0.5):
    if word_masks is None:
        return
    word_mask_combine, word_group_mask_combine, line_mask_combine, para_mask_combine, layout_mask_combine = (np.zeros_like(image) for _ in range(5))
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    word_masks = word_masks.astype(np.uint8)

    for word_mask in word_masks:
        color_group = np.random.uniform(0, 256, size=(3,))
        contours, _ = cv2.findContours(word_mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        for cont in contours:
            color_word = np.random.uniform(0, 256, size=(3,))
            pts = process_contour(cont, image.shape[:2])
            if pts is None:
                continue
            cv2.fillPoly(word_group_mask_combine, [pts], color_group)
            cv2.fillPoly(word_mask_combine, [pts], color_word)

    for line_mask in line_masks:
        color = np.random.randint(0, 256, size=(1,1,3), dtype=np.uint8)
        line_mask_combine += line_mask[:,:,None] * color

    for para_mask in para_masks:
        color = np.random.randint(0, 256, size=(1, 1, 3), dtype=np.uint8)
        para_mask_combine += para_mask[:, :, None] * color

    # 组合mask和image
    word_mask_combine = masked_add(image, word_mask_combine, alpha)
    word_group_mask_combine = masked_add(image, word_group_mask_combine, alpha)
    line_mask_combine = masked_add(image, line_mask_combine, alpha)
    para_mask_combine = masked_add(image, para_mask_combine, alpha)

    cv2.imwrite(f"{output_path}_word.jpg", word_mask_combine)
    cv2.imwrite(f"{output_path}_word_group.jpg", word_group_mask_combine)
    cv2.imwrite(f"{output_path}_line.jpg", line_mask_combine)
    cv2.imwrite(f"{output_path}_para.jpg", para_mask_combine)

    if layout is not None:
        for para in layout['paragraphs']:
            layout_color = np.random.uniform(0, 255, size=(3,))
            for line in para['lines']:
                for word in line['words']:
                    word_poly = np.array(word["vertices"], dtype=np.int32)
                    cv2.fillPoly(layout_mask_combine, [word_poly], layout_color)
        layout_mask_combine = masked_add(image, layout_mask_combine, alpha)
        cv2.imwrite(f"{output_path}_layout.jpg", layout_mask_combine)

def visualize_masks(image, masks, output_path, task=1, alpha=0.5):
    if masks is None:
        return
    mask_combine = np.zeros_like(image)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    masks = masks.astype(np.uint8)

    # 绘制单行单词和文本行
    for mask in masks:
        if task==1:
            color = np.random.uniform(0, 256, size=(3,))
            contours, _ = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
            pts = process_contour(contours[0], image.shape[:2])
            if pts is None:
                continue
            cv2.fillPoly(mask_combine, [pts], color)
        else:
            color = np.random.randint(0, 256, size=(1, 1, 3), dtype=np.uint8)
            mask_combine += mask[:, :, None] * color
    # 组合mask和image
    mask_with_image = masked_add(image, mask_combine, alpha)
    output_type = "word" if task==1 else "line"
    cv2.imwrite(f"{output_path}_{output_type}.jpg", mask_with_image)


def save_hierText_masks(masks, affinity, image_shape, image_name, output_dir, layout_threshold=0.5):
    img_id = image_name.split('.')[0]
    if masks is None:
        lines = [{'words': [{'text': '', 'vertices': [[0, 0], [1, 0], [1, 1], [0, 1]]}], 'text': ''}]
        paragraphs = [{'lines': lines}]
        result = {
            'image_id': img_id,
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
            'image_id': img_id,
            "paragraphs": paragraphs
        }
    output_dir = os.path.join(output_dir, "results")
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, img_id + '.jsonl'), 'w', encoding='utf-8') as fw:
        json.dump(result, fw)
    return result

def get_line_masks_dict(line_masks, scores, image_shape, image_name):
    result_list = []
    if line_masks is None:
        return result_list
    line_masks = line_masks.astype(np.uint8)
    scores = scores.astype(float)
    for line_mask, score in zip(line_masks, scores):
        contours, _ = cv2.findContours(line_mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        max_contour = max(contours, key=cv2.contourArea)[:, 0]  # 去掉多余的小点，确保只有一个结果
        max_contour[:, 0] = np.clip(max_contour[:, 0], 0, image_shape[1])  # w
        max_contour[:, 1] = np.clip(max_contour[:, 1], 0, image_shape[0])  # h
        mask_dict = {"category_id": 1, "rec": "", "image_id": image_name, "score": score, "polys": max_contour.tolist()}
        result_list.append(mask_dict)
    return result_list

def get_word_masks_dict(word_masks, scores, image_shape, image_name):
    result_list = []
    if word_masks is None:
        return result_list
    word_masks = word_masks.astype(np.uint8)
    scores = scores.astype(float)
    for word_mask, score in zip(word_masks, scores):
        contours, _ = cv2.findContours(word_mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            pts = process_contour(contour, image_shape)
            if pts is None:
                continue
            mask_dict = {"category_id": 1, "rec": "", "image_id": image_name, "score": score, "polys": pts.tolist()}
            result_list.append(mask_dict)
    return result_list

def save_word_masks(word_masks, image_shape, image_name, output_dir):
    lines = []
    output_dir = os.path.join(output_dir, "results")
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, 'res_' + image_name + '.txt')
    if word_masks is None:
        with open(output_path, 'w', encoding='utf-8') as fw:
            fw.writelines(lines)
        return
    word_masks = word_masks.astype(np.uint8)
    for word_mask in word_masks:
        contours, _ = cv2.findContours(word_mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            pts = process_contour(contour, image_shape)
            if pts is None:
                continue
            rect = cv2.minAreaRect(pts)
            box = cv2.boxPoints(rect)
            box = box.round().astype(np.int32).flatten().tolist()
            box = ",".join(map(str, box)) + "\n"
            lines.append(box)

    with open(output_path, 'w', encoding='utf-8') as fw:
        fw.writelines(lines)

def log_output(image_num, point_num, point_threshold):
    max_memory = torch.cuda.max_memory_allocated() / 1024 ** 3
    total_time = 0
    print("===========Inference finished.===========")
    print(f"Image number: {image_num}\n",
          f"Point threshold: {point_threshold}\n")
    print("Average inference latency:")
    for key, value in GLOBAL_TIMER.items():
        print(f'-{key}: {value / image_num:.5f}s')
        total_time += value
    print(f"-total: {total_time / image_num:.5f}s\n",
          f"Average prompt points number: {point_num / image_num:.2f}",
          f"Max memory usage: {max_memory:.2f}GB")

def main(args):
    torch.cuda.set_device(args.device)
    images_name = os.listdir(args.test_image_dir)
    pred_point_num = 0
    predictor = Predictor(args)
    result_list = []
    output_dir = os.path.join(args.output_dir, os.path.basename(args.checkpoint).split('.')[0])
    output_image_dir = os.path.join(output_dir, 'visualizations')
    os.makedirs(output_image_dir, exist_ok=True)
    layout_flag = True
    assert args.task_type in [0,1,2], f"Invalid task_type: {args.task_type}, must be one of [0, 1, 2]."

    for image_name in tqdm(images_name):
        image_path = os.path.join(args.test_image_dir, image_name)
        image_name = image_name.split('.')[0]
        output_image_path = os.path.join(output_image_dir, image_name)

        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_shape = image.shape[:2]
        outputs = predictor.predict(image, task=args.task_type)

        if not args.hier_det:
            points, heatmap = outputs
            if args.visualize:
                visualize_points(image, points, output_image_path)
                visualize_heatmap(image, heatmap, output_image_path)
        else:
            points, mask_outputs = outputs
            if args.task_type == 0:
                line_word_masks, line_masks, para_masks, hi_scores, affinity = mask_outputs
                layout = None
                if args.eval:
                    layout = save_hierText_masks(line_word_masks, affinity, image_shape, image_name, output_dir, args.layout_threshold)
                if args.visualize:
                    if layout is None and layout_flag:
                        print("If you need layout analysis visualization results, set args.eval to True.")
                        layout_flag = False
                    visualize_hi_masks(image, line_word_masks, line_masks, para_masks, layout, output_image_path)
            else:
                masks, scores = mask_outputs
                if args.eval:
                    if "CTW1500" in args.test_image_dir:
                        result_list += get_line_masks_dict(masks, scores, image_shape, image_name)
                    elif "TotalText" in args.test_image_dir:
                        result_list += get_word_masks_dict(masks, scores, image_shape, image_name)
                    elif "ICDAR" in args.test_image_dir:
                        save_word_masks(masks, image_shape, image_name, output_dir)
                if args.visualize:
                    visualize_masks(image, masks, output_image_path, args.task_type)
        pred_point_num += len(points)
    if len(result_list) > 0:
        with open(os.path.join(output_dir, 'results.json'), 'w', encoding='utf-8') as fw:
            json.dump(result_list, fw)

    log_output(len(images_name), pred_point_num, args.point_threshold)

def get_args():
    parser = argparse.ArgumentParser()
    # paths
    parser.add_argument('--test_image_dir', type=str, required=True)
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--output_dir', type=str, required=True)

    # model
    parser.add_argument('--model_type', type=str, default="vit_l")
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--hier_det', action='store_true', default=False)

    # task settings
    parser.add_argument('--use_task_prompt', action='store_true', default=False)
    parser.add_argument('--task_type', type=int, default=0)

    # post-processing
    parser.add_argument('--point_batch_size', type=int, default=100)
    parser.add_argument('--nms_kernel_size', type=int, default=3)
    parser.add_argument('--point_threshold', type=float, default=0.6)
    parser.add_argument('--layout_threshold', type=float, default=0.5)

    # eval / vis
    parser.add_argument('--visualize', action='store_true', default=False)
    parser.add_argument('--eval', action='store_true', default=False)
    
    args = parser.parse_args()
    args.device = f"cuda:{args.device}"
    return args


if __name__ == '__main__':
    args = get_args()
    main(args)
