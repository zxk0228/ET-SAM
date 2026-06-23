import math
import os
import json

import tqdm
from scipy.io import loadmat

from HierText_process import shrink_polygon, get_args
from heatmap_tools import *

if __name__ == '__main__':
    root_dir = get_args().root_dir
    image_dir = os.path.join(root_dir, "train")
    gt_dir = os.path.join(root_dir, "gt", "train_gt")
    output_heatmap_dir = os.path.join(root_dir, "train_heatmap")
    os.makedirs(output_heatmap_dir, exist_ok=True)
    output_path = os.path.join(root_dir, "gt", "train_gt.json")

    mat_names = os.listdir(gt_dir)
    output_dict = dict()
    for i, mat_name in enumerate(tqdm.tqdm(mat_names)):
        idx = mat_name.lstrip("gt_").split(".")[0]
        image_path = os.path.join(image_dir, idx + ".jpg")
        image = cv2.imread(image_path)
        image_shape = image.shape[:2]
        heatmap_path = os.path.join(output_heatmap_dir, idx + ".npy")

        mat_path = os.path.join(gt_dir, mat_name)
        data = loadmat(mat_path)['gt']
        mat_dict = {
            "word_masks": [],
        }
        center_points_list = []
        word_size_list = []
        for word in data:
            x = np.array(word[1])[0]
            y = np.array(word[3])[0]
            word_mask = np.stack((x, y), axis=1, dtype=np.int32)

            half_num = math.ceil(len(word_mask) / 2)
            slice_1 = slice(0, half_num)
            slice_2 = slice(len(word_mask) - 1, len(word_mask) // 2 - 1, -1)
            center_points = (word_mask[slice_1] + word_mask[slice_2]) / 2
            center_points = center_points.round(2).tolist()
            center_points_list.append(center_points)

            word_size = (np.sum((word_mask[0] - word_mask[-1]) ** 2)) ** 0.5
            word_size = word_size.round(2).tolist()
            word_size_list.append(word_size)

            if word[4][0] != '#':
                word_mask = shrink_polygon(word_mask) if len(word_mask) >= 4 else word_mask.tolist()
                mat_dict["word_masks"].append(word_mask)

        output_dict[str(i).zfill(7)] = mat_dict

        center_points_list = sample_center_points_list(center_points_list, word_size_list)
        heatmap = get_heatmap_by_center_line_gpu(center_points_list, word_size_list, image_shape)
        np.save(heatmap_path, heatmap)

    with open(output_path, "w") as f:
        json.dump(output_dict, f)