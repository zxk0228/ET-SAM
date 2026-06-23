import os
import json
import xml.etree.ElementTree as ET

import tqdm
from HierText_process import get_args
from heatmap_tools import *

if __name__ == '__main__':
    root_dir = get_args().root_dir
    gt_dir = os.path.join(root_dir, "gt", "train_gt")
    image_dir = os.path.join(root_dir, "train")
    output_path = os.path.join(root_dir, "gt", "train_gt.json")

    # 根据原始标注得到计算热图所需的中心点和宽度
    gt_names = os.listdir(gt_dir)
    output_dict = dict()
    for gt_name in tqdm.tqdm(gt_names):
        image_path = os.path.join(image_dir, gt_name.split(".")[0]+".jpg")
        image = cv2.imread(image_path)
        image_shape = image.shape[:2]

        gt_path = os.path.join(gt_dir, gt_name)
        xml_data = ET.parse(gt_path).getroot().find("image")
        gt_dict = {
            "line_masks": [],
        }
        for box in xml_data.findall("box"):
            line_mask = np.array(box.find("segs").text.split(","), dtype=np.int32).reshape((-1, 2))
            gt_dict["line_masks"].append(line_mask.tolist())

        output_dict[gt_name.split('.')[0]] = gt_dict

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output_dict, f)