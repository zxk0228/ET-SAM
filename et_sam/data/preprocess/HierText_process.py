import argparse
import copy
import os
import json

import pyclipper
from shapely.geometry import Polygon
from tqdm import tqdm

from heatmap_tools import *


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root_dir', type=str, required=True)
    return parser.parse_args()


def get_word_size(polygon):
    polygon = np.array(polygon)
    word_size = (np.sum((polygon[0] - polygon[-1]) ** 2)) ** 0.5
    return word_size.round(2).tolist()

def shrink_polygon(polygon, shrink_ratio=0.4):
    # from DB (https://github.com/MhLiao/DB)
    np_poly = np.array(polygon)
    polygon_shape = Polygon(np_poly)
    distance = polygon_shape.area * (1 - np.power(shrink_ratio, 2)) / polygon_shape.length
    subject = [tuple(l) for l in np_poly]
    padding = pyclipper.PyclipperOffset()
    padding.AddPath(subject, pyclipper.JT_ROUND, pyclipper.ET_CLOSEDPOLYGON)
    shrinked = padding.Execute(-distance)
    if shrinked == []:
        return np_poly.tolist()
    else:
        shrinked = shrinked[0]
        distance = np.sum((np_poly[0][None,:] - shrinked)**2, axis=1)
        index = np.argmin(distance)
        shrinked = np.roll(shrinked, -index, axis=0)
        return shrinked.tolist()

if __name__ == '__main__':
    ''' 
    An example script for processing the original gt.
    Step(1) filter empty paragraphs to avoid training error 
    '''
    root_dir = get_args().root_dir
    json_type = 'train'
    heatmap_dir = os.path.join(root_dir, "train_heatmap")
    os.makedirs(heatmap_dir, exist_ok=True)
    ANN_PATH = os.path.join(root_dir, "gt", json_type + '.jsonl')
    with open(ANN_PATH, 'r', encoding='utf-8') as f:
        anns = json.load(f)
    new_json = dict()
    new_json['info'] = anns['info']
    new_annotations = []
    for old_ann in tqdm(anns['annotations']):
        # per image
        paragraphs = []
        old_paras = old_ann['paragraphs']
        assert len(old_paras) > 0
        for old_para in old_paras:
            if len(old_para['lines']) > 0:
                paragraphs.append(old_para)
        if len(paragraphs) > 0:
            new_ann = copy.deepcopy(old_ann)
            new_ann['paragraphs'] = paragraphs
            new_annotations.append(new_ann)
        else:
            continue
    new_json.update(annotations=new_annotations)


    ''' 
    Step(2) shrink the word polygon and reorganize the dict
    '''
    new_all_dict = {'info': new_json['info']}
    new_annotations = []
    old_annotations = new_json.pop('annotations')

    for old_anno in tqdm(old_annotations):
        new_dict = old_anno
        paras = new_dict.pop('paragraphs')  # [{pa1}, {pa2}, ...]
        w = new_dict['image_width']
        h = new_dict['image_height']
        paragraph_masks, line_masks, word_masks = [], [], []
        line2paragraph_index = []
        para_legible, line_legible = [], []
        word_size_list = []
        lineindex = 0
        for para_idx, para in enumerate(paras):
            assert len(para['vertices']) > 2
            paragraph_masks.append(para['vertices'])
            para_legible.append(para['legible'])
            for line in para['lines']:
                line_masks.append(line['vertices'])
                line2paragraph_index.append(para_idx)
                line_legible.append(line['legible'])
                word_mask_per_line = []
                word_size_per_line = []
                for word in line['words']:
                    word_size = max(get_word_size(word["vertices"]), 1.0)
                    shr_word_mask = shrink_polygon(word['vertices'], 0.4)
                    assert len(shr_word_mask) > 2
                    word_mask_per_line.append(shr_word_mask)
                    word_size_per_line.append(word_size)
                word_masks.append(word_mask_per_line)
                word_size_list.append(word_size_per_line)
                lineindex += 1
        new_dict.update(
            paragraph_masks=paragraph_masks,
            line_masks=line_masks,
            word_masks=word_masks,
            line2paragraph_index=line2paragraph_index,
            para_legible=para_legible,
            line_legible=line_legible
        )

        if json_type=='train':
            heatmap_path = os.path.join(heatmap_dir, new_dict['image_id'] + ".npy")
            line_points_list, word_size_list = sample_center_points_list_hier(word_masks, word_size_list)
            heatmap = get_heatmap_by_center_line_gpu(line_points_list, word_size_list, [h, w])
            np.save(heatmap_path, heatmap)

        assert len(line_masks) == len(word_masks)
        new_annotations.append(new_dict)
    new_all_dict.update(annotations=new_annotations)
    print(len(new_annotations))
    with open(root_dir + json_type + '_gt.json', 'w', encoding='utf-8') as fw:
        json.dump(new_all_dict, fw)
