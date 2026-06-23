import os
import json
import random

from torch.utils.data import Dataset

from et_sam.data.my_transforms import *
from et_sam.data.preprocess.heatmap_tools import *


def get_one_mask(vertices, w, h):
    mask = np.zeros((h, w), dtype=np.float32)
    mask = cv2.fillPoly(mask, [np.array(vertices)], [1])
    return mask


def get_line_word_mask(vertices, w, h):
    mask = np.zeros((h, w), dtype=np.float32)
    for ver in vertices:
        mask = cv2.fillPoly(mask, [np.array(ver)], [1])
    return mask

def get_word_mask(vertices, w, h):
    mask = np.zeros((h, w), dtype=np.float32)
    ver = random.choice(vertices)
    mask = cv2.fillPoly(mask, [np.array(ver)], [1])
    return mask


class HierTextDataset(Dataset):
    def __init__(self, root_dir, transform, hier_det=False, heatmap_scale=4, min_sigma=1, sample_num=10):
        self.image_dir = os.path.join(root_dir, "train")
        self.images_name = os.listdir(self.image_dir)
        self.heatmap_dir = os.path.join(root_dir, "train_heatmap")
        json_path = os.path.join(root_dir, "gt", "train_gt.json")
        self.transform = transform
        self.hier_det = hier_det
        self.heatmap_scale = heatmap_scale
        self.min_sigma = min_sigma
        self.sample_num = sample_num
        with open(json_path, 'r', encoding='utf-8') as f:
            annotations = json.load(f)['annotations']  # 得到注释的列表，每个注释对应一张图片
        self.annotations = {anns['image_id']: anns for anns in annotations}
        del annotations
        print(f"Loaded {len(self.images_name)} samples from {root_dir}.")

    def __len__(self):
        return len(self.images_name)

    def __getitem__(self, idx):
        image = cv2.imread(os.path.join(self.image_dir, self.images_name[idx]))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_idx = self.images_name[idx].split('.')[0]  # 从路径获取图片名称

        sample = {'image': image, 'task': 0}

        heatmap_path = os.path.join(self.heatmap_dir, image_idx + ".npy")
        sample['heatmap'] = np.load(heatmap_path)

        if self.hier_det:  # 随机筛选适量的掩码，sample增加了word_masks line_masks paragraph_masks line2paragraph_index
            anns = self.annotations[image_idx]
            w, h = anns['image_width'], anns['image_height']
            line_num = len(anns['line_masks'])
            if line_num > self.sample_num:  # 文本行数量大于10，随机选取10个文本行掩码
                select_line_idx = random.sample(range(line_num), self.sample_num)
                select_line_idx.sort()

                masks = [get_one_mask(anns['line_masks'][l_idx], w, h) for l_idx in select_line_idx]
                masks = np.array(masks).transpose((1, 2, 0))
                sample['line_masks'] = masks

                masks = [get_line_word_mask(anns['word_masks'][l_idx], w, h) for l_idx in select_line_idx]
                masks = np.array(masks).transpose((1, 2, 0))
                sample['line_word_masks'] = masks

                masks = [get_word_mask(anns['word_masks'][l_idx], w, h) for l_idx in select_line_idx]
                masks = np.array(masks).transpose((1, 2, 0))
                sample['word_masks'] = masks

                line2para_index = anns['line2paragraph_index']  # ordered
                line2para_index = [line2para_index[l_idx] for l_idx in select_line_idx]
                line2para_index_set = set(line2para_index)

                masks = [get_one_mask(anns['paragraph_masks'][l2p_idx], w, h) for l2p_idx in line2para_index_set]
                masks = np.array(masks).transpose((1, 2, 0))
                sample['paragraph_masks'] = masks

                new_l2p_index = []
                for ii, jj in enumerate(line2para_index_set):
                    new_l2p_index.extend([ii] * line2para_index.count(jj))
                sample['line2paragraph_index'] = torch.tensor(new_l2p_index)
            else:  # 文本行数量不足10个，选取所有掩码
                sample['line2paragraph_index'] = torch.tensor(anns['line2paragraph_index'])
                masks = [get_one_mask(ver, w, h) for ver in anns['paragraph_masks']]
                masks = np.array(masks).transpose((1, 2, 0))
                sample['paragraph_masks'] = masks

                masks = [get_one_mask(ver, w, h) for ver in anns['line_masks']]
                masks = np.array(masks).transpose((1, 2, 0))
                sample['line_masks'] = masks

                masks = [get_line_word_mask(ver, w, h) for ver in anns['word_masks']]
                masks = np.array(masks).transpose((1, 2, 0))
                sample['line_word_masks'] = masks

                masks = [get_word_mask(ver, w, h) for ver in anns['word_masks']]
                masks = np.array(masks).transpose((1, 2, 0))
                sample['word_masks'] = masks
        # 数据增广及处理
        self.transform(sample)
        return sample


class WordLevelDataset(Dataset):
    def __init__(self, root_dir, transform, hier_det=False, heatmap_scale=4, min_sigma=1, sample_num=10):
        self.image_dir = os.path.join(root_dir, "train")
        self.images_name = os.listdir(self.image_dir)
        self.heatmap_dir = os.path.join(root_dir, "train_heatmap")
        json_path = os.path.join(root_dir, "gt", "train_gt.json")
        self.transform = transform
        self.hier_det = hier_det
        self.heatmap_scale = heatmap_scale
        self.min_sigma = min_sigma
        self.sample_num = sample_num
        with open(json_path, 'r', encoding='utf-8') as f:
            self.annotations = json.load(f)  # 得到注释的列表，每个注释对应一张图片
        print(f"Loaded {len(self.images_name)} samples from {root_dir}.")

    def __len__(self):
        return len(self.images_name)

    def __getitem__(self, idx):
        image = cv2.imread(os.path.join(self.image_dir, self.images_name[idx]))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_idx = self.images_name[idx].split('.')[0]  # 从路径获取图片名称
        image_shape = np.array(image.shape[:2])

        sample = {'image': image,
                  'task': 1}
        heatmap_path = os.path.join(self.heatmap_dir, image_idx + ".npy")
        annotation = self.annotations[image_idx]
        if os.path.exists(heatmap_path):
            sample['heatmap'] = np.load(heatmap_path)
        else:
            grid = get_grid(image_shape, heatmap_scale=self.heatmap_scale)
            center_points_list, word_size_list = annotation["center_points"], annotation["word_size"]
            center_points_list = sample_center_points_list(center_points_list, word_size_list)
            sample['heatmap'] = get_heatmap_by_center_line(center_points_list, word_size_list, grid, self.heatmap_scale,
                                                           self.min_sigma)
            np.save(heatmap_path, sample['heatmap'])

        if self.hier_det:  # 随机筛选适量的掩码，sample增加了word_masks
            w, h = image_shape[1], image_shape[0]
            word_num = len(annotation['word_masks'])
            if word_num > self.sample_num:  # 文本行数量大于10，随机选取10个文本行掩码
                select_word_idx = random.sample(range(word_num), self.sample_num)
                select_word_idx.sort()
                masks = [get_one_mask(annotation['word_masks'][idx], w, h) for idx in select_word_idx]
            elif word_num > 0:  # 文本行数量不足10个，选取所有掩码
                masks = [get_one_mask(ver, w, h) for ver in annotation['word_masks']]
            else:
                masks = np.zeros((1, h, w))
            masks = np.array(masks).transpose((1, 2, 0))
            sample['word_masks'] = masks
        # 数据增广及处理
        self.transform(sample)
        return sample


class LineLevelDataset(Dataset):
    def __init__(self, root_dir, transform, hier_det=False, heatmap_scale=4, min_sigma=1, sample_num =10):
        self.image_dir = os.path.join(root_dir, "train")
        self.images_name = os.listdir(self.image_dir)
        self.heatmap_dir = os.path.join(root_dir, "train_heatmap")
        json_path = os.path.join(root_dir, "gt", "train_gt.json")
        self.transform = transform
        self.hier_det = hier_det
        self.heatmap_scale = heatmap_scale
        self.min_sigma = min_sigma
        self.sample_num = sample_num
        with open(json_path, 'r', encoding='utf-8') as f:
            self.annotations = json.load(f)  # 得到注释的列表，每个注释对应一张图片
        print(f"Loaded {len(self.images_name)} samples from {root_dir}.")

    def __len__(self):
        return len(self.images_name)

    def __getitem__(self, idx):
        image = cv2.imread(os.path.join(self.image_dir, self.images_name[idx]))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_idx = self.images_name[idx].split('.')[0]  # 从路径获取图片名称
        image_shape = np.array(image.shape[:2])

        sample = {'image': image,
                  'task': 2}
        heatmap_path = os.path.join(self.heatmap_dir, image_idx + ".npy")
        annotation = self.annotations[image_idx]
        sample['heatmap'] = np.load(heatmap_path)

        if self.hier_det:  # 随机筛选适量的掩码，sample增加了word_masks
            w, h = image_shape[1], image_shape[0]
            line_num = len(annotation['line_masks'])
            if line_num > self.sample_num:  # 文本行数量大于10，随机选取10个文本行掩码
                select_word_idx = random.sample(range(line_num), self.sample_num)
                select_word_idx.sort()
                masks = [get_one_mask(annotation['line_masks'][idx], w, h) for idx in select_word_idx]
            elif line_num > 0:  # 文本行数量不足10个，选取所有掩码
                masks = [get_one_mask(ver, w, h) for ver in annotation['line_masks']]
            else:
                masks = np.zeros((1, h, w))
            masks = np.array(masks).transpose((1, 2, 0))
            sample['line_masks'] = masks
        # 数据增广及处理
        self.transform(sample)
        return sample


class UnifiedDataset(Dataset):
    def __init__(self, hier_dataset, word_dataset, line_dataset):
        self.hier_dataset = hier_dataset
        self.word_dataset = word_dataset
        self.line_dataset = line_dataset
        self.max_length = max(len(self.hier_dataset), len(self.word_dataset), len(self.line_dataset))

    def __len__(self):
        return self.max_length

    def __getitem__(self, idx):
        hier_index = idx % len(self.hier_dataset)
        word_index = idx % len(self.word_dataset)
        line_index = idx % len(self.line_dataset)
        hier_data = self.hier_dataset[hier_index]
        word_data = self.word_dataset[word_index]
        line_data = self.line_dataset[line_index]

        batch_data = [hier_data, word_data, line_data]
        batch_data = concat_collate_fn(batch_data)
        return batch_data


def concat_collate_fn(batch_samples):
    """将一个batch的sample合并为一个sample"""
    image, heatmap, heatmap_shape = [], [], []
    para_masks, line_masks, word_masks, line_word_masks, line2para_idx = [], [], [], [], []
    tasks = []
    for sample in batch_samples:
        image.append(sample['image'])
        heatmap.append(sample['heatmap'])
        heatmap_shape.append(sample['heatmap_shape'])
        word_masks.append(sample.get('word_masks', None))
        line_word_masks.append(sample.get('line_word_masks', None))
        line_masks.append(sample.get('line_masks', None))
        para_masks.append(sample.get('paragraph_masks', None))
        line2para_idx.append(sample.get('line2paragraph_index', None))
        tasks.append(sample['task'])

    batch_data = {
        'image': torch.stack(image),
        'heatmap': torch.stack(heatmap),
        'heatmap_shape': torch.stack(heatmap_shape),
        'tasks': tasks,

        'word_masks': word_masks,
        'line_word_masks': line_word_masks,
        'line_masks': line_masks,
        'paragraph_masks': para_masks,
        'line2paragraph_index': line2para_idx
    }
    return batch_data


def unified_collate_fn(batch_samples):
    return batch_samples[0]