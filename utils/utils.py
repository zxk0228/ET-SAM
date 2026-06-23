import collections
import random
from typing import List

import numpy as np
import torch
import torch.nn.functional as F


GLOBAL_TIMER = {
    "image_encoder": 0,
    "point_decoder": 0,
    "hier_seg": 0,
    "postprocess": 0
}

def sample_single_mask(batch_heatmaps, batch_masks, np_per_mask=2):
    batch_points = peak_selection(batch_heatmaps)
    dev = batch_heatmaps.device
    select_points, select_masks = [], []
    for points, masks in zip(batch_points, batch_masks):
        points = points.round().to(torch.int32)
        masks = masks.to(dev)
        h, w = masks.shape[-2:]
        intersection = (
                (masks[:, points[:, 1], points[:, 0]]) > 0
        )

        intersec_num = torch.sum(intersection, dim=1)  # 计算每个文本行掩码与像素标签的交点数量
        keep_index = torch.nonzero(intersec_num).reshape(-1)  # 获取有交点的文本行掩码的索引
        keep_num = keep_index.numel()  # 计算有交点的文本行掩码的数量

        # 过滤无交点的部分
        intersection = intersection[keep_index]
        intersec_num = intersec_num[keep_index]
        masks = masks[keep_index]
        if keep_num == 0:  # 一个交点都没有
            sampled_xy = torch.tensor([[w / 2, h / 2]], dtype=torch.float32, device=dev)
            select_points.append(sampled_xy)
            masks = torch.zeros((1, h, w)).to(masks)
            select_masks.append(masks.unsqueeze(1))
            continue

        # 筛选出交点的横纵坐标
        x_idx = torch.masked_select(points[:, 0], intersection)
        y_idx = torch.masked_select(points[:, 1], intersection)
        del intersection

        intersec_cumsum = torch.cumsum(intersec_num, dim=0)  # 做累加，计算出每个文本行掩码对应的交点坐标的终点索引
        start_idx = intersec_cumsum - intersec_num  # 计算出每个文本行掩码对应的交点坐标的起始索引
        np_per_mask = min(np_per_mask, intersec_num.min().item())  # 计算每行采样点的数量
        masks = torch.repeat_interleave(masks, np_per_mask, dim=0)

        pos_idx = np.ravel([
            random.sample(range(start_idx[p_i].item(), intersec_cumsum[p_i].item()), np_per_mask)
            for p_i in range(keep_num)
        ])

        # 根据采样点坐标的索引获取采样点坐标，并组合
        sampled_x = x_idx[pos_idx]
        sampled_y = y_idx[pos_idx]
        sampled_xy = torch.cat((sampled_x[:, None], sampled_y[:, None]), dim=1)  # (k, 2)

        # 打乱采样点顺序，对相应的mask也打乱，使其一一对应
        perm_index = torch.randperm(sampled_xy.shape[0])
        sampled_xy = sampled_xy[perm_index]
        masks = masks[perm_index]

        # 收集输出结果
        select_points.append(sampled_xy)
        select_masks.append(masks.unsqueeze(1))
    select_masks = torch.cat(select_masks, dim=0) if len(select_masks) > 0 else None
    return select_points, select_masks



def sample_train_points(batch_heatmaps, batch_para_masks, batch_line_masks,
                        batch_line_word_masks, batch_word_masks, batch_line2para_idx, np_per_line=2):
    """
    对生成的关键点随机采样，平均每个文本行采样两个点。
    输出的提示点列表中，每个元素包含对应图像的提示点；输出的其他掩码均为(total_num,1,1024,1024)，数量为所有提示点的数量

    :param batch_heatmaps: (bs,1,256,256)
    :param batch_para_masks: (bs,para_num,1024,1024)
    :param batch_line_masks: (bs,line_num,1024,1024)
    :param batch_word_masks: (bs,line_num,1024,1024)
    :param batch_line2para_idx: (bs,line_num), 文本行和段落的对应关系
    :param np_per_line: 每个文本行选取点的数量
    """
    batch_points = peak_selection(batch_heatmaps)
    select_points, select_line_masks, select_para_masks, select_line_word_masks, select_word_masks = [], [], [], [], []

    dev = batch_heatmaps.device

    for points, word_masks, line_word_masks, line_masks, para_masks, line2para_idx in (
            zip(batch_points, batch_word_masks, batch_line_word_masks, batch_line_masks, batch_para_masks, batch_line2para_idx)
    ):
        points = points.round().to(torch.int32)
        have_word, have_line_word, have_line, have_para = word_masks is not None, line_word_masks is not None, line_masks is not None, para_masks is not None
        if have_word:
            word_masks = word_masks.to(dev)
            h, w = word_masks.shape[-2:]
        if have_line_word:
            line_word_masks = line_word_masks.to(dev)
        if have_line:
            line_masks = line_masks.to(dev)
            h, w = line_masks.shape[-2:]
        if have_para:
            para_masks = para_masks.to(dev)
            line2para_idx = line2para_idx.to(dev)

        # 过滤掉没有交点的文本行掩码
        if have_word:
            intersection = (word_masks[:, points[:, 1], points[:, 0]]) > 0 # (nl, points_num)
        else:
            intersection = (line_masks[:, points[:, 1], points[:, 0]]) > 0
        intersec_num = torch.sum(intersection, dim=1)  # 计算每个文本行掩码与像素标签的交点数量
        keep_index = torch.nonzero(intersec_num).reshape(-1)  # 获取有交点的文本行掩码的索引
        keep_num = keep_index.numel()  # 计算有交点的文本行掩码的数量

        # 过滤无交点的部分
        intersection = intersection[keep_index]
        intersec_num = intersec_num[keep_index]
        if have_word:
            word_masks = word_masks[keep_index]
        if have_line_word:
            line_word_masks = line_word_masks[keep_index]
        if have_line:
            line_masks = line_masks[keep_index]
        if have_para:
            line2para_idx = line2para_idx[keep_index]
            para_masks = para_masks[line2para_idx]

        if keep_num == 0:  # 一个交点都没有
            sampled_xy = torch.tensor([[w / 2, h / 2]], dtype=torch.float32, device=dev)
            select_points.append(sampled_xy)
            if have_word:
                word_masks = torch.zeros((1, h, w)).to(word_masks)
                select_word_masks.append(word_masks.unsqueeze(1))
            if have_line_word:
                line_word_masks = torch.zeros((1, h, w)).to(line_word_masks)
                select_line_word_masks.append(line_word_masks.unsqueeze(1))
            if have_line:
                line_masks = torch.zeros((1, h, w)).to(line_masks)
                select_line_masks.append(line_masks.unsqueeze(1))
            if have_para:
                para_masks = torch.zeros((1, h, w)).to(para_masks)
                select_para_masks.append(para_masks.unsqueeze(1))
            continue

        # 筛选出交点的横纵坐标
        x_idx = torch.masked_select(points[:, 0], intersection)
        y_idx = torch.masked_select(points[:, 1], intersection)
        del intersection

        intersec_cumsum = torch.cumsum(intersec_num, dim=0)  # 做累加，计算出每个文本行掩码对应的交点坐标的终点索引
        start_idx = intersec_cumsum - intersec_num  # 计算出每个文本行掩码对应的交点坐标的起始索引
        np_per_line = min(np_per_line, intersec_num.min().item())  # 计算每行采样点的数量

        # 按选取点数量复制掩码
        if have_word:
            word_masks = torch.repeat_interleave(word_masks, np_per_line, dim=0)
        if have_line_word:
            line_word_masks = torch.repeat_interleave(line_word_masks, np_per_line, dim=0)
        if have_line:
            line_masks = torch.repeat_interleave(line_masks, np_per_line, dim=0)
        if have_para:
            para_masks = torch.repeat_interleave(para_masks, np_per_line, dim=0)

        # 在每个起点和终点索引之间随机选n个索引，即为采样点坐标的索引
        pos_idx = np.ravel([
            random.sample(range(start_idx[p_i].item(), intersec_cumsum[p_i].item()), np_per_line)
            for p_i in range(keep_num)
        ])

        # 根据采样点坐标的索引获取采样点坐标，并组合
        sampled_x = x_idx[pos_idx]
        sampled_y = y_idx[pos_idx]
        sampled_xy = torch.cat((sampled_x[:, None], sampled_y[:, None]), dim=1)  # (k, 2)

        # 打乱采样点顺序，对相应的mask也打乱，使其一一对应
        perm_index = torch.randperm(sampled_xy.shape[0])
        sampled_xy = sampled_xy[perm_index]
        if have_word:
            word_masks = word_masks[perm_index]
            select_word_masks.append(word_masks.unsqueeze(1))
        if have_line_word:
            line_word_masks = line_word_masks[perm_index]
            select_line_word_masks.append(line_word_masks.unsqueeze(1))
        if have_line:
            line_masks = line_masks[perm_index]
            select_line_masks.append(line_masks.unsqueeze(1))
        if have_para:
            para_masks = para_masks[perm_index]
            select_para_masks.append(para_masks.unsqueeze(1))  # (k, 1, h, w)

        # 收集输出结果
        select_points.append(sampled_xy)

    select_word_masks = torch.cat(select_word_masks, dim=0) if len(select_word_masks) > 0 else None
    select_line_word_masks = torch.cat(select_line_word_masks, dim=0) if len(select_line_word_masks) > 0 else None
    select_line_masks = torch.cat(select_line_masks, dim=0) if len(select_line_masks) > 0 else None
    select_para_masks = torch.cat(select_para_masks, dim=0) if len(select_para_masks) > 0 else None
    return select_points, select_para_masks, select_line_masks, select_line_word_masks, select_word_masks


def peak_selection(heatmaps, max_points=1000, nms_kernel_size=3, point_threshold=0.5):
    device = heatmaps.device
    batch_size = heatmaps.shape[0]
    heatmaps_nms = nms(heatmaps.detach().clone(), nms_kernel_size)
    selected_points = []
    for i in range(batch_size):
        points = torch.nonzero((heatmaps_nms[i] > point_threshold).squeeze())
        points.to(device)
        points = torch.flip(points, dims=(-1,))
        points_score_ = heatmaps_nms[i, 0, points[:, 1], points[:, 0]].flatten(0)

        idx = torch.argsort(points_score_, dim=0, descending=True)[:min(max_points, points_score_.size(0))]

        points = points[idx]
        selected_points.append(points * 4)

    return selected_points


def postprocess_masks(image, input_size, ori_size, target_size=(1024, 1024), mode='bilinear'):
    if image.shape[0] == 0:
        return None
    if len(image.shape) < 4:
        image = torch.unsqueeze(image, 1)
    image = F.interpolate(image, size=target_size, mode=mode)
    image = image[..., :input_size[0], :input_size[1]]
    image = F.interpolate(image, size=ori_size, mode=mode)
    return image


def nms(heat, kernel=3):
    pad = (kernel - 1) // 2

    hmax = F.max_pool2d(
        heat, (kernel, kernel), stride=1, padding=pad)
    keep = (hmax == heat).float()
    return heat * keep


def matrix_nms(seg_masks, scores, kernel='gaussian', sigma=2.0, sum_masks=None):
    """Matrix NMS from SOLOv2

    Args:
        seg_masks (Tensor): shape (n, h, w)
        scores (Tensor): shape (n)
        kernel (str): 'linear' or 'gaussian'
        sigma (float): std in gaussian method
        sum_masks (Tensor): the sum of seg_masks
    """
    n_samples = len(seg_masks)
    if sum_masks is None:
        sum_masks = seg_masks.sum((1, 2)).float()
    seg_masks = seg_masks.reshape(n_samples, -1).float()
    # inter
    inter_matrix = torch.mm(seg_masks, seg_masks.transpose(1, 0))
    del seg_masks
    # union
    sum_masks = sum_masks.expand(n_samples, n_samples)
    # iou
    iou_matrix = (inter_matrix / (sum_masks + sum_masks.transpose(1, 0) - inter_matrix)).triu(diagonal=1)
    # IOU compensation
    compensate_iou, _ = iou_matrix.max(0)
    compensate_iou = compensate_iou.expand(n_samples, n_samples).transpose(1, 0)
    # IOU decay
    decay_iou = iou_matrix  # no label matrix because there is only one foreground class

    if kernel == 'gaussian':
        decay_matrix = torch.exp(-1 * sigma * (decay_iou ** 2))
        compensate_matrix = torch.exp(-1 * sigma * (compensate_iou ** 2))
        decay_coef, _ = (decay_matrix / compensate_matrix).min(0)
    elif kernel == 'linear':
        decay_matrix = (1 - decay_iou) / (1 - compensate_iou)
        decay_coef, _ = decay_matrix.min(0)
    else:
        raise NotImplementedError
    updated_score = scores * decay_coef
    return updated_score


def get_iou_matrix(masks):
    """
        Args:
            masks (Tensor): shape (n, h, w)
    """
    n_samples = len(masks)
    sum_masks = masks.sum((1, 2)).float()
    masks = masks.reshape(n_samples, -1).float()
    inter_matrix = torch.mm(masks, masks.transpose(1, 0))
    # del para_masks
    sum_masks = sum_masks.expand(n_samples, n_samples)
    iou_matrix = (inter_matrix / (sum_masks + sum_masks.transpose(1, 0) - inter_matrix))

    return iou_matrix


class DisjointSet:
    """
    A disjoint set implementation from HierText
    github.com/tensorflow/models/blob/master/official/projects/unified_detector/utils/utilities.py
    """

    def __init__(self, num_elements: int):
        self._num_elements = num_elements
        self._parent = list(range(num_elements))

    def find(self, item: int) -> int:
        if self._parent[item] == item:
            return item
        else:
            self._parent[item] = self.find(self._parent[item])
            return self._parent[item]

    def union(self, i1: int, i2: int) -> None:
        r1 = self.find(i1)
        r2 = self.find(i2)
        self._parent[r1] = r2

    def to_group(self) -> List[List[int]]:
        """Return the grouping results.

        Returns:
            A list of integer lists. Each list represents the IDs belonging to the
          same group.
        """
        groups = collections.defaultdict(list)
        for i in range(self._num_elements):
            r = self.find(i)
            groups[r].append(i)
        return list(groups.values())
