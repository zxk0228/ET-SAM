import cv2
import numpy as np
import torch


def get_grid(image_shape, heatmap_scale=4):
    image_shape = np.array(image_shape)
    heatmap_shape = (image_shape / heatmap_scale).round().astype(int)
    x = np.arange(0, heatmap_shape[1], 1)
    y = np.arange(0, heatmap_shape[0], 1)
    x, y = np.meshgrid(x, y, indexing='xy')
    grid = np.stack((x, y), axis=-1)[None, ...]  # (1,h/4,w/4,2)
    return grid


def get_rect_list(word_masks):
    point_list, word_size_list, angle_list = [], [], []
    for line_word_mask in word_masks:
        for box in line_word_mask:
            box = np.array(box)
            center_point, size, angle = cv2.minAreaRect(box)
            point_list.append(center_point)
            word_size_list.append(size)
            angle_list.append(angle)
    return point_list, word_size_list, angle_list


def uniform_sampling(center_points, word_size):
    lengths = np.sum((center_points[:-1] - center_points[1:]) ** 2, axis=1) ** 0.5
    sample_nums = (3 * lengths / word_size + 2).round().astype(np.int32)  # 根据单词框的宽度计算采样点数量，至少采样起点和终点
    sample_points = np.zeros((0, 2))
    for point1, point2, sample_num in zip(center_points[:-1], center_points[1:], sample_nums):
        t = np.linspace(0, 1, sample_num)[:-1, None]
        sample_points = np.concatenate((sample_points, (1 - t) * point1[None, :] + t * point2[None, :]), axis=0)
    sample_points = np.concatenate((sample_points, center_points[-1][None, :]), axis=0)  # 增加最后一个点
    return sample_points


def sample_center_points_list_hier(word_masks, word_sizes):
    line_points_list, word_size_list = [], []
    for line_word_mask, line_word_size in zip(word_masks,word_sizes):
        for word_poly, word_size in zip(line_word_mask,line_word_size):
            point_num = len(word_poly)
            word_poly = np.array(word_poly)
            slice1 = slice(0, (point_num + 1) // 2)
            slice2 = slice(point_num-1, point_num//2-1, -1)
            center_points = (word_poly[slice1] + word_poly[slice2]) / 2
            sample_points = uniform_sampling(center_points, word_size)

            line_points_list.append(sample_points)
            word_size_list.append(word_size)
    return line_points_list, word_size_list


def sample_center_points_list(center_points_list, word_size_list):
    sample_points_list = []
    for center_points, word_size in zip(center_points_list, word_size_list):
        center_points = np.array(center_points)
        sample_points = uniform_sampling(center_points, word_size)
        if len(sample_points) >= 7:
            sample_points = sample_points[2:-2]
        elif len(sample_points) >= 3:
            sample_points = sample_points[1:-1]
        elif len(sample_points) == 2:
            sample_points = (sample_points[0:1] + sample_points[1:]) / 2
        sample_points_list.append(sample_points)
    return sample_points_list


def get_heatmap_by_center_point(point_list, word_size_list, angle_list, image_shape, heatmap_scale=4, min_sigma=1, boundary=1.5):
    points = np.array(point_list, dtype=np.float32) / heatmap_scale # (n,2)
    words_size = np.array(word_size_list, dtype=np.float32) / heatmap_scale # (n,2)
    angles = np.array(angle_list, dtype=np.float32)
    grid = get_grid(image_shape)

    distance = grid - points[:, None, None, :]  # 计算热图上每个点与候选点的坐标差, (n,h/4,w/4,2)

    angles = np.deg2rad(angles)
    angles_sin = np.sin(angles)
    angles_cos = np.cos(angles)
    val1 = np.sum(distance * np.stack([angles_cos, angles_sin], axis=-1)[:, None, None, :], axis=-1)
    val2 = np.sum(distance * np.stack([angles_sin, -angles_cos], axis=-1)[:, None, None, :], axis=-1)
    val = np.stack([val1, val2], axis=-1)

    sigma = words_size / (8 * boundary) ** 0.5
    sigma = np.maximum(sigma, np.array(min_sigma))

    heatmaps = np.exp(-np.sum((val / sigma[:, None, None, :]) ** 2 / 2, axis=-1))  # 计算每个候选点对应的热图
    heatmap = np.max(heatmaps, axis=0)  # 合并所有热图，取每个点的最大值
    return heatmap


def get_heatmap_by_center_point_gpu(point_list, word_size_list, angle_list, image_shape, heatmap_scale=4, min_sigma=1, boundary=1.5,):
    device = "cuda"

    points = torch.as_tensor(point_list, dtype=torch.float32, device=device) / heatmap_scale
    words_size = torch.as_tensor(word_size_list, dtype=torch.float32, device=device) / heatmap_scale
    angles = torch.as_tensor(angle_list, dtype=torch.float32, device=device)
    grid = torch.as_tensor(get_grid(image_shape), dtype=torch.float32, device=device)

    distance = grid - points[:, None, None, :]

    angles = torch.deg2rad(angles)
    sin, cos = torch.sin(angles), torch.cos(angles)

    val1 = (distance * torch.stack([cos, sin], dim=-1)[:, None, None]).sum(dim=-1)
    val2 = (distance * torch.stack([sin, -cos], dim=-1)[:, None, None]).sum(dim=-1)
    val = torch.stack([val1, val2], dim=-1)

    sigma = words_size / (8 * boundary) ** 0.5
    sigma = torch.clamp(sigma, min=min_sigma)

    heatmaps = torch.exp(-((val / sigma[:, None, None, :]) ** 2).sum(dim=-1) / 2)
    heatmap = heatmaps.max(dim=0).values

    return heatmap.cpu().numpy()


def get_heatmap_by_center_line(line_points_list, word_size_list, image_shape, batch_size=512, heatmap_scale=4, min_sigma=1, boundary=1.5):
    points, words_size = np.zeros((0, 2)), np.zeros((0,))
    for line_points, word_size in zip(line_points_list, word_size_list):
        points = np.append(points, line_points, axis=0)
        points_num = len(line_points)
        word_size = np.repeat(word_size, points_num, axis=0)
        words_size = np.append(words_size, word_size, axis=0)

    points = points / heatmap_scale  # 计算热图上候选点坐标
    words_size = words_size / heatmap_scale
    grid = get_grid(image_shape)

    heatmap = np.zeros(grid.shape[:3])
    for start_idx in range(0, len(points), batch_size):
        batch_slice = slice(start_idx, min(start_idx + batch_size, len(points)))

        distance = grid - points[batch_slice, None, None, :]  # 计算热图上每个点与候选点的坐标差, (n,h/4,w/4,2)

        sigma = words_size[batch_slice, None, None, None] / (8 * boundary) ** 0.5
        sigma = np.maximum(sigma, np.array(min_sigma))

        heatmaps = np.exp(-np.sum((distance / sigma) ** 2 / 2, axis=-1))
        heatmap = np.max(np.append(heatmaps, heatmap, axis=0), axis=0, keepdims=True)  # 合并所有热图，取每个点的最大值
    return heatmap[0]


def get_heatmap_by_center_line_gpu(line_points_list, word_size_list, image_shape, batch_size=512, heatmap_scale=4, min_sigma=1, boundary=1.5):
    device = "cuda"
    points, words_size = np.zeros((0, 2)), np.zeros((0,))
    for line_points, word_size in zip(line_points_list, word_size_list):
        points = np.append(points, line_points, axis=0)
        points_num = len(line_points)
        word_size = np.repeat(word_size, points_num, axis=0)
        words_size = np.append(words_size, word_size, axis=0)

    points = torch.as_tensor(points, dtype=torch.float32, device=device) / heatmap_scale
    words_size = torch.as_tensor(words_size, dtype=torch.float32, device=device) / heatmap_scale
    grid = torch.as_tensor(get_grid(image_shape), dtype=torch.float32, device=device)

    heatmap = torch.zeros(grid.shape[1:3], device=device)
    for i in range(0, len(points), batch_size):
        p, s = points[i:i+batch_size], words_size[i:i+batch_size]
        distance = grid - p[:, None, None, :]
        sigma = torch.clamp(s[:, None, None, None] / (8 * boundary) ** 0.5, min=min_sigma)

        heatmaps = torch.exp(-((distance / sigma) ** 2).sum(dim=-1) / 2)
        heatmap = torch.maximum(heatmap, heatmaps.max(dim=0).values)

    return heatmap.cpu().numpy()

