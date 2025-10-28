#!/usr/bin/env python3
import os
import yaml
import math
import numpy as np
import cv2
from skimage.morphology import skeletonize


def extract_centerline_image(inp_path: str, out_path: str):
    img = cv2.imread(inp_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Map image not found: {inp_path}")

    # 1) Threshold walls (black) and close gaps
    _, walls = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    walls = cv2.dilate(walls, kernel, 1)

    # 2) Free space
    free = cv2.bitwise_not(walls)

    # 3) Remove outside free space via flood fill, keep enclosed areas
    h, w = free.shape
    ff = free.copy()
    mask = np.zeros((h + 2), np.uint8)
    mask = np.zeros((h + 2, w + 2), np.uint8)
    cv2.floodFill(ff, mask, (0, 0), 64)
    enclosed = np.where(ff == 255, 255, 0).astype(np.uint8)

    # 4) Choose enclosed component with longest skeleton as the track
    n, labels = cv2.connectedComponents(enclosed)
    best = np.zeros_like(enclosed)
    best_len = -1
    for i in range(1, n):
        comp = (labels == i).astype(np.uint8)
        skel_i = skeletonize(comp.astype(bool))
        length = int(skel_i.sum())
        if length > best_len:
            best_len = length
            best = comp

    # 5) Skeletonize selected area
    skeleton = (skeletonize(best.astype(bool)) * 255).astype(np.uint8)

    # 6) Save overlay
    result = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    result[skeleton == 255] = (0, 0, 255)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    cv2.imwrite(out_path, result)
    return skeleton


def _neighbors8(v, u, h, w):
    for dv in (-1, 0, 1):
        for du in (-1, 0, 1):
            if dv == 0 and du == 0:
                continue
            vv = v + dv
            uu = u + du
            if 0 <= vv < h and 0 <= uu < w:
                yield vv, uu


def order_skeleton_points(skeleton_mask: np.ndarray):
    pts = np.column_stack(np.where(skeleton_mask > 0))  # (v, u)
    if len(pts) == 0:
        return []
    h, w = skeleton_mask.shape
    skel_set = set((int(v), int(u)) for v, u in pts)
    start = (int(pts[0][0]), int(pts[0][1]))
    path = [start]
    prev = None
    cur = start
    visited = set([start])
    max_steps = len(pts) * 2
    steps = 0
    while steps < max_steps:
        steps += 1
        nbrs = [p for p in _neighbors8(cur[0], cur[1], h, w) if p in skel_set]
        if prev is not None and len(nbrs) > 1:
            nbrs = [p for p in nbrs if p != prev] + [p for p in nbrs if p == prev]
        nxt = None
        for p in nbrs:
            if p not in visited:
                nxt = p
                break
        if nxt is None:
            break
        path.append(nxt)
        visited.add(nxt)
        prev = cur
        cur = nxt
    if len(path) < len(pts):
        remaining = [p for p in skel_set if p not in visited]
        while remaining:
            cur = path[-1]
            dists = [((p[0] - cur[0]) ** 2 + (p[1] - cur[1]) ** 2, p) for p in remaining]
            dists.sort(key=lambda x: x[0])
            nxt = dists[0][1]
            path.append(nxt)
            visited.add(nxt)
            remaining.remove(nxt)
    return [(int(u), int(v)) for v, u in path]


def load_map_info(map_yaml_path: str):
    with open(map_yaml_path, 'r') as f:
        info = yaml.safe_load(f)
    resolution = float(info['resolution'])
    origin = info['origin']
    width = None
    height = None
    image_path = info['image']
    if not os.path.isabs(image_path):
        image_path = os.path.join(os.path.dirname(map_yaml_path), image_path)
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is not None:
        height, width = img.shape
    return {
        'resolution': resolution,
        'origin': origin,
        'image_path': image_path,
        'width': width,
        'height': height,
    }


def pixel_to_world(u: int, v: int, map_info: dict):
    res = map_info['resolution']
    ox, oy, _ = map_info['origin']
    H = map_info['height']
    if H is None:
        raise RuntimeError('Map image size unknown for pixel->world conversion.')
    x = ox + (u * res)
    y = oy + ((H - v) * res)
    return x, y


def downsample_loop(points_uv, map_info, step_meters=0.2):
    if not points_uv:
        return []
    world = np.array([pixel_to_world(u, v, map_info) for (u, v) in points_uv], dtype=np.float32)
    diffs = np.diff(world, axis=0)
    seglen = np.sqrt((diffs ** 2).sum(axis=1))
    s = np.concatenate([[0.0], np.cumsum(seglen)])
    total = s[-1]
    if total < 1e-6:
        return world.tolist()
    target_s = np.arange(0.0, total, step_meters)
    sampled = []
    idx = 0
    for ts in target_s:
        while idx < len(s) - 1 and not (s[idx] <= ts <= s[idx + 1]):
            idx += 1
        if idx >= len(s) - 1:
            break
        t = 0.0 if s[idx + 1] == s[idx] else (ts - s[idx]) / (s[idx + 1] - s[idx])
        p = (1 - t) * world[idx] + t * world[idx + 1]
        sampled.append((float(p[0]), float(p[1])))
    return sampled


def extract_centerline_cli():
    inp = os.path.expanduser('~/sim_ws/src/f1tenth_gym_ros/maps/Spielberg_map.png')
    outp = os.path.expanduser('~/sim_ws/src/f1tenth_gym_ros/maps/Spielberg_map_centerline.png')
    skeleton = extract_centerline_image(inp, outp)
    print(f'Saved centerline overlay: {outp}  (skeleton pixels={int((skeleton>0).sum())})')

