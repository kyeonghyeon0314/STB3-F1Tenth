#!/usr/bin/env python3
"""
맵 이미지에서 중심선(centerline)을 추출하고 CSV 파일로 저장하는 스크립트
"""

import cv2
import numpy as np
import yaml
import argparse
from pathlib import Path
from scipy import ndimage


def load_map_config(yaml_path):
    """YAML 파일에서 맵 설정 로드"""
    with open(yaml_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def pixel_to_world(pixel_coords, origin, resolution):
    """
    픽셀 좌표를 월드 좌표로 변환

    Args:
        pixel_coords: (N, 2) 배열 [row, col]
        origin: [x, y, theta] 맵의 원점
        resolution: 미터/픽셀

    Returns:
        world_coords: (N, 2) 배열 [x, y]
    """
    # OpenCV 이미지는 (row, col) = (y, x) 형식
    # 월드 좌표계로 변환: x = origin_x + col * resolution
    #                    y = origin_y + (height - row) * resolution
    world_x = origin[0] + pixel_coords[:, 1] * resolution
    world_y = origin[1] + pixel_coords[:, 0] * resolution

    return np.column_stack([world_x, world_y])


def extract_skeleton(img_bin):
    """이미지를 스켈레톤화하여 중심선 추출"""
    # 흰색 영역(free space)을 스켈레톤화
    skeleton = cv2.ximgproc.thinning(img_bin)
    return skeleton


def order_skeleton_points(skeleton):
    """
    스켈레톤 점들을 순서대로 정렬 (트랙을 따라가도록)

    Args:
        skeleton: 이진화된 스켈레톤 이미지

    Returns:
        ordered_points: (N, 2) 배열 [row, col], 순서대로 정렬됨
    """
    # 스켈레톤의 모든 점 찾기
    points = np.column_stack(np.where(skeleton > 0))

    if len(points) == 0:
        raise ValueError("스켈레톤에서 점을 찾을 수 없습니다.")

    # 시작점: 가장 왼쪽 위 점
    start_idx = np.argmin(points[:, 0] + points[:, 1])

    # 가장 가까운 이웃을 따라가며 경로 생성
    ordered_points = [points[start_idx]]
    remaining_points = np.delete(points, start_idx, axis=0)

    while len(remaining_points) > 0:
        # 현재 점에서 가장 가까운 점 찾기
        current = ordered_points[-1]
        distances = np.linalg.norm(remaining_points - current, axis=1)
        nearest_idx = np.argmin(distances)

        # 너무 멀면 중단 (트랙이 끊긴 경우)
        if distances[nearest_idx] > 10:  # 10 픽셀 이상 떨어진 경우
            break

        ordered_points.append(remaining_points[nearest_idx])
        remaining_points = np.delete(remaining_points, nearest_idx, axis=0)

    return np.array(ordered_points)


def smooth_centerline(points, window_size=5):
    """
    중심선을 부드럽게 만들기 (이동 평균)

    Args:
        points: (N, 2) 배열
        window_size: 평활화 윈도우 크기

    Returns:
        smoothed_points: (N, 2) 배열
    """
    if len(points) < window_size:
        return points

    # 각 차원에 대해 이동 평균 적용
    smoothed = np.zeros_like(points, dtype=float)
    for i in range(2):
        smoothed[:, i] = ndimage.uniform_filter1d(points[:, i], size=window_size, mode='nearest')

    return smoothed


def subsample_points(points, target_spacing):
    """
    점들을 균일한 간격으로 재샘플링

    Args:
        points: (N, 2) 배열
        target_spacing: 목표 간격 (미터)

    Returns:
        subsampled_points: (M, 2) 배열
    """
    if len(points) < 2:
        return points

    # 누적 거리 계산
    distances = np.sqrt(np.sum(np.diff(points, axis=0)**2, axis=1))
    cumulative_distances = np.concatenate([[0], np.cumsum(distances)])

    # 균일한 간격으로 샘플링할 거리 계산
    total_distance = cumulative_distances[-1]
    num_samples = int(total_distance / target_spacing) + 1
    sample_distances = np.linspace(0, total_distance, num_samples)

    # 각 차원에 대해 보간
    subsampled = np.zeros((num_samples, 2))
    for i in range(2):
        subsampled[:, i] = np.interp(sample_distances, cumulative_distances, points[:, i])

    return subsampled


def main():
    parser = argparse.ArgumentParser(description='맵 이미지에서 중심선 추출 및 CSV 저장')
    parser.add_argument('--map-dir', type=str,
                        default='f1tenth_racetracks/Underground',
                        help='맵 디렉토리 경로')
    parser.add_argument('--map-name', type=str,
                        default='underground',
                        help='맵 이름 (확장자 제외)')
    parser.add_argument('--smooth', type=int, default=5,
                        help='평활화 윈도우 크기 (0이면 평활화 안 함)')
    parser.add_argument('--spacing', type=float, default=0.1,
                        help='중심선 점들 간의 간격 (미터)')
    parser.add_argument('--visualize', action='store_true',
                        help='중간 결과 시각화')

    args = parser.parse_args()

    # 경로 설정
    map_dir = Path(args.map_dir)
    yaml_path = map_dir / f"{args.map_name}.yaml"
    png_path = map_dir / f"{args.map_name}_map.png"
    output_csv = map_dir / f"{args.map_name}_centerline.csv"

    print(f"맵 설정 로드: {yaml_path}")
    config = load_map_config(yaml_path)

    print(f"맵 이미지 로드: {png_path}")
    img = cv2.imread(str(png_path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"이미지 파일을 찾을 수 없습니다: {png_path}")

    print("맵 정보:")
    print(f"  - 해상도: {config['resolution']} m/pixel")
    print(f"  - 원점: {config['origin']}")
    print(f"  - 이미지 크기: {img.shape}")

    # 이진화: 흰색 영역(free space)을 255로
    print("\n이미지 이진화...")
    _, img_bin = cv2.threshold(img, 128, 255, cv2.THRESH_BINARY)

    # 스켈레톤화
    print("스켈레톤 추출 (중심선 계산)...")
    skeleton = extract_skeleton(img_bin)

    if args.visualize:
        cv2.imwrite(str(map_dir / f"{args.map_name}_skeleton.png"), skeleton)
        print(f"  - 스켈레톤 이미지 저장: {args.map_name}_skeleton.png")

    # 점들을 순서대로 정렬
    print("스켈레톤 점들을 경로로 정렬...")
    try:
        ordered_pixel_coords = order_skeleton_points(skeleton)
        print(f"  - {len(ordered_pixel_coords)}개의 점 추출")
    except ValueError as e:
        print(f"오류: {e}")
        return

    # 평활화
    if args.smooth > 0:
        print(f"중심선 평활화 (윈도우 크기: {args.smooth})...")
        ordered_pixel_coords = smooth_centerline(ordered_pixel_coords, args.smooth)

    # 픽셀 좌표를 월드 좌표로 변환
    print("픽셀 좌표를 월드 좌표로 변환...")
    world_coords = pixel_to_world(
        ordered_pixel_coords,
        config['origin'],
        config['resolution']
    )

    # 균일한 간격으로 재샘플링
    print(f"균일한 간격({args.spacing}m)으로 재샘플링...")
    world_coords_resampled = subsample_points(world_coords, args.spacing)
    print(f"  - 재샘플링 후 점 개수: {len(world_coords_resampled)}")

    # CSV 파일로 저장
    print(f"\nCSV 파일 저장: {output_csv}")
    np.savetxt(
        output_csv,
        world_coords_resampled,
        delimiter=',',
        header='x,y',
        comments='',
        fmt='%.6f'
    )

    print("✓ 중심선 추출 완료!")
    print(f"  - 총 {len(world_coords_resampled)}개의 웨이포인트")
    print(f"  - 파일: {output_csv}")

    # 시각화
    if args.visualize:
        print("\n결과 시각화 생성...")
        vis_img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

        # 중심선을 이미지에 그리기
        for i in range(len(ordered_pixel_coords) - 1):
            pt1 = tuple(ordered_pixel_coords[i][::-1].astype(int))  # [col, row]
            pt2 = tuple(ordered_pixel_coords[i+1][::-1].astype(int))
            cv2.line(vis_img, pt1, pt2, (0, 0, 255), 2)

        # 시작점 표시
        start_pt = tuple(ordered_pixel_coords[0][::-1].astype(int))
        cv2.circle(vis_img, start_pt, 5, (0, 255, 0), -1)

        vis_output = map_dir / f"{args.map_name}_centerline_visualization.png"
        cv2.imwrite(str(vis_output), vis_img)
        print(f"  - 시각화 저장: {vis_output}")


if __name__ == "__main__":
    main()
