import argparse
import gc
import os
import traceback
from concurrent import futures

# from pathlib import Path
from typing import Optional, Tuple

import cv2
import matplotlib
import numpy as np
import open3d
from matplotlib import pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg
from os import PathLike
from pathlib import Path

matplotlib.use("Agg")


use_cache = False
point_cloud_range = [-80, -80, -5, 80, 80, 3]
# 根据相机命名排序修改这个
cams_mapping = {
    "front": 0,
    "front_left": 1,
    "front_right": 2,
    "rear": 3,
    "rear_left": 4,
    "rear_right": 5,
}


def load_points(lidar_path):
    if lidar_path.endswith(".npy"):
        points = np.load(lidar_path)
    elif lidar_path.endswith(".pcd"):
        pcd = open3d.t.io.read_point_cloud(lidar_path)  # >0.15.2
        points = pcd.point["positions"].numpy()
        inst = pcd.point["intensity"].numpy()
        points = np.hstack((points, inst))
        points = points.astype(np.float32)

    else:
        points = np.fromfile(lidar_path, dtype=np.float32)

    return points


def load_image(image_path):
    return np.array(cv2.imread(image_path, cv2.IMREAD_COLOR))


def visualize_lidar_bev(
    lidar: Optional[np.ndarray] = None,
    xlim: Tuple[float, float] = (-50, 50),
    ylim: Tuple[float, float] = (-50, 50),
    radius: float = 15,
):
    size = (xlim[1] - xlim[0], ylim[1] - ylim[0])
    fig = plt.figure(figsize=size, facecolor="black")
    fig.set_dpi(10)
    ax = plt.gca()
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_aspect(1)
    ax.set_axis_off()

    if lidar is not None:
        plt.scatter(
            lidar[:, 0],
            lidar[:, 1],
            s=radius,
            c="white",
        )
    buff, shape = FigureCanvasAgg(plt.gcf()).print_to_buffer()
    image_rgba = np.frombuffer(buff, dtype=np.uint8).reshape(shape[0], shape[1], 4)
    h = image_rgba.shape[0]
    # some optional crop
    image_rgba = image_rgba[int(0.2 * h) : int(0.8 * h), ...]
    # image_rgba = cv2.resize(image_rgba, (shape[0] // 5, shape[1] // 5))
    fig.clf()
    plt.close()
    gc.collect()
    return image_rgba


def tar_pic_infos(camera_path_info, maxtri=[2, 3]):
    """
    图片合成为3行2列还是2行三列
    """
    cam_list = []

    for cam_i in camera_path_info:
        image = cv2.imread(camera_path_info[cam_i], cv2.IMREAD_COLOR)
        h, w, c = image.shape
        cam_list.append(image)

    if maxtri == [2, 3]:
        img_1 = np.hstack(
            [
                cam_list[cams_mapping["front_left"]],
                cam_list[cams_mapping["front"]],
                cam_list[cams_mapping["front_right"]],
            ]
        )
        img_2 = np.hstack(
            [
                cam_list[cams_mapping["rear_right"]],
                cam_list[cams_mapping["rear"]],
                cam_list[cams_mapping["rear_left"]],
            ]
        )
        img_a = np.vstack((img_1, img_2))
    elif maxtri == [3, 2]:
        img_1 = np.hstack(
            [cam_list[cams_mapping["front"]], cam_list[cams_mapping["rear"]]]
        )
        img_2 = np.hstack(
            [
                cam_list[cams_mapping["front_left"]],
                cam_list[cams_mapping["front_right"]],
            ]
        )
        img_3 = np.hstack(
            [
                cam_list[cams_mapping["rear_left"]],
                cam_list[cams_mapping["rear_right"]],
            ]
        )
        img_a = np.vstack((img_1, img_2, img_3))
    return img_a


def fusion_pic(true_cam_pic, true_lidar_pic, point_cloud_range):
    """
    相机和图片融合
    """
    # h_r = point_cloud_range[4] - point_cloud_range[1]
    # d_r = point_cloud_range[3] - point_cloud_range[0]
    # scale = 1
    # start_ = int((h_r - d_r) / (2 * h_r) * true_lidar_pic_.shape[1])
    # end_ = int((h_r + d_r) / (2 * h_r) * true_lidar_pic_.shape[1])

    aspect = true_lidar_pic.shape[0] / true_lidar_pic.shape[1]

    # true_lidar_pic_ = true_lidar_pic_[start_:end_, start_:end_, :]
    true_lidar_pic = cv2.resize(
        true_lidar_pic,
        (int(true_cam_pic.shape[1]), int(true_cam_pic.shape[1] * aspect)),
    )

    fused_pic = np.zeros(
        (true_cam_pic.shape[0] + true_lidar_pic.shape[0], true_cam_pic.shape[1], 3)
    )
    fused_pic[: true_cam_pic.shape[0], :, :3] = true_cam_pic[..., :3]
    fused_pic[true_cam_pic.shape[0] :, :, :3] = true_lidar_pic[..., :3]

    fused_pic = cv2.resize(
        fused_pic, (fused_pic.shape[1] // 2, fused_pic.shape[0] // 2)
    )
    # vis_save_path = os.path.join(save_path, "fusion.jpg")
    return fused_pic


def stitching_pic(lidar_path, camera_path_info, save_path):
    """
    camera_path_info = {
        front_cam = '',
        front_left_cam = '',
        front_right_cam = '',
        rear_cam = '',
        rear_left_cam = '',
        rear_right_cam = '',
    }

    """
    lidar = load_points(lidar_path)

    pic = tar_pic_infos(camera_path_info, [2, 3])

    true_lidar_pic_ = visualize_lidar_bev(
        lidar,
        xlim=(point_cloud_range[0], point_cloud_range[3]),
        ylim=(point_cloud_range[1], point_cloud_range[4]),
    )
    fused_pic = fusion_pic(pic, true_lidar_pic_, point_cloud_range)
    cv2.imwrite(save_path, fused_pic)  #

    return np.array(fused_pic, dtype=np.uint8)


def parser_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("cam_dir", type=str, help="camera dir")
    parser.add_argument("lidar_dir", type=str, help="lidar dir")
    parser.add_argument("save_dir", type=str, help="output dir")

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parser_args()
    cam_dir = args.cam_dir
    lidar_dir = args.lidar_dir
    save_dir = args.save_dir

    os.makedirs(save_dir, exist_ok=True)

    timestamps = [
        os.path.split(lidar_path)[1].split(".pcd")[0]
        for lidar_path in os.listdir(lidar_dir)
    ]
    timestamps = sorted(timestamps, key=lambda x: float(x))[:20]

    def task(timestamp: str, frame_idx: int):
        cam_paths = {}
        lidar_path = os.path.join(lidar_dir, f"{timestamp}.pcd")
        output_path = os.path.join(save_dir, f"{frame_idx:04d}.png")

        if os.path.isfile(output_path) and use_cache:
            print(f"use cached {output_path}")
            return np.array(cv2.imread(output_path), dtype=np.uint8)
        print(f"visualizing {timestamp} to frame {frame_idx}")
        for cam in sorted(os.listdir(cam_dir)):
            cam_paths[cam] = os.path.join(cam_dir, cam, f"{timestamp}.jpg")
        return stitching_pic(lidar_path, cam_paths, output_path)

    # task(timestamps[0], 0)  # debug
    frames: list[tuple] = []
    with futures.ProcessPoolExecutor(max_workers=2) as executor:
        task_futures = {
            executor.submit(task, timestamp, idx): idx
            for idx, timestamp in enumerate(timestamps)
        }
        for f in futures.as_completed(task_futures.keys()):
            frames.append((f.result(), task_futures[f]))

    print("converting to video ...")
    frames = sorted(frames, key=lambda x: x[1])
    h, w = frames[0][0].shape[:2]

    video = cv2.VideoWriter(
        os.path.join(save_dir, "output.mp4"),
        cv2.VideoWriter_fourcc(*"mp4v"),
        5,
        (w, h),
    )
    for frame_img, _ in frames:
        video.write(frame_img)

    video.release()

# python save_video.py scene-xxx/camera/ scene-xxx/lidar/ outputs
