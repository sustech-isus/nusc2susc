# check converted data correctness
from os import PathLike
import pypcd4
import rerun as rr

import argparse
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation as R
from utils import Label, LidarPose


def log_susc_scene(scene_root: PathLike | str):
    scene_root = Path(scene_root)
    lidar_dir = scene_root / "lidar"
    lidar_pose_dir = scene_root / "lidar_pose"
    cam_dir = scene_root / "cameras" / "front"
    label_dir = scene_root / "label"

    timestamps = [p.stem for p in lidar_dir.glob("*.pcd")]

    for idx, ts in enumerate(timestamps):
        rr.set_time_sequence("frame_id", idx)
        lidar_path = lidar_dir / f"{ts}.pcd"
        camera_path = cam_dir / f"{ts}.jpg"
        lidar_pose_path = lidar_pose_dir / f"{ts}.json"
        label_path = label_dir / f"{ts}.json"

        # ----------------- lidar pose -----------------
        with open(lidar_pose_path, "r") as f:
            lidar_pose = LidarPose.model_validate_json(f.read()).lidarPose
        lidar_pose = np.array(lidar_pose).reshape(4, 4)

        # ----------------- lidar -----------------
        pc = pypcd4.PointCloud.from_path(lidar_path).numpy()
        pc_homogeneous = np.hstack((pc[:, :3], np.ones((pc.shape[0], 1))))
        pc_world = (lidar_pose @ pc_homogeneous.T).T[:, :3]
        rr.log("pointcloud", rr.Points3D(pc_world, colors=[255, 255, 255]))

        # ----------------- label -----------------
        with open(label_path, "r") as f:
            obj_labels = Label.model_validate_json(f.read())

        rr.log("box/", rr.Clear(recursive=True))
        for obj in obj_labels.objs:
            size = np.array([obj.psr.scale.x, obj.psr.scale.y, obj.psr.scale.z])
            centers = np.array(
                [obj.psr.position.x, obj.psr.position.y, obj.psr.position.z, 1]
            )
            center_world = (lidar_pose @ centers)[:3]
            euler_angles = np.array(
                [obj.psr.rotation.x, obj.psr.rotation.y, obj.psr.rotation.z]
            )
            combined_rotation = R.from_matrix(lidar_pose[:3, :3]) * R.from_euler(
                "xyz", euler_angles
            )
            _, _, yaw = combined_rotation.as_euler("xyz")

            rr.log(
                f"box/{obj.obj_id}",
                rr.Boxes3D(
                    sizes=size,
                    centers=center_world,
                    rotations=rr.RotationAxisAngle(axis=[0, 0, 1], angle=yaw),
                    labels=obj.obj_type,
                ),
            )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--scene-root",
        type=Path,
        default=Path("data/suscape_samples/scene-000000"),
        help="Root directory of SUSCape Scene",
    )

    rr.script_add_args(parser)
    args = parser.parse_args()

    rr.script_setup(args, "SUSCAPE-VISUALIZATION")
    log_susc_scene(args.scene_root)
    rr.script_teardown(args)


if __name__ == "__main__":
    main()
