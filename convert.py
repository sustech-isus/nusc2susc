from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
import json
from os import PathLike
from pathlib import Path

import numpy as np
import pypcd4 as pypcd
from nuscenes.nuscenes import NuScenes
from PIL import Image
from PIL.ImageFile import ImageFile
from pyquaternion import Quaternion
from scipy.spatial.transform import Rotation as R
from nuscenes.utils.splits import create_splits_logs
from utils import PSR, XYZ, CalibCamera, Label, LabelObject, LidarPose
from nuscenes.utils.splits import create_splits_scenes


class SUSCapeConverter:
    def __init__(
        self,
        nusc_path: PathLike | str = "data/nusc",
        output_path: PathLike | str = "output/nusc_susc_mini",
        nusc_version: str = "v1.0-mini",
        nusc_split: str = "mini_train",
        use_aligned_timestamp: bool = True,
        use_simpler_instance_id: bool = True,
        workers: int = 4,
    ) -> None:
        nusc_path = Path(nusc_path)
        nusc_susc_root = Path(output_path)
        nusc_susc_root.mkdir(exist_ok=True, parents=True)

        self.nusc = NuScenes(nusc_version, nusc_path.__fspath__())
        self.nusc_susc_root = nusc_susc_root
        self.split = nusc_split
        self.use_aligned_timestamp = use_aligned_timestamp
        self.use_simpler_instance_id = use_simpler_instance_id
        self.workers = workers

        self.lidar_name = "LIDAR_TOP"
        self.camera_mappings = {
            "CAM_FRONT": "front",
            "CAM_FRONT_LEFT": "front_left",
            "CAM_FRONT_RIGHT": "front_right",
            "CAM_BACK": "rear",
            "CAM_BACK_LEFT": "rear_left",
            "CAM_BACK_RIGHT": "rear_right",
        }
        self.category_mappings = {
            "human.pedestrian.adult": "Pedestrian",
            "human.pedestrian.child": "Pedestrian",
            # "human.pedestrian.wheelchair": "ignore",
            # "human.pedestrian.stroller": "ignore",
            # "human.pedestrian.personal_mobility": "ignore",
            "human.pedestrian.police_officer": "Pedestrian",
            "human.pedestrian.construction_worker": "Pedestrian",
            "vehicle.car": "Vehicle",
            "vehicle.motorcycle": "Bicycle",
            "vehicle.bicycle": "Bicycle",
            "vehicle.bus.bendy": "Vehicle",
            "vehicle.bus.rigid": "Vehicle",
            "vehicle.truck": "Vehicle",
            "vehicle.construction": "Vehicle",
            "vehicle.emergency.ambulance": "Vehicle",
            "vehicle.emergency.police": "Vehicle",
            "vehicle.trailer": "Vehicle",
            # "movable_object.barrier": "barrier",
        }
        self.scene2instance_mappings = defaultdict(lambda: {})

    def nusc_to_susc(self):
        def task(scene_token):
            scene_rec = self.nusc.get("scene", scene_token)
            scene_name = scene_rec["name"]
            print(f"{scene_name} converting")
            first_sample_rec = self.nusc.get("sample", scene_rec["first_sample_token"])
            # last_sample_rec = self.nusc.get("sample", scene_rec["last_sample_token"])

            # TODO convert calibration

            # collect all sample tokens in current scene
            sample_tokens = []
            sample_token = first_sample_rec["token"]

            self.process_and_save_calib(scene_name, first_sample_rec)
            while sample_token:
                sample_rec = self.nusc.get("sample", sample_token)
                sample_tokens.append(sample_token)
                sample_token = sample_rec["next"]

            for sample_token in sample_tokens:
                sample_rec = self.nusc.get("sample", sample_token)

                timestamp = sample_rec["timestamp"]
                if self.use_aligned_timestamp:
                    timestamp = round(timestamp / 500_000) * 500_000
                frame_name = str(timestamp / 1_000_000.0)

                # Process and save the data as needed
                self.process_and_save_single_frame(
                    scene_name,
                    sample_token,
                    self.read_lidar(sample_rec),
                    self.read_cameras(sample_rec),
                    self.read_object_labels_lidar(sample_rec),
                    frame_name,
                )

                # Move to the next sample
                sample_token = sample_rec["next"]

            scene_instance_mappings = {
                new_id: nusc_id
                for nusc_id, new_id in self.scene2instance_mappings[scene_name].items()
            }

            with open(
                self.nusc_susc_root / scene_name / "instaceid_mappings.json", "w"
            ) as f:
                json.dump(scene_instance_mappings, f)

        scene_tokens = [
            s["token"]
            for s in self.nusc.scene
            if s["name"] in create_splits_scenes()[self.split]
        ]
        with ThreadPoolExecutor(max_workers=self.workers) as executor:
            executor.map(task, scene_tokens)

    def read_lidar(self, sample_rec):
        lidar_token = sample_rec["data"][self.lidar_name]
        lidar_rec = self.nusc.get("sample_data", lidar_token)
        lidar_path = self.nusc.get_sample_data_path(lidar_token)

        # Read LiDAR points
        points = np.fromfile(lidar_path, dtype=np.float32).reshape((-1, 5))

        # Get LiDAR-to-vehicle transformation
        calib_rec = self.nusc.get(
            "calibrated_sensor", lidar_rec["calibrated_sensor_token"]
        )
        lidar_to_vehicle = {
            "translation": np.array(calib_rec["translation"]),
            "rotation": Quaternion(calib_rec["rotation"]),
        }

        # vehicle-to-world transformation (ego pose)
        pose_rec = self.nusc.get("ego_pose", lidar_rec["ego_pose_token"])
        vehicle_to_world = {
            "translation": np.array(pose_rec["translation"]),
            "rotation": Quaternion(pose_rec["rotation"]),
        }

        lidar_to_world = {
            "translation": vehicle_to_world["rotation"].rotate(
                lidar_to_vehicle["translation"]
            )
            + vehicle_to_world["translation"],
            "rotation": vehicle_to_world["rotation"] * lidar_to_vehicle["rotation"],
        }

        return {
            "points": points,
            "lidar_to_vehicle": lidar_to_vehicle,
            "vehicle_to_world": vehicle_to_world,
            "lidar_to_world": lidar_to_world,
        }

    def read_cameras(self, sample_rec):
        camera_data = {}
        for cam in self.camera_mappings.keys():
            cam_token = sample_rec["data"][cam]
            cam_rec = self.nusc.get("sample_data", cam_token)
            cam_path = self.nusc.get_sample_data_path(cam_token)

            image = None
            try:
                image = Image.open(cam_path)
            except IOError:
                print(f"Error opening image: {cam_path}")
                continue

            # camera pose (instead of ego pose)
            calib_rec = self.nusc.get(
                "calibrated_sensor", cam_rec["calibrated_sensor_token"]
            )
            pose = {
                "translation": calib_rec["translation"],
                "rotation": Quaternion(calib_rec["rotation"]),
            }

            intrinsic = np.array(calib_rec["camera_intrinsic"])

            camera_data[cam] = {
                "image": image,
                "pose": pose,
                "intrinsic": intrinsic,
            }

        return camera_data

    def read_object_labels_lidar(self, sample_rec):
        object_labels = []
        annotation_tokens = sample_rec["anns"]

        # LiDAR pose
        lidar_token = sample_rec["data"][self.lidar_name]
        lidar_rec = self.nusc.get("sample_data", lidar_token)

        # ego pose
        ego_pose = self.nusc.get("ego_pose", lidar_rec["ego_pose_token"])
        ego_rotation = Quaternion(ego_pose["rotation"])
        ego_translation = np.array(ego_pose["translation"])

        # LiDAR calibration
        calib_sensor = self.nusc.get(
            "calibrated_sensor", lidar_rec["calibrated_sensor_token"]
        )
        lidar_rotation = Quaternion(calib_sensor["rotation"])
        lidar_translation = np.array(calib_sensor["translation"])

        full_lidar_rotation = ego_rotation * lidar_rotation
        full_lidar_translation = (
            ego_rotation.rotate(lidar_translation) + ego_translation
        )
        for ann_token in annotation_tokens:
            ann_rec = self.nusc.get("sample_annotation", ann_token)
            category = ann_rec["category_name"]
            if category not in self.category_mappings.keys():
                continue

            box = self.nusc.get_box(ann_rec["token"])

            box.translate(-full_lidar_translation)  # type: ignore
            box.rotate(full_lidar_rotation.inverse)

            # Nuscenes object is in global coordinate
            object_labels.append(
                {
                    "category": self.category_mappings[category],
                    # "box": box,
                    "position": box.center.tolist(),
                    "size": box.wlh.tolist(),
                    "rotation": box.orientation,
                    "instance_id": ann_rec["instance_token"],
                }
            )

        return object_labels

    def process_and_save_calib(self, scene_name, first_sample):
        scene_root = self.nusc_susc_root / f"{scene_name}"

        calib_dir = scene_root / "calib"
        calib_cameras_dir = calib_dir / "camera"
        calib_cameras_dir.mkdir(parents=True, exist_ok=True)

        # Get lidar calibration data
        lidar_data = self.nusc.get("sample_data", first_sample["data"]["LIDAR_TOP"])
        lidar_calib = self.nusc.get(
            "calibrated_sensor", lidar_data["calibrated_sensor_token"]
        )

        # Get lidar-to-vehicle transform
        lidar_translation = lidar_calib["translation"]
        lidar_rotation = R.from_quat(lidar_calib["rotation"]).as_matrix()

        # Construct 4x4 vehicle-to-lidar transform
        vehicle_T_lidar = np.eye(4)
        vehicle_T_lidar[:3, :3] = lidar_rotation
        vehicle_T_lidar[:3, 3] = lidar_translation

        for nusc_cam_name, cam_name in self.camera_mappings.items():
            cam_data = self.nusc.get("sample_data", first_sample["data"][nusc_cam_name])
            cam_calib = self.nusc.get(
                "calibrated_sensor", cam_data["calibrated_sensor_token"]
            )

            intrinsic = cam_calib["camera_intrinsic"]

            cam_translation = cam_calib["translation"]
            cam_rotation = R.from_quat(cam_calib["rotation"]).as_matrix()

            # Construct 4x4 vehicle-to-camera transform
            vehicle_T_cam = np.eye(4)
            vehicle_T_cam[:3, :3] = cam_rotation
            vehicle_T_cam[:3, 3] = cam_translation

            lidar_T_cam = vehicle_T_cam @ np.linalg.inv(vehicle_T_lidar)
            coord_transform = np.array(
                [
                    [0, 0, -1],  # x = -y
                    [0, 1, 0],  # y = -z
                    [-1, 0, 0],  # z = x
                ]
            )
            transform_matrix = np.eye(4)
            transform_matrix[:3, :3] = coord_transform
            lidar_T_cam = transform_matrix @ lidar_T_cam @ np.linalg.inv(transform_matrix)
            # Create CalibCamera object
            cam_calib = CalibCamera(
                extrinsic=lidar_T_cam.flatten().tolist(),
                intrinsic=np.array(intrinsic).flatten().tolist(),
            )

            # Save calibration to file
            calib_path = calib_cameras_dir / f"{cam_name}.json"
            with open(calib_path, "w") as f:
                f.write(cam_calib.model_dump_json(indent=2))

    def process_and_save_single_frame(
        self,
        scene_name,
        sample_token,
        lidar_data,
        camera_data,
        object_labels,
        frame_name,
    ):
        # print(timestamp / 1_000_000.0)
        # Implement your logic to process and save the data
        # This could involve transforming coordinates, saving to specific formats, etc.
        scene_root = self.nusc_susc_root / f"{scene_name}"
        lidar_dir = scene_root / "lidar"
        lidar_pose_dir = scene_root / "lidar_pose"
        cameras_dir = scene_root / "camera"
        label_dir = scene_root / "label"

        lidar_dir.mkdir(exist_ok=True, parents=True)
        lidar_pose_dir.mkdir(exist_ok=True, parents=True)
        cameras_dir.mkdir(exist_ok=True, parents=True)
        label_dir.mkdir(exist_ok=True, parents=True)

        # 定义坐标转换矩阵
        coord_transform = np.array(
            [
                [-1, 0, 0],  # x = -x
                [0, -1, 0],  # y = -y
                [0, 0, 1],  # z = z
            ]
        )
        # # # 点在lidar坐标系下，只需要处理轴变换，不需要负号
        # point_transform = np.abs(coord_transform)

        # 1. lidar points
        points = lidar_data["points"][:, :4].copy()

        xyz_points = points[:, :3]
        transformed_xyz = (coord_transform @ xyz_points.T).T
        points[:, :3] = transformed_xyz

        # Save transformed point cloud
        pc = pypcd.PointCloud.from_xyzi_points(points)
        pc.save(lidar_dir / f"{frame_name}.pcd")

        # 2. lidar pose transformation
        lidar_pose_trans = lidar_data["lidar_to_world"]["translation"]
        lidar_pose_rot: Quaternion = lidar_data["lidar_to_world"]["rotation"]
        original_trans = np.array(lidar_pose_trans)
        original_rot = lidar_pose_rot.rotation_matrix

        # transform
        transformed_trans = coord_transform @ original_trans.T
        transformed_rot = coord_transform @ original_rot @ coord_transform.T

        lidar_pose_se3 = np.eye(4)
        lidar_pose_se3[:3, 3] = transformed_trans
        lidar_pose_se3[:3, :3] = transformed_rot

        with open(lidar_pose_dir / f"{frame_name}.json", "w") as f:
            content = LidarPose(
                lidarPose=list(lidar_pose_se3.flatten())
            ).model_dump_json(indent=2)
            f.write(content)

        # 3. Transform objects
        susc_objects = []
        for obj in object_labels:
            # Transform position
            original_pos = np.array([obj["position"][0], obj["position"][1], obj["position"][2]])
            transformed_pos = (coord_transform @ original_pos.T).T
            # position
            obj_pos = XYZ(
                x=transformed_pos[0], 
                y=transformed_pos[1], 
                z=transformed_pos[2]
            )
            # size
            obj_size = XYZ(x=obj["size"][1], y=obj["size"][0], z=obj["size"][2])
            # rot
            rot = R.from_quat(obj["rotation"].elements, scalar_first=True)
            original_rot_mat = rot.as_matrix()
            # Transform rotation matrix similar to how we transform lidar pose rotation
            transformed_rot_mat = coord_transform @ original_rot_mat @ coord_transform.T
            transformed_rot = R.from_matrix(transformed_rot_mat)
            
            # obj_rot = XYZ(x=rot_q)
            nusc_id = obj["instance_id"]
            if self.use_simpler_instance_id:
                if nusc_id in self.scene2instance_mappings[scene_name]:
                    instance_id = self.scene2instance_mappings[scene_name][nusc_id]
                else:
                    instance_id = f"{len(self.scene2instance_mappings[scene_name])}"
                    self.scene2instance_mappings[scene_name][nusc_id] = instance_id
            else:
                instance_id = nusc_id
            susc_objects.append(
                LabelObject(
                    obj_id=instance_id,
                    obj_type=obj["category"],
                    psr=PSR(
                        position=obj_pos,
                        rotation=XYZ(x=0, y=0, z=rot.as_euler("xyz", False)[2]),
                        scale=obj_size,
                    ),
                )
            )
        with open(label_dir / f"{frame_name}.json", "w") as f:
            content = Label(frame=f"{frame_name}", objs=susc_objects).model_dump_json(
                indent=2
            )
            f.write(content)

        # cameras
        for cam in self.camera_mappings.keys():
            cam_dir = cameras_dir / self.camera_mappings[cam]
            cam_dir.mkdir(exist_ok=True)

            image: ImageFile = camera_data[cam]["image"]
            image.save(cam_dir / f"{frame_name}.jpg")


if __name__ == "__main__":
    import tyro

    converter = tyro.cli(SUSCapeConverter)
    converter.nusc_to_susc()
