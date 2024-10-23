from concurrent.futures import ThreadPoolExecutor
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
from utils import PSR, XYZ, Label, LabelObject, LidarPose
from nuscenes.utils.splits import create_splits_scenes

class SUSCapeConverter:
    def __init__(
        self,
        nusc_path: PathLike | str = "data/nusc",
        output_path: PathLike | str = "output/nusc_susc",
        nusc_version: str = "v1.0-mini",
        nusc_split: str = "mini_train",
    ) -> None:
        nusc_path = Path(nusc_path)
        nusc_susc_root = Path(output_path)
        nusc_susc_root.mkdir(exist_ok=True, parents=True)

        self.nusc = NuScenes(nusc_version, nusc_path.__fspath__())
        self.nusc_susc_root = nusc_susc_root
        self.split = nusc_split
        
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
            "human.pedestrian.adult": "pedestrian",
            "human.pedestrian.child": "pedestrian",
            # "human.pedestrian.wheelchair": "ignore",
            # "human.pedestrian.stroller": "ignore",
            # "human.pedestrian.personal_mobility": "ignore",
            "human.pedestrian.police_officer": "pedestrian",
            "human.pedestrian.construction_worker": "pedestrian",
            "vehicle.car": "car",
            "vehicle.motorcycle": "motorcycle",
            "vehicle.bicycle": "bicycle",
            "vehicle.bus.bendy": "bus",
            "vehicle.bus.rigid": "bus",
            "vehicle.truck": "truck",
            "vehicle.construction": "construction_vehicle",
            # "vehicle.emergency.ambulance": "ignore",
            # "vehicle.emergency.police": "ignore",
            "vehicle.trailer": "trailer",
            "movable_object.barrier": "barrier",
        }

        
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
            while sample_token:
                sample_rec = self.nusc.get("sample", sample_token)
                sample_tokens.append(sample_token)
                sample_token = sample_rec["next"]

            for sample_token in sample_tokens:
                sample_rec = self.nusc.get("sample", sample_token)

                timestamp = sample_rec["timestamp"]

                lidar_data = self.read_lidar(sample_rec)
                camera_data = self.read_cameras(sample_rec)
                object_labels = self.read_object_labels_lidar(sample_rec)

                # Process and save the data as needed
                self.process_and_save_single_frame(
                    scene_name,
                    sample_token,
                    lidar_data,
                    camera_data,
                    object_labels,
                    timestamp,
                )

                # Move to the next sample
                sample_token = sample_rec["next"]
                
        scene_tokens = [s['token'] for s in self.nusc.scene if s['name'] in create_splits_scenes()[self.split]]
        with ThreadPoolExecutor(max_workers=4) as executor:
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

    def process_and_save_calib(self, scene_name):
        scene_root = self.nusc_susc_root / f"{scene_name}"

        calib_dir = scene_root / "calib"
        calib_cameras_dir = calib_dir / "camera"

    def process_and_save_single_frame(
        self,
        scene_name,
        sample_token,
        lidar_data,
        camera_data,
        object_labels,
        timestamp_milonsecs,
    ):
        timestamp = timestamp_milonsecs / 1_000_000.0
        # print(timestamp / 1_000_000.0)
        # Implement your logic to process and save the data
        # This could involve transforming coordinates, saving to specific formats, etc.
        scene_root = self.nusc_susc_root / f"{scene_name}"
        lidar_dir = scene_root / "lidar"
        lidar_pose_dir = scene_root / "lidar_pose"
        cameras_dir = scene_root / "cameras"
        label_dir = scene_root / "label"

        lidar_dir.mkdir(exist_ok=True, parents=True)
        lidar_pose_dir.mkdir(exist_ok=True, parents=True)
        cameras_dir.mkdir(exist_ok=True, parents=True)
        label_dir.mkdir(exist_ok=True, parents=True)

        # lidar
        pc = pypcd.PointCloud.from_xyzi_points(lidar_data["points"][:, :4])
        pc.save(lidar_dir / f"{timestamp}.pcd")

        # lidar pose
        lidar_pose_rot: Quaternion = lidar_data["lidar_to_world"]["rotation"]
        lidar_pose_trans = lidar_data["lidar_to_world"]["translation"]
        lidar_pose_se3 = np.eye(4)
        lidar_pose_se3[:3, :3] = lidar_pose_rot.rotation_matrix
        lidar_pose_se3[:3, 3] = lidar_pose_trans
        with open(lidar_pose_dir / f"{timestamp}.json", "w") as f:
            content = LidarPose(
                lidarPose=list(lidar_pose_se3.flatten())
            ).model_dump_json(indent=2)
            f.write(content)

        susc_objects = []
        for obj in object_labels:
            obj_pos = XYZ(
                x=obj["position"][0], y=obj["position"][1], z=obj["position"][2]
            )
            obj_size = XYZ(x=obj["size"][1], y=obj["size"][2], z=obj["size"][0])
            rot = R.from_quat(obj["rotation"].elements, scalar_first=True)
            # obj_rot = XYZ(x=rot_q)
            susc_objects.append(
                LabelObject(
                    obj_id=obj["instance_id"],
                    obj_type=obj["category"],
                    psr=PSR(
                        position=obj_pos,
                        rotation=XYZ(x=0, y=0, z=rot.as_euler("xyz", False)[2]),
                        scale=obj_size,
                    ),
                )
            )
        with open(label_dir / f"{timestamp}.json", "w") as f:
            content = Label(frame=f"{timestamp}", objs=susc_objects).model_dump_json(
                indent=2
            )
            f.write(content)
        # cameras
        for cam in self.camera_mappings.keys():
            cam_dir = cameras_dir / self.camera_mappings[cam]
            cam_dir.mkdir(exist_ok=True)

            image: ImageFile = camera_data[cam]["image"]
            image.save(cam_dir / f"{timestamp}.jpg")


if __name__ == "__main__":
    import tyro

    converter = tyro.cli(SUSCapeConverter)
    converter.nusc_to_susc()
