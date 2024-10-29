from pydantic import BaseModel
import numpy as np
from scipy.spatial.transform import Rotation as R
import cv2


class LidarPose(BaseModel):
    lidarPose: list[float]


class CalibCamera(BaseModel):
    extrinsic: list[float]
    intrinsic: list[float]


class XYZ(BaseModel):
    x: float
    y: float
    z: float


class PSR(BaseModel):
    position: XYZ
    rotation: XYZ
    scale: XYZ


class LabelObject(BaseModel):
    obj_id: str
    obj_type: str
    psr: PSR


class Label(BaseModel):
    frame: str
    objs: list[LabelObject]


def susc_project_box3d(box3d_world, calib_camera, img_shape):
    """Project 3D box corners to 2D image plane.

    Args:
        box3d_world: (8,3) array of box corners in world coordinates
        calib_camera: CalibCamera object containing extrinsic and intrinsic matrices
        img_shape: (height, width) of image

    Returns:
        box2d: (8,2) array of box corners in image coordinates
        is_valid: bool indicating if box is in front of camera
    """
    # Convert camera calibration to matrices
    extrinsic = np.array(calib_camera.extrinsic).reshape(4, 4)
    intrinsic = np.array(calib_camera.intrinsic).reshape(3, 3)

    # Transform points from world to camera coordinates
    box3d_homo = np.concatenate([box3d_world, np.ones((8, 1))], axis=1)
    box3d_camera = (np.linalg.inv(extrinsic) @ box3d_homo.T).T

    # Check if box is in front of camera
    if np.any(box3d_camera[:, 2] < 0):
        return None, False

    # Project to image plane
    box2d_homo = (intrinsic @ box3d_camera[:, :3].T).T
    box2d = box2d_homo[:, :2] / box2d_homo[:, 2:3]

    # Check if box is within image bounds
    h, w = img_shape
    if (
        np.any(box2d[:, 0] < 0)
        or np.any(box2d[:, 0] >= w)
        or np.any(box2d[:, 1] < 0)
        or np.any(box2d[:, 1] >= h)
    ):
        return None, False

    return box2d.astype(np.int32), True


def get_3d_box_corners(position, scale, rotation):
    """Get 8 corners of 3D bounding box in object coordinates."""
    # Box corners in object coordinates
    x, y, z = scale.x / 2, scale.y / 2, scale.z / 2
    corners = np.array(
        [
            [x, y, z],
            [x, y, -z],
            [x, -y, z],
            [x, -y, -z],
            [-x, y, z],
            [-x, y, -z],
            [-x, -y, z],
            [-x, -y, -z],
        ]
    )

    # Rotate corners
    r = R.from_euler("xyz", [rotation.x, rotation.y, rotation.z])
    corners = r.apply(corners)

    # Translate corners
    corners = corners + np.array([position.x, position.y, position.z])
    return corners


def draw_3d_boxes(img, obj_labels, calib_camera, lidar_pose):
    """Draw 3D boxes projected to 2D image with instance IDs."""
    img_copy = img.copy()
    h, w = img.shape[:2]

    for obj in obj_labels.objs:
        # Get box corners in object coordinates
        corners = get_3d_box_corners(obj.psr.position, obj.psr.scale, obj.psr.rotation)

        # Transform to world coordinates
        corners_homo = np.concatenate([corners, np.ones((8, 1))], axis=1)
        corners_world = (lidar_pose @ corners_homo.T).T[:, :3]

        # Project to image
        box2d, valid = susc_project_box3d(corners_world, calib_camera, (h, w))
        if not valid:
            continue

        # Draw box
        for i in range(4):
            # Draw bottom face
            cv2.line(
                img_copy, tuple(box2d[i]), tuple(box2d[(i + 1) % 4]), (0, 255, 0), 2
            )
            # Draw top face
            cv2.line(
                img_copy,
                tuple(box2d[i + 4]),
                tuple(box2d[(i + 1) % 4 + 4]),
                (0, 255, 0),
                2,
            )
            # Draw vertical lines
            cv2.line(img_copy, tuple(box2d[i]), tuple(box2d[i + 4]), (0, 255, 0), 2)

        # Add instance ID text
        center = box2d.mean(axis=0).astype(int)
        cv2.putText(
            img_copy,
            obj.obj_id,
            tuple(center),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            2,
        )

    return img_copy
