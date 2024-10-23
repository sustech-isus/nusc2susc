from pydantic import BaseModel


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
