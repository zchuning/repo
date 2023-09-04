import numpy as np
import pybullet as p


class DefaultCamera:
    def __init__(self):
        self.view_matrix = p.computeViewMatrixFromYawPitchRoll(
            cameraTargetPosition=[0, 0, 0],
            distance=0.9,
            yaw=90,
            pitch=-25,
            roll=0,
            upAxisIndex=2,
        )

        self.proj_matrix = (
            (0.7634194493293762, 0.0, 0.0, 0.0)
            + (0.0, 1.0, 0.0, 0.0)
            + (0.0, 0.0, -1.0000200271606445, -1.0)
            + (0.0, 0.0, -0.02000020071864128, 0.0)
        )


class RealSenseD415:
    """Default configuration with 3 RealSense RGB-D cameras."""

    # Mimic RealSense D415 RGB-D camera parameters.
    image_size = (480, 640)
    intrinsics = (450.0, 0, 320.0, 0, 450.0, 240.0, 0, 0, 1)

    # Set default camera poses.
    front_position = (1.0, 0, 0.75)
    front_rotation = (np.pi / 4, np.pi, -np.pi / 2)
    front_rotation = p.getQuaternionFromEuler(front_rotation)
    left_position = (0, 0.5, 0.75)
    left_rotation = (np.pi / 4.5, np.pi, np.pi / 4)
    left_rotation = p.getQuaternionFromEuler(left_rotation)
    right_position = (0, -0.5, 0.75)
    right_rotation = (np.pi / 4.5, np.pi, 3 * np.pi / 4)
    right_rotation = p.getQuaternionFromEuler(right_rotation)

    # Default camera configs.
    CONFIG = [
        {
            "image_size": image_size,
            "intrinsics": intrinsics,
            "position": front_position,
            "rotation": front_rotation,
            "zrange": (0.01, 10.0),
            "noise": False,
        },
        {
            "image_size": image_size,
            "intrinsics": intrinsics,
            "position": left_position,
            "rotation": left_rotation,
            "zrange": (0.01, 10.0),
            "noise": False,
        },
        {
            "image_size": image_size,
            "intrinsics": intrinsics,
            "position": right_position,
            "rotation": right_rotation,
            "zrange": (0.01, 10.0),
            "noise": False,
        },
    ]


class LowResRealSenseD415:
    # Mimic RealSense D415 RGB-D camera parameters.
    image_size = (120, 160)
    intrinsics = (450.0, 0, 320.0, 0, 450.0, 240.0, 0, 0, 1)

    # Set default camera poses.
    front_position = (1.0, 0, 0.75)
    front_rotation = (np.pi / 4, np.pi, -np.pi / 2)
    front_rotation = p.getQuaternionFromEuler(front_rotation)
    left_position = (0, 0.5, 0.75)
    left_rotation = (np.pi / 4.5, np.pi, np.pi / 4)
    left_rotation = p.getQuaternionFromEuler(left_rotation)
    right_position = (0, -0.5, 0.75)
    right_rotation = (np.pi / 4.5, np.pi, 3 * np.pi / 4)
    right_rotation = p.getQuaternionFromEuler(right_rotation)

    # Default camera configs.
    CONFIG = [
        {
            "image_size": image_size,
            "intrinsics": intrinsics,
            "position": front_position,
            "rotation": front_rotation,
            "zrange": (0.01, 10.0),
            "noise": False,
        },
        {
            "image_size": image_size,
            "intrinsics": intrinsics,
            "position": left_position,
            "rotation": left_rotation,
            "zrange": (0.01, 10.0),
            "noise": False,
        },
        {
            "image_size": image_size,
            "intrinsics": intrinsics,
            "position": right_position,
            "rotation": right_rotation,
            "zrange": (0.01, 10.0),
            "noise": False,
        },
    ]


class Oracle:
    """Top-down noiseless image used only by the oracle demonstrator."""

    # Near-orthographic projection.
    image_size = (480, 640)
    intrinsics = (63e4, 0, 320.0, 0, 63e4, 240.0, 0, 0, 1)
    position = (0.5, 0, 1000.0)
    rotation = p.getQuaternionFromEuler((0, np.pi, -np.pi / 2))

    # Camera config.
    CONFIG = {
        "image_size": image_size,
        "intrinsics": intrinsics,
        "position": position,
        "rotation": rotation,
        "zrange": (999.7, 1001.0),
        "noise": False,
    }

    @property
    def view_matrix(self):
        return compute_view_matrix(self.CONFIG)

    @property
    def proj_matrix(self):
        return compute_proj_matrix(self.CONFIG)


def compute_view_matrix(config):
    # Compute view matrix
    rotation = p.getMatrixFromQuaternion(config["rotation"])
    rotm = np.float32(rotation).reshape(3, 3)
    lookdir = np.float32([0, 0, 1]).reshape(3, 1)
    lookdir = (rotm @ lookdir).reshape(-1)
    lookat = config["position"] + lookdir
    updir = np.float32([0, -1, 0]).reshape(3, 1)
    updir = (rotm @ updir).reshape(-1)
    viewm = p.computeViewMatrix(config["position"], lookat, updir)
    return viewm


def compute_proj_matrix(config):
    # Compute rotation matrix
    focal_len = config["intrinsics"][0]
    fovh = (config["image_size"][0] / 2) / focal_len
    fovh = 180 * np.arctan(fovh) * 2 / np.pi
    aspect_ratio = config["image_size"][1] / config["image_size"][0]
    znear, zfar = config["zrange"]
    projm = p.computeProjectionMatrixFOV(fovh, aspect_ratio, znear, zfar)
    return projm

