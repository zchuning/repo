from .robot_push import (
    RobotPushRedEnv,
    RobotPushGreenEnv,
    RobotPushBlueEnv,
    RobotPushMultitaskEnv,
)

from .pointmass import PointmassReachEnv

FRANKA_ENVIRONMENTS = {
    "push-red": RobotPushRedEnv,
    "push-green": RobotPushGreenEnv,
    "push-blue": RobotPushBlueEnv,
    "push-multitask": RobotPushMultitaskEnv,
}
