import numpy as np
import pybullet as p
import sys
import time

sys.path.append("..")

from environments.tabletop.pointmass import PointmassReachEnv


env = PointmassReachEnv("tabletop/assets", "train_red")
env.reset()
forward, right = 0, 0
while 1:
    reset = 0
    keys = p.getKeyboardEvents()
    for k, v in keys.items():
        if k == p.B3G_RIGHT_ARROW and (v & p.KEY_WAS_TRIGGERED):
            right = 1
        if k == p.B3G_RIGHT_ARROW and (v & p.KEY_WAS_RELEASED):
            right = 0
        if k == p.B3G_LEFT_ARROW and (v & p.KEY_WAS_TRIGGERED):
            right = -1
        if k == p.B3G_LEFT_ARROW and (v & p.KEY_WAS_RELEASED):
            right = 0
        if k == p.B3G_UP_ARROW and (v & p.KEY_WAS_TRIGGERED):
            forward = 1
        if k == p.B3G_UP_ARROW and (v & p.KEY_WAS_RELEASED):
            forward = 0
        if k == p.B3G_DOWN_ARROW and (v & p.KEY_WAS_TRIGGERED):
            forward = -1
        if k == p.B3G_DOWN_ARROW and (v & p.KEY_WAS_RELEASED):
            forward = 0
        if k == p.B3G_RETURN and (v & p.KEY_WAS_RELEASED):
            reset = 1
    if reset:
        env.reset()
    action = np.array([forward, right])
    _, reward, _, _ = env.step(action)
    time.sleep(1.0 / 240.0)
