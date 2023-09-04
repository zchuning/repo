import abc
import gym
import numpy as np
import os
import pkgutil
from gym import spaces

import pybullet as p
import pybullet_utils.bullet_client as bc

from .utils.cameras import Oracle
from .utils.colors import COLORS

PYBULLET_CONNECTION_MODE = os.environ.get("PYBULLET_CONNECTION_MODE", "direct")
PYBULLET_RENDERER = os.environ.get("PYBULLET_RENDERER", "egl")

# range: (0.25, 0.75); (-0.5, 0.5)
QUADS = [
    [[0.29, -0.46], [0.46, -0.04]],
    [[0.29, 0.04], [0.46, 0.46]],
    [[0.54, 0.04], [0.71, 0.46]],
    [[0.54, -0.46], [0.71, -0.04]],
]

GOALS = [
    [0.375, -0.25],
    [0.375, 0.25],
    [0.625, 0.25],
    [0.625, -0.25],
]


class PyBulletEnv(abc.ABC, gym.Env):
    def __init__(self, assets_root, state_dim, action_dim, pixel_obs):
        self.assets_root = assets_root
        self.pixel_obs = pixel_obs
        self.sim_steps = 10
        self.render_size = (64, 64)
        self.camera = Oracle()
        self.reset_state = None

        if self.pixel_obs:
            self.observation_space = spaces.Box(
                0, 255, (3,) + self.render_size, dtype=np.uint8
            )
        else:
            self.observation_space = spaces.Box(
                -np.inf, np.inf, (state_dim,), dtype=np.float32
            )
        self.action_space = spaces.Box(-1, 1, (action_dim,), dtype=np.float32)

        # Start PyBullet
        if PYBULLET_CONNECTION_MODE == "gui":
            self.p = bc.BulletClient(p.SHARED_MEMORY)
            if self.p._client < 0:
                self.p = bc.BulletClient(p.GUI)
            self.p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
            self.p.resetDebugVisualizerCamera(
                cameraDistance=0.9,
                cameraYaw=90,
                cameraPitch=-25,
                cameraTargetPosition=[0, 0, 0],
            )
        elif PYBULLET_CONNECTION_MODE == "direct":
            self.p = bc.BulletClient(p.DIRECT)
            # Load EGL plugin for headless rendering
            self.egl_plugin = None
            if PYBULLET_RENDERER == "egl":
                egl = pkgutil.get_loader("eglRenderer")
                self.egl_plugin = self.p.loadPlugin(
                    egl.get_filename(), "_eglRendererPlugin"
                )
        else:
            raise ValueError("Unsupported PyBullet connection mode")

    def close(self):
        if PYBULLET_CONNECTION_MODE == "direct" and self.egl_plugin:
            self.p.unloadPlugin(self.egl_plugin)
        self.p.disconnect()

    def render(self, mode="rgb_array"):
        if mode != "rgb_array":
            raise NotImplementedError("Unsupported render mode")
        _, _, image, _, _ = self.p.getCameraImage(
            width=self.render_size[1],
            height=self.render_size[0],
            viewMatrix=self.camera.view_matrix,
            projectionMatrix=self.camera.proj_matrix,
            flags=p.ER_NO_SEGMENTATION_MASK,
            renderer=p.ER_BULLET_HARDWARE_OPENGL,
        )
        image_size = self.render_size + (4,)
        image = np.array(image, dtype=np.uint8).reshape(image_size)
        image = image[:, :, :3]
        return image

    def _step_simulation(self):
        for _ in range(self.sim_steps):
            self.p.stepSimulation()

    def _get_obs(self):
        if self.pixel_obs:
            return self.render().transpose(2, 0, 1).copy()
        else:
            return self._get_state()

    @abc.abstractmethod
    def _get_state(self):
        pass


class PointmassReachEnv(PyBulletEnv):
    def __init__(self, assets_root, task="train_red", pixel_obs=True):
        super().__init__(assets_root, 2, 2, pixel_obs)
        self.block_urdf = os.path.join(self.assets_root, "puck/puck.urdf")
        self.block_colors = ["red", "green", "blue"]
        self.block_scale = 1.5
        self.action_scale = 10

        self.mode, self.color = task.split("_", 1)
        if self.mode == "train":
            quadrants = [0, 1, 2]
        else:
            if self.color == "red":
                quadrants = [0, 2, 1]
            elif self.color == "green":
                quadrants = [2, 1, 0]
            elif self.color == "blue":
                quadrants = [1, 0, 2]

        self.bounds = {
            "low": [QUADS[q][0] for q in quadrants],
            "high": [QUADS[q][1] for q in quadrants],
        }
        self.goals = [GOALS[q] for q in quadrants]

    def _load_objects(self):
        self.blocks = []
        for i in range(3):
            # Load block out of view
            block_id = self.p.loadURDF(
                self.block_urdf, [-100, -100, 0], globalScaling=self.block_scale
            )
            # Change color
            block_color = COLORS[self.block_colors[i]] + [1]
            self.p.changeVisualShape(block_id, -1, rgbaColor=block_color)
            self.blocks.append(block_id)
        active_block_ind = self.block_colors.index(self.color)
        self.active_block = self.blocks[active_block_ind]
        self.active_goal = self.goals[active_block_ind]

    def _get_random_pos(self, xy_low, xy_high):
        xy_pos = np.random.uniform(xy_low, xy_high)
        pos = np.concatenate((xy_pos, [0.03]))
        return pos

    def _shuffle_objects(self):
        for i in range(3):
            if self.blocks[i] == self.active_block:
                # Randomize position of active block
                pos = self._get_random_pos(
                    self.bounds["low"][i],
                    self.bounds["high"][i],
                )
            else:
                if self.mode == "train":
                    # Set distractor blocks to goal positions
                    pos = np.concatenate((self.goals[i], [0.03]))
                else:
                    # Randomize position of distractor blocks
                    pos = self._get_random_pos(
                        self.bounds["low"][i],
                        self.bounds["high"][i],
                    )
            self.p.resetBasePositionAndOrientation(self.blocks[i], pos, [0, 0, 0, 1])

    def reset(self):
        if self.reset_state is None:
            # Reset simulation
            self.p.resetSimulation()
            self.p.setGravity(0, 0, -9.8)
            # Load scene
            self.plane = self.p.loadURDF(
                os.path.join(self.assets_root, "plane/plane.urdf"),
                [0, 0, -0.001],
            )
            self.table = self.p.loadURDF(
                os.path.join(self.assets_root, "workspace/workspace_wall.urdf"),
                [0.5, 0, 0],
            )
            # Load objects
            self._load_objects()
            # Save reset state
            self.reset_state = self.p.saveState()
        # Restore reset state
        self.p.restoreState(self.reset_state)
        # Shuffle objects
        self._shuffle_objects()
        # Step simulation
        self._step_simulation()
        return self._get_obs()

    def step(self, action):
        action = np.clip(action, -1, 1) * self.action_scale
        action = np.concatenate((action, [0]))
        pos, _ = self.p.getBasePositionAndOrientation(self.active_block)
        self.p.applyExternalForce(
            self.active_block, -1, action, pos, flags=p.WORLD_FRAME
        )
        self._step_simulation()

        reward, success = self._compute_reward()
        info = {"success": success}
        return self._get_obs(), reward, False, info

    def _compute_reward(self):
        pos = self._get_state()
        goal_dist = np.linalg.norm(pos - self.active_goal)
        reward = np.exp(-goal_dist * 10)
        success = float(goal_dist < 0.03)
        return reward, success

    def _get_state(self):
        pos, _ = self.p.getBasePositionAndOrientation(self.active_block)
        return np.array(pos)[:2]
