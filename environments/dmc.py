import glob
import gym
import numpy as np
import os

from gym import spaces

import local_dm_control_suite as suite
from .img_sources import make_img_source


# https://github.com/denisyarats/dmc2gym/blob/master/dmc2gym/wrappers.py
# https://github.com/facebookresearch/deep_bisim4control/blob/main/dmc2gym/wrappers.py


def _flatten_obs(obs):
    obs_pieces = []
    for v in obs.values():
        flat = np.array([v]) if np.isscalar(v) else v.ravel()
        obs_pieces.append(flat)
    return np.concatenate(obs_pieces, axis=0)


class DMCEnv(gym.Env):
    def __init__(
        self, name, pixel_obs, img_source, resource_files, total_frames, reset_bg
    ):
        domain, task = name.split("-", 1)
        self._env = suite.load(domain, task)
        self._pixel_obs = pixel_obs
        self._img_source = img_source
        self._reset_bg = reset_bg
        self._resolution = 64
        self._camera_id = 0

        if pixel_obs:
            img_shape = (3, self._resolution, self._resolution)
            self.observation_space = spaces.Box(
                low=0, high=255, shape=img_shape, dtype=np.uint8
            )
        else:
            obs_spec = self._env.observation_spec()
            obs_len = sum([int(np.prod(s.shape)) for s in obs_spec.values()])
            self.observation_space = spaces.Box(
                low=-np.inf, high=np.inf, shape=(obs_len,), dtype=np.float32
            )

        act_spec = self._env.action_spec()
        self.action_space = spaces.Box(
            low=act_spec.minimum.astype(np.float32),
            high=act_spec.maximum.astype(np.float32),
            dtype=np.float32,
        )

        # Change background
        if img_source is not None:
            img_shape = (self._resolution, self._resolution)
            self._bg_source = make_img_source(
                src_type=img_source,
                img_shape=img_shape,
                resource_files=resource_files,
                total_frames=total_frames,
                grayscale=True,
            )

    def seed(self, seed):
        self.observation_space.seed(seed)
        self.action_space.seed(seed)

    def step(self, action):
        time_step = self._env.step(action)
        obs = self._get_obs(time_step)
        reward = time_step.reward
        info = dict(discount=time_step.discount)
        return obs, reward, False, info

    def reset(self):
        if self._img_source is not None and self._reset_bg:
            self._bg_source.reset()
        time_step = self._env.reset()
        obs = self._get_obs(time_step)
        return obs

    def render(self, mode="rgb_array", height=64, width=64, camera_id=0):
        assert mode == "rgb_array", "DMC only supports rgb_array render mode"
        return self._env.physics.render(height=height, width=width, camera_id=camera_id)

    def _get_obs(self, time_step):
        if self._pixel_obs:
            obs = self.render(
                mode="rgb_array",
                height=self._resolution,
                width=self._resolution,
                camera_id=self._camera_id,
            )
            if self._img_source is not None:
                # Hardcoded mask for dmc
                mask = np.logical_and(
                    (obs[:, :, 2] > obs[:, :, 1]), (obs[:, :, 2] > obs[:, :, 0])
                )
                bg = self._bg_source.get_image()
                obs[mask] = bg[mask]
            obs = obs.transpose(2, 0, 1).copy()
        else:
            obs = _flatten_obs(time_step.observation)
        return obs
