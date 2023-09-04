import glob
import gym
import numpy as np
import os
import random
import torch
from gym.spaces import Box

from dm_control.utils.rewards import tolerance
from sapien.core import Pose
from mani_skill2.envs.pick_and_place.pick_cube import (
    PickCubeEnv,
    LiftCubeEnv,
)
from mani_skill2.envs.misc.turn_faucet import TurnFaucetEnv
from mani_skill2.utils.common import flatten_dict_space_keys, flatten_state_dict
from mani_skill2.utils.registration import register_env
from mani_skill2.utils.sapien_utils import look_at
from transforms3d.euler import euler2quat


camera_poses = {
    "PickCube": look_at([0.2, 0.4, 0.4], [0.0, 0.0, 0.3]),
    "TurnFaucet": look_at([0.2, 0.4, 0.4], [0.0, 0.0, 0.3]),
    "PushCubeMatterport": look_at([0.2, -0.4, 0.4], [0.0, 0.0, 0.3]),
    "LiftCubeMatterport": look_at([0.2, -0.4, 0.4], [0.0, 0.0, 0.3]),
    "PickCubeMatterport": look_at([0.2, -0.4, 0.4], [0.0, 0.0, 0.3]),
    "TurnFaucetMatterport": look_at([0.2, -0.4, 0.4], [0.0, 0.0, 0.3]),
}

env_kwargs = {
    "PickCube": {},
    "TurnFaucet": {"model_ids": "5021"},
    "PushCubeMatterport": {},
    "LiftCubeMatterport": {},
    "PickCubeMatterport": {},
    "TurnFaucetMatterport": {"model_ids": "5021"},
}

QPOS_LOW = np.array(
    [0.0, np.pi * 2 / 8, 0, -np.pi * 5 / 8, 0, np.pi * 7 / 8, np.pi / 4, 0.04, 0.04]
)
QPOS_HIGH = np.array(
    [0.0, np.pi * 1 / 8, 0, -np.pi * 5 / 8, 0, np.pi * 6 / 8, np.pi / 4, 0.04, 0.04]
)
BASE_POSE = Pose([-0.615, 0, 0.05])
CUBE_HALF_SIZE = 0.02
xyz = np.hstack([0.0, 0.0, CUBE_HALF_SIZE])
quat = np.array([1.0, 0.0, 0.0, 0.0])
OBJ_INIT_POSE = Pose(xyz, quat)


def load_ReplicaCAD(builder):
    paths = sorted(
        list(
            glob.glob(
                os.path.join("./data/hab2_bench_assets/stages_uncompressed/*.glb")
            )
        )
    )
    path = random.choice(paths)
    pose = Pose(q=[0.707, 0.707, 0, 0])  # y-axis up for Habitat scenes
    builder.add_visual_from_file(path, pose)
    arena = builder.build_static()
    # Add offset to workspace
    offset = np.array([-2.0, -3.0, 0.8])
    arena.set_pose(Pose(-offset))
    return arena


def load_Matterport(builder):
    paths = sorted(list(glob.glob(os.path.join("./data/matterport3d/*.glb"))))
    path = random.choice(paths)
    pose = Pose(q=[0, 0, 0, 1])  # y-axis up for Matterport scenes
    builder.add_visual_from_file(path, pose)
    arena = builder.build_static()
    # Add offset to workspace
    offset = np.array([0, 0, 0.8])
    arena.set_pose(Pose(-offset))
    return arena


@register_env("PickCubeMatterport-v0", max_episode_steps=100, override=True)
class PickCubeMatterport(PickCubeEnv):
    def _clear(self):
        # Release cached resources
        self._renderer.clear_cached_resources()
        super()._clear()

    def _initialize_task(self):
        # Fix goal position
        self.goal_pos = np.array([0.1, 0.0, 0.3])
        self.goal_site.set_pose(Pose(self.goal_pos))

    def _initialize_agent(self):
        # Set ee to be near the object
        self.agent.reset(QPOS_LOW)
        self.agent_init_pose = BASE_POSE
        self.agent.robot.set_pose(self.agent_init_pose)

    def _initialize_actors(self):
        self.obj_init_pose = OBJ_INIT_POSE
        self.obj.set_pose(self.obj_init_pose)

    def _load_actors(self):
        # Load invisible ground
        self._add_ground(render=False)
        # Load cube
        self.obj = self._build_cube(self.cube_half_size)
        # Add goal indicator
        self.goal_site = self._build_sphere_site(self.goal_thresh)
        # Load arena
        builder = self._scene.create_actor_builder()
        self.arena = load_Matterport(builder)

    def get_done(self, info, **kwargs):
        # Disable done from task completion
        return False

    def compute_dense_reward(self, info, **kwargs):
        _CUBE_HALF_SIZE = self.cube_half_size[0]
        _GOAL_THRESH = self.goal_thresh
        _LIFT_THRESH = 0.1

        tcp_to_obj = np.linalg.norm(self.obj.pose.p - self.tcp.pose.p)
        obj_to_goal_xy = np.linalg.norm(self.goal_pos[:2] - self.obj.pose.p[:2])
        obj_to_goal_z = np.abs(self.goal_pos[2] - self.obj.pose.p[2])
        gripper_dist = np.linalg.norm(
            self.agent.finger1_link.pose.p - self.agent.finger2_link.pose.p
        )

        reaching_reward = tolerance(
            tcp_to_obj,
            bounds=(0, _CUBE_HALF_SIZE),
            margin=np.linalg.norm(self.obj_init_pose.p - self.agent_init_pose.p),
            sigmoid="long_tail",
        )
        reward = reaching_reward

        # Only issue gripping reward if agent is close to object
        if tcp_to_obj < _CUBE_HALF_SIZE:
            # Encourage agent to close gripper
            gripping_reward = tolerance(
                gripper_dist,
                bounds=(0, _CUBE_HALF_SIZE * 2),
                margin=_CUBE_HALF_SIZE,
                sigmoid="linear",
            )
            reward += 0.5 * gripping_reward

        # Only issue placing reward if object is grasped
        if self.agent.check_grasp(self.obj, max_angle=30):
            # Add lifting reward
            lifting_reward = tolerance(
                obj_to_goal_z,
                bounds=(0, _GOAL_THRESH),
                margin=self.goal_pos[2] - self.obj_init_pose.p[2],
                sigmoid="linear",
            )
            reward += 5 * lifting_reward

            if np.abs(self.goal_pos[2] - self.obj.pose.p[2]) < _GOAL_THRESH:
                # Add placing reward
                placing_reward = tolerance(
                    obj_to_goal_xy,
                    bounds=(0, _GOAL_THRESH),
                    margin=np.linalg.norm(self.goal_pos[:2] - self.obj_init_pose.p[:2]),
                    sigmoid="linear",
                )
                reward += 5 * placing_reward
        return reward
    
@register_env("PushCubeMatterport-v0", max_episode_steps=100, override=True)
class PushCubeMatterport(PickCubeMatterport):
    def _initialize_task(self):
        # Fix goal position
        self.goal_pos = np.array([0.2, 0.2, 0.0])
        self.goal_site.set_pose(Pose(self.goal_pos))

    def compute_dense_reward(self, info, **kwargs):
        _CUBE_HALF_SIZE = self.cube_half_size[0]
        _GOAL_THRESH = self.goal_thresh

        tcp_to_obj = np.linalg.norm(self.obj.pose.p - self.tcp.pose.p)
        obj_to_goal = np.linalg.norm(self.goal_pos - self.obj.pose.p)
        gripper_dist = np.linalg.norm(
            self.agent.finger1_link.pose.p - self.agent.finger2_link.pose.p
        )

        reaching_reward = tolerance(
            tcp_to_obj,
            bounds=(0, _CUBE_HALF_SIZE),
            margin=np.linalg.norm(self.obj_init_pose.p - self.agent_init_pose.p),
            sigmoid="long_tail",
        )
        reward = reaching_reward

        # Only issue gripping reward if agent is close to object
        if tcp_to_obj < _CUBE_HALF_SIZE:
            # Encourage agent to close gripper
            gripping_reward = tolerance(
                gripper_dist,
                bounds=(0, _CUBE_HALF_SIZE * 2),
                margin=_CUBE_HALF_SIZE,
                sigmoid="linear",
            )
            reward += 0.5 * gripping_reward

        # Only issue pushing reward if object is grasped
        if self.agent.check_grasp(self.obj, max_angle=30):
            # Add placing reward
            pushing_reward = tolerance(
                obj_to_goal,
                bounds=(0, _GOAL_THRESH),
                margin=np.linalg.norm(self.goal_pos - self.obj_init_pose.p),
                sigmoid="linear",
            )
            reward += 5 * pushing_reward
        return reward


@register_env("LiftCubeMatterport-v0", max_episode_steps=100, override=True)
class LiftCubeMatterport(LiftCubeEnv):
    def _clear(self):
        # Release cached resources
        self._renderer.clear_cached_resources()
        super()._clear()

    def _initialize_task(self):
        # Fix goal position
        self.goal_pos = np.array([0.0, 0.0, 0.3])
        self.goal_site.set_pose(Pose(self.goal_pos))

    def _initialize_agent(self):
        # Set ee to be near the object
        self.agent.reset(QPOS_LOW)
        self.agent_init_pose = BASE_POSE
        self.agent.robot.set_pose(self.agent_init_pose)

    def _initialize_actors(self):
        self.obj_init_pose = OBJ_INIT_POSE
        self.obj.set_pose(self.obj_init_pose)

    def _load_actors(self):
        # Load invisible ground
        self._add_ground(render=False)
        # Load cube
        self.obj = self._build_cube(self.cube_half_size)
        # Add goal indicator
        self.goal_site = self._build_sphere_site(self.goal_thresh)
        # Load arena
        builder = self._scene.create_actor_builder()
        self.arena = load_Matterport(builder)

    def get_done(self, info, **kwargs):
        # Disable done from task completion
        return False

    def compute_dense_reward(self, info, **kwargs):
        _CUBE_HALF_SIZE = self.cube_half_size[0]
        _GOAL_THRESH = self.goal_thresh

        tcp_to_obj = np.linalg.norm(self.obj.pose.p - self.tcp.pose.p)
        obj_to_goal_z = np.abs(self.goal_pos[2] - self.obj.pose.p[2])
        gripper_dist = np.linalg.norm(
            self.agent.finger1_link.pose.p - self.agent.finger2_link.pose.p
        )

        reaching_reward = tolerance(
            tcp_to_obj,
            bounds=(0, _CUBE_HALF_SIZE),
            margin=np.linalg.norm(self.obj_init_pose.p - self.agent_init_pose.p),
            sigmoid="long_tail",
        )
        reward = reaching_reward

        # Only issue gripping reward if agent is close to object
        if tcp_to_obj < _CUBE_HALF_SIZE:
            # Encourage agent to close gripper
            gripping_reward = tolerance(
                gripper_dist,
                bounds=(0, _CUBE_HALF_SIZE * 2),
                margin=_CUBE_HALF_SIZE,
                sigmoid="linear",
            )
            reward += 0.5 * gripping_reward

        # Only issue placing reward if object is grasped
        if self.agent.check_grasp(self.obj, max_angle=30):
            # Add lifting reward
            lifting_reward = tolerance(
                obj_to_goal_z,
                bounds=(0, _GOAL_THRESH),
                margin=self.goal_pos[2] - self.obj_init_pose.p[2],
                sigmoid="linear",
            )
            reward += 5 * lifting_reward
        return reward
    

@register_env("TurnFaucetMatterport-v0", max_episode_steps=100, override=True)
class TurnFaucetMatterport(TurnFaucetEnv):
    def _clear(self):
        # Release cached resources
        self._renderer.clear_cached_resources()
        super()._clear()

    def _initialize_agent(self):
        # Set ee to be above the faucet
        self.agent.reset(QPOS_HIGH)
        self.agent_init_pose = BASE_POSE
        self.agent.robot.set_pose(self.agent_init_pose)
    
    def _initialize_articulations(self):
        q = euler2quat(0, 0, 0)
        p = np.array([0.1, 0.0, 0.0])
        self.faucet.set_pose(Pose(p, q))

    def _load_actors(self):
        # Add invisible ground
        self._add_ground(render=False)
        # Load arena
        builder = self._scene.create_actor_builder()
        self.arena = load_Matterport(builder)

    def get_done(self, info, **kwargs):
        # Disable done from task completion
        return False


class ManiSkillWrapper(gym.Wrapper):
    def __init__(self, env, pixel_obs):
        super().__init__(env)
        assert env.obs_mode == "rgbd"
        self._pixel_obs = pixel_obs
        if pixel_obs:
            self._observation_space = Box(
                low=0, high=255, shape=(3, 64, 64), dtype=np.uint8
            )
        else:
            # States include robot proprioception (agent) and task information (extra)
            obs_space = self.env.observation_space
            state_spaces = []
            state_spaces.extend(
                flatten_dict_space_keys(obs_space["agent"]).spaces.values()
            )
            state_spaces.extend(
                flatten_dict_space_keys(obs_space["extra"]).spaces.values()
            )
            # Concatenate all the state spaces
            state_size = sum([space.shape[0] for space in state_spaces])
            self._observation_space = Box(-np.inf, np.inf, shape=(state_size,))

    def observation(self, observation):
        if self._pixel_obs:
            obs = observation["image"]["base_camera"]["rgb"]
            obs = obs.transpose(2, 0, 1).copy()
            return obs
        else:
            # Concatenate all the states
            state = np.hstack(
                [
                    flatten_state_dict(observation["agent"]),
                    flatten_state_dict(observation["extra"]),
                ]
            )
            return state

    def reset(self, **kwargs):
        return self.observation(self.env.reset(reconfigure=True, **kwargs))

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        return self.observation(obs), reward, done, info
