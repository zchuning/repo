import gym
import sys
from functools import partial

sys.path.append("./environments")

from .mt_env import MultitaskEnv, MultitaskVecEnv
from .vec_env import AsyncVecEnv
from .wrappers import (
    CastObs,
    TimeLimit,
    ActionRepeat,
    NormalizeAction,
    MetaWorldWrapper,
    FrankaWrapper,
    MazeWrapper,
)

gym.logger.set_level(40)


def make_env(env_id, seed, pixel_obs=False):
    suite, task = env_id.split("-", 1)
    if suite == "mw":
        from metaworld.envs import ALL_V2_ENVIRONMENTS_GOAL_HIDDEN

        env = ALL_V2_ENVIRONMENTS_GOAL_HIDDEN[f"{task}-v2-goal-hidden"]()
        env = MetaWorldWrapper(env, pixel_obs)
        env = TimeLimit(env, 150)
    elif suite[:3] == "dmc":
        from .dmc import DMCEnv

        img_source = None
        resource_files = None
        reset_bg = False
        if suite == "dmc_static":
            img_source = "images"
            resource_files = "../data/imagenet/*.JPEG"
        elif suite == "dmc_static_reset":
            img_source = "images"
            resource_files = "../data/imagenet/*.JPEG"
            reset_bg = True
        elif suite == "dmc_distracted":
            img_source = "video"
            resource_files = "../kinetics-downloader/dataset/train/driving_car/*.mp4"

        env = DMCEnv(
            name=task,
            pixel_obs=pixel_obs,
            img_source=img_source,
            resource_files=resource_files,
            total_frames=1000,
            reset_bg=reset_bg,
        )
        env = NormalizeAction(env)
        env = TimeLimit(env, 1000)
        env = ActionRepeat(env, 2)
    elif suite == "franka":
        from .tabletop import FRANKA_ENVIRONMENTS

        env = FRANKA_ENVIRONMENTS[task]("environments/tabletop/assets")
        env = FrankaWrapper(env, pixel_obs)
        env = TimeLimit(env, 200)
    elif suite == "pointmass":
        from .tabletop import PointmassReachEnv

        env = PointmassReachEnv("environments/tabletop/assets", task, pixel_obs)
        env = TimeLimit(env, 50)
    elif suite == "maze" or suite == "maze_distracted":
        import maze

        env_kwargs = {}
        if task == "obstacle":
            env_kwargs["reset_locations"] = [(3, 1)]
        env = gym.make(f"maze2d-{task}-v0", **env_kwargs)
        env = MazeWrapper(
            env=env,
            pixel_obs=pixel_obs,
            img_source="video" if "distracted" in suite else None,
            resource_files="../kinetics-downloader/dataset/train/driving_car/*.mp4",
            total_frames=1000,
        )
    elif suite == "maniskill":
        import mani_skill2.envs
        from maniskill import camera_poses, env_kwargs
        from .maniskill import ManiSkillWrapper

        pose = camera_poses[task]
        kwargs = env_kwargs[task]
        env = gym.make(
            f"{task}-v0",
            obs_mode="rgbd",
            control_mode="pd_ee_delta_pose",
            reward_mode="dense",
            camera_cfgs=dict(base_camera=dict(width=64, height=64, p=pose.p, q=pose.q)),
            **kwargs,
        )
        env = ManiSkillWrapper(env, pixel_obs)
    else:
        env = gym.make(env_id)

    # Cast state observations to float32
    if not pixel_obs:
        env = CastObs(env)

    # Set seed
    env.seed(seed)
    env.action_space.seed(seed)
    env.observation_space.seed(seed)
    return env


def make_vec_env(env_id, num_envs, seed, pixel_obs=False):
    env_fns = [
        partial(make_env, env_id=env_id, seed=seed + i, pixel_obs=pixel_obs)
        for i in range(num_envs)
    ]
    return AsyncVecEnv(env_fns)


def make_multitask_env(env_id, seed, pixel_obs=False, vec_env=False):
    suite, domain = env_id.split("-", 1)
    if suite == "dmc" or suite == "dmc_distracted":
        if domain == "walker":
            tasks = ["walker-walk", "walker-run", "walker-stand"]
        elif domain == "mixed":
            tasks = ["finger-turn_hard", "ball_in_cup-catch", "reacher-hard"]
        else:
            raise ValueError("Unsupported multitask DMC env")
        env_dict = {
            task: partial(make_env, f"{suite}-{task}", seed, pixel_obs)
            for task in tasks
        }
    elif suite == "franka":
        # TODO: move franka multitask env here
        pass
    elif suite == "pointmass":
        if domain == "train":
            tasks = ["train_red", "train_green", "train_blue"]
        elif domain == "test":
            tasks = ["test_red", "test_green", "test_blue"]
        else:
            raise ValueError("Unsupported multitask pointmass env")
        env_dict = {
            task: partial(make_env, f"pointmass-{task}", seed, pixel_obs)
            for task in tasks
        }

    # Make multitask environments
    if vec_env:
        env = MultitaskVecEnv(env_dict)
    else:
        env = MultitaskEnv(env_dict)
    return env
