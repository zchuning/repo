import gym
import numpy as np
from copy import deepcopy
from gym import spaces

from setup import AttrDict, parse_arguments, set_seed, set_device, setup_logger
from algorithms.repo import (
    Dreamer,
    RePo,
    FinetunedRePo,
    CalibratedRePo,
)
from environments import make_env
from environments.dmc import DMCEnv
from environments.wrappers import NormalizeAction, TimeLimit, ActionRepeat, MazeWrapper


class PairedDMCEnv(DMCEnv):
    def __init__(
        self, name, pixel_obs, img_source, resource_files, total_frames, reset_bg
    ):
        super().__init__(
            name, pixel_obs, img_source, resource_files, total_frames, reset_bg
        )
        img_shape = (6, self._resolution, self._resolution)
        self.observation_space = spaces.Box(
            low=0, high=255, shape=img_shape, dtype=np.uint8
        )

    def _get_obs(self, time_step):
        if self._pixel_obs:
            src_obs = self.render(
                mode="rgb_array",
                height=self._resolution,
                width=self._resolution,
                camera_id=self._camera_id,
            )
            tgt_obs = src_obs.copy()
            if self._img_source is not None:
                # Hardcoded mask for dmc
                mask = np.logical_and(
                    (tgt_obs[:, :, 2] > tgt_obs[:, :, 1]),
                    (tgt_obs[:, :, 2] > tgt_obs[:, :, 0]),
                )
                bg = self._bg_source.get_image()
                tgt_obs[mask] = bg[mask]
            obs = np.concatenate((src_obs, tgt_obs), 2).transpose(2, 0, 1).copy()
        else:
            raise ValueError("Paired DMC only supports pixel obs")
        return obs


class PairedMazeWrapper(MazeWrapper):
    def __init__(self, env, pixel_obs, img_source, resource_files, total_frames):
        super().__init__(env, pixel_obs, img_source, resource_files, total_frames)
        img_shape = (6, self._resolution, self._resolution)
        self._observation_space = spaces.Box(
            low=0, high=255, shape=img_shape, dtype=np.uint8
        )

    def _get_pixel_obs(self):
        src_obs = self.render("rgb_array")
        tgt_obs = src_obs.copy()
        if self._img_source is not None:
            # Hardcoded mask for maze
            mask = np.logical_and(
                (tgt_obs[:, :, 2] > tgt_obs[:, :, 1]),
                (tgt_obs[:, :, 2] > tgt_obs[:, :, 0]),
            )
            bg = self._bg_source.get_image()
            tgt_obs[mask] = bg[mask]
        obs = np.concatenate((src_obs, tgt_obs), 2).transpose(2, 0, 1).copy()
        return obs


def make_paired_env(env_id, seed, pixel_obs):
    suite, task = env_id.split("-", 1)
    if suite == "dmc" or suite == "dmc_distracted":
        env = PairedDMCEnv(
            name=task,
            pixel_obs=pixel_obs,
            img_source="video" if "distracted" in suite else None,
            resource_files="../kinetics-downloader/dataset/train/driving_car/*.mp4",
            total_frames=1000,
            reset_bg=False,
        )
        env = NormalizeAction(env)
        env = TimeLimit(env, 1000)
        env = ActionRepeat(env, 2)
    else:
        env_kwargs = {}
        if task == "obstacle":
            env_kwargs["reset_locations"] = [(3, 1)]
        env = gym.make(f"maze2d-{task}-v0", **env_kwargs)
        env = PairedMazeWrapper(
            env=env,
            pixel_obs=pixel_obs,
            img_source="video" if "distracted" in suite else None,
            resource_files="../kinetics-downloader/dataset/train/driving_car/*.mp4",
            total_frames=1000,
        )

    # Set seed
    env.seed(seed)
    env.action_space.seed(seed)
    env.observation_space.seed(seed)
    return env


def get_config():
    config = AttrDict()
    config.algo = "repo_calibrate"
    config.env_id = "dmc_distracted-walker-walk"
    config.expr_name = "default"
    config.seed = 0
    config.use_gpu = True
    config.gpu_id = 0

    # Dreamer
    config.pixel_obs = True
    config.num_steps = 50000
    config.replay_size = 50000
    config.prefill = 5000
    config.train_every = 500
    config.train_steps = 100
    config.eval_every = 5000
    config.checkpoint_every = 25000
    config.log_every = 500
    config.embedding_size = 1024
    config.hidden_size = 200
    config.belief_size = 200
    config.state_size = 30
    config.dense_activation_function = "elu"
    config.cnn_activation_function = "relu"
    config.batch_size = 50
    config.chunk_size = 50
    config.horizon = 15
    config.gamma = 0.99
    config.gae_lambda = 0.95
    config.action_noise = 0.0
    config.action_ent_coef = 3e-4
    config.latent_ent_coef = 0.0
    config.free_nats = 3
    config.model_lr = 3e-4
    config.actor_lr = 8e-5
    config.value_lr = 8e-5
    config.grad_clip_norm = 100.0
    config.load_checkpoint = False
    config.load_offline = False
    config.offline_dir = "data"
    config.offline_truncate_size = 1000000
    config.save_buffer = False

    # RePo
    config.target_kl = 3.0
    config.beta_lr = 1e-4
    config.init_beta = 1e-5
    config.prior_train_steps = 5

    # Disagreement model
    config.disag_model = False
    config.ensemble_size = 6
    config.disag_lr = 3e-4
    config.disag_coef = 0.0

    # Inverse dynamics
    config.inv_dynamics = False
    config.inv_dynamics_lr = 3e-4
    config.inv_dynamics_hidden_size = 512

    # Transfer parameters
    config.source_dir = ""
    config.eval_episodes = 10
    config.calibration_buffer_size = 5000
    config.expert_calib_data = True
    config.calib_time_limit = 500
    config.calibration_mode = "simple_pair"
    config.alignment_mode = "support"
    config.aln_coef = 1.0
    config.dyn_coef = 1.0
    config.calib_coef = 1.0

    # Alignment parameters
    config.f_lr = 3e-4
    config.f_latent_size = 64
    config.f_target_kl = 0.1
    config.f_hidden_size = 256
    config.tau_lr = 5e-5
    config.u_lr = 5e-3
    config.init_u = 1e-4
    return parse_arguments(config)


if __name__ == "__main__":
    config = get_config()
    set_seed(config.seed)
    set_device(config.use_gpu, config.gpu_id)

    # Logger
    logger = setup_logger(config)

    # Environment
    env = make_env(config.env_id, config.seed, config.pixel_obs)
    eval_env = make_env(config.env_id, config.seed, config.pixel_obs)

    # Sync video distractors
    if getattr(eval_env.unwrapped, "_img_source", None) is not None:
        eval_env.unwrapped._bg_source = deepcopy(env.unwrapped._bg_source)

    # Agent
    if config.algo == "dreamer":
        algo = Dreamer(config, env, eval_env, logger)
        algo.load_checkpoint(config.source_dir)
        algo.step = 0
    elif config.algo == "repo":
        algo = RePo(config, env, eval_env, logger)
        algo.load_checkpoint(config.source_dir)
        algo.step = 0
    elif config.algo == "repo_finetune":
        algo = FinetunedRePo(config, env, eval_env, logger)
    elif config.algo == "repo_calibrate":
        calib_env = make_paired_env(config.env_id, config.seed, config.pixel_obs)
        # Sync distractors
        calib_env.unwrapped._bg_source = deepcopy(env.unwrapped._bg_source)
        algo = CalibratedRePo(config, env, eval_env, calib_env, logger)
    else:
        raise NotImplementedError("Unsupported algorithm")
    algo.train()
