from copy import deepcopy

from setup import AttrDict, parse_arguments, set_seed, set_device, setup_logger
from algorithms.bisim import Bisim, DeepMDP
from environments import make_env
from environments.wrappers import FrameStack


def get_config():
    config = AttrDict()
    config.algo = "bisim"
    config.env_id = "dmc_distracted-walker-walk"
    config.expr_name = "default"
    config.seed = 0
    config.use_gpu = True
    config.gpu_id = 0

    # SAC
    config.pixel_obs = True
    config.frame_stack = 3
    config.num_steps = 500000
    config.replay_size = 500000
    config.init_steps = 1000
    config.train_every = 1
    config.eval_every = 5000
    config.save_every = 25000
    config.log_every = 500
    config.gamma = 0.99
    config.batch_size = 128
    config.hidden_size = 1024
    config.bisim_coef = 0.5

    # Encoder
    config.encoder_lr = 1e-3
    config.encoder_tau = 0.05
    config.feature_size = 50

    # Decoder
    config.decoder_lr = 1e-3
    config.decoder_wd = 1e-7
    config.transition_model_type = "deterministic"

    # Actor
    config.actor_lr = 1e-3
    config.actor_update_freq = 2

    # Critic
    config.critic_lr = 1e-3
    config.critic_tau = 0.01
    config.critic_target_update_freq = 2

    # Entropy tuning
    config.init_temperature = 0.1
    config.alpha_lr = 1e-4
    config.alpha_beta = 0.5
    return parse_arguments(config)


if __name__ == "__main__":
    config = get_config()
    set_seed(config.seed)
    set_device(config.use_gpu, config.gpu_id)

    # Logger
    logger = setup_logger(config)

    # Environment
    env = make_env(config.env_id, config.seed, config.pixel_obs)
    env = FrameStack(env, config.frame_stack)
    eval_env = make_env(config.env_id, config.seed, config.pixel_obs)
    eval_env = FrameStack(eval_env, config.frame_stack)

    # Sync video distractors
    if getattr(eval_env.unwrapped, "_img_source", None) is not None:
        eval_env.unwrapped._bg_source = deepcopy(env.unwrapped._bg_source)

    # Agent
    if config.algo == "bisim":
        algo = Bisim(config, env, eval_env, logger)
    elif config.algo == "deepmdp":
        algo = DeepMDP(config, env, eval_env, logger)
    else:
        raise ValueError("Unsupported algorithm")
    algo.train()
