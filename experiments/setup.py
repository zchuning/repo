import argparse
import glob
import json
import numpy as np
import os
import pathlib
import pipes
import random
import sys
import torch
import wandb

sys.path.append(".")
os.environ["WANDB_START_METHOD"] = "thread"

from common.logger import configure_logger
from common.utils import set_gpu_mode


class AttrDict(dict):
    __getattr__ = dict.__getitem__
    __setattr__ = dict.__setitem__


def arg_type(value):
    if isinstance(value, bool):
        return lambda x: bool(["False", "True"].index(x))
    if isinstance(value, int):
        return lambda x: float(x) if ("e" in x or "." in x) else int(x)
    if isinstance(value, pathlib.Path):
        return lambda x: pathlib.Path(x).expanduser()
    return type(value)


def parse_arguments(config):
    parser = argparse.ArgumentParser()
    for key, value in config.items():
        parser.add_argument(f"--{key}", type=arg_type(value), default=value)
    return parser.parse_args()


def get_latest_run_id(path):
    max_run_id = 0
    for path in glob.glob(os.path.join(path, "[0-9]*")):
        id = path.split(os.sep)[-1]
        if id.isdigit() and int(id) > max_run_id:
            max_run_id = int(id)
    return max_run_id


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def set_device(use_gpu, gpu_id=0):
    set_gpu_mode(use_gpu, gpu_id)


def save_cmd(base_dir):
    cmd_path = os.path.join(base_dir, "cmd.txt")
    cmd = "python " + " ".join([sys.argv[0]] + [pipes.quote(s) for s in sys.argv[1:]])
    cmd += "\n"
    print("\n" + "*" * 80)
    print("Training command:\n" + cmd)
    print("*" * 80 + "\n")
    with open(cmd_path, "w") as f:
        f.write(cmd)


def save_git(base_dir):
    git_path = os.path.join(base_dir, "git.txt")
    print("Save git commit and diff to {}".format(git_path))
    cmds = [
        "echo `git rev-parse HEAD` > {}".format(git_path),
        "git diff >> {}".format(git_path),
    ]
    os.system("\n".join(cmds))


def save_cfg(base_dir, cfg):
    cfg_path = os.path.join(base_dir, "cfg.json")
    print("Save config to {}".format(cfg_path))
    cfg_dict = vars(cfg).copy()
    for key, val in cfg_dict.items():
        if isinstance(val, pathlib.PosixPath):
            cfg_dict[key] = str(val)
    with open(cfg_path, "w") as f:
        json.dump(cfg_dict, f, indent=4)


def setup_logger(config):
    # Initialize WANDB
    wandb.init(
        project=os.environ.get("WANDB_PROJECT", "minimal-rl"),
        entity=os.environ.get("MY_WANDB_ID", None),
        group=config.env_id,
        job_type=config.algo,
        config=config,
    )

    # Configure logger
    logdir = os.path.join(
        "logdir",
        config.algo,
        config.env_id,
        config.expr_name,
        str(config.seed),
    )
    logger = configure_logger(logdir, ["stdout", "tensorboard", "wandb"])

    # Log experiment info
    save_cmd(logdir)
    save_git(logdir)
    save_cfg(logdir, config)
    return logger
