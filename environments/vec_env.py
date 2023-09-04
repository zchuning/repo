import cloudpickle
import multiprocessing as mp
import numpy as np

# https://github.com/DLR-RM/stable-baselines3/blob/master/stable_baselines3/common/vec_env/subproc_vec_env.py


class SyncVecEnv:
    def __init__(self, env_fns):
        self.num_envs = len(env_fns)
        self.envs = [env_fn() for env_fn in env_fns]
        self.observation_space = self.envs[0].observation_space
        self.action_space = self.envs[0].action_space

    def step(self, actions):
        results = [env.step(action) for env, action in zip(self.envs, actions)]
        obs, rews, dones, infos = map(list, zip(*results))
        for i, done in enumerate(dones):
            if done:
                infos[i]["terminal_obs"] = obs[i]
                obs[i] = self.envs[i].reset()
        return np.stack(obs), np.vstack(rews), np.vstack(dones), infos

    def seed(self, seed=None):
        return [env.seed(seed + i) for i, env in enumerate(self.envs)]

    def reset(self):
        obs = [env.reset() for env in self.envs]
        return np.stack(obs)

    def close(self):
        for env in self.envs:
            env.close()

    def render(self):
        imgs = [env.render() for env in self.envs]
        return np.stack(imgs)

    def get_attr(self, attr_name, indices=None):
        """Return attribute from vectorized environment (see base class)."""
        return [getattr(self.envs[i], attr_name) for i in self._get_indices(indices)]

    def set_attr(self, attr_name, value, indices=None):
        """Set attribute inside vectorized environments (see base class)."""
        for i in self._get_indices(indices):
            setattr(self.envs[i], attr_name, value)

    def env_method(self, method_name, *method_args, indices=None, **method_kwargs):
        """Call instance methods of vectorized environments."""
        return [
            getattr(self.envs[i], method_name)(*method_args, **method_kwargs)
            for i in self._get_indices(indices)
        ]

    def _get_indices(self, indices):
        """Convert a flexibly-typed reference to environment indices to an implied list of indices."""
        if indices is None:
            indices = range(self.num_envs)
        elif isinstance(indices, int):
            indices = [indices]
        return indices


class CloudpickleWrapper:
    """
    Uses cloudpickle to serialize contents (otherwise multiprocessing tries to use pickle)

    :param var: the variable you wish to wrap for pickling with cloudpickle
    """

    def __init__(self, var):
        self.var = var

    def __getstate__(self):
        return cloudpickle.dumps(self.var)

    def __setstate__(self, var):
        self.var = cloudpickle.loads(var)


def _worker(remote, parent_remote, env_fn_wrapper):
    parent_remote.close()
    env = env_fn_wrapper.var()
    rewards, successes = [], []
    while True:
        try:
            cmd, data = remote.recv()
            if cmd == "step":
                obs, reward, done, info = env.step(data)
                rewards.append(reward)
                successes.append(info.get("success", 0))
                if done:
                    # Save terminal observation in info then reset
                    info["terminal_obs"] = obs
                    info["episode_return"] = sum(rewards)
                    info["episode_success"] = float(sum(successes) > 0)
                    obs = env.reset()
                    rewards, successes = [], []
                remote.send((obs, reward, done, info))
            elif cmd == "seed":
                remote.send(env.seed(data))
            elif cmd == "reset":
                obs = env.reset()
                rewards, successes = [], []
                remote.send(obs)
            elif cmd == "render":
                remote.send(env.render(data))
            elif cmd == "close":
                env.close()
                remote.close()
                break
            elif cmd == "get_spaces":
                remote.send((env.observation_space, env.action_space))
            elif cmd == "env_method":
                method = getattr(env, data[0])
                remote.send(method(*data[1], **data[2]))
            elif cmd == "get_attr":
                remote.send(getattr(env, data))
            elif cmd == "set_attr":
                remote.send(setattr(env, data[0], data[1]))
            else:
                raise NotImplementedError(f"`{cmd}` is not implemented in the worker")
        except EOFError:
            break


class AsyncVecEnv:
    """
    Creates a multiprocess vectorized wrapper for multiple environments, distributing each environment to its own
    process, allowing significant speed up when the environment is computationally complex.

    For performance reasons, if your environment is not IO bound, the number of environments should not exceed the
    number of logical cores on your CPU.

    .. warning::

        Only 'forkserver' and 'spawn' start methods are thread-safe,
        which is important when TensorFlow sessions or other non thread-safe
        libraries are used in the parent (see issue #217). However, compared to
        'fork' they incur a small start-up cost and have restrictions on
        global variables. With those methods, users must wrap the code in an
        ``if __name__ == "__main__":`` block.
        For more information, see the multiprocessing documentation.

    :param env_fns: Environments to run in subprocesses
    :param start_method: method used to start the subprocesses.
           Must be one of the methods returned by multiprocessing.get_all_start_methods().
           Defaults to 'forkserver' on available platforms, and 'spawn' otherwise.
    """

    def __init__(self, env_fns, start_method=None):
        self.num_envs = len(env_fns)
        self.waiting = False
        self.closed = False

        if start_method is None:
            forkserver_available = "forkserver" in mp.get_all_start_methods()
            start_method = "forkserver" if forkserver_available else "spawn"
        ctx = mp.get_context(start_method)

        # Launch processes
        self.remotes, self.work_remotes = zip(
            *[ctx.Pipe() for _ in range(self.num_envs)]
        )
        self.processes = []
        for work_remote, remote, env_fn in zip(
            self.work_remotes, self.remotes, env_fns
        ):
            args = (work_remote, remote, CloudpickleWrapper(env_fn))
            # daemon=True: if the main process crashes, we should not cause things to hang
            process = ctx.Process(target=_worker, args=args, daemon=True)
            process.start()
            self.processes.append(process)
            work_remote.close()

        # Get observation and action spaces
        self.remotes[0].send(("get_spaces", None))
        self.observation_space, self.action_space = self.remotes[0].recv()

    def step_async(self, actions):
        for remote, action in zip(self.remotes, actions):
            remote.send(("step", action))
        self.waiting = True

    def step_wait(self):
        results = [remote.recv() for remote in self.remotes]
        self.waiting = False
        obs, rews, dones, infos = zip(*results)
        # Use vstack for rewards and dones to keep dimension
        return np.stack(obs), np.vstack(rews), np.vstack(dones), list(infos)

    def step(self, actions):
        self.step_async(actions)
        return self.step_wait()

    def seed(self, seed=None):
        for idx, remote in enumerate(self.remotes):
            remote.send(("seed", seed + idx))
        return [remote.recv() for remote in self.remotes]

    def reset(self):
        for remote in self.remotes:
            remote.send(("reset", None))
        obs = [remote.recv() for remote in self.remotes]
        return np.stack(obs)

    def close(self):
        if self.closed:
            return
        if self.waiting:
            for remote in self.remotes:
                remote.recv()
        for remote in self.remotes:
            remote.send(("close", None))
        for process in self.processes:
            process.join()
        self.closed = True

    def render(self):
        for pipe in self.remotes:
            pipe.send(("render", "rgb_array"))
        imgs = [pipe.recv() for pipe in self.remotes]
        return np.stack(imgs)

    def get_attr(self, attr_name, indices=None):
        """Return attribute from vectorized environment (see base class)."""
        target_remotes = self._get_target_remotes(indices)
        for remote in target_remotes:
            remote.send(("get_attr", attr_name))
        return [remote.recv() for remote in target_remotes]

    def set_attr(self, attr_name, value, indices=None):
        """Set attribute inside vectorized environments (see base class)."""
        target_remotes = self._get_target_remotes(indices)
        for remote in target_remotes:
            remote.send(("set_attr", (attr_name, value)))
        for remote in target_remotes:
            remote.recv()

    def env_method(self, method_name, *method_args, indices=None, **method_kwargs):
        """Call instance methods of vectorized environments."""
        target_remotes = self._get_target_remotes(indices)
        for remote in target_remotes:
            remote.send(("env_method", (method_name, method_args, method_kwargs)))
        return [remote.recv() for remote in target_remotes]

    def _get_indices(self, indices):
        """Convert a flexibly-typed reference to environment indices to an implied list of indices."""
        if indices is None:
            indices = range(self.num_envs)
        elif isinstance(indices, int):
            indices = [indices]
        return indices

    def _get_target_remotes(self, indices):
        """Get the connection object to communicate with the wanted envs that are in subprocesses."""
        indices = self._get_indices(indices)
        return [self.remotes[i] for i in indices]
