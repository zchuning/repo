import gym
import numpy as np

from .vec_env import AsyncVecEnv


class MultitaskEnv(gym.Env):
    """
    A multitask environment that samples a new task at each reset.
    """

    def __init__(self, env_dict):
        # env_dict maps environment names to functions
        self._tasks, self._envs = zip(*[(k, v()) for (k, v) in env_dict.items()])
        self._task_ind = None
        self._env = None

        self.observation_space = self._envs[0].observation_space
        self.action_space = self._envs[0].action_space

    @property
    def num_tasks(self):
        return len(self._tasks)

    @property
    def task(self):
        return self._tasks[self._task_ind]

    @property
    def task_one_hot(self):
        # Return one-hot task vector
        task_one_hot = np.zeros(self.num_tasks, dtype=np.float32)
        task_one_hot[self._task_ind] = 1
        return task_one_hot

    def sample_task(self, round_robin=False):
        if round_robin:
            # Sample next task in round robin fashion
            if self._task_ind is None:
                # Initialize to the first task
                next_task_ind = 0
            else:
                # Select next task in round robin fashion
                next_task_ind = (self._task_ind + 1) % self.num_tasks
        else:
            # Sample next task uniformly at random
            next_task_ind = self.np_random.randint(self.num_tasks)
        return self._tasks[next_task_ind]

    def seed(self, seed):
        for env in self._envs:
            env.seed(seed)

    def step(self, action):
        obs, reward, done, info = self._env.step(action)
        info["task"] = self.task
        info["task_ind"] = self._task_ind
        return obs, reward, done, info

    def reset(self, task=None):
        if not task:
            task = self.sample_task()
        self._task_ind = self._tasks.index(task)
        self._env = self._envs[self._task_ind]
        return self._env.reset()

    def render(self, mode="rgb_array"):
        return self._env.render(mode)

    def close(self):
        for env in self._envs:
            env.close()


class MultitaskVecEnv(AsyncVecEnv):
    """
    A vectorized multitask environment that simulates all environments at once.
    """

    def __init__(self, env_dict):
        # env_dict maps environment names to functions
        self._tasks, env_fns = zip(*[(k, v) for (k, v) in env_dict.items()])
        super().__init__(env_fns)

    @property
    def tasks(self):
        return self._tasks

    @property
    def num_tasks(self):
        return len(self._tasks)

    @property
    def task_one_hots(self):
        # Return one-hot task vectors for all tasks
        return np.identity(self.num_tasks)

    def step(self, actions):
        obs, reward, done, info = super().step(actions)
        for i in range(self.num_tasks):
            info[i]["task"] = self._tasks[i]
            info[i]["task_ind"] = i
        return obs, reward, done, info
