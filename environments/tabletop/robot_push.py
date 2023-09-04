import numpy as np
import os

from .robot import RobotEnv
from .utils.colors import COLORS


class RobotPushEnv(RobotEnv):
    def __init__(self, assets_root="assets"):
        super().__init__(assets_root=assets_root)
        self.colors = ["red", "green", "blue"]
        self.block_urdf = os.path.join(self.assets_root, "block/block.urdf")
        self.goal_urdf = os.path.join(self.assets_root, "cross/cross.urdf")
        self.block_pos = np.array(
            [[0.4, -0.3, 0.02], [0.4, 0.0, 0.02], [0.4, 0.3, 0.02]]
        )
        self.goal_pos = np.array([[0.6, -0.3, 0], [0.6, 0.0, 0], [0.6, 0.3, 0]])

    def _load_objects(self):
        self.blocks = []
        self.goals = []
        for i in range(3):
            color = COLORS[self.colors[i]] + [1]
            # Load block
            block_id = self.p.loadURDF(self.block_urdf, self.block_pos[i])
            self.p.changeVisualShape(block_id, -1, rgbaColor=color)
            self.blocks.append(block_id)
            # Load goal
            goal_id = self.p.loadURDF(
                self.goal_urdf, self.goal_pos[i], useFixedBase=True
            )
            self.p.changeVisualShape(goal_id, -1, rgbaColor=color)
            self.goals.append(goal_id)

    def _compute_reward_single_block(self, block_index):
        ee_center_pos = np.array(
            self.p.getLinkState(self.robot.body, self.robot.ee_center)[0]
        )
        ee_gripper_pos = np.array(
            self.p.getLinkState(self.robot.body, self.robot.ee_grippers[0])[0]
        )
        block_id, goal_id = self.blocks[block_index], self.goals[block_index]
        obj_pos = np.array(self.p.getBasePositionAndOrientation(block_id)[0])
        goal_pos = np.array(self.p.getBasePositionAndOrientation(goal_id)[0])

        reach_dist = np.linalg.norm(ee_center_pos - obj_pos)
        push_dist = np.linalg.norm(obj_pos - goal_pos)
        grip_dist = np.linalg.norm(ee_gripper_pos - ee_center_pos)

        # reach_rew = np.exp(-reach_dist)
        # if reach_dist < 0.03:
        #     push_rew = 10 * np.exp(-push_dist / 0.1) + np.exp(-grip_dist / 0.05)
        # else:
        #     push_rew = 0
        # reward = reach_rew + push_rew

        max_push_dist = np.linalg.norm(
            self.block_pos[block_index] - self.goal_pos[block_index]
        )

        reach_rew = -reach_dist
        if reach_dist < 0.03:
            # Incentive to close fingers when reach_dist is small
            reach_rew += np.exp(-(grip_dist**2) / 0.05)
            push_rew = 1000 * (max_push_dist - push_dist) + 1000 * (
                np.exp(-(push_dist**2) / 0.01) + np.exp(-(push_dist**2) / 0.001)
            )
            push_rew = max(push_rew, 0)
        else:
            push_rew = 0
        reward = reach_rew + push_rew
        success = float(push_dist < 0.05)
        return reward, success

    def _get_object_states(self):
        object_states = []
        for block_id in self.blocks:
            pos, rot = self.p.getBasePositionAndOrientation(block_id)
            object_states.extend([pos, rot])
        for goal_id in self.goals:
            pos = self.p.getBasePositionAndOrientation(goal_id)[0]
            object_states.append(pos)
        return np.concatenate(object_states, 0).astype(np.float32)


class RobotPushRedEnv(RobotPushEnv):
    def _compute_reward(self):
        return self._compute_reward_single_block(0)


class RobotPushGreenEnv(RobotPushEnv):
    def _compute_reward(self):
        return self._compute_reward_single_block(1)


class RobotPushBlueEnv(RobotPushEnv):
    def _compute_reward(self):
        return self._compute_reward_single_block(2)


class RobotPushMultitaskEnv(RobotPushEnv):
    def __init__(self, assets_root):
        super().__init__(assets_root)
        self._tasks = ["red", "green", "blue"]
        self._task_ind = None

    @property
    def tasks(self):
        return self._tasks

    @property
    def num_tasks(self):
        return len(self._tasks)

    def get_task(self):
        return self._tasks[self._task_ind]

    def get_task_one_hot(self):
        # Returns one-hot task vector
        task_one_hot = np.zeros(self.num_tasks, dtype=np.float32)
        task_one_hot[self._task_ind] = 1
        return task_one_hot

    def sample_next_task(self, random=False):
        if random:
            next_task_ind = self.np_random.randint(self.num_tasks)
        else:
            if self._task_ind is None:
                next_task_ind = 0
            else:
                next_task_ind = (self._task_ind + 1) % self.num_tasks
        return self._tasks[next_task_ind]

    def reset(self, task):
        self._task_ind = self._tasks.index(task)
        return super().reset()

    def step(self, action):
        obs, reward, done, info = super().step(action)
        info["task"] = self.get_task()
        info["task_ind"] = self._task_ind
        return obs, reward, done, info

    def _compute_reward(self):
        return self._compute_reward_single_block(self._task_ind)
