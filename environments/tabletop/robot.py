import abc
import gym
import numpy as np
import os
import pkgutil
import pybullet as p
import pybullet_utils.bullet_client as bc

from gym import spaces

from .utils.cameras import DefaultCamera


PYBULLET_CONNECTION_MODE = os.environ.get("PYBULLET_CONNECTION_MODE", "direct")
PYBULLET_RENDERER = os.environ.get("PYBULLET_RENDERER", "egl")


class RobotEnv(gym.Env, abc.ABC):
    def __init__(self, assets_root="assets"):
        self.assets_root = assets_root
        self.render_size = (64, 64)
        self.camera = DefaultCamera()
        self.observation_space = spaces.Box(
            0, 255, (3,) + self.render_size, dtype=np.uint8
        )
        self.action_space = spaces.Box(-1, 1, (5,), dtype=np.float32)
        self.reset_state = None

        # Start PyBullet
        if PYBULLET_CONNECTION_MODE == "gui":
            self.p = bc.BulletClient(p.SHARED_MEMORY)
            if self.p._client < 0:
                self.p = bc.BulletClient(p.GUI)
            self.p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
            self.p.resetDebugVisualizerCamera(
                cameraDistance=self.camera.distance,
                cameraYaw=self.camera.yaw,
                cameraPitch=self.camera.pitch,
                cameraTargetPosition=self.camera.target,
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

    def reset(self):
        if self.reset_state is None:
            # Reset simulation
            self.p.resetSimulation(p.RESET_USE_DEFORMABLE_WORLD)
            self.p.setGravity(0, 0, -9.8)
            # Load scene
            self.p.loadURDF(
                os.path.join(self.assets_root, "plane/plane.urdf"), [0, 0, -0.001]
            )
            self.p.loadURDF(
                os.path.join(self.assets_root, "workspace/workspace.urdf"), [0.5, 0, 0]
            )
            # Load robot
            self.robot = FrankaPanda(self.assets_root, self.p)
            # Load objects
            self._load_objects()
            # Save state for fast reset
            self.reset_state = self.p.saveState()
        else:
            self.p.restoreState(self.reset_state)
        return self._get_obs()

    def step(self, action):
        action = np.clip(action, -1, 1)
        self.robot.apply_action(action)

        obs = self._get_obs()
        reward, success = self._compute_reward()
        info = {"success": success, "state_obs": self._get_state()}
        return obs, reward, False, info

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

    def _get_obs(self):
        return self.render().transpose(2, 0, 1).copy()

    def _get_state(self):
        robot_state = self.robot.get_joint_pos()
        object_states = self._get_object_states()
        return np.concatenate((robot_state, object_states), 0)

    @abc.abstractmethod
    def _compute_reward(self):
        pass

    @abc.abstractmethod
    def _load_objects(self):
        pass

    @abc.abstractmethod
    def _get_object_states(self):
        pass


class FrankaPanda:
    def __init__(self, assets_root, bullet_client):
        self.assets_root = assets_root
        self.p = bullet_client
        self.action_scale = np.array([0.01, 0.01, 0.01, 0.1, 0.01], dtype=np.float32)
        self.bounds = np.array([[0.25, -0.5, 0.008], [0.75, 0.5, 0.3]])
        self.sim_steps = 10

        # Load robot
        self.body = self.p.loadURDF(
            os.path.join(self.assets_root, "franka_panda/panda.urdf"),
            useFixedBase=True,
        )

        # Collect movable joints
        num_joints = self.p.getNumJoints(self.body)
        self.joints = []
        for joint in range(num_joints):
            if self.p.getJointInfo(self.body, joint)[2] != p.JOINT_FIXED:
                self.joints.append(joint)
        self.ee_angle = 6
        self.ee_grippers = [9, 10]
        self.ee_center = 11

        # Joint limits, ranges, and resting pose for null space
        self.ll = [-0.96, -1.83, -0.96, -3.14, -1.57, 0, -1.57, 0, 0]
        self.ul = [0.96, 1.83, 0.96, 0, 1.57, 3.8, 1.57, 0.04, 0.04]
        self.jr = [u - l for (u, l) in zip(self.ul, self.ll)]
        self.rp = [0, 0, 0, -0.75 * np.pi, 0, 0.75 * np.pi, 0, 0.04, 0.04]

        # Reset joint positions
        for j in range(len(self.joints)):
            self.p.resetJointState(self.body, self.joints[j], self.rp[j])

    def apply_action(self, action):
        # Scale actions
        action *= self.action_scale

        # Endeffector pose
        ee_pos, ee_orn = self.p.getLinkState(self.body, self.ee_center)[4:6]
        target_ee_pos = np.array(ee_pos) + action[:3]
        target_ee_pos = np.clip(target_ee_pos, self.bounds[0], self.bounds[1])
        # Keep endeffector facing downwards
        target_ee_orn = np.array(self.p.getEulerFromQuaternion(ee_orn))
        target_ee_orn[0] = -np.pi
        target_ee_orn = np.array(self.p.getQuaternionFromEuler(target_ee_orn))
        # Use regular IK because null space IK causes drifting
        target_joint_poses = self.p.calculateInverseKinematics(
            bodyUniqueId=self.body,
            endEffectorLinkIndex=self.ee_center,
            targetPosition=target_ee_pos,
            targetOrientation=target_ee_orn,
            maxNumIterations=100,
            residualThreshold=1e-4,
        )
        self.p.setJointMotorControlArray(
            bodyIndex=self.body,
            jointIndices=self.joints,
            controlMode=p.POSITION_CONTROL,
            targetPositions=target_joint_poses,
            targetVelocities=[0] * len(self.joints),
            forces=[200] * len(self.joints),
            positionGains=[0.4] * len(self.joints),
            velocityGains=[1] * len(self.joints),
        )

        # Endeffector angle
        ee_angle = self.p.getJointState(self.body, self.ee_angle)[0]
        target_ee_angle = np.clip(ee_angle + action[3], -1.57, 1.57)
        self.p.setJointMotorControl2(
            bodyIndex=self.body,
            jointIndex=self.ee_angle,
            controlMode=p.POSITION_CONTROL,
            targetPosition=target_ee_angle,
            targetVelocity=0,
            force=200,
            positionGain=0.8,
            velocityGain=1,
        )

        # Endeffector grippers
        # Make sure grippers are symmetric
        ee_gripper = self.p.getJointState(self.body, self.ee_grippers[0])[0]
        target_ee_gripper = np.clip(ee_gripper + action[4], 0, 0.04)
        self.p.setJointMotorControlArray(
            bodyIndex=self.body,
            jointIndices=self.ee_grippers,
            controlMode=p.POSITION_CONTROL,
            targetPositions=[target_ee_gripper] * 2,
            targetVelocities=[0] * 2,
            forces=[40] * 2,
        )

        # Simulate for multiple steps
        for _ in range(self.sim_steps):
            self.p.stepSimulation()

    def get_joint_pos(self):
        joint_states = self.p.getJointStates(self.body, self.joints)
        joint_pos = [s[0] for s in joint_states]
        return np.array(joint_pos).astype(np.float32)
