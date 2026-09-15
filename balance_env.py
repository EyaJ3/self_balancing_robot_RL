
"""PyBullet reinforcement-learning environment for the two-wheel robot."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np
import pybullet as p
import pybullet_data
from gymnasium import spaces


class TwoWheelBalanceEnv(gym.Env[np.ndarray, np.ndarray]):
    """Train the robot to remain upright using wheel torque control."""

    metadata = {"render_modes": ["human", "none"], "render_fps": 240}

    def __init__(
        self,
        render_mode: str | None = None,
        episode_seconds: float = 10.0,
        frame_skip: int = 4,
        randomize: bool = False,
        observation_mode: str = "full",
        velocity_penalty: float = 0.001,
        position_penalty: float = 0.0,
    ) -> None:
        super().__init__()
        if render_mode not in (None, "human"):
            raise ValueError("render_mode must be None or 'human'")
        self.render_mode = render_mode
        self.episode_seconds = episode_seconds
        self.frame_skip = frame_skip
        # Keep Bullet's stable physics timestep independent of controller rate.
        self.dt = 1.0 / 240.0
        self.control_dt = self.dt * self.frame_skip
        self.randomize = randomize
        self.velocity_penalty = float(velocity_penalty)
        self.position_penalty = float(position_penalty)
        if observation_mode not in (
            "full",
            "hardware",
            "hardware_accel",
            "hardware_stacked",
            "hardware_encoder",
        ):
            raise ValueError(
                "observation_mode must be 'full', 'hardware', "
                "'hardware_accel', or 'hardware_stacked'"
                " or 'hardware_encoder'"
            )
        self.observation_mode = observation_mode
        self.max_steps = int(episode_seconds / (self.dt * frame_skip))
        self.max_tilt = math.radians(35.0)
        # This is motor torque in N*m, not the velocity-controller force cap
        # used by simulate_balance.py.
        self.max_action_torque = 0.315
        self.torque_scale = 1.0
        self.left_torque_scale = 1.0
        self.right_torque_scale = 1.0
        self.left_motor_deadband = 0.0
        self.right_motor_deadband = 0.0
        self.motor_delay_steps = 0
        self.action_buffer: list[tuple[float, float]] = []
        self.pitch_sensor_bias = 0.0
        self.gyro_sensor_bias = 0.0
        self.pitch_sensor_noise = 0.0
        self.gyro_sensor_noise = 0.0
        self.acceleration_sensor_noise = 0.0
        self.initial_tilt_limit = math.radians(7.0 if randomize else 2.3)
        self.urdf_path = Path(__file__).with_name("self_balancing_robot.urdf")

        if self.observation_mode in (
            "hardware",
            "hardware_accel",
            "hardware_stacked",
            "hardware_encoder",
        ):
            # Encoder mode: [pitch, pitch rate, yaw rate, wheel speeds, command].
            # The gyro channel is clipped to this range in _state().
            low = [-math.pi, -5.0, -1.0]
            high = [math.pi, 5.0, 1.0]
            if observation_mode == "hardware_accel":
                low.append(-5.0)
                high.append(5.0)
            if observation_mode == "hardware_stacked":
                low *= 4
                high *= 4
            if observation_mode == "hardware_encoder":
                low = [-math.pi, -5.0, -5.0, -100.0, -100.0, -1.0]
                high = [math.pi, 5.0, 5.0, 100.0, 100.0, 1.0]
            self.observation_space = spaces.Box(
                low=np.asarray(low, dtype=np.float32),
                high=np.asarray(high, dtype=np.float32),
                dtype=np.float32,
            )
        else:
            # [pitch, pitch rate, x velocity, left wheel speed, right wheel speed, x]
            self.observation_space = spaces.Box(
                low=np.array(
                    [-math.pi, -50.0, -10.0, -100.0, -100.0, -5.0],
                    dtype=np.float32,
                ),
                high=np.array(
                    [math.pi, 50.0, 10.0, 100.0, 100.0, 5.0],
                    dtype=np.float32,
                ),
                dtype=np.float32,
            )
        # Independent wheel commands let the policy compensate for asymmetric
        # motor deadbands while still learning the shared balancing behavior.
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(2,), dtype=np.float32
        )

        self.client_id = -1
        self.robot_id = -1
        self.left_joint = -1
        self.right_joint = -1
        self.steps = 0
        self.command_state = 0.0
        self.acceleration_state = 0.0
        self.previous_x_velocity = 0.0
        self.hardware_observation_history: list[float] = []

    def _connect(self) -> None:
        if self.client_id >= 0:
            return
        connection = p.GUI if self.render_mode == "human" else p.DIRECT
        self.client_id = p.connect(connection)
        p.setAdditionalSearchPath(pybullet_data.getDataPath(), physicsClientId=self.client_id)
        p.setGravity(0.0, 0.0, -9.81, physicsClientId=self.client_id)
        p.setTimeStep(self.dt, physicsClientId=self.client_id)

    def _find_wheels(self) -> None:
        joints = {
            p.getJointInfo(self.robot_id, index, physicsClientId=self.client_id)[1].decode()
            : index
            for index in range(p.getNumJoints(self.robot_id, physicsClientId=self.client_id))
        }
        try:
            self.left_joint = joints["left_wheel_joint"]
            self.right_joint = joints["right_wheel_joint"]
        except KeyError as exc:
            raise RuntimeError("The URDF must define both wheel joints") from exc

        for joint in (self.left_joint, self.right_joint):
            p.setJointMotorControl2(
                self.robot_id,
                joint,
                p.VELOCITY_CONTROL,
                force=0.0,
                physicsClientId=self.client_id,
            )

    def _physical_state(
        self,
    ) -> tuple[float, float, float, float, float, float, float]:
        position, orientation = p.getBasePositionAndOrientation(
            self.robot_id, physicsClientId=self.client_id
        )
        linear_velocity, angular_velocity = p.getBaseVelocity(
            self.robot_id, physicsClientId=self.client_id
        )
        # The wheel axle is the Y axis, so forward/backward pitch is Euler Y.
        pitch = p.getEulerFromQuaternion(orientation)[1]
        pitch_rate = angular_velocity[1]
        yaw_rate = angular_velocity[2]
        left_speed = p.getJointState(
            self.robot_id, self.left_joint, physicsClientId=self.client_id
        )[1]
        right_speed = p.getJointState(
            self.robot_id, self.right_joint, physicsClientId=self.client_id
        )[1]
        return (
            pitch,
            pitch_rate,
            yaw_rate,
            linear_velocity[0],
            left_speed,
            right_speed,
            position[0],
        )

    def _state(self) -> np.ndarray:
        pitch, pitch_rate, yaw_rate, x_velocity, left_speed, right_speed, position_x = (
            self._physical_state()
        )
        if self.randomize:
            pitch += self.pitch_sensor_bias + float(
                self.np_random.normal(0.0, self.pitch_sensor_noise)
            )
            pitch_rate += self.gyro_sensor_bias + float(
                self.np_random.normal(0.0, self.gyro_sensor_noise)
            )
            yaw_rate += float(
                self.np_random.normal(0.0, self.gyro_sensor_noise)
            )
        if self.observation_mode in (
            "hardware",
            "hardware_accel",
            "hardware_stacked",
            "hardware_encoder",
        ):
            gyro_observation = float(
                np.clip(math.degrees(pitch_rate) / 100.0, -5.0, 5.0)
            )
            values = [
                pitch,
                gyro_observation,
                self.command_state,
            ]
            if self.observation_mode == "hardware_accel":
                values.append(self.acceleration_state)
            if self.observation_mode == "hardware_encoder":
                values = [
                    pitch,
                    gyro_observation,
                    float(np.clip(math.degrees(yaw_rate) / 100.0, -5.0, 5.0)),
                    left_speed,
                    right_speed,
                    self.command_state,
                ]
            if self.observation_mode == "hardware_stacked":
                if not self.hardware_observation_history:
                    self.hardware_observation_history = values * 4
                else:
                    self.hardware_observation_history.extend(values)
                self.hardware_observation_history = (
                    self.hardware_observation_history[-12:]
                )
                values = self.hardware_observation_history
        else:
            values = [
                pitch,
                pitch_rate,
                x_velocity,
                left_speed,
                right_speed,
                position_x,
            ]
        return np.asarray(values, dtype=np.float32)

    def _body_tilt(self) -> float:
        """Return the angle between the robot's local up axis and world up."""
        _, orientation = p.getBasePositionAndOrientation(
            self.robot_id, physicsClientId=self.client_id
        )
        rotation = p.getMatrixFromQuaternion(orientation)
        world_up_dot_body_up = float(np.clip(rotation[8], -1.0, 1.0))
        return math.acos(world_up_dot_body_up)

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[np.ndarray, dict[str, Any]]:
        super().reset(seed=seed)
        self._connect()
        p.resetSimulation(physicsClientId=self.client_id)
        p.setAdditionalSearchPath(pybullet_data.getDataPath(), physicsClientId=self.client_id)
        p.setGravity(0.0, 0.0, -9.81, physicsClientId=self.client_id)
        p.setTimeStep(self.dt, physicsClientId=self.client_id)
        p.loadURDF("plane.urdf", physicsClientId=self.client_id)

        initial_pitch = float(
            self.np_random.uniform(-self.initial_tilt_limit, self.initial_tilt_limit)
        )
        self.robot_id = p.loadURDF(
            str(self.urdf_path),
            [0.0, 0.0, 0.04],
            p.getQuaternionFromEuler([0.0, initial_pitch, 0.0]),
            physicsClientId=self.client_id,
        )
        self._find_wheels()
        self._randomize_dynamics()
        self.action_buffer = [(0.0, 0.0)] * self.motor_delay_steps
        self.steps = 0
        self.command_state = 0.0
        self.acceleration_state = 0.0
        self.hardware_observation_history = []
        self.previous_x_velocity = self._physical_state()[3]
        return self._state(), {}

    def _randomize_dynamics(self) -> None:
        if not self.randomize:
            self.torque_scale = 1.0
            self.left_torque_scale = 1.0
            self.right_torque_scale = 1.0
            self.left_motor_deadband = 0.0
            self.right_motor_deadband = 0.0
            self.motor_delay_steps = 0
            self.pitch_sensor_bias = 0.0
            self.gyro_sensor_bias = 0.0
            self.pitch_sensor_noise = 0.0
            self.gyro_sensor_noise = 0.0
            self.acceleration_sensor_noise = 0.0
            return

        self.torque_scale = float(self.np_random.uniform(0.9, 1.1))
        self.left_torque_scale = float(self.np_random.uniform(0.95, 1.05))
        self.right_torque_scale = float(self.np_random.uniform(0.95, 1.05))
        # The right motor needs more command to overcome static friction under load.
        self.left_motor_deadband = float(self.np_random.uniform(0.00, 0.04))
        self.right_motor_deadband = float(self.np_random.uniform(0.06, 0.11))
        self.motor_delay_steps = int(self.np_random.integers(0, 3))
        self.pitch_sensor_bias = float(self.np_random.uniform(-0.004, 0.004))
        self.gyro_sensor_bias = float(self.np_random.uniform(-0.03, 0.03))
        self.pitch_sensor_noise = float(self.np_random.uniform(0.001, 0.003))
        self.gyro_sensor_noise = float(self.np_random.uniform(0.01, 0.03))
        self.acceleration_sensor_noise = float(self.np_random.uniform(0.01, 0.05))
        for link in range(-1, p.getNumJoints(self.robot_id, physicsClientId=self.client_id)):
            if link >= 0:
                link_name = p.getJointInfo(
                    self.robot_id, link, physicsClientId=self.client_id
                )[12].decode()
                if link_name == "mpu6050":
                    continue
            mass = p.getDynamicsInfo(
                self.robot_id, link, physicsClientId=self.client_id
            )[0]
            p.changeDynamics(
                self.robot_id,
                link,
                mass=float(mass) * float(self.np_random.uniform(0.95, 1.05)),
                lateralFriction=float(self.np_random.uniform(0.8, 1.2)),
                physicsClientId=self.client_id,
            )

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        clipped_action = np.asarray(action, dtype=np.float32).clip(-1.0, 1.0)
        self.action_buffer.append((float(clipped_action[0]), float(clipped_action[1])))
        delayed_left, delayed_right = self.action_buffer.pop(0)
        left_action = delayed_left
        if abs(left_action) < self.left_motor_deadband:
            left_action = 0.0
        right_action = delayed_right
        if abs(right_action) < self.right_motor_deadband:
            right_action = 0.0
        delayed_command = 0.5 * (delayed_left + delayed_right)
        self.command_state = 0.9 * self.command_state + 0.1 * delayed_command
        for _ in range(self.frame_skip):
            # TORQUE_CONTROL must be refreshed for every Bullet substep.
            for joint, wheel_action, wheel_scale in (
                (self.left_joint, left_action, self.left_torque_scale),
                (self.right_joint, right_action, self.right_torque_scale),
            ):
                p.setJointMotorControl2(
                    self.robot_id,
                    joint,
                    p.TORQUE_CONTROL,
                    force=(
                        wheel_action
                        * self.max_action_torque
                        * self.torque_scale
                        * wheel_scale
                    ),
                    physicsClientId=self.client_id,
                )
            p.stepSimulation(physicsClientId=self.client_id)

        self.steps += 1
        current_x_velocity = self._physical_state()[3]
        forward_acceleration = (
            current_x_velocity - self.previous_x_velocity
        ) / self.control_dt
        self.previous_x_velocity = current_x_velocity
        acceleration_observation = forward_acceleration / 9.81
        if self.randomize:
            acceleration_observation += float(
                self.np_random.normal(0.0, self.acceleration_sensor_noise)
            )
        self.acceleration_state = 0.8 * self.acceleration_state + 0.2 * float(
            np.clip(acceleration_observation, -5.0, 5.0)
        )
        observation = self._state()
        pitch, pitch_rate, yaw_rate, x_velocity, _, _, position_x = (
            self._physical_state()
        )
        fallen = (
            self._body_tilt() > self.max_tilt
            or float(position_x) > 5.0
            or float(position_x) < -5.0
        )
        terminated = bool(fallen)
        truncated = self.steps >= self.max_steps

        # Upright posture is the main objective; movement and large torque are penalized.
        angle_error = float(pitch)
        reward = (
            1.0
            - 2.0 * angle_error**2
            - 0.02 * float(pitch_rate) ** 2
            - 0.02 * float(yaw_rate) ** 2
            - self.velocity_penalty * float(x_velocity) ** 2
            - self.position_penalty * float(position_x) ** 2
            - 0.0005 * (delayed_left**2 + delayed_right**2)

        )
        if terminated:
            reward -= 10.0

        if self.render_mode == "human":
            import time

            time.sleep(self.dt * self.frame_skip)
        return observation, float(reward), terminated, truncated, {}

    def close(self) -> None:
        if self.client_id >= 0 and p.isConnected(self.client_id):
            p.disconnect(self.client_id)
        self.client_id = -1
