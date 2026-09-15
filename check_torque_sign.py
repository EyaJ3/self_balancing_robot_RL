"""Quick sanity check: does positive wheel torque actually correct forward pitch?

No policy, no training -- just apply a known constant torque to both wheels
from a fixed initial lean and watch what pitch does. Takes a few seconds to run.

Run:
    python check_torque_sign.py
"""

from __future__ import annotations

import math
from pathlib import Path

import pybullet as p
import pybullet_data

URDF_PATH = Path(__file__).with_name("self_balancing_robot.urdf")
DT = 1.0 / 240.0
TEST_TORQUE = 0.2  # N*m, well below max_action_torque so it doesn't saturate instantly
INITIAL_TILT_DEG = 5.0  # lean forward at the start
NUM_STEPS = 240  # 1 second of sim time
PRINT_EVERY = 20


def run_trial(torque_sign: float) -> None:
    client = p.connect(p.DIRECT)
    p.setAdditionalSearchPath(pybullet_data.getDataPath(), physicsClientId=client)
    p.setGravity(0.0, 0.0, -9.81, physicsClientId=client)
    p.setTimeStep(DT, physicsClientId=client)
    p.loadURDF("plane.urdf", physicsClientId=client)

    initial_pitch = math.radians(INITIAL_TILT_DEG)
    robot_id = p.loadURDF(
        str(URDF_PATH),
        [0.0, 0.0, 0.04],
        p.getQuaternionFromEuler([0.0, initial_pitch, 0.0]),
        physicsClientId=client,
    )

    joints = {
        p.getJointInfo(robot_id, i, physicsClientId=client)[1].decode(): i
        for i in range(p.getNumJoints(robot_id, physicsClientId=client))
    }
    left = joints["left_wheel_joint"]
    right = joints["right_wheel_joint"]
    for j in (left, right):
        p.setJointMotorControl2(robot_id, j, p.VELOCITY_CONTROL, force=0.0, physicsClientId=client)

    torque = torque_sign * TEST_TORQUE
    print(f"\n--- Trial: constant torque = {torque:+.3f} N*m, starting pitch = {INITIAL_TILT_DEG:+.1f} deg ---")

    for step in range(NUM_STEPS):
        for j in (left, right):
            p.setJointMotorControl2(
                robot_id, j, p.TORQUE_CONTROL, force=torque, physicsClientId=client
            )
        p.stepSimulation(physicsClientId=client)

        if step % PRINT_EVERY == 0:
            _, orn = p.getBasePositionAndOrientation(robot_id, physicsClientId=client)
            pitch_deg = math.degrees(p.getEulerFromQuaternion(orn)[1])
            print(f"  t={step * DT:5.2f}s  pitch={pitch_deg:+7.2f} deg")

    _, orn = p.getBasePositionAndOrientation(robot_id, physicsClientId=client)
    final_pitch_deg = math.degrees(p.getEulerFromQuaternion(orn)[1])
    print(f"  FINAL pitch = {final_pitch_deg:+.2f} deg (started at {INITIAL_TILT_DEG:+.1f} deg)")

    p.disconnect(client)
    return final_pitch_deg


def main() -> None:
    print("Robot starts leaning FORWARD (positive pitch).")
    print("A correct controller should apply torque that DECREASES pitch back toward 0.\n")

    final_positive = run_trial(torque_sign=+1.0)
    final_negative = run_trial(torque_sign=-1.0)

    print("\n=== RESULT ===")
    if abs(final_positive - INITIAL_TILT_DEG) < abs(final_negative - INITIAL_TILT_DEG):
        print("Positive torque reduced pitch more -> POSITIVE torque = corrective for forward lean.")
        print("Check that your policy/env applies positive action when pitch > 0 (leaning forward).")
    else:
        print("Negative torque reduced pitch more -> NEGATIVE torque = corrective for forward lean.")
        print("If your reward/env assumes positive action corrects forward lean, THAT'S THE SIGN BUG.")


if __name__ == "__main__":
    main()