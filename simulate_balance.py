"""
Simulate the self-balancing 2-wheel robot in PyBullet and try to keep it
upright with a simple PD controller on the pitch angle (classic inverted
pendulum control, same idea your real robot's MPU6050 + firmware would use).

Run:
    python simulate_balance.py

LEGACY: early prototype, not used by the RL pipeline. It reads euler[0]
(roll) as pitch and uses velocity control; the environment in balance_env.py
uses euler[1] (rotation about the wheel axle) and torque control.
"""

import pybullet as p
import pybullet_data
import time

URDF_PATH = "self_balancing_robot.urdf"  # keep in the same folder, or give a full path

# ---- PD gains: tune these until it balances nicely ----
KP = 220.0   # proportional gain on pitch error (rad -> motor velocity)
KD = 8.0     # damping on pitch rate
MAX_WHEEL_VEL = 40.0  # rad/s cap on wheel motor velocity

def main():
    p.connect(p.GUI)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setGravity(0, 0, -9.81)
    p.setTimeStep(1.0 / 240.0)

    plane_id = p.loadURDF("plane.urdf")

    # Start slightly above ground, near-vertical with a tiny initial tilt
    start_pos = [0, 0, 0.05]
    start_orn = p.getQuaternionFromEuler([0.05, 0, 0])  # 0.05 rad ~= 3 deg initial tilt
    robot_id = p.loadURDF(URDF_PATH, start_pos, start_orn)

    # Find wheel joints by name
    left_wheel_joint = None
    right_wheel_joint = None
    for j in range(p.getNumJoints(robot_id)):
        info = p.getJointInfo(robot_id, j)
        joint_name = info[1].decode("utf-8")
        if joint_name == "left_wheel_joint":
            left_wheel_joint = j
        elif joint_name == "right_wheel_joint":
            right_wheel_joint = j

    # Let wheels spin freely (no default motor friction) so torque control works
    for j in [left_wheel_joint, right_wheel_joint]:
        p.setJointMotorControl2(robot_id, j, p.VELOCITY_CONTROL, force=0)

    print("Left wheel joint index :", left_wheel_joint)
    print("Right wheel joint index:", right_wheel_joint)
    print("Balancing... close the window or Ctrl+C to stop.")

    prev_pitch = 0.0
    dt = 1.0 / 240.0

    try:
        while True:
            pos, orn = p.getBasePositionAndOrientation(robot_id)
            euler = p.getEulerFromQuaternion(orn)
            pitch = euler[0]  # rotation about the wheel axis (x-axis here) = lean angle

            pitch_rate = (pitch - prev_pitch) / dt
            prev_pitch = pitch

            # PD control: if leaning forward (positive pitch), drive wheels forward
            # to move the base back under the center of mass.
            wheel_vel = KP * pitch + KD * pitch_rate
            wheel_vel = max(-MAX_WHEEL_VEL, min(MAX_WHEEL_VEL, wheel_vel))

            p.setJointMotorControl2(
                robot_id, left_wheel_joint, p.VELOCITY_CONTROL,
                targetVelocity=wheel_vel, force=2.5
            )
            p.setJointMotorControl2(
                robot_id, right_wheel_joint, p.VELOCITY_CONTROL,
                targetVelocity=wheel_vel, force=2.5
            )

            p.stepSimulation()
            time.sleep(dt)

    except KeyboardInterrupt:
        pass

    p.disconnect()


if __name__ == "__main__":
    main()
