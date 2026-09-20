"""Cautious encoder-free policy runner for Raspberry Pi and IBT-2 drivers.

Default mode is sensor-only. Add --arm only after the IMU and motor wiring
have been checked with the wheels lifted and an emergency power disconnect ready.
"""

from __future__ import annotations

import argparse
import csv
import datetime
import math
import time
from pathlib import Path

import numpy as np
import serial
import smbus2
import RPi.GPIO as GPIO


MPU_ADDRESS = 0x68
RPWM_RIGHT, LPWM_RIGHT, REN_RIGHT, LEN_RIGHT = 12, 13, 24, 25
RPWM_LEFT, LPWM_LEFT, REN_LEFT, LEN_LEFT = 19, 18, 26, 27
MOTOR_PINS = [
    RPWM_RIGHT,
    LPWM_RIGHT,
    REN_RIGHT,
    LEN_RIGHT,
    RPWM_LEFT,
    LPWM_LEFT,
    REN_LEFT,
    LEN_LEFT,
]
TARGET_ANGLE_DEG = 1.6
SAFETY_LIMIT_DEG = 25.0
GYRO_LSB_PER_DEG_S = 131.0
CONTROL_FREQUENCY_HZ = 120.0
LOG_COLUMNS = [
    "t_s",
    "dt_s",
    "angle_deg",
    "error_deg",
    "gyro_deg_s",
    "left_rad_s",
    "right_rad_s",
    "action_left",
    "action_right",
    "pwm_left_pct",
    "pwm_right_pct",
    "accel_angle_deg",
]


def read_word(bus: smbus2.SMBus, register: int) -> int:
    value = (bus.read_byte_data(MPU_ADDRESS, register) << 8) | bus.read_byte_data(
        MPU_ADDRESS, register + 1
    )
    return value - 65536 if value >= 0x8000 else value


def read_imu(
    bus: smbus2.SMBus,
    gyro_bias: float,
    yaw_gyro_bias: float,
    gyro_sign: float = 1.0,
) -> tuple[float, float, float, float]:
    ax = read_word(bus, 0x3B) / 16384.0
    az = read_word(bus, 0x3F) / 16384.0
    gyro_deg_s = gyro_sign * (read_word(bus, 0x45) / GYRO_LSB_PER_DEG_S - gyro_bias)
    yaw_gyro_deg_s = read_word(bus, 0x47) / GYRO_LSB_PER_DEG_S - yaw_gyro_bias
    angle_deg = math.degrees(math.atan2(ax, az))
    forward_acceleration = (ax * 9.81) - (9.81 * math.sin(math.radians(angle_deg)))
    return angle_deg, gyro_deg_s, yaw_gyro_deg_s, forward_acceleration


def load_policy(path: Path) -> tuple[list[tuple[np.ndarray, np.ndarray]], np.ndarray, np.ndarray]:
    data = np.load(path)
    layers: list[tuple[np.ndarray, np.ndarray]] = []
    index = 0
    while f"w{index}" in data:
        layers.append((data[f"w{index}"], data[f"b{index}"]))
        index += 1
    return layers + [(data["w_out"], data["b_out"])], data["obs_mean"], data["obs_var"]


def policy_action(
    layers: list[tuple[np.ndarray, np.ndarray]],
    obs: np.ndarray,
    obs_mean: np.ndarray,
    obs_var: np.ndarray,
) -> np.ndarray:
    x = (obs - obs_mean) / np.sqrt(obs_var + 1e-8)
    for index, (weights, bias) in enumerate(layers):
        x = x @ weights.T + bias
        if index < len(layers) - 1:
            x = np.tanh(x)  # au lieu de np.maximum(x, 0.0)
    return np.clip(x, -1.0, 1.0).astype(np.float32)


def stop_motors(pwms: list[object]) -> None:
    if not pwms:
        return
    for pwm in pwms:
        pwm.ChangeDutyCycle(0)
    for pin in [REN_RIGHT, LEN_RIGHT, REN_LEFT, LEN_LEFT]:
        GPIO.output(pin, GPIO.LOW)


def set_motors(pwms: list[object], commands: np.ndarray, max_pwm: float) -> None:
    left_pwm, right_pwm = np.clip(commands * max_pwm, -max_pwm, max_pwm)
    GPIO.output(REN_RIGHT, GPIO.HIGH)
    GPIO.output(LEN_RIGHT, GPIO.HIGH)
    GPIO.output(REN_LEFT, GPIO.HIGH)
    GPIO.output(LEN_LEFT, GPIO.HIGH)
    for channel in pwms:
        channel.ChangeDutyCycle(0)
    if right_pwm > 0:
        pwms[1].ChangeDutyCycle(float(right_pwm))
    elif right_pwm < 0:
        pwms[0].ChangeDutyCycle(float(-right_pwm))
    if left_pwm > 0:
        pwms[2].ChangeDutyCycle(float(left_pwm))
    elif left_pwm < 0:
        pwms[3].ChangeDutyCycle(float(-left_pwm))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--arm", action="store_true", help="Enable motor outputs.")
    parser.add_argument(
        "--nominal",
        action="store_true",
        help="Use the non-randomized hardware policy export.",
    )
    parser.add_argument(
        "--accel",
        action="store_true",
        help="Use the four-observation filtered-acceleration policy.",
    )
    parser.add_argument("--encoder", action="store_true",
        help="Use the wheel-speed policy and read encoder telemetry from the ESP32.",
    )
    parser.add_argument("--encoder-port", default="/dev/ttyUSB0")
    parser.add_argument("--max-pwm", type=float, default=20.0)
    parser.add_argument("--seconds", type=float, default=30.0)
    parser.add_argument("--policy", type=Path, default=None)
    parser.add_argument(
        "--log",
        type=Path,
        default=None,
        help="CSV file for the per-step log (default: logs/deploy_<timestamp>.csv).",
    )
    parser.add_argument("--no-log", action="store_true", help="Do not write a CSV log.")
    parser.add_argument(
        "--invert-gyro",
        action="store_true",
        help="Negate the pitch gyro so it equals d(angle)/dt (sim convention).",
    )
    args = parser.parse_args()
    if args.accel and args.encoder:
        parser.error("--accel and --encoder cannot be combined")
    if not 0.0 < args.max_pwm <= 100.0:
        raise ValueError("--max-pwm must be between 0 and 100")

    policy_path = args.policy
    if policy_path is None:
        output_name = (
            "training_output_hardware_encoder"
            if args.encoder
            else
            "training_output_hardware_accel_nominal"
            if args.accel and args.nominal
            else "training_output_hardware_accel"
            if args.accel
            else "training_output_hardware_nominal"
            if args.nominal
            else "training_output_hardware"
        )
        policy_path = Path(__file__).with_name(output_name) / "hardware_policy.npz"
    layers, obs_mean, obs_var = load_policy(policy_path)
    encoder_device = (
        serial.Serial(args.encoder_port, 115200, timeout=0)
        if args.encoder
        else None
    )
    bus = smbus2.SMBus(1)
    bus.write_byte_data(MPU_ADDRESS, 0x6B, 0)
    GPIO.setmode(GPIO.BCM)
    GPIO.setwarnings(False)
    pwms: list[object] = []
    if args.arm:
        for pin in MOTOR_PINS:
            GPIO.setup(pin, GPIO.OUT, initial=GPIO.LOW)
        pwms = [
            GPIO.PWM(pin, 1000)
            for pin in [RPWM_RIGHT, LPWM_RIGHT, RPWM_LEFT, LPWM_LEFT]
        ]
        for pwm in pwms:
            pwm.start(0)

    try:
        gyro_samples = []
        yaw_gyro_samples = []
        for _ in range(200):
            gyro_samples.append(read_word(bus, 0x45) / GYRO_LSB_PER_DEG_S)
            yaw_gyro_samples.append(read_word(bus, 0x47) / GYRO_LSB_PER_DEG_S)
            time.sleep(0.005)
        gyro_bias = float(np.mean(gyro_samples))
        yaw_gyro_bias = float(np.mean(yaw_gyro_samples))
        gyro_sign = -1.0 if args.invert_gyro else 1.0
        angle, _, _, _ = read_imu(bus, gyro_bias, yaw_gyro_bias, gyro_sign)
        filtered_angle = angle
        filtered_acceleration = 0.0
        left_speed = 0.0
        right_speed = 0.0
        encoder_buffer = b""
        command_state = 0.0
        log_rows: list[list[float]] = []
        start = time.perf_counter()
        last = start
        print(f"Gyro bias: {gyro_bias:.3f} deg/s; initial angle: {angle:.2f} deg")
        print("Sensor-only mode." if not args.arm else f"ARMED, max PWM={args.max_pwm:.1f}%")
        if args.invert_gyro:
            print("Pitch gyro inverted.")
        while time.perf_counter() - start < args.seconds:
            now = time.perf_counter()
            dt = max(now - last, 1e-4)
            last = now
            angle, gyro, yaw_gyro, forward_acceleration = read_imu(
                bus, gyro_bias, yaw_gyro_bias, gyro_sign
            )
            filtered_angle = 0.98 * (filtered_angle + gyro * dt) + 0.02 * angle
            filtered_acceleration = (
                0.8 * filtered_acceleration + 0.2 * forward_acceleration / 9.81
            )
            error_deg = filtered_angle - TARGET_ANGLE_DEG
            if abs(error_deg) > SAFETY_LIMIT_DEG:
                stop_motors(pwms)
                raise RuntimeError(f"Safety tilt limit exceeded: {error_deg:.1f} deg")
            observation_values = [math.radians(error_deg), gyro / 100.0, command_state]
            if args.accel:
                observation_values.append(
                    float(np.clip(filtered_acceleration, -5.0, 5.0))
                )
            if args.encoder:
                if encoder_device is None:
                    raise RuntimeError("Encoder serial device is not open")
                waiting = encoder_device.in_waiting
                if waiting:
                    encoder_buffer += encoder_device.read(waiting)
                    complete_lines = encoder_buffer.split(b"\n")
                    encoder_buffer = complete_lines.pop()
                    for raw_line in complete_lines:
                        fields = raw_line.decode(
                            "ascii", errors="replace"
                        ).strip().split(",")
                        if len(fields) != 6 or fields[0] != "ENC":
                            continue
                        try:
                            new_left_speed = float(fields[4])
                            new_right_speed = float(fields[5])
                        except ValueError:
                            continue
                        left_speed = new_left_speed
                        right_speed = new_right_speed
                observation_values = [
                    math.radians(error_deg),
                    gyro / 100.0,
                    float(np.clip(yaw_gyro / 100.0, -5.0, 5.0)),
                    left_speed,
                    right_speed,
                    command_state,
                ]
            observation = np.asarray(observation_values, dtype=np.float32)
            action = policy_action(layers, observation, obs_mean, obs_var)
            if args.arm:
                set_motors(pwms, action, args.max_pwm)
            # PWM percent actually sent to each wheel (0 when not armed).
            left_pwm, right_pwm = (
                np.clip(action * args.max_pwm, -args.max_pwm, args.max_pwm)
                if args.arm
                else (0.0, 0.0)
            )
            log_rows.append(
                [
                    now - start,
                    dt,
                    filtered_angle,
                    error_deg,
                    gyro,
                    left_speed,
                    right_speed,
                    float(action[0]),
                    float(action[1]),
                    float(left_pwm),
                    float(right_pwm),
                    angle,
                ]
            )
            command_state = 0.9 * command_state + 0.1 * float(np.mean(action))
            print(
                f"\rangle={filtered_angle:6.2f} deg gyro={gyro:7.2f} "
                f"action=({action[0]:+.3f},{action[1]:+.3f}) "
                f"pwm=({left_pwm:+5.1f},{right_pwm:+5.1f})%",
                end="",
                flush=True,
            )
            time.sleep(max((1.0 / CONTROL_FREQUENCY_HZ) - (time.perf_counter() - now), 0.0))
    finally:
        stop_motors(pwms)
        for pwm in pwms:
            try:
                pwm.stop()
            except (AttributeError, TypeError):
                pass
        pwm = None
        pwms.clear()
        GPIO.cleanup()
        bus.close()
        if encoder_device is not None:
            encoder_device.close()
        print("\nMotors stopped.")
        if log_rows and not args.no_log:
            try:
                log_path = args.log or Path(__file__).with_name("logs") / (
                    datetime.datetime.now().strftime("deploy_%Y%m%d_%H%M%S.csv")
                )
                log_path.parent.mkdir(parents=True, exist_ok=True)
                with open(log_path, "w", newline="") as log_file:
                    writer = csv.writer(log_file)
                    writer.writerow(LOG_COLUMNS)
                    writer.writerows(log_rows)
                print(f"Log saved: {log_path} ({len(log_rows)} rows)")
            except OSError as exc:
                print(f"Could not write log: {exc}")


if __name__ == "__main__":
    main()
