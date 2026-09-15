"""Run a saved PPO balance policy in the PyBullet GUI."""

from __future__ import annotations

import argparse
from pathlib import Path

import pybullet as p
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecNormalize

from balance_env import TwoWheelBalanceEnv


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--hardware",
        action="store_true",
        help="View the encoder-free policy from training_output_hardware.",
    )
    parser.add_argument(
        "--nominal",
        action="store_true",
        help="View the non-randomized hardware policy.",
    )
    parser.add_argument(
        "--hardware-accel",
        action="store_true",
        help="View the four-observation acceleration policy.",
    )
    parser.add_argument(
        "--hardware-stacked",
        action="store_true",
        help="View the four-frame stacked hardware policy.",
    )
    parser.add_argument(
        "--hardware-encoder",
        action="store_true",
        help="View the wheel-speed hardware policy.",
    )
    parser.add_argument(
        "--full-baseline",
        action="store_true",
        help="View the separate full-observation baseline policy.",
    )
    args = parser.parse_args()
    if sum(
        [
            args.hardware,
            args.hardware_accel,
            args.hardware_stacked,
            args.hardware_encoder,
            args.full_baseline,
        ]
    ) > 1:
        parser.error(
            "Choose only one of --hardware, --hardware-accel, "
            "--hardware-stacked, --hardware-encoder, or --full-baseline"
        )
    if args.nominal and not (
        args.hardware
        or args.hardware_accel
        or args.hardware_stacked
        or args.hardware_encoder
    ):
        parser.error("--nominal requires a hardware mode")
    if args.full_baseline:
        output_name = "training_output_full_baseline"
    elif args.hardware_stacked:
        output_name = "training_output_hardware_stacked"
    elif args.hardware_encoder:
        output_name = "training_output_hardware_encoder"
    elif args.hardware_accel:
        output_name = (
            "training_output_hardware_accel_nominal"
            if args.nominal
            else "training_output_hardware_accel"
        )
    elif args.hardware:
        output_name = (
            "training_output_hardware_nominal"
            if args.nominal
            else "training_output_hardware"
        )
    else:
        output_name = "training_output"
    output_dir = Path(__file__).with_name(output_name)
    model_path = output_dir / "ppo_balance_robot"
    raw_env = make_vec_env(
        lambda: TwoWheelBalanceEnv(
            render_mode="human",
            episode_seconds=30.0,
            frame_skip=2
            if (
                args.hardware
                or args.hardware_accel
                or args.hardware_stacked
                or args.hardware_encoder
            )
            else 4,
            observation_mode=(
                "hardware_accel"
                if args.hardware_accel
                else "hardware_stacked"
                if args.hardware_stacked
                else "hardware_encoder"
                if args.hardware_encoder
                else "hardware"
                if args.hardware
                else "full"
            ),
            randomize=(
                args.hardware or args.hardware_accel
            )
            and not args.nominal,
            velocity_penalty=0.05,
            position_penalty=0.02,
        ),
        n_envs=1,
    )
    bullet_env = raw_env.envs[0].unwrapped
    stats_path = output_dir / "vecnormalize.pkl"
    if stats_path.exists():
        env = VecNormalize.load(str(stats_path), raw_env)
        env.training = False
        env.norm_reward = False
        print("Loaded observation normalization statistics.")
    else:
        env = raw_env
        print(
            "WARNING: vecnormalize.pkl was not found. "
            "Running without observation normalization."
        )
    model = PPO.load(str(model_path), env=env)
    try:
        observation = env.reset()
        while True:
            if not p.isConnected(bullet_env.client_id):
                print("PyBullet window was closed; stopping.")
                break
            action, _ = model.predict(observation, deterministic=True)
            try:
                # VecEnv uses the SB3 API: observations, rewards, dones, infos.
                observation, _, done, _ = env.step(action)
            except p.error:
                print("PyBullet window was closed; stopping.")
                break
    finally:
        env.close()


if __name__ == "__main__":
    main()
