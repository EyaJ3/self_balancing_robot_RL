"""Train the two-wheel robot with PPO.

Usage:
    python train_balance.py
    python train_balance.py --timesteps 1000000
"""

from __future__ import annotations

import argparse
from pathlib import Path

from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecNormalize

from balance_env import TwoWheelBalanceEnv


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--timesteps", type=int, default=2_000_000)
    parser.add_argument("--envs", type=int, default=8)
    parser.add_argument("--check-env", action="store_true")
    parser.add_argument(
        "--hardware",
        action="store_true",
        help="Train the encoder-free three-observation policy for Raspberry Pi deployment.",
    )
    parser.add_argument(
        "--hardware-accel",
        action="store_true",
        help="Train the four-observation hardware policy with filtered acceleration.",
    )
    parser.add_argument(
        "--hardware-stacked",
        action="store_true",
        help="Train a hardware policy using four stacked three-observation frames.",
    )
    parser.add_argument(
        "--hardware-encoder",
        action="store_true",
        help="Train a hardware policy using real wheel-speed observations.",
    )
    parser.add_argument(
        "--full-baseline",
        action="store_true",
        help="Train a separate full-observation baseline policy.",
    )
    parser.add_argument(
        "--no-randomization",
        action="store_true",
        help="Disable domain randomization for a nominal hardware learning test.",
    )
    args = parser.parse_args()
    modes = sum(
        [
            args.hardware,
            args.hardware_accel,
            args.hardware_stacked,
            args.hardware_encoder,
            args.full_baseline,
        ]
    )
    if modes > 1:
        parser.error(
            "Choose only one of --hardware, --hardware-accel, "
            "--hardware-stacked, --hardware-encoder, or --full-baseline"
        )
    if args.no_randomization and not (
        args.hardware
        or args.hardware_accel
        or args.hardware_stacked
        or args.hardware_encoder
    ):
        parser.error("--no-randomization requires a hardware mode")
    hardware_mode = (
        args.hardware
        or args.hardware_accel
        or args.hardware_stacked
        or args.hardware_encoder
    )
    observation_mode = (
        "hardware_accel"
        if args.hardware_accel
        else "hardware_stacked"
        if args.hardware_stacked
        else "hardware_encoder"
        if args.hardware_encoder
        else "hardware"
        if args.hardware
        else "full"
    )

    if args.check_env:
        check_env(
            TwoWheelBalanceEnv(
                frame_skip=2 if hardware_mode else 4,
                observation_mode=observation_mode,
                randomize=hardware_mode and not args.no_randomization,
            )
        )
        return

    if args.hardware_stacked:
        output_name = "training_output_hardware_stacked"
    elif args.hardware_encoder:
        output_name = "training_output_hardware_encoder"
    elif args.hardware_accel:
        output_name = (
            "training_output_hardware_accel_nominal"
            if args.no_randomization
            else "training_output_hardware_accel"
        )
    elif args.hardware:
        output_name = (
            "training_output_hardware_nominal"
            if args.no_randomization
            else "training_output_hardware"
        )
    elif args.full_baseline:
        output_name = "training_output_full_baseline"
    else:
        output_name = "training_output"
    output_dir = Path(__file__).with_name(output_name)
    output_dir.mkdir(exist_ok=True)
    env = make_vec_env(
        lambda: TwoWheelBalanceEnv(
            randomize=not args.no_randomization if hardware_mode else True,
            frame_skip=2 if hardware_mode else 4,
            observation_mode=observation_mode,
            velocity_penalty=0.05,
            position_penalty=0.02,
        ),
        n_envs=args.envs,
    )
    env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=10.0)
    model = PPO(
        "MlpPolicy",
        env,
        policy_kwargs={"net_arch": [64, 64]},
        verbose=1,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=256,
        gamma=0.99,
        gae_lambda=0.95,
        ent_coef=0.0,
        tensorboard_log=str(output_dir / "tensorboard"),
    )
    try:
        model.learn(total_timesteps=args.timesteps)
        model.save(str(output_dir / "ppo_balance_robot"))
        env.save(str(output_dir / "vecnormalize.pkl"))
    finally:
        env.close()


if __name__ == "__main__":
    main()
