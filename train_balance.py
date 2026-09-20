"""Train the two-wheel robot with PPO.

Usage:
    python train_balance.py
    python train_balance.py --timesteps 1000000
"""

from __future__ import annotations

import argparse
from pathlib import Path

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecNormalize

from balance_env import TwoWheelBalanceEnv


class SaveNormalizationOnBest(BaseCallback):
    """Save VecNormalize stats next to the best model whenever it improves."""

    def __init__(self, path: Path) -> None:
        super().__init__()
        self.path = path

    def _on_step(self) -> bool:
        self.model.get_vec_normalize_env().save(str(self.path))
        return True


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
    parser.add_argument(
        "--output-suffix",
        default="",
        help="Append to the output folder name (e.g. _v2) so existing models are kept.",
    )
    parser.add_argument(
        "--episode-seconds",
        type=float,
        default=10.0,
        help="Training episode length; longer episodes cover behaviour past 10 s.",
    )
    parser.add_argument("--velocity-penalty", type=float, default=0.05)
    parser.add_argument("--position-penalty", type=float, default=0.02)
    parser.add_argument(
        "--lr-decay",
        action="store_true",
        help="Decay the learning rate linearly to zero over training.",
    )
    parser.add_argument(
        "--eval-freq",
        type=int,
        default=50_000,
        help="Total timesteps between evaluations that pick the best checkpoint (0 = off).",
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
                velocity_penalty=args.velocity_penalty,
                position_penalty=args.position_penalty,
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
    output_dir = Path(__file__).with_name(output_name + args.output_suffix)
    output_dir.mkdir(exist_ok=True)

    def make_env() -> TwoWheelBalanceEnv:
        return TwoWheelBalanceEnv(
            randomize=not args.no_randomization if hardware_mode else True,
            frame_skip=2 if hardware_mode else 4,
            observation_mode=observation_mode,
            episode_seconds=args.episode_seconds,
            velocity_penalty=args.velocity_penalty,
            position_penalty=args.position_penalty,
        )

    env = make_vec_env(make_env, n_envs=args.envs)
    env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=10.0)
    model = PPO(
        "MlpPolicy",
        env,
        policy_kwargs={"net_arch": [64, 64]},
        verbose=1,
        learning_rate=(lambda progress: 3e-4 * progress) if args.lr_decay else 3e-4,
        n_steps=2048,
        batch_size=256,
        gamma=0.99,
        gae_lambda=0.95,
        ent_coef=0.0,
        tensorboard_log=str(output_dir / "tensorboard"),
    )
    callback = None
    eval_env = None
    if args.eval_freq > 0:
        # Same randomization as training; stats are synced from the train env.
        eval_env = VecNormalize(
            make_vec_env(make_env, n_envs=1),
            training=False,
            norm_obs=True,
            norm_reward=False,
            clip_obs=10.0,
        )
        best_dir = output_dir / "best"
        best_dir.mkdir(exist_ok=True)
        callback = EvalCallback(
            eval_env,
            best_model_save_path=str(best_dir),
            callback_on_new_best=SaveNormalizationOnBest(best_dir / "vecnormalize.pkl"),
            eval_freq=max(args.eval_freq // args.envs, 1),
            n_eval_episodes=10,
            deterministic=True,
        )
    try:
        model.learn(total_timesteps=args.timesteps, callback=callback)
        model.save(str(output_dir / "ppo_balance_robot"))
        env.save(str(output_dir / "vecnormalize.pkl"))
    finally:
        env.close()
        if eval_env is not None:
            eval_env.close()


if __name__ == "__main__":
    main()
