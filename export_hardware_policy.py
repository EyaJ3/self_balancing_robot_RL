"""Export the encoder-free PPO policy for NumPy-only Raspberry Pi inference."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import argparse
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecNormalize

from balance_env import TwoWheelBalanceEnv


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--nominal",
        action="store_true",
        help="Export the non-randomized hardware policy.",
    )
    parser.add_argument(
        "--accel",
        action="store_true",
        help="Export the four-observation acceleration policy.",
    )
    parser.add_argument(
        "--encoder",
        action="store_true",
        help="Export the wheel-speed hardware policy.",
    )
    args = parser.parse_args()
    if args.accel and args.encoder:
        parser.error("--accel and --encoder cannot be combined")
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
    output_dir = Path(__file__).with_name(output_name)
    raw_env = make_vec_env(
        lambda: TwoWheelBalanceEnv(
            randomize=not args.nominal,
            frame_skip=2,
            observation_mode=(
                "hardware_encoder"
                if args.encoder
                else "hardware_accel"
                if args.accel
                else "hardware"
            ),
        ),
        n_envs=1,
    )
    stats = VecNormalize.load(
        str(output_dir / "vecnormalize.pkl"),
        raw_env,
    )
    model = PPO.load(str(output_dir / "ppo_balance_robot"), env=stats)
    policy = model.policy
    weights: dict[str, np.ndarray] = {}
    layer_number = 0
    for layer in policy.mlp_extractor.policy_net:
        if hasattr(layer, "weight") and hasattr(layer, "bias"):
            weights[f"w{layer_number}"] = layer.weight.detach().cpu().numpy()
            weights[f"b{layer_number}"] = layer.bias.detach().cpu().numpy()
            layer_number += 1
    weights["w_out"] = policy.action_net.weight.detach().cpu().numpy()
    weights["b_out"] = policy.action_net.bias.detach().cpu().numpy()
    weights["obs_mean"] = stats.obs_rms.mean.astype(np.float32)
    weights["obs_var"] = stats.obs_rms.var.astype(np.float32)
    np.savez(output_dir / "hardware_policy.npz", **weights)
    stats.close()
    print(f"Exported {output_dir / 'hardware_policy.npz'}")


if __name__ == "__main__":
    main()
