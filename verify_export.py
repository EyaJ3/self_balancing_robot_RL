"""Cross-check: does the NumPy hardware_policy.npz reconstruction match the
real SB3 model's output for the same observation?

Run this on your TRAINING machine (needs stable_baselines3 installed).

Usage:
    python3 verify_export.py --encoder
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecNormalize

from balance_env import TwoWheelBalanceEnv


def load_npz_policy(path: Path):
    data = np.load(path)
    layers = []
    index = 0
    while f"w{index}" in data:
        layers.append((data[f"w{index}"], data[f"b{index}"]))
        index += 1
    return layers + [(data["w_out"], data["b_out"])], data["obs_mean"], data["obs_var"]


def numpy_policy_action(layers, obs, obs_mean, obs_var) -> np.ndarray:
    x = (obs - obs_mean) / np.sqrt(obs_var + 1e-8)
    for index, (weights, bias) in enumerate(layers):
        x = x @ weights.T + bias
        if index < len(layers) - 1:
            x = np.tanh(x)
    return np.clip(x, -1.0, 1.0).astype(np.float32)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--encoder", action="store_true")
    parser.add_argument("--accel", action="store_true")
    parser.add_argument("--nominal", action="store_true")
    parser.add_argument(
        "--output-suffix",
        default="",
        help="Load from the output folder with this suffix (e.g. _v2).",
    )
    parser.add_argument(
        "--best",
        action="store_true",
        help="Use the best checkpoint saved during training (best/ subfolder).",
    )
    args = parser.parse_args()

    output_name = (
        "training_output_hardware_encoder"
        if args.encoder
        else "training_output_hardware_accel_nominal"
        if args.accel and args.nominal
        else "training_output_hardware_accel"
        if args.accel
        else "training_output_hardware_nominal"
        if args.nominal
        else "training_output_hardware"
    )
    output_dir = Path(__file__).with_name(output_name + args.output_suffix)
    source_dir = output_dir / "best" if args.best else output_dir
    model_name = "best_model" if args.best else "ppo_balance_robot"

    # --- Load the real SB3 model + VecNormalize, exactly like export does ---
    raw_env = make_vec_env(
        lambda: TwoWheelBalanceEnv(
            randomize=not args.nominal,
            frame_skip=2,
            observation_mode=(
                "hardware_encoder" if args.encoder
                else "hardware_accel" if args.accel
                else "hardware"
            ),
        ),
        n_envs=1,
    )
    stats = VecNormalize.load(str(source_dir / "vecnormalize.pkl"), raw_env)
    stats.training = False
    model = PPO.load(str(source_dir / model_name), env=stats)

    # --- Load the exported NumPy reconstruction ---
    layers, obs_mean, obs_var = load_npz_policy(output_dir / "hardware_policy.npz")

    print(f"obs_mean (npz) = {obs_mean}")
    print(f"stats.obs_rms.mean (SB3) = {stats.obs_rms.mean}")
    print(f"Match: {np.allclose(obs_mean, stats.obs_rms.mean)}\n")

    # --- Test a handful of realistic observations, including a near-zero one ---
    n_obs = obs_mean.shape[0]
    tiny_angle = np.zeros(n_obs, dtype=np.float32)
    tiny_angle[0] = 0.005
    test_cases = [
        np.zeros(n_obs, dtype=np.float32),  # perfectly still
        tiny_angle,
        np.random.uniform(-0.1, 0.1, size=n_obs).astype(np.float32),
    ]

    for i, raw_obs in enumerate(test_cases):
        numpy_action = numpy_policy_action(layers, raw_obs, obs_mean, obs_var)

        # SB3's own normalization + prediction pipeline
        normalized_obs = stats.normalize_obs(raw_obs.reshape(1, -1))
        sb3_action, _ = model.predict(normalized_obs, deterministic=True)

        print(f"Test {i}: raw_obs={raw_obs}")
        print(f"  NumPy action = {numpy_action}")
        print(f"  SB3 action   = {sb3_action[0]}")
        print(f"  Match: {np.allclose(numpy_action, sb3_action[0], atol=1e-3)}\n")

    stats.close()


if __name__ == "__main__":
    main()