"""Headless evaluation of a saved PPO balance policy over many seeds.

Reports survival time, fall rate and drift, so policies can be compared.

Usage:
    python evaluate_policy.py --hardware-encoder
    python evaluate_policy.py --hardware-encoder --output-suffix _v2 --best
    python evaluate_policy.py --hardware-encoder --nominal --stochastic
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecNormalize

from balance_env import TwoWheelBalanceEnv

FOLDERS = {
    "full": "training_output",
    "hardware": "training_output_hardware",
    "hardware_accel": "training_output_hardware_accel",
    "hardware_stacked": "training_output_hardware_stacked",
    "hardware_encoder": "training_output_hardware_encoder",
    "full_baseline": "training_output_full_baseline",
}


def main() -> None:
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group()
    for name in FOLDERS:
        if name != "full":
            group.add_argument(f"--{name.replace('_', '-')}", dest="mode", action="store_const", const=name)
    parser.add_argument("--nominal", action="store_true", help="Disable randomization.")
    parser.add_argument("--stochastic", action="store_true", help="Sample actions instead of using the mean.")
    parser.add_argument("--episodes", type=int, default=30)
    parser.add_argument("--seconds", type=float, default=30.0)
    parser.add_argument("--seed", type=int, default=1000)
    parser.add_argument("--output-suffix", default="")
    parser.add_argument("--best", action="store_true")
    args = parser.parse_args()
    mode = args.mode or "full"

    output_dir = Path(__file__).with_name(FOLDERS[mode] + args.output_suffix)
    if args.best:
        output_dir = output_dir / "best"
    observation_mode = "full" if mode == "full_baseline" else mode
    hardware = observation_mode != "full"

    raw = make_vec_env(
        lambda: TwoWheelBalanceEnv(
            randomize=hardware and not args.nominal,
            frame_skip=2 if hardware else 4,
            observation_mode=observation_mode,
            episode_seconds=args.seconds,
            velocity_penalty=0.05,
            position_penalty=0.02,
        ),
        n_envs=1,
    )
    env = VecNormalize.load(str(output_dir / "vecnormalize.pkl"), raw)
    env.training = False
    env.norm_reward = False
    model = PPO.load(str(output_dir / ("best_model" if args.best else "ppo_balance_robot")), env=env)
    base = raw.envs[0].unwrapped

    lengths, drifts, falls, speeds = [], [], 0, []
    max_steps = base.max_steps
    for episode in range(args.episodes):
        raw.seed(args.seed + episode)
        obs = env.reset()
        steps, drift = 0, 0.0
        while True:
            action, _ = model.predict(obs, deterministic=not args.stochastic)
            obs, _, done, info = env.step(action)
            steps += 1
            state = base._physical_state()
            drift = max(drift, abs(state[6]))
            speeds.append(max(abs(state[4]), abs(state[5])))
            if done[0]:
                falls += steps < max_steps
                break
        lengths.append(steps)
        drifts.append(drift)

    seconds = np.array(lengths) * base.control_dt
    print(f"policy      : {output_dir}")
    print(f"environment : {'nominal' if args.nominal or not hardware else 'randomized'}, "
          f"{'stochastic' if args.stochastic else 'deterministic'}, {args.episodes} episodes of {args.seconds:.0f}s")
    print(f"fell        : {falls}/{args.episodes}")
    print(f"survival    : mean {seconds.mean():.1f}s  min {seconds.min():.1f}s")
    print(f"max |x|     : mean {np.mean(drifts):.3f} m  worst {np.max(drifts):.3f} m")
    print(f"wheel speed : 99th pct {np.percentile(speeds, 99):.1f} rad/s")
    env.close()


if __name__ == "__main__":
    main()
