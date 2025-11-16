# generate_dataset_4ball.py
import argparse
import json
import os
import random
from typing import Dict

import torch

from env_balls_pymunk import PymunkBallWorldSimulator


def generate_windows_4ball(
    num_scenes: int,
    steps: int,
    seed: int,
) -> torch.Tensor:
    """
    Generate all (t-1, t, t+1) windows for the 4-ball, equal-mass prediction task.
    Returns a tensor of shape [num_samples, 3, 4, D].
    """
    random.seed(seed)
    torch.manual_seed(seed)

    # World params must match the paper
    width = 800.0
    height = 600.0
    ball_radius = 60.0
    v_max = 60.0
    dt = 0.1
    num_balls = 4
    variable_mass = False  # equal mass for prediction task

    sim = PymunkBallWorldSimulator(
        width=width,
        height=height,
        ball_radius=ball_radius,
        v_max=v_max,
        dt=dt,
        num_balls=num_balls,
        variable_mass=variable_mass,
        seed=seed,
    )

    all_windows = []
    for scene_idx in range(num_scenes):
        traj = sim.simulate(steps=steps)  # [T, N, D]
        T, N, D = traj.shape
        assert N == num_balls, "Unexpected num_balls in trajectory"

        # sliding 3-frame windows: (t-1, t, t+1), t = 1..T-2
        for t in range(1, T - 1):
            window = traj[t - 1 : t + 2]  # [3, N, D]
            all_windows.append(window)

        if (scene_idx + 1) % 1000 == 0:
            print(f"  generated {scene_idx + 1} / {num_scenes} scenes")

    windows = torch.stack(all_windows)  # [num_samples, 3, N, D]
    return windows


def main():
    parser = argparse.ArgumentParser(
        description="Generate 4-ball equal-mass prediction dataset (NPE)"
    )
    parser.add_argument(
        "--num_scenes",
        type=int,
        default=50000,
        help="Number of trajectories (scenes) to simulate (paper uses 50000)",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=60,
        help="Number of timesteps per trajectory (paper uses 60)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default="data/balls_n4_t60_uniform",
        help="Output directory for dataset",
    )
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    print(
        f"Generating 4-ball equal-mass dataset with "
        f"{args.num_scenes} scenes, {args.steps} steps each..."
    )
    windows = generate_windows_4ball(
        num_scenes=args.num_scenes,
        steps=args.steps,
        seed=args.seed,
    )
    num_samples, T, N, D = windows.shape
    print(f"Total windows: {num_samples}, window shape: [{T}, {N}, {D}]")

    # Shuffle and split 70/15/15 like the paper
    perm = torch.randperm(num_samples)
    windows = windows[perm]

    num_train = int(0.7 * num_samples)
    num_val = int(0.15 * num_samples)
    num_test = num_samples - num_train - num_val

    train_windows = windows[:num_train]
    val_windows = windows[num_train : num_train + num_val]
    test_windows = windows[num_train + num_val :]

    prefix = os.path.join(args.out_dir, "balls_n4_t60_uniform")

    train_path = prefix + "_train.pt"
    val_path = prefix + "_val.pt"
    test_path = prefix + "_test.pt"

    print(f"Saving train set: {train_path} ({train_windows.shape[0]} examples)")
    torch.save(train_windows, train_path)

    print(f"Saving val set:   {val_path} ({val_windows.shape[0]} examples)")
    torch.save(val_windows, val_path)

    print(f"Saving test set:  {test_path} ({test_windows.shape[0]} examples)")
    torch.save(test_windows, test_path)

    meta: Dict = {
        "num_scenes": args.num_scenes,
        "steps": args.steps,
        "num_samples": num_samples,
        "num_train": int(num_train),
        "num_val": int(num_val),
        "num_test": int(num_test),
        "num_balls": 4,
        "state_dim": int(D),
        "world": {
            "width": 800.0,
            "height": 600.0,
            "ball_radius": 60.0,
            "v_max": 60.0,
            "dt": 0.1,
        },
        "variable_mass": False,
        "description": "4-ball equal-mass prediction dataset for NPE (triplets t-1,t,t+1)",
    }

    meta_path = os.path.join(args.out_dir, "meta.json")
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"Saved meta info to {meta_path}")

    print("Done.")


if __name__ == "__main__":
    main()
