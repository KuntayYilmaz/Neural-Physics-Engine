# compare_rollouts_classic_vs_modern.py
import argparse
import math
from pathlib import Path

import numpy as np
import torch
import matplotlib.pyplot as plt

from models import NeuralPhysicsEngine
from models_modern import ModernNeuralPhysicsEngine
from env_balls_pymunk import PymunkBallWorldSimulator


# -------------------- utils -------------------- #

def load_ckpt(path, device):
    ckpt = torch.load(path, map_location=device)
    world_params = ckpt.get("world_params", {})
    width = world_params.get("width", 800.0)
    height = world_params.get("height", 600.0)
    ball_radius = world_params.get("ball_radius", 60.0)
    v_max = world_params.get("v_max", 60.0)
    dt = world_params.get("dt", 0.1)
    config = ckpt.get("config", {})
    state = ckpt["model_state"]
    return ckpt, state, config, width, height, ball_radius, v_max, dt


def maybe_strip_orig_prefix(state_dict):
    """Handle checkpoints saved from torch.compile models (_orig_mod.* keys)."""
    needs_strip = any(k.startswith("_orig_mod.") for k in state_dict.keys())
    if not needs_strip:
        return state_dict
    new_sd = {}
    for k, v in state_dict.items():
        if k.startswith("_orig_mod."):
            new_sd[k[len("_orig_mod."):]] = v
        else:
            new_sd[k] = v
    return new_sd


def build_classic_npe(ckpt_path, device):
    ckpt, state, config, width, height, ball_radius, v_max, dt = load_ckpt(
        ckpt_path, device
    )

    ball_radius_norm = ball_radius / width
    neighborhood_radius = 3.5 * ball_radius_norm
    state_dim = 5  # [x, y, vx, vy, mass]

    model = NeuralPhysicsEngine(
        state_dim=state_dim,
        neighborhood_radius=neighborhood_radius,
    ).to(device)

    state = maybe_strip_orig_prefix(state)
    model.load_state_dict(state)
    model.eval()

    return model, width, height, ball_radius, v_max, dt


def build_modern_npe(ckpt_path, device):
    ckpt, state, config, width, height, ball_radius, v_max, dt = load_ckpt(
        ckpt_path, device
    )

    ball_radius_norm = ball_radius / width
    neighborhood_radius = 3.5 * ball_radius_norm
    state_dim = 5

    msg_dim = config.get("msg_dim", 96)
    enc_blocks = config.get("enc_blocks", 3)
    dec_blocks = config.get("dec_blocks", 3)
    dropout = config.get("dropout", 0.0)
    expansion = config.get("expansion", 2)

    model = ModernNeuralPhysicsEngine(
        state_dim=state_dim,
        neighborhood_radius=neighborhood_radius,
        msg_dim=msg_dim,
        enc_blocks=enc_blocks,
        dec_blocks=dec_blocks,
        dropout=dropout,
        expansion=expansion,
    ).to(device)

    state = maybe_strip_orig_prefix(state)
    model.load_state_dict(state)
    model.eval()

    return model, width, height, ball_radius, v_max, dt


def rollout_model_on_scene(
    model,
    sim,
    device,
    rollout_steps,
    v_max,
    width,
    dt,
):
    """
    Single-scene rollout for one model.

    Returns:
        cos_sims:        [T] tensor
        rel_mag_errors:  [T] tensor
    """
    use_autocast = (device.type == "cuda")
    autocast_dtype = torch.bfloat16

    # Need T_total = rollout_steps future + 2 seed frames
    T_total = rollout_steps + 2
    with torch.no_grad():
        traj = sim.simulate(steps=T_total)   # [T_total, N, D], normalized

    traj = traj.to(device)
    T, N, D = traj.shape
    assert T == T_total
    assert D >= 4

    integration_factor = v_max * dt / width

    # Seed states
    state_tm1 = traj[0]   # [N,D] at t=0
    state_t   = traj[1]   # [N,D] at t=1

    pred_states = [state_tm1, state_t]

    # autoregressive rollout
    for step in range(rollout_steps):
        s_tm1 = pred_states[-2]
        s_t   = pred_states[-1]

        # Build window [1,3,N,D] – the third frame is a dummy.
        window = torch.stack([s_tm1, s_t, s_t], dim=0)[None, ...]

        if use_autocast:
            with torch.autocast(device_type=device.type, dtype=autocast_dtype):
                pred_v, _ = model(window)    # [1,N,2]
        else:
            pred_v, _ = model(window)

        pred_v = pred_v[0]                   # [N,2]

        pos_t = s_t[..., 0:2]
        pos_next = pos_t + pred_v * integration_factor

        s_next = s_t.clone()
        s_next[..., 0:2] = pos_next
        s_next[..., 2:4] = pred_v

        pred_states.append(s_next)

    pred_traj = torch.stack(pred_states, dim=0)  # [T_total,N,D]

    cos_sims = []
    rel_mag_errs = []
    eps = 1e-8

    for step in range(rollout_steps):
        t_idx = step + 2  # compare at timestep t_idx

        gt_state   = traj[t_idx]      # [N,D]
        pred_state = pred_traj[t_idx] # [N,D]

        gt_v   = gt_state[..., 2:4]
        pred_v = pred_state[..., 2:4]

        # cosine similarity
        dot = (gt_v * pred_v).sum(dim=-1)          # [N]
        gt_norm   = gt_v.norm(dim=-1) + eps
        pred_norm = pred_v.norm(dim=-1) + eps
        cos = dot / (gt_norm * pred_norm)
        cos_sims.append(cos.mean())

        # relative magnitude error
        mag_diff = (gt_norm - pred_norm).abs()
        rel_err = mag_diff / (gt_norm + eps)
        rel_mag_errs.append(rel_err.mean())

    cos_sims = torch.stack(cos_sims, dim=0)           # [T]
    rel_mag_errs = torch.stack(rel_mag_errs, dim=0)   # [T]
    return cos_sims, rel_mag_errs


def plot_two_metric_curves(
    timesteps,
    cos_classic,
    cos_modern,
    relmag_classic,
    relmag_modern,
    out_path="fig_prediction_task_classic_vs_modern_2panels.png",
):
    timesteps = np.asarray(timesteps)

    fig, axes = plt.subplots(
        2, 1, figsize=(7, 8), sharex=True,
        gridspec_kw={"hspace": 0.15}
    )

    # Top panel: cosine similarity
    ax = axes[0]
    ax.plot(timesteps, cos_classic, label="Classic NPE (4 → 4)")
    ax.plot(timesteps, cos_modern,  label="Modern NPE (4 → 4)")
    ax.set_ylabel("Cosine similarity ↑")
    ax.set_ylim(0.0, 1.0)
    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.3)
    ax.legend(loc="lower left")

    # Bottom panel: relative error in |v|
    ax = axes[1]
    ax.plot(timesteps, relmag_classic, label="Classic NPE (4 → 4)")
    ax.plot(timesteps, relmag_modern,  label="Modern NPE (4 → 4)")
    ax.set_ylabel("Relative error in |v| ↓")
    ax.set_xlabel("Timesteps")
    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.3)

    fig.suptitle("Prediction Task — Classic vs Modern NPE (Train 4 → Test 4)",
                 y=0.98)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    print(f"Saved plot to {out_path}")
    plt.show()


# -------------------- main -------------------- #

def main():
    parser = argparse.ArgumentParser(
        description="Compare Classic vs Modern NPE on 4-ball prediction task "
                    "using cosine similarity & relative magnitude error."
    )
    parser.add_argument(
        "--classic_ckpt",
        type=str,
        default="checkpoints/npe_balls_n4_t60_uniform_logged.pt",
        help="Checkpoint for classic NPE"
    )
    parser.add_argument(
        "--modern_ckpt",
        type=str,
        default="checkpoints/npe_modern_balls_n4_medium_adamw_cosine_120M.pt",
        help="Checkpoint for modern NPE"
    )
    parser.add_argument(
        "--num_scenes",
        type=int,
        default=200,
        help="Number of random scenes to roll out"
    )
    parser.add_argument(
        "--rollout_steps",
        type=int,
        default=50,
        help="Number of future steps to roll out"
    )
    parser.add_argument(
        "--base_seed",
        type=int,
        default=123,
        help="Base seed for scene generation (different seed per scene)."
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument(
        "--out_png",
        type=str,
        default="fig_prediction_task_classic_vs_modern_2panels.png",
    )

    args = parser.parse_args()
    device = torch.device(args.device)
    print("Using device:", device)

    # build models
    print("Loading classic NPE from:", args.classic_ckpt)
    classic_model, width_c, height_c, ball_r_c, v_max_c, dt_c = build_classic_npe(
        args.classic_ckpt, device
    )

    print("Loading modern NPE from:", args.modern_ckpt)
    modern_model, width_m, height_m, ball_r_m, v_max_m, dt_m = build_modern_npe(
        args.modern_ckpt, device
    )

    # sanity check same world params
    assert width_c == width_m
    assert height_c == height_m
    assert ball_r_c == ball_r_m
    assert math.isclose(v_max_c, v_max_m)
    assert math.isclose(dt_c, dt_m)

    width = width_c
    height = height_c
    ball_radius = ball_r_c
    v_max = v_max_c
    dt = dt_c

    all_cos_classic = []
    all_cos_modern = []
    all_relmag_classic = []
    all_relmag_modern = []

    for i in range(args.num_scenes):
        seed = args.base_seed + i

        sim = PymunkBallWorldSimulator(
            width=width,
            height=height,
            ball_radius=ball_radius,
            v_max=v_max,
            dt=dt,
            num_balls=4,
            variable_mass=False,
            seed=seed,
        )

        with torch.no_grad():
            cos_c, relmag_c = rollout_model_on_scene(
                classic_model,
                sim,
                device,
                rollout_steps=args.rollout_steps,
                v_max=v_max,
                width=width,
                dt=dt,
            )

            # re-create simulator with same seed so GT trajectory is identical
            sim = PymunkBallWorldSimulator(
                width=width,
                height=height,
                ball_radius=ball_radius,
                v_max=v_max,
                dt=dt,
                num_balls=4,
                variable_mass=False,
                seed=seed,
            )

            cos_m, relmag_m = rollout_model_on_scene(
                modern_model,
                sim,
                device,
                rollout_steps=args.rollout_steps,
                v_max=v_max,
                width=width,
                dt=dt,
            )

        all_cos_classic.append(cos_c.cpu())
        all_cos_modern.append(cos_m.cpu())
        all_relmag_classic.append(relmag_c.cpu())
        all_relmag_modern.append(relmag_m.cpu())

        if (i + 1) % 20 == 0:
            print(f"  evaluated {i+1}/{args.num_scenes} scenes")

    all_cos_classic = torch.stack(all_cos_classic, dim=0)       # [S,T]
    all_cos_modern = torch.stack(all_cos_modern, dim=0)
    all_relmag_classic = torch.stack(all_relmag_classic, dim=0)
    all_relmag_modern = torch.stack(all_relmag_modern, dim=0)

    mean_cos_classic = all_cos_classic.mean(dim=0).numpy()
    mean_cos_modern = all_cos_modern.mean(dim=0).numpy()
    mean_relmag_classic = all_relmag_classic.mean(dim=0).numpy()
    mean_relmag_modern = all_relmag_modern.mean(dim=0).numpy()

    timesteps = np.arange(1, args.rollout_steps + 1)

    plot_two_metric_curves(
        timesteps,
        mean_cos_classic,
        mean_cos_modern,
        mean_relmag_classic,
        mean_relmag_modern,
        out_path=args.out_png,
    )


if __name__ == "__main__":
    main()
