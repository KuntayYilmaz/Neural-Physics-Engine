# render_rollout_triple.py
import argparse
import os

import numpy as np
import torch
import matplotlib.pyplot as plt
import imageio.v2 as imageio

from env_balls_pymunk import PymunkBallWorldSimulator
from models import NeuralPhysicsEngine
from models_modern import ModernNeuralPhysicsEngine


def load_classic_npe(ckpt_path, device):
    ckpt = torch.load(ckpt_path, map_location=device)
    wp = ckpt["world_params"]
    width = wp["width"]
    height = wp["height"]
    ball_radius = wp["ball_radius"]
    v_max = wp["v_max"]
    dt = wp["dt"]

    # same neighborhood radius as training
    ball_radius_norm = ball_radius / width
    neighborhood_radius = 3.5 * ball_radius_norm

    state_dim = 5  # [x, y, vx, vy, mass]
    model = NeuralPhysicsEngine(
        state_dim=state_dim,
        neighborhood_radius=neighborhood_radius,
    ).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    return model, width, height, ball_radius, v_max, dt


def load_modern_npe(ckpt_path, device, width, ball_radius):
    ckpt = torch.load(ckpt_path, map_location=device)
    cfg = ckpt["config"]

    msg_dim = cfg.get("msg_dim", 96)
    enc_blocks = cfg.get("enc_blocks", 3)
    dec_blocks = cfg.get("dec_blocks", 3)
    dropout = cfg.get("dropout", 0.0)
    expansion = cfg.get("expansion", 2)

    ball_radius_norm = ball_radius / width
    neighborhood_radius = 3.5 * ball_radius_norm

    state_dim = 5
    model = ModernNeuralPhysicsEngine(
        state_dim=state_dim,
        neighborhood_radius=neighborhood_radius,
        msg_dim=msg_dim,
        enc_blocks=enc_blocks,
        dec_blocks=dec_blocks,
        dropout=dropout,
        expansion=expansion,
    ).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    return model


def rollout_with_models(
    classic_model,
    modern_model,
    gt_traj,
    device,
    rollout_steps,
    v_max,
    dt,
    width,
):
    """
    gt_traj: [T_total, N, D] normalized states from simulator
    Returns:
        gt_traj_cpu, classic_traj_cpu, modern_traj_cpu
        each [T_total, N, D]
    """
    T_total, N, D = gt_traj.shape
    assert T_total == rollout_steps + 2

    integration_factor = v_max * dt / width

    def rollout_one(model):
        states = [gt_traj[0], gt_traj[1]]  # s_{t-1}, s_t
        use_autocast = device.type == "cuda"

        for _ in range(rollout_steps):
            s_tm1 = states[-2]
            s_t = states[-1]
            window = torch.stack([s_tm1, s_t, s_t], dim=0)[None, ...]  # [1,3,N,D]

            with torch.no_grad():
                if use_autocast:
                    with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
                        pred_v, _ = model(window)
                else:
                    pred_v, _ = model(window)

            pred_v = pred_v[0]  # [N,2]

            pos_t = s_t[..., 0:2]
            pos_next = pos_t + pred_v * integration_factor

            s_next = s_t.clone()
            s_next[..., 0:2] = pos_next
            s_next[..., 2:4] = pred_v
            states.append(s_next)

        return torch.stack(states, dim=0)

    classic_traj = rollout_one(classic_model)
    modern_traj = rollout_one(modern_model)

    return gt_traj.cpu(), classic_traj.cpu(), modern_traj.cpu()


def render_frame_triplet(
    pos_gt_px,
    pos_classic_px,
    pos_modern_px,
    width,
    height,
    ball_radius,
    colors,
    t_index,
    dpi=120,
):
    """
    pos_*_px: [N,2] in pixel space
    Returns RGB uint8 frame.
    """
    fig, axes = plt.subplots(1, 3, figsize=(9, 3), dpi=dpi)
    titles = ["Ground truth", "Classic NPE", "Modern NPE"]

    for ax, pos, title in zip(
        axes, [pos_gt_px, pos_classic_px, pos_modern_px], titles
    ):
        ax.set_xlim(0, width)
        ax.set_ylim(0, height)
        ax.invert_yaxis()
        ax.set_aspect("equal", adjustable="box")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(title, fontsize=10)

        # draw balls
        for i, (x, y) in enumerate(pos):
            circ = plt.Circle((x, y), radius=ball_radius, color=colors[i])
            ax.add_patch(circ)

        # draw timestep label
        ax.text(
            0.02,
            0.96,
            f"t = {t_index}",
            transform=ax.transAxes,
            fontsize=8,
            ha="left",
            va="top",
        )

    fig.tight_layout(pad=0.3)

    # convert figure to RGB array
    fig.canvas.draw()
    w, h = fig.canvas.get_width_height()
    buf = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
    img = buf.reshape(h, w, 4)[..., :3]  # drop alpha

    plt.close(fig)
    return img


def main():
    parser = argparse.ArgumentParser(
        description="Render GT vs Classic NPE vs Modern NPE rollouts (4 balls)."
    )
    parser.add_argument(
        "--classic_ckpt",
        type=str,
        default="checkpoints/npe_balls_n4_t60_uniform_logged.pt",
        help="Classic NPE checkpoint",
    )
    parser.add_argument(
        "--modern_ckpt",
        type=str,
        required=True,
        help="Modern NPE checkpoint",
    )
    parser.add_argument(
        "--num_scenes",
        type=int,
        default=3,
        help="Number of random scenes to render",
    )
    parser.add_argument(
        "--rollout_steps",
        type=int,
        default=50,
        help="Future steps to roll out",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=20,
        help="Video FPS",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default="videos",
        help="Output directory",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
    )

    args = parser.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    device = torch.device(args.device)
    print("Using device:", device)

    # ---- load models ----
    classic_model, width, height, ball_radius, v_max, dt = load_classic_npe(
        args.classic_ckpt, device
    )
    modern_model = load_modern_npe(
        args.modern_ckpt, device, width=width, ball_radius=ball_radius
    )

    # ---- simulator ----
    sim = PymunkBallWorldSimulator(
        width=width,
        height=height,
        ball_radius=ball_radius,
        v_max=v_max,
        dt=dt,
        num_balls=4,
        variable_mass=False,
        seed=None,  # different random scenes each run
    )

    colors = plt.cm.tab10(np.linspace(0, 1, 4))

    T_total = args.rollout_steps + 2

    for scene_idx in range(args.num_scenes):
        print(f"Rendering scene {scene_idx+1}/{args.num_scenes} ...")

        # ground-truth trajectory from simulator
        with torch.no_grad():
            traj = sim.simulate(T_total)  # [T_total, N, D]

        gt_traj = traj.to(device)

        gt_traj_cpu, classic_traj, modern_traj = rollout_with_models(
            classic_model,
            modern_model,
            gt_traj,
            device=device,
            rollout_steps=args.rollout_steps,
            v_max=v_max,
            dt=dt,
            width=width,
        )

        frames = []
        for t in range(T_total):
            gt_pos_px = gt_traj_cpu[t, :, 0:2] * width
            classic_pos_px = classic_traj[t, :, 0:2] * width
            modern_pos_px = modern_traj[t, :, 0:2] * width

            frame = render_frame_triplet(
                gt_pos_px.numpy(),
                classic_pos_px.numpy(),
                modern_pos_px.numpy(),
                width=width,
                height=height,
                ball_radius=ball_radius,
                colors=colors,
                t_index=t,
                dpi=120,
            )
            frames.append(frame)

        out_path = os.path.join(
            args.out_dir, f"rollout_gt_classic_modern_scene{scene_idx+1}.mp4"
        )
        imageio.mimsave(out_path, frames, fps=args.fps)
        print("  Saved:", out_path)


if __name__ == "__main__":
    main()
