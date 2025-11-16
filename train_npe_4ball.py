# train_npe_4ball.py
import argparse
import os
import random
import time
import csv
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from env_balls_pymunk import WindowDataset
from models import NeuralPhysicsEngine


def _unwrap_for_saving(model: torch.nn.Module) -> torch.nn.Module:
    """Return the underlying uncompiled model for clean state_dict saving."""
    return getattr(model, "_orig_mod", model)


def train_npe_4ball(argv=None):
    parser = argparse.ArgumentParser(
        description="Train NPE on 4-ball equal-mass prediction dataset (with CSV logging)"
    )

    parser.add_argument(
        "--dataset_prefix",
        type=str,
        default="data/balls_n4_t60_uniform/balls_n4_t60_uniform",
        help="Prefix for train/val/test .pt files (expects *_train.pt, *_val.pt, *_test.pt)",
    )
    parser.add_argument("--batch_size", type=int, default=50, help="Paper uses 50")
    parser.add_argument(
        "--max_steps",
        type=int,
        default=1_200_000,
        help="Total optimization steps (paper uses 1,200,000)",
    )
    parser.add_argument("--eval_every", type=int, default=5_000)
    parser.add_argument("--log_every", type=int, default=500)
    parser.add_argument("--lr", type=float, default=3e-4, help="Paper: 3e-4 (RMSprop)")
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    parser.add_argument("--save_dir", type=str, default="checkpoints")
    # NEW DEFAULT NAME so you don't overwrite your previous run
    parser.add_argument("--name", type=str, default="npe_balls_n4_t60_uniform_logged")

    parser.add_argument("--metrics_csv", type=str, default="logs/npe_balls_n4_t60_uniform_logged.csv",
                        help="Path to write train/val/test metrics CSV")
    parser.add_argument("--seed", type=int, default=0)

    # dataloader performance knobs
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--pin_memory", action="store_true")
    parser.add_argument("--no_pin_memory", action="store_true")
    parser.set_defaults(pin_memory=True)  # default: True unless --no_pin_memory

    args = parser.parse_args(argv)

    # ---------- global backend / GPU optimization ----------
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True  # allow TF32 for matmul on Ampere+
    torch.set_float32_matmul_precision("high")        # prefer fast matmuls

    device = torch.device(args.device)
    print(f"Using device: {device}")

    # ---------- seeds ----------
    torch.manual_seed(args.seed)
    random.seed(args.seed)

    # world params (match generator)
    width = 800.0
    height = 600.0
    ball_radius = 60.0
    v_max = 60.0
    dt = 0.1

    # neighborhood radius: 3.5 ball radii (positions normalized by width)
    ball_radius_norm = ball_radius / width
    neighborhood_radius = 3.5 * ball_radius_norm

    # ---------- load dataset ----------
    prefix = args.dataset_prefix
    train_path = prefix + "_train.pt"
    val_path = prefix + "_val.pt"
    test_path = prefix + "_test.pt"

    print(f"Loading train set from {train_path}")
    train_windows = torch.load(train_path)
    print(f"Loading val set from {val_path}")
    val_windows = torch.load(val_path)
    print(f"Loading test set from {test_path}")
    test_windows = torch.load(test_path)

    state_dim = train_windows.shape[-1]
    print(f"State dim D = {state_dim}, window shape = {train_windows.shape[1:]}")

    train_ds = WindowDataset(train_windows)
    val_ds = WindowDataset(val_windows)
    test_ds = WindowDataset(test_windows)

    # pin_memory defaults to True unless explicitly disabled
    pin_memory = args.pin_memory and not args.no_pin_memory

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
        persistent_workers=True if args.num_workers > 0 else False,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
        persistent_workers=True if args.num_workers > 0 else False,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
        persistent_workers=True if args.num_workers > 0 else False,
    )

    # ---------- model ----------
    model = NeuralPhysicsEngine(
        state_dim=state_dim,
        neighborhood_radius=neighborhood_radius,
    ).to(device)

    # compile for speed; we'll unwrap on save to avoid _orig_mod keys
    model = torch.compile(model)

    optimizer = torch.optim.RMSprop(model.parameters(), lr=args.lr)

    # autocast setup: bfloat16 on GPU
    use_autocast = device.type == "cuda"
    autocast_dtype = torch.bfloat16

    # ---------- metrics CSV ----------
    metrics_path = Path(args.metrics_csv)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)

    def _csv_write_header_if_needed():
        if not metrics_path.exists():
            with open(metrics_path, "w", newline="") as f:
                w = csv.writer(f)
                w.writerow(["step", "split", "loss", "lr", "minutes_elapsed"])

    def _csv_log(step, split, loss_value, lr_value, minutes_elapsed):
        with open(metrics_path, "a", newline="") as f:
            w = csv.writer(f)
            w.writerow([step, split, f"{loss_value:.8f}", f"{lr_value:.6e}", f"{minutes_elapsed:.2f}"])

    _csv_write_header_if_needed()

    # ---------- evaluation helper ----------
    def evaluate(loader):
        model.eval()
        total_loss = 0.0
        total_samples = 0
        with torch.inference_mode():
            for batch in loader:
                batch = batch.to(device, non_blocking=True)
                if use_autocast:
                    with torch.autocast(device_type=device.type, dtype=autocast_dtype):
                        pred_v, target_v = model(batch)
                        loss = F.mse_loss(pred_v, target_v, reduction="mean")
                else:
                    pred_v, target_v = model(batch)
                    loss = F.mse_loss(pred_v, target_v, reduction="mean")

                bs = batch.size(0)
                total_loss += loss.item() * bs
                total_samples += bs
        model.train()
        return total_loss / max(1, total_samples)

    # ---------- train loop ----------
    os.makedirs(args.save_dir, exist_ok=True)
    ckpt_best = os.path.join(args.save_dir, f"{args.name}.pt")
    ckpt_last = os.path.join(args.save_dir, f"{args.name}_last.pt")

    global_step = 0
    best_val_loss = float("inf")
    start_time = time.time()

    print("Starting training...")
    model.train()
    while global_step < args.max_steps:
        for batch in train_loader:
            global_step += 1
            batch = batch.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            if use_autocast:
                with torch.autocast(device_type=device.type, dtype=autocast_dtype):
                    pred_v, target_v = model(batch)
                    loss = F.mse_loss(pred_v, target_v, reduction="mean")
            else:
                pred_v, target_v = model(batch)
                loss = F.mse_loss(pred_v, target_v, reduction="mean")

            loss.backward()
            optimizer.step()

            # LR schedule: decay 0.99 every 2500 steps after 50k steps (paper)
            if global_step > 50_000 and global_step % 2_500 == 0:
                for pg in optimizer.param_groups:
                    pg["lr"] *= 0.99

            # logging
            if global_step % args.log_every == 0:
                elapsed_min = (time.time() - start_time) / 60.0
                cur_lr = optimizer.param_groups[0]["lr"]
                print(f"[step {global_step}] train_loss={loss.item():.6f} elapsed={elapsed_min:.2f} min lr={cur_lr:.1e}")
                _csv_log(global_step, "train", loss.item(), cur_lr, elapsed_min)

            # validation
            if global_step % args.eval_every == 0:
                val_loss = evaluate(val_loader)
                cur_lr = optimizer.param_groups[0]["lr"]
                elapsed_min = (time.time() - start_time) / 60.0
                print(f"[step {global_step}] VAL MSE={val_loss:.6f}")
                _csv_log(global_step, "val", val_loss, cur_lr, elapsed_min)

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    to_save = _unwrap_for_saving(model)
                    torch.save(
                        {
                            "model_state": to_save.state_dict(),
                            "optimizer_state": optimizer.state_dict(),
                            "step": global_step,
                            "val_loss": best_val_loss,
                            "config": vars(args),
                            "world_params": {
                                "width": width,
                                "height": height,
                                "ball_radius": ball_radius,
                                "v_max": v_max,
                                "dt": dt,
                            },
                        },
                        ckpt_best,
                    )
                    print(f"  Saved new BEST checkpoint -> {ckpt_best}")

            if global_step >= args.max_steps:
                break

    print("Training finished.")
    print(f"Best validation MSE: {best_val_loss:.6f}")

    # save LAST checkpoint too (for completeness)
    to_save = _unwrap_for_saving(model)
    torch.save(
        {
            "model_state": to_save.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "step": global_step,
            "val_loss": best_val_loss,
            "config": vars(args),
            "world_params": {
                "width": width,
                "height": height,
                "ball_radius": ball_radius,
                "v_max": v_max,
                "dt": dt,
            },
        },
        ckpt_last,
    )
    print(f"Saved LAST checkpoint -> {ckpt_last}")

    # final test
    test_loss = evaluate(test_loader)
    elapsed_min = (time.time() - start_time) / 60.0
    cur_lr = optimizer.param_groups[0]["lr"]
    print(f"Final test MSE: {test_loss:.6f}")
    _csv_log(global_step, "test_final", test_loss, cur_lr, elapsed_min)

    return model


if __name__ == "__main__":
    train_npe_4ball()
