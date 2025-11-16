# train_npe_4ball_modern.py
import argparse
import csv
import math
import os
import random
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from env_balls_pymunk import WindowDataset
from models_modern import ModernNeuralPhysicsEngine


def _unwrap_for_saving(model: torch.nn.Module) -> torch.nn.Module:
    """Handle torch.compile wrapping when saving state_dict."""
    return getattr(model, "_orig_mod", model)


def build_optimizer(params, opt_name: str, lr: float, weight_decay: float):
    opt_name = opt_name.lower()
    if opt_name == "adamw":
        return torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)
    elif opt_name == "adam":
        return torch.optim.Adam(params, lr=lr, weight_decay=weight_decay)
    elif opt_name == "rmsprop":
        return torch.optim.RMSprop(params, lr=lr, weight_decay=weight_decay)
    else:
        raise ValueError(f"Unknown optimizer '{opt_name}'")


def train_npe_4ball_modern(argv=None):
    parser = argparse.ArgumentParser(
        description="Train modern NPE on 4-ball equal-mass prediction dataset "
                    "(modern-medium: msg_dim≈96, residual MLPs)."
    )

    # ----- data / IO -----
    parser.add_argument(
        "--dataset_prefix",
        type=str,
        default="data/balls_n4_t60_uniform/balls_n4_t60_uniform",
        help="Prefix for train/val/test .pt files "
             "(expects *_train.pt, *_val.pt, *_test.pt)",
    )
    parser.add_argument("--save_dir", type=str, default="checkpoints")
    parser.add_argument("--name", type=str,
                        default="npe_modern_balls_n4_medium")
    parser.add_argument("--metrics_csv", type=str,
                        default="logs/npe_modern_n4_medium.csv")

    # ----- training hyperparams -----
    parser.add_argument("--batch_size", type=int, default=2048)
    parser.add_argument(
        "--total_samples",
        type=int,
        default=120_000_000,  # for best results; set 60M to match paper exactly
        help="Total windows to process.",
    )
    parser.add_argument("--optimizer", type=str, default="adamw",
                        choices=["adamw", "adam", "rmsprop"])
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--eval_every", type=int, default=5000,
                        help="Evaluate val set every N steps (updates).")
    parser.add_argument("--log_every", type=int, default=500,
                        help="Log training loss every N steps.")

    # ----- model size (modern-medium defaults) -----
    parser.add_argument("--msg_dim", type=int, default=96)
    parser.add_argument("--enc_blocks", type=int, default=3)
    parser.add_argument("--dec_blocks", type=int, default=3)
    parser.add_argument("--expansion", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.0)

    # ----- system / misc -----
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--pin_memory", action="store_true")
    parser.add_argument("--no_pin_memory", action="store_true")
    parser.set_defaults(pin_memory=True)

    args = parser.parse_args(argv)

    # ---------- global backend / GPU optimization ----------
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
    torch.set_float32_matmul_precision("high")

    device = torch.device(args.device)
    print(f"Using device: {device}")

    # ---------- seeds ----------
    torch.manual_seed(args.seed)
    random.seed(args.seed)

    # world params (should match generator)
    width = 800.0
    height = 600.0
    ball_radius = 60.0
    v_max = 60.0
    dt = 0.1

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

    pin_memory = args.pin_memory and not args.no_pin_memory
    persistent = True if args.num_workers > 0 else False

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent,
    )

    # ---------- model ----------
    model = ModernNeuralPhysicsEngine(
        state_dim=state_dim,
        neighborhood_radius=neighborhood_radius,
        msg_dim=args.msg_dim,
        enc_blocks=args.enc_blocks,
        dec_blocks=args.dec_blocks,
        dropout=args.dropout,
        expansion=args.expansion,
    ).to(device)

    # compile for speed
    model = torch.compile(model)

    # ---------- optimizer + scheduler ----------
    optimizer = build_optimizer(
        params=model.parameters(),
        opt_name=args.optimizer,
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    # estimate total training steps
    total_steps_est = math.ceil(args.total_samples / args.batch_size)
    warmup_steps = max(1, int(0.1 * total_steps_est))
    main_steps = max(1, total_steps_est - warmup_steps)

    sched_warm = torch.optim.lr_scheduler.LinearLR(
        optimizer, start_factor=0.1, total_iters=warmup_steps
    )
    sched_main = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=main_steps
    )
    scheduler = torch.optim.lr_scheduler.SequentialLR(
        optimizer, [sched_warm, sched_main], milestones=[warmup_steps]
    )

    use_autocast = device.type == "cuda"
    autocast_dtype = torch.bfloat16

    # ---------- metrics CSV ----------
    metrics_path = Path(args.metrics_csv)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)

    if not metrics_path.exists():
        with open(metrics_path, "w", newline="") as f:
            csv.writer(f).writerow(
                ["step", "samples_seen", "split", "loss", "lr", "minutes_elapsed"]
            )

    def log_metric(step, samples_seen, split, loss_value, lr_value, minutes_elapsed):
        with open(metrics_path, "a", newline="") as f:
            csv.writer(f).writerow(
                [
                    step,
                    samples_seen,
                    split,
                    f"{loss_value:.8f}",
                    f"{lr_value:.6e}",
                    f"{minutes_elapsed:.2f}",
                ]
            )

    # ---------- evaluation helper ----------
    def evaluate(loader):
        model.eval()
        total_loss = 0.0
        total_samples = 0
        with torch.inference_mode():
            for batch in loader:
                batch = batch.to(device, non_blocking=True)
                if use_autocast:
                    with torch.autocast(
                        device_type=device.type, dtype=autocast_dtype
                    ):
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
    samples_seen = 0
    best_val_loss = float("inf")
    start_time = time.time()

    print(
        f"Starting training with batch_size={args.batch_size}, "
        f"target total_samples={args.total_samples}, "
        f"optimizer={args.optimizer}, lr={args.lr}"
    )

    model.train()

    while samples_seen < args.total_samples:
        for batch in train_loader:
            bs = batch.size(0)
            global_step += 1
            samples_seen += bs

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
            if args.max_grad_norm is not None and args.max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            optimizer.step()
            scheduler.step()

            # logging
            if global_step % args.log_every == 0:
                elapsed_min = (time.time() - start_time) / 60.0
                cur_lr = optimizer.param_groups[0]["lr"]
                print(
                    f"[step {global_step}] samples={samples_seen} "
                    f"train_loss={loss.item():.6f} "
                    f"elapsed={elapsed_min:.2f} min "
                    f"lr={cur_lr:.1e}"
                )
                log_metric(global_step, samples_seen, "train",
                           loss.item(), cur_lr, elapsed_min)

            # validation
            if global_step % args.eval_every == 0:
                elapsed_min = (time.time() - start_time) / 60.0
                cur_lr = optimizer.param_groups[0]["lr"]

                val_loss = evaluate(val_loader)
                print(
                    f"[step {global_step}] samples={samples_seen} "
                    f"VAL MSE={val_loss:.6f}"
                )
                log_metric(global_step, samples_seen, "val",
                           val_loss, cur_lr, elapsed_min)

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    base_model = _unwrap_for_saving(model)
                    torch.save(
                        {
                            "model_state": base_model.state_dict(),
                            "optimizer_state": optimizer.state_dict(),
                            "step": global_step,
                            "samples_seen": samples_seen,
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

            if samples_seen >= args.total_samples:
                break

    print("Training finished.")
    print(f"Best validation MSE: {best_val_loss:.6f}")

    # save LAST
    base_model = _unwrap_for_saving(model)
    torch.save(
        {
            "model_state": base_model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "step": global_step,
            "samples_seen": samples_seen,
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

    # final test on 4-ball test split
    test_loss = evaluate(test_loader)
    elapsed_min = (time.time() - start_time) / 60.0
    cur_lr = optimizer.param_groups[0]["lr"]
    print(f"Final test MSE: {test_loss:.6f}")
    log_metric(global_step, samples_seen, "test_final",
               test_loss, cur_lr, elapsed_min)

    return model


if __name__ == "__main__":
    train_npe_4ball_modern()
