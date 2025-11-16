# plot_compare_classic_vs_modern_samples.py
import argparse
import pandas as pd
import matplotlib.pyplot as plt


def ema(values, alpha):
    out = []
    m = None
    for v in values:
        m = v if m is None else alpha * v + (1.0 - alpha) * m
        out.append(m)
    return out


def prepare_df(df, batch_size_for_step=None, smooth_alpha=0.98):
    """
    Expects columns:
      - 'step'
      - 'split' in {'train','val','test_final'}
      - 'loss'
      - optional 'samples_seen'
    """
    df = df[df["split"].isin(["train", "val", "test_final"])]

    # compute samples_seen if not present (classic run)
    if "samples_seen" not in df.columns:
        if batch_size_for_step is None:
            raise ValueError("samples_seen missing and no batch_size_for_step provided")
        df["samples_seen"] = df["step"] * batch_size_for_step

    # split
    df_train = df[df["split"] == "train"].copy()
    df_val   = df[df["split"] == "val"].copy()
    df_test  = df[df["split"] == "test_final"].copy()

    # EMA smoothing
    if smooth_alpha > 0 and not df_train.empty:
        df_train["loss_ema"] = ema(df_train["loss"].tolist(), smooth_alpha)
    else:
        df_train["loss_ema"] = df_train["loss"]

    if smooth_alpha > 0 and not df_val.empty:
        df_val["loss_ema"] = ema(df_val["loss"].tolist(), smooth_alpha)
    else:
        df_val["loss_ema"] = df_val["loss"]

    return df_train, df_val, df_test


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--classic_csv", required=True,
                    help="metrics CSV from classic train_npe_4ball.py")
    ap.add_argument("--modern_csv", required=True,
                    help="metrics CSV from modern train_npe_4ball_modern.py")
    ap.add_argument("--classic_batch_size", type=int, default=50,
                    help="batch size used in classic NPE training (for samples_seen=step*B)")
    ap.add_argument("--smooth", type=float, default=0.98,
                    help="EMA alpha; 0 disables smoothing")
    ap.add_argument("--out", type=str,
                    default="figs/classic_vs_modern_train4_test4_samples.png")
    args = ap.parse_args()

    # --- load & prepare classic ---
    df_classic = pd.read_csv(args.classic_csv)
    c_train, c_val, c_test = prepare_df(
        df_classic,
        batch_size_for_step=args.classic_batch_size,
        smooth_alpha=args.smooth,
    )

    # --- load & prepare modern ---
    df_modern = pd.read_csv(args.modern_csv)
    m_train, m_val, m_test = prepare_df(
        df_modern,
        batch_size_for_step=None,   # modern logs already have samples_seen
        smooth_alpha=args.smooth,
    )

    fig, ax = plt.subplots(figsize=(10, 5))

    # x-axis in millions of samples
    if not c_train.empty:
        ax.plot(
            c_train["samples_seen"] / 1e6,
            c_train["loss_ema"],
            label="Classic NPE – train (EMA)",
            alpha=0.7,
        )
    if not c_val.empty:
        ax.plot(
            c_val["samples_seen"] / 1e6,
            c_val["loss_ema"],
            label="Classic NPE – val (EMA)",
            alpha=0.9,
        )

    if not m_train.empty:
        ax.plot(
            m_train["samples_seen"] / 1e6,
            m_train["loss_ema"],
            label="Modern NPE – train (EMA)",
            alpha=0.7,
        )
    if not m_val.empty:
        ax.plot(
            m_val["samples_seen"] / 1e6,
            m_val["loss_ema"],
            label="Modern NPE – val (EMA)",
            alpha=0.9,
        )

    # mark final test points if present
    if not c_test.empty:
        sx = c_test["samples_seen"].iloc[-1] / 1e6
        sy = c_test["loss"].iloc[-1]
        ax.scatter([sx], [sy], marker="x", s=80,
                   label=f"Classic test_final={sy:.6e}")

    if not m_test.empty:
        sx = m_test["samples_seen"].iloc[-1] / 1e6
        sy = m_test["loss"].iloc[-1]
        ax.scatter([sx], [sy], marker="x", s=80,
                   label=f"Modern test_final={sy:.6e}")

    ax.set_xlabel("samples seen (millions)")
    ax.set_ylabel("MSE (normalized velocity)")
    ax.set_title("Train: 4 objects \u2192 Test: 4 objects")

    ax.set_yscale("log")
    ax.set_ylim(1e-5, 1e-2)

    ax.grid(True, which="both", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(args.out, dpi=200)
    print(f"Saved comparison plot to {args.out}")


if __name__ == "__main__":
    main()
