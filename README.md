# Neural Physics Engine

A PyTorch re-implementation of the ICLR 2017 paper **"A Compositional Object-Based Approach to Learning Physical Dynamics"** by Watters et al.

**Paper**: [arXiv:1612.00341](https://arxiv.org/abs/1612.00341)

This repository includes both the original architecture (classic NPE) and an enhanced modern version with residual connections, layer normalization, and improved training techniques.






![Training curves](https://github.com/user-attachments/assets/3432b75c-85b1-4e1e-81a0-6250df33cf48)






![Simulation Gif-1](https://github.com/user-attachments/assets/661fd1bd-f140-4e6b-9538-045c4ba43988)






![Simulation Gif-2](https://github.com/user-attachments/assets/9e43263e-8ac5-4eaa-97ca-96394cef0377)




## Requirements

* Python 3.8+
* [PyTorch](https://pytorch.org/) (tested with 2.0+)
* CUDA-compatible GPU (recommended for training)

### Dependencies

To install Python dependencies, run:

```bash
pip install -r requirements.txt
```

The `requirements.txt` includes:
- `torch` - PyTorch for neural networks
- `pymunk` - 2D physics simulation engine
- `numpy` - Numerical operations
- `pandas` - CSV logging and data analysis
- `matplotlib` - Plotting and visualization
- `imageio` - Video generation
- `tqdm` - Progress bars

## Instructions

_NOTE: This is a PyTorch re-implementation. The original paper used Torch7/Lua with matter-js for data generation._

### Generating Data

The code uses PyMunk (Chipmunk2D) to generate physics simulations. Data generation creates sliding 3-frame windows `(t-1, t, t+1)` for training.

This is an example of generating 12000 trajectories of 4 balls of equal mass over 60 timesteps. It will create files in the `data/balls_n4_t60_uniform/` folder.

```bash
python generate_dataset_4ball.py `
  --num_scenes 12000 `
  --out_prefix data/balls_n4_t60_uniform/balls_n4_t60_uniform `
  --seed 0
```

This generates approximately 700K training windows (12000 scenes × 58 windows per scene), split into 70% train / 15% val / 15% test.

For reproducing paper results, generate 50000 trajectories:

```bash
python generate_dataset_4ball.py `
  --num_scenes 50000 `
  --out_prefix data/balls_n4_t60_uniform/balls_n4_t60_uniform `
  --seed 0
```

For debugging purposes, 200-1000 trajectories is sufficient:

```bash
python generate_dataset_4ball.py `
  --num_scenes 1000 `
  --out_prefix data/balls_n4_t60_uniform/balls_n4_t60_uniform `
  --seed 0
```

**Data Format**: Each state vector contains `[x_norm, y_norm, vx_norm, vy_norm, mass]` where positions and velocities are normalized to [0,1] range. Output files are saved as PyTorch tensors with shape `[num_windows, 3, 4, 5]`.

### Training the Model

If training on Windows without setting up Triton, do not use torch.compile().
https://github.com/woct0rdho/triton-windows

#### Classic NPE (Original Paper Architecture)

This is an example of training the classic NPE model on the `balls_n4_t60_uniform` dataset. Model checkpoints are saved in `checkpoints/npe_balls_n4_t60_uniform_logged.pt`. Training metrics are logged to `logs/npe_n4.csv`.

```bash
python train_npe_4ball.py `
  --dataset_prefix data/balls_n4_t60_uniform/balls_n4_t60_uniform `
  --batch_size 50 `
  --total_samples 60000000 `
  --eval_every 2000 `
  --name npe_balls_n4_t60_uniform_logged `
  --metrics_csv logs/npe_n4.csv
```
#### Modern NPE (Enhanced Architecture)

This is an example of training the modern NPE with residual connections and advanced optimization. Model checkpoints are saved in `checkpoints/npe_modern_balls_n4_medium_adamw_cosine_200M.pt`.

```bash
python train_npe_4ball_modern.py `
  --dataset_prefix data/balls_n4_t60_uniform/balls_n4_t60_uniform `
  --batch_size 2048 `
  --total_samples 200000000 `
  --optimizer adamw `
  --lr 1e-3 `
  --weight_decay 1e-4 `
  --dropout 0.0 `
  --max_grad_norm 1.0 `
  --msg_dim 96 `
  --enc_blocks 3 `
  --dec_blocks 3 `
  --expansion 2 `
  --name npe_modern_balls_n4_medium_adamw_cosine_200M `
  --metrics_csv logs/npe_modern_n4_medium_adamw_cosine_200M.csv
```
**GPU Memory**: If you encounter OOM errors, reduce `--batch_size` (e.g., 2048→1024→512) or reduce `--msg_dim` (e.g., 96→64). Use `--no_pin_memory` flag if needed. The code defaults to GPU if available, but you can force CPU with `--device cpu`.

### Visualization

#### Training Curves

This is an example of plotting training and validation loss curves for both classic and modern NPE models. The plot is saved in `figs/train_val_classic_vs_modern.png`.

```bash
python plot_compare_classic_vs_modern_samples.py `
  --classic_csv logs/npe_n4.csv `
  --modern_csv logs/npe_modern_n4_medium_adamw_cosine_200M.csv `
  --out_png figs/train_val_classic_vs_modern.png
```

Optional parameters:
- `--classic_batch_size 50`: Batch size used in classic training (for computing samples_seen)
- `--smooth 0.98`: EMA smoothing factor (0 = no smoothing)

The output shows log-scale MSE loss vs. training samples (in millions) for both train and validation splits.

#### Rollout Videos

This is an example of rendering side-by-side comparison videos of ground truth vs. model predictions. Videos are saved in `videos/rollout_gt_classic_modern_scene{N}.mp4`.

```bash
python render_rollout_triple.py `
  --classic_ckpt checkpoints/npe_balls_n4_t60_uniform_logged.pt `
  --modern_ckpt checkpoints/npe_modern_balls_n4_medium_adamw_cosine_200M.pt `
  --num_scenes 5 `
  --rollout_steps 60 `
  --out_dir videos
```

Each video shows three panels: Ground Truth | Classic NPE | Modern NPE, allowing visual comparison of autoregressive prediction quality.

### Prediction (Rollout Evaluation)

This is an example of quantitatively evaluating prediction quality using cosine similarity and relative magnitude error metrics. The plot is saved in `figs/prediction_task_classic_vs_modern_2panels.png`.

```bash
python compare_rollouts_classic_vs_modern.py `
  --classic_ckpt checkpoints/npe_balls_n4_t60_uniform_logged.pt `
  --modern_ckpt checkpoints/npe_modern_balls_n4_medium_adamw_cosine_200M.pt `
  --num_scenes 200 `
  --rollout_steps 50 `
  --out_png figs/prediction_task_classic_vs_modern_2panels.png
```

**Evaluation Metrics**:
1. **Cosine Similarity** (higher is better, max=1.0): Measures alignment between predicted and ground-truth velocity vectors
   - Formula: `cos(v_pred, v_gt) = (v_pred · v_gt) / (|v_pred| |v_gt|)`
2. **Relative Magnitude Error** (lower is better, min=0.0): Measures error in velocity magnitude
   - Formula: `|∥v_gt∥ - ∥v_pred∥| / ∥v_gt∥`

The output is a 2-panel figure showing how these metrics degrade over 50 autoregressive timesteps.
