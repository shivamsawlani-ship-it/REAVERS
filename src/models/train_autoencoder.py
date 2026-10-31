"""
CAVShield LSTM Autoencoder Training Pipeline.

Trains a temporal LSTM Autoencoder on normal CAN sequence windows,
computes validation reconstruction errors, establishes the 99.5th percentile
anomaly threshold, and evaluates separation against cyber-physical attacks.

Outputs:
- Model weights: output/models/anomaly_ae.pt
- Metadata and normalization: output/models/ae_metadata.json
- Loss and error distribution plots: output/figures/ae_reconstruction_error.html and .png
"""

from __future__ import annotations

import argparse
import json
import logging
import math
from pathlib import Path
import struct
import sys
import time
from typing import Any, Dict, List, Optional, Tuple
import zlib

import numpy as np
import pandas as pd
import plotly.graph_objects as go  # type: ignore
from plotly.subplots import make_subplots  # type: ignore
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from src import load_config
from src.can_simulator import KNOWN_IDS
from src.data_loader import (
    build_window_dataset,
    generate_synthetic_dataset,
    split_train_test_by_run,
)
from src.models.lstm_autoencoder import LSTMAutoencoder

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def _write_simple_png(
    filename: Path,
    width: int,
    height: int,
    pixels: bytes,
) -> None:
    """Write an RGB bitmap to a standard PNG file without third-party imaging libraries."""
    def chunk(chunk_type: bytes, data: bytes) -> bytes:
        c = chunk_type + data
        crc = zlib.crc32(c) & 0xFFFFFFFF
        return struct.pack(">I", len(data)) + c + struct.pack(">I", crc)

    # PNG signature
    header = b"\x89PNG\r\n\x1a\n"
    # IHDR chunk: width, height, bit_depth=8, color_type=2 (RGB), compression=0, filter=0, interlace=0
    ihdr = chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0))

    # Raw scanlines with filter byte 0 (None)
    scanlines = bytearray()
    row_bytes = width * 3
    for y in range(height):
        scanlines.append(0)  # filter type None
        scanlines.extend(pixels[y * row_bytes : (y + 1) * row_bytes])

    compressed = zlib.compress(bytes(scanlines), level=6)
    idat = chunk(b"IDAT", compressed)
    iend = chunk(b"IEND", b"")

    with open(filename, "wb") as f:
        f.write(header + ihdr + idat + iend)


def _render_bitmap_figure(
    train_losses: List[float],
    val_losses: List[float],
    normal_errors: np.ndarray,
    attack_errors: np.ndarray,
    threshold: float,
    output_png: Path,
) -> None:
    """Render a clean summary bitmap showing training loss and reconstruction histograms."""
    width, height = 800, 400
    # Create white canvas (RGB)
    canvas = bytearray([255, 255, 255] * (width * height))

    def set_pixel(x: int, y: int, r: int, g: int, b: int) -> None:
        if 0 <= x < width and 0 <= y < height:
            idx = (y * width + x) * 3
            canvas[idx] = r
            canvas[idx + 1] = g
            canvas[idx + 2] = b

    def draw_line(x0: int, y0: int, x1: int, y1: int, r: int, g: int, b: int) -> None:
        dx = abs(x1 - x0)
        dy = abs(y1 - y0)
        sx = 1 if x0 < x1 else -1
        sy = 1 if y0 < y1 else -1
        err = dx - dy
        while True:
            set_pixel(x0, y0, r, g, b)
            if x0 == x1 and y0 == y1:
                break
            e2 = 2 * err
            if e2 > -dy:
                err -= dy
                x0 += sx
            if e2 < dx:
                err += dx
                y0 += sy

    # Plot 1: Loss curves (left side: x in [50, 360], y in [50, 340])
    l_x0, l_x1, l_y0, l_y1 = 60, 360, 50, 340
    # Axes
    draw_line(l_x0, l_y1, l_x1, l_y1, 100, 100, 100)
    draw_line(l_x0, l_y0, l_x0, l_y1, 100, 100, 100)

    n_epochs = len(train_losses)
    max_loss = max(max(train_losses), max(val_losses)) if n_epochs > 0 else 1.0
    min_loss = min(min(train_losses), min(val_losses)) if n_epochs > 0 else 0.0
    loss_range = max(max_loss - min_loss, 1e-6)

    def loss_to_y(loss: float) -> int:
        norm = (loss - min_loss) / loss_range
        return int(l_y1 - norm * (l_y1 - l_y0))

    def epoch_to_x(ep: int) -> int:
        return int(l_x0 + (ep / max(n_epochs - 1, 1)) * (l_x1 - l_x0))

    for i in range(n_epochs - 1):
        # Train loss (blue)
        draw_line(epoch_to_x(i), loss_to_y(train_losses[i]), epoch_to_x(i + 1), loss_to_y(train_losses[i + 1]), 30, 100, 220)
        # Val loss (green)
        draw_line(epoch_to_x(i), loss_to_y(val_losses[i]), epoch_to_x(i + 1), loss_to_y(val_losses[i + 1]), 30, 180, 50)

    # Plot 2: Histogram of Reconstruction Errors (right side: x in [440, 750], y in [50, 340])
    r_x0, r_x1, r_y0, r_y1 = 440, 750, 50, 340
    draw_line(r_x0, r_y1, r_x1, r_y1, 100, 100, 100)
    draw_line(r_x0, r_y0, r_x0, r_y1, 100, 100, 100)

    max_err = max(float(np.percentile(attack_errors, 98)) if len(attack_errors) else 1.0, threshold * 2.0)
    min_err = 0.0
    bins = 30
    norm_counts, bin_edges = np.histogram(normal_errors, bins=bins, range=(min_err, max_err))
    att_counts, _ = np.histogram(attack_errors, bins=bins, range=(min_err, max_err))

    max_count = max(int(np.max(norm_counts)), int(np.max(att_counts)), 1)
    bin_w = (r_x1 - r_x0) / bins

    for b in range(bins):
        bx0 = int(r_x0 + b * bin_w)
        bx1 = int(bx0 + bin_w * 0.45)
        bx2 = int(bx0 + bin_w * 0.9)

        # Normal bar (green)
        bar_h_norm = int((norm_counts[b] / max_count) * (r_y1 - r_y0))
        for x in range(bx0, bx1):
            for y in range(r_y1 - bar_h_norm, r_y1):
                set_pixel(x, y, 40, 180, 40)

        # Attack bar (red)
        bar_h_att = int((att_counts[b] / max_count) * (r_y1 - r_y0))
        for x in range(bx1, bx2):
            for y in range(r_y1 - bar_h_att, r_y1):
                set_pixel(x, y, 220, 50, 50)

    # Threshold line (dashed red/purple)
    thresh_x = int(r_x0 + ((threshold - min_err) / max(max_err - min_err, 1e-6)) * (r_x1 - r_x0))
    if r_x0 <= thresh_x <= r_x1:
        for y in range(r_y0, r_y1):
            set_pixel(thresh_x, y, 160, 20, 160)

    _write_simple_png(output_png, width, height, bytes(canvas))


def train_autoencoder(
    cfg: Optional[Any] = None,
    dataset_path: Optional[str] = None,
    quick: bool = False,
    epochs: int = 12,
    batch_size: int = 128,
    lr: float = 0.001,
) -> Dict[str, Any]:
    """Train PyTorch LSTM Autoencoder on normal sequences and evaluate anomaly detection.

    Parameters
    ----------
    cfg : Optional[Any]
        Project configuration.
    dataset_path : Optional[str]
        Path to pre-existing dataset CSV.
    quick : bool
        If True, trains fewer epochs on smaller subset for test speed.
    epochs : int
        Maximum training epochs.
    batch_size : int
        Batch size.
    lr : float
        Initial Adam learning rate.

    Returns
    -------
    Dict[str, Any]
        Training and evaluation results.
    """
    if cfg is None:
        cfg = load_config()

    output_models_dir = Path(getattr(cfg.paths, "output_models", "output/models"))
    output_figures_dir = Path(getattr(cfg.paths, "output_figures", "output/figures"))
    output_models_dir.mkdir(parents=True, exist_ok=True)
    output_figures_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load or Generate Dataset
    data_proc_dir = Path(getattr(cfg.paths, "data_processed", "data/processed"))
    target_csv = (
        Path(dataset_path)
        if dataset_path
        else data_proc_dir / "synthetic_can_dataset.csv.gz"
    )

    if not target_csv.exists():
        logger.info("Dataset not found at %s. Generating synthetic dataset...", target_csv)
        df_can = generate_synthetic_dataset(cfg, quick=quick)
    else:
        logger.info("Loading dataset from %s...", target_csv)
        df_can = pd.read_csv(target_csv)

    # 2. Build Window Features
    X_seq, X_stat, y, subtypes, run_ids, t_ends = build_window_dataset(
        df_can, known_ids=KNOWN_IDS, cfg=cfg
    )
    if len(y) == 0:
        raise RuntimeError("No windows generated from dataset!")

    test_split_pct = float(getattr(cfg.training, "test_split_pct", 25.0))
    train_mask, test_mask = split_train_test_by_run(run_ids, test_split_pct=test_split_pct)

    # 3. Filter Normal Sequences
    normal_train_mask = train_mask & (y == "Normal")
    normal_test_mask = test_mask & (y == "Normal")
    attack_test_mask = test_mask & (y != "Normal")

    X_train_normal = X_seq[normal_train_mask]
    X_val_normal = X_seq[normal_test_mask]
    X_test_attack = X_seq[attack_test_mask]

    logger.info(
        "Sequences: Train Normal=%d, Val Normal=%d, Test Attack=%d",
        len(X_train_normal),
        len(X_val_normal),
        len(X_test_attack),
    )

    # Subsample if necessary to guarantee fast CPU execution (< 3 minutes)
    max_train_samples = 4000 if quick else 8000
    if len(X_train_normal) > max_train_samples:
        rng = np.random.default_rng(42)
        idx_sub = rng.choice(len(X_train_normal), size=max_train_samples, replace=False)
        X_train_normal = X_train_normal[idx_sub]

    # 4. Compute Feature Normalization (mean and std across training sequences)
    # X shape: (N, 64, 17) -> compute mean and std along axis (0, 1) -> (17,)
    feat_mean = np.mean(X_train_normal, axis=(0, 1)).astype(np.float32)
    feat_std = np.std(X_train_normal, axis=(0, 1)).astype(np.float32)
    feat_std[feat_std < 1e-4] = 1.0  # prevent division by zero

    # Normalize data
    def normalize(data: np.ndarray) -> np.ndarray:
        return (data - feat_mean) / feat_std

    X_train_norm = normalize(X_train_normal)
    X_val_norm = normalize(X_val_normal)
    X_attack_norm = normalize(X_test_attack)

    # Convert to PyTorch tensors (CPU)
    device = torch.device("cpu")
    train_dataset = TensorDataset(torch.from_numpy(X_train_norm).float())
    val_dataset = TensorDataset(torch.from_numpy(X_val_norm).float())

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # 5. Initialize Model
    model = LSTMAutoencoder(
        input_dim=17,
        seq_len=64,
        hidden_dim=64,
        bottleneck_dim=24,
        num_layers=2,
        dropout=0.1,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    criterion = nn.MSELoss()

    best_val_loss = float("inf")
    best_weights: Optional[Dict[str, torch.Tensor]] = None
    patience = 3
    patience_counter = 0
    actual_epochs = 4 if quick else epochs

    train_losses: List[float] = []
    val_losses: List[float] = []

    logger.info("Training LSTM Autoencoder on CPU (max %d epochs)...", actual_epochs)
    t0_train = time.time()

    for epoch in range(1, actual_epochs + 1):
        model.train()
        total_train_loss = 0.0
        n_batches = 0
        for (batch_x,) in train_loader:
            batch_x = batch_x.to(device)
            optimizer.zero_grad()
            recon = model(batch_x)
            loss = criterion(recon, batch_x)
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item()
            n_batches += 1

        avg_train_loss = total_train_loss / max(n_batches, 1)
        train_losses.append(avg_train_loss)

        # Validation
        model.eval()
        total_val_loss = 0.0
        n_val_batches = 0
        with torch.no_grad():
            for (batch_val_x,) in val_loader:
                batch_val_x = batch_val_x.to(device)
                recon = model(batch_val_x)
                loss = criterion(recon, batch_val_x)
                total_val_loss += loss.item()
                n_val_batches += 1

        avg_val_loss = total_val_loss / max(n_val_batches, 1)
        val_losses.append(avg_val_loss)

        logger.info(
            "Epoch %d/%d | Train MSE: %.6f | Val MSE: %.6f",
            epoch,
            actual_epochs,
            avg_train_loss,
            avg_val_loss,
        )

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_weights = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                logger.info("Early stopping triggered at epoch %d.", epoch)
                break

    dt_train = time.time() - t0_train
    logger.info("LSTM Autoencoder training completed in %.2f s.", dt_train)

    if best_weights is not None:
        model.load_state_dict(best_weights)
    model.eval()

    # 6. Compute Reconstruction Errors
    def get_errors(data_norm: np.ndarray) -> np.ndarray:
        if len(data_norm) == 0:
            return np.array([], dtype=np.float32)
        errs: List[float] = []
        batch_sz = 256
        with torch.no_grad():
            for i in range(0, len(data_norm), batch_sz):
                bx = torch.from_numpy(data_norm[i : i + batch_sz]).float().to(device)
                err_t = model.reconstruction_error(bx, reduction="none")
                errs.extend(err_t.cpu().numpy().tolist())
        return np.array(errs, dtype=np.float32)

    val_errors = get_errors(X_val_norm)
    attack_errors = get_errors(X_attack_norm)

    # 99.5th percentile threshold from normal validation error
    threshold = float(np.percentile(val_errors, 99.5)) if len(val_errors) > 0 else 0.05
    val_mean = float(np.mean(val_errors)) if len(val_errors) > 0 else 0.01
    val_std = float(np.std(val_errors)) if len(val_errors) > 0 else 0.01

    # Separation evaluation
    att_detected = np.sum(attack_errors >= threshold) if len(attack_errors) > 0 else 0
    detection_rate = float(att_detected / max(len(attack_errors), 1))
    att_mean = float(np.mean(attack_errors)) if len(attack_errors) > 0 else 0.0
    att_median = float(np.median(attack_errors)) if len(attack_errors) > 0 else 0.0

    print("\n" + "=" * 70)
    print("      CAVSHIELD LSTM AUTOENCODER SEPARATION EVALUATION")
    print("=" * 70)
    print(f"Normal Sequences (Val):   count={len(val_errors)}, mean={val_mean:.6f}, std={val_std:.6f}")
    print(f"Normal 99.5th Percentile: threshold={threshold:.6f}")
    print(f"Attack Sequences:         count={len(attack_errors)}, mean={att_mean:.6f}, median={att_median:.6f}")
    print(f"Attack Detection Rate:    {detection_rate * 100:.2f}% above normal threshold")
    print("-" * 70)
    print(f"Histogram Separation Ratio (Attack Mean / Normal Threshold): {att_mean / max(threshold, 1e-6):.2f}x")
    print("=" * 70 + "\n")

    # 7. Save Model and Metadata
    model_pt_path = output_models_dir / "anomaly_ae.pt"
    meta_json_path = output_models_dir / "ae_metadata.json"

    torch.save(model.state_dict(), model_pt_path)

    ae_meta = {
        "threshold": threshold,
        "mean_error": val_mean,
        "std_error": val_std,
        "feat_mean": feat_mean.tolist(),
        "feat_std": feat_std.tolist(),
        "input_dim": 17,
        "seq_len": 64,
        "hidden_dim": 64,
        "bottleneck_dim": 24,
        "num_layers": 2,
        "detection_rate": detection_rate,
        "created_at": time.time(),
    }
    with open(meta_json_path, "w", encoding="utf-8") as f:
        json.dump(ae_meta, f, indent=2)

    logger.info("Saved Autoencoder model to %s and metadata to %s", model_pt_path, meta_json_path)

    # 8. Plot Loss and Error Histograms (HTML & PNG)
    fig_html_path = output_figures_dir / "ae_reconstruction_error.html"
    fig_png_path = output_figures_dir / "ae_reconstruction_error.png"

    # Plotly interactive figure
    fig = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=("Training & Validation Loss", "Reconstruction Error Distribution (Normal vs Attack)"),
    )

    fig.add_trace(
        go.Scatter(y=train_losses, mode="lines+markers", name="Train MSE", line=dict(color="#1f77b4")),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(y=val_losses, mode="lines+markers", name="Val MSE", line=dict(color="#2ca02c")),
        row=1,
        col=1,
    )

    fig.add_trace(
        go.Histogram(x=val_errors, nbinsx=40, name="Normal (Val)", marker_color="green", opacity=0.7),
        row=1,
        col=2,
    )
    fig.add_trace(
        go.Histogram(x=attack_errors, nbinsx=40, name="Attack Sequences", marker_color="red", opacity=0.7),
        row=1,
        col=2,
    )
    fig.add_vline(
        x=threshold,
        line_dash="dash",
        line_color="purple",
        annotation_text=f"Threshold: {threshold:.4f}",
        row=1,
        col=2,
    )

    fig.update_layout(
        title_text="CAVShield LSTM Autoencoder Anomaly Detection",
        barmode="overlay",
        template="plotly_white",
        height=450,
        width=950,
    )
    fig.write_html(str(fig_html_path))
    logger.info("Saved interactive plot to %s", fig_html_path)

    # Bitmap PNG plot
    _render_bitmap_figure(
        train_losses=train_losses,
        val_losses=val_losses,
        normal_errors=val_errors,
        attack_errors=attack_errors,
        threshold=threshold,
        output_png=fig_png_path,
    )
    logger.info("Saved static bitmap figure to %s", fig_png_path)

    return {
        "threshold": threshold,
        "detection_rate": detection_rate,
        "normal_mean": val_mean,
        "attack_mean": att_mean,
        "epochs": len(train_losses),
        "train_time_s": dt_train,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train CAVShield LSTM Autoencoder.")
    parser.add_argument("--dataset", type=str, default=None, help="Dataset CSV path")
    parser.add_argument("--quick", action="store_true", help="Quick mode for testing")
    parser.add_argument("--epochs", type=int, default=12, help="Max epochs")
    args = parser.parse_args()

    train_autoencoder(dataset_path=args.dataset, quick=args.quick, epochs=args.epochs)

