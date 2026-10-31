"""
CAVShield Data Loader and Dataset Generator.

Provides:
- load_hcrl(path): Loads and normalizes HCRL Car Hacking 2020 dataset CSVs.
- generate_synthetic_dataset(cfg): Offline simulation of the World across normal
  and attack runs, saving compressed CSV and marker files.
- build_window_dataset(df, known_ids, cfg): Constructs sequential (X_seq) and
  statistical (X_stat) datasets, resetting feature extractor per run to prevent leakage.
- split_train_test_by_run: Splits window datasets strictly by run_id.
"""

from __future__ import annotations

import argparse
import collections
import gzip
import os
from pathlib import Path
import random
import sys
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd

from src import load_config
from src.can_simulator import KNOWN_IDS
from src.engine import World
from src.feature_engineering import CANFeatureExtractor
from src.preprocessing import clean_dataframe
from src.windowing import make_windows, window_stat_features


# ---------------------------------------------------------------------------
# HCRL Dataset Loader
# ---------------------------------------------------------------------------

def load_hcrl(path: Union[str, Path]) -> pd.DataFrame:
    """Load and standardize an HCRL 2020 Car Hacking Challenge CSV dataset.

    Parameters
    ----------
    path : str or Path
        Path to CSV (or CSV.GZ) file.

    Returns
    -------
    pd.DataFrame
        Standardized DataFrame with columns ['ts', 'arb_id', 'dlc', 'data', 'label', 'subtype'].
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"HCRL dataset file not found: {path}")

    # Read CSV
    df_raw = pd.read_csv(path)
    return clean_dataframe(df_raw)


# ---------------------------------------------------------------------------
# Synthetic Dataset Generator
# ---------------------------------------------------------------------------

def generate_synthetic_dataset(
    cfg: Optional[Any] = None,
    output_dir: Optional[Union[str, Path]] = None,
    quick: bool = False,
) -> pd.DataFrame:
    """Generate a synthetic multi-run CAN dataset using the offline World.

    Runs virtual time stepped simulation for:
    - Normal runs with randomized cruise speeds.
    - Attack runs per attack type with active attacks.

    Parameters
    ----------
    cfg : Optional[Any]
        Project configuration.
    output_dir : Optional[Union[str, Path]]
        Target directory to save dataset files (defaults to cfg.paths.data_processed).
    quick : bool
        If True, runs a shortened parameter set for quick verification.

    Returns
    -------
    pd.DataFrame
        Generated dataset with run_id column.
    """
    if cfg is None:
        cfg = load_config()

    train_cfg = getattr(cfg, "training", {})
    sim_cfg = getattr(cfg, "simulation", {})
    world_hz = float(getattr(sim_cfg, "world_hz", 200.0))
    dt = 1.0 / world_hz

    if quick:
        normal_runs = 2
        normal_duration_s = 10.0
        attack_runs_per_type = 1
        attack_duration_s = 10.0
    else:
        normal_runs = int(getattr(train_cfg, "normal_runs", 8))
        normal_duration_s = float(getattr(train_cfg, "normal_duration_s", 45.0))
        attack_runs_per_type = int(getattr(train_cfg, "attack_runs_per_type", 4))
        attack_duration_s = float(getattr(train_cfg, "attack_duration_s", 30.0))

    attack_types = [
        "flood",
        "fuzz",
        "replay",
        "spoof_brake",
        "spoof_accel",
        "spoof_speed",
        "spoof_rpm",
        "spoof_gear",
    ]

    all_rows: List[Dict[str, Any]] = []
    run_counter = 0

    # 1. Normal Runs
    for i in range(normal_runs):
        run_id = f"normal_{i+1:02d}"
        seed = 1000 + i
        rng = random.Random(seed)
        world = World(cfg=cfg, rng=rng)

        # Randomize initial cruise speed within [60, 85] km/h
        cruise_speed = rng.uniform(62.0, 82.0)
        world.twin.cruise_kmh = cruise_speed
        world.twin.target_speed_kmh = cruise_speed

        total_steps = int(normal_duration_s * world_hz)
        for step_idx in range(total_steps):
            now = step_idx * dt
            can_frames, _ = world.step(now=now, dt=dt)
            for f in can_frames:
                all_rows.append({
                    "ts": f.timestamp,
                    "arb_id": f.arb_id,
                    "dlc": f.dlc,
                    "data": f.data,
                    "label": f.label,
                    "subtype": f.subtype,
                    "run_id": run_id,
                })
        run_counter += 1

    # 2. Attack Runs
    for atk_type in attack_types:
        for j in range(attack_runs_per_type):
            run_id = f"attack_{atk_type}_{j+1:02d}"
            seed = 2000 + run_counter * 10 + j
            rng = random.Random(seed)
            world = World(cfg=cfg, rng=rng)

            # Prime replay buffer early if replay attack
            if atk_type == "replay":
                world.twin.state.driver_or_adas_decel_cmd = 3.5
                world.twin.state.brake_cmd_source = 0

            # Activate attack midway through the run
            atk_start_t = 3.0 if quick else 6.0
            atk_duration = 5.0 if quick else 18.0
            world.injector.start(atk_type, duration=atk_duration, now=atk_start_t)

            total_steps = int(attack_duration_s * world_hz)
            for step_idx in range(total_steps):
                now = step_idx * dt
                can_frames, _ = world.step(now=now, dt=dt)
                for f in can_frames:
                    all_rows.append({
                        "ts": f.timestamp,
                        "arb_id": f.arb_id,
                        "dlc": f.dlc,
                        "data": f.data,
                        "label": f.label,
                        "subtype": f.subtype,
                        "run_id": run_id,
                    })
            run_counter += 1

    df_dataset = pd.DataFrame(all_rows)

    # Resolve output directories
    paths_cfg = getattr(cfg, "paths", {})
    proc_dir = Path(output_dir) if output_dir else Path(paths_cfg.get("data_processed", "data/processed"))
    sample_dir = Path(paths_cfg.get("data_sample", "data/sample"))
    proc_dir.mkdir(parents=True, exist_ok=True)
    sample_dir.mkdir(parents=True, exist_ok=True)

    # Save full dataset as compressed CSV
    csv_path = proc_dir / "synthetic_can_dataset.csv.gz"
    # Format data bytes as hex strings for readable CSV storage
    df_to_save = df_dataset.copy()
    df_to_save["data"] = df_to_save["data"].apply(lambda b: b.hex() if isinstance(b, (bytes, bytearray)) else "")
    df_to_save.to_csv(csv_path, index=False, compression="gzip")

    # Write SYNTHETIC marker file
    marker_path = proc_dir / "SYNTHETIC.txt"
    with open(marker_path, "w", encoding="utf-8") as f:
        f.write("CAVShield Synthetic CAN Dataset generated via World digital twin simulation.\n")
        f.write(f"Total rows: {len(df_dataset)}\n")
        f.write(f"Total runs: {run_counter}\n")

    # Write tiny sample file (500 rows)
    sample_path = sample_dir / "sample_can.csv"
    df_to_save.head(500).to_csv(sample_path, index=False)

    return df_dataset


# ---------------------------------------------------------------------------
# Window Dataset Builder
# ---------------------------------------------------------------------------

def build_window_dataset(
    df: pd.DataFrame,
    known_ids: Sequence[int] = KNOWN_IDS,
    cfg: Optional[Any] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Build sequential and statistical window datasets from message DataFrame.

    Resets the feature extractor per run to guarantee no temporal data leakage.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with ['ts', 'arb_id', 'dlc', 'data', 'label', 'subtype', 'run_id'].
    known_ids : Sequence[int]
        Known arbitration IDs for summary statistics.
    cfg : Optional[Any]
        Project configuration.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]
        X_seq: float32 array (N, 64, 17)
        X_stat: float32 array (N, D_stat)
        y: string array (N,) window labels ('Normal', 'Flooding', etc.)
        subtypes: string array (N,) window subtypes
        run_ids: string array (N,)
        t_ends: float64 array (N,) window end timestamps
    """
    if cfg is None:
        cfg = load_config()

    feat_cfg = getattr(cfg, "features", {})
    train_cfg = getattr(cfg, "training", {})
    window_size = int(getattr(feat_cfg, "window_size", 64))
    window_stride = int(getattr(feat_cfg, "window_stride", 16))
    attack_fraction = float(getattr(train_cfg, "attack_window_fraction", 0.35))

    if "run_id" not in df.columns:
        df = df.copy()
        df["run_id"] = "run_default"

    seq_list: List[np.ndarray] = []
    stat_list: List[np.ndarray] = []
    label_list: List[str] = []
    subtype_list: List[str] = []
    run_id_list: List[str] = []
    t_end_list: List[float] = []

    # Process each run independently
    for run_id, run_df in df.groupby("run_id", sort=False):
        extractor = CANFeatureExtractor(cfg=cfg)
        extractor.reset()

        # Sort within run by timestamp
        run_df = run_df.sort_values(by="ts")

        ts_vals = run_df["ts"].to_numpy()
        id_vals = run_df["arb_id"].to_numpy(dtype=int)
        dlc_vals = run_df["dlc"].to_numpy(dtype=int)
        data_vals = list(run_df["data"])
        labels = list(run_df["label"])
        subtypes = list(run_df["subtype"])

        n_rows = len(run_df)
        if n_rows < window_size:
            continue

        # Extract 17 features per message
        features = np.zeros((n_rows, 17), dtype=np.float32)
        for idx in range(n_rows):
            features[idx] = extractor.extract(
                arb_id=id_vals[idx],
                data=data_vals[idx],
                ts=ts_vals[idx],
                dlc=dlc_vals[idx],
            )

        # Batch create windows
        windows = make_windows(
            feature_matrix=features,
            timestamps=ts_vals,
            labels=labels,
            subtypes=subtypes,
            raw_ids=list(id_vals),
            raw_payloads=data_vals,
            size=window_size,
            stride=window_stride,
            attack_window_fraction=attack_fraction,
        )

        for w in windows:
            stat_vec = window_stat_features(w, known_ids=known_ids)
            seq_list.append(w.features)
            stat_list.append(stat_vec)
            label_list.append(w.label)
            subtype_list.append(w.subtype)
            run_id_list.append(str(run_id))
            t_end_list.append(w.t_end)

    if not seq_list:
        return (
            np.zeros((0, window_size, 17), dtype=np.float32),
            np.zeros((0, 103), dtype=np.float32),
            np.array([], dtype=object),
            np.array([], dtype=object),
            np.array([], dtype=object),
            np.array([], dtype=np.float64),
        )

    return (
        np.array(seq_list, dtype=np.float32),
        np.array(stat_list, dtype=np.float32),
        np.array(label_list, dtype=object),
        np.array(subtype_list, dtype=object),
        np.array(run_id_list, dtype=object),
        np.array(t_end_list, dtype=np.float64),
    )


def split_train_test_by_run(
    run_ids: np.ndarray,
    test_split_pct: float = 25.0,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray]:
    """Split window dataset into train and test boolean masks strictly by run_id."""
    unique_runs = sorted(list(set(run_ids)))
    rng = random.Random(seed)
    rng.shuffle(unique_runs)

    n_test = max(1, int(len(unique_runs) * (test_split_pct / 100.0)))
    test_runs = set(unique_runs[:n_test])
    train_runs = set(unique_runs[n_test:])

    train_mask = np.array([r in train_runs for r in run_ids], dtype=bool)
    test_mask = np.array([r in test_runs for r in run_ids], dtype=bool)

    return train_mask, test_mask


# ---------------------------------------------------------------------------
# CLI Command Entrypoint
# ---------------------------------------------------------------------------

def main() -> None:
    """CLI entrypoint for data generation and dataset verification."""
    parser = argparse.ArgumentParser(description="CAVShield Data Loader & Dataset Generator")
    parser.add_argument("--generate", action="store_true", help="Generate synthetic dataset and build windows")
    parser.add_argument("--quick", action="store_true", help="Run in quick mode with shortened durations")
    parser.add_argument("--output-dir", type=str, default=None, help="Directory to save generated datasets")

    args = parser.parse_args()

    if args.generate:
        print("Generating synthetic CAN dataset using World digital twin simulation...")
        cfg = load_config()
        df = generate_synthetic_dataset(cfg=cfg, output_dir=args.output_dir, quick=args.quick)

        print("\n" + "=" * 60)
        print("SYNTHETIC DATASET GENERATION SUMMARY")
        print("=" * 60)
        print(f"Total Rows Generated: {len(df)}")
        print("\nPer-Label Row Counts:")
        label_counts = df["label"].value_counts()
        for label, count in label_counts.items():
            print(f"  {label:15s}: {count:6d} ({count/len(df)*100:5.1f}%)")

        print("\nBuilding window dataset...")
        X_seq, X_stat, y, subtypes, run_ids, t_ends = build_window_dataset(df, known_ids=KNOWN_IDS, cfg=cfg)

        print("\nWINDOW DATASET SUMMARY")
        print("=" * 60)
        print(f"Total Windows: {len(y)}")
        print(f"X_seq Shape:   {X_seq.shape} (N, size, features)")
        print(f"X_stat Shape:  {X_stat.shape} (N, stats)")
        print("\nPer-Label Window Counts:")
        window_counts = collections.Counter(y)
        for label, count in window_counts.most_common():
            print(f"  {label:15s}: {count:6d} ({count/len(y)*100:5.1f}%)")
        print("=" * 60)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
