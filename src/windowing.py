"""
CAVShield Windowing and Aggregation.

Implements SlidingWindower for online streaming window creation,
window_stat_features for statistical tabular feature generation per CONTRACT item 45,
and offline batch windowing with ground-truth window label aggregation.
"""

from __future__ import annotations

import collections
from dataclasses import dataclass, field
import math
from typing import Any, Deque, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

from src import load_config
from src.feature_engineering import FEATURE_NAMES


# ---------------------------------------------------------------------------
# Window Dataclass
# ---------------------------------------------------------------------------

@dataclass
class Window:
    """A sliding window of sequential CAN message feature vectors.

    Attributes
    ----------
    features : np.ndarray
        Array of shape (size, 17) containing normalized per-message features.
    t_start : float
        Timestamp of the first message in the window.
    t_end : float
        Timestamp of the last message in the window.
    labels : List[str]
        Ground-truth labels of individual frames in the window.
    subtypes : List[str]
        Ground-truth subtypes of individual frames in the window.
    raw_ids : List[int]
        Arbitration IDs of messages in the window.
    raw_payloads : List[bytes]
        Raw 8-byte payloads of messages in the window.
    label : str
        Aggregated window-level label.
    subtype : str
        Aggregated window-level subtype.
    """
    features: np.ndarray
    t_start: float
    t_end: float
    labels: List[str] = field(default_factory=list)
    subtypes: List[str] = field(default_factory=list)
    raw_ids: List[int] = field(default_factory=list)
    raw_payloads: List[bytes] = field(default_factory=list)
    label: str = "Normal"
    subtype: str = ""


# ---------------------------------------------------------------------------
# Window Statistical Feature Extraction
# ---------------------------------------------------------------------------

def STAT_FEATURE_NAMES(known_ids: Sequence[int]) -> List[str]:
    """Generate exact feature names for the window-stat vector."""
    names: List[str] = []

    # 1. 17 columns x 4 statistics (mean, std, min, max) = 68
    for stat in ["mean", "std", "min", "max"]:
        for fname in FEATURE_NAMES:
            names.append(f"{stat}_{fname}")

    # 2. Global stats = 5
    names.extend([
        "n_unique_ids",
        "id_entropy",
        "top_id_share",
        "fraction_unknown_ids",
        "duplicate_payload_share",
    ])

    # 3. Per-known-ID stats = 3 * K
    for kid in known_ids:
        hex_id = f"0x{kid:03X}"
        names.extend([
            f"{hex_id}_count",
            f"{hex_id}_u16_mean",
            f"{hex_id}_u16_std",
        ])

    return names


def window_stat_features(window: Window, known_ids: Sequence[int]) -> np.ndarray:
    """Compute 1D summary statistical feature vector from a Window.

    Parameters
    ----------
    window : Window
        Window of shape (64, 17).
    known_ids : Sequence[int]
        Known arbitration IDs to generate per-ID statistics.

    Returns
    -------
    np.ndarray
        1D float32 vector containing (68 + 5 + 3 * len(known_ids)) features.
    """
    if isinstance(window, np.ndarray):
        feats = window
        raw_ids = []
        raw_payloads = []
    else:
        feats = getattr(window, "features", window)
        raw_ids = getattr(window, "raw_ids", [])
        raw_payloads = getattr(window, "raw_payloads", [])
    n_rows = feats.shape[0]

    # 1. Column-wise statistics: mean, std, min, max (68 features)
    mean_vec = np.mean(feats, axis=0)
    std_vec = np.std(feats, axis=0)
    min_vec = np.min(feats, axis=0)
    max_vec = np.max(feats, axis=0)

    # 2. Global window statistics (5 features)
    if not raw_ids:
        # Fallback: recover approximate IDs from normalized feature column 0
        raw_ids = [int(round(norm_id * 2047.0)) for norm_id in feats[:, 0]]

    id_counter = collections.Counter(raw_ids)
    n_unique_ids = float(len(id_counter))

    # Shannon entropy of IDs in window
    entropy = 0.0
    for count in id_counter.values():
        p = count / n_rows
        entropy -= p * math.log2(p)

    top_id_share = float(max(id_counter.values(), default=0)) / n_rows

    known_set = set(known_ids)
    unknown_count = sum(1 for arb_id in raw_ids if arb_id not in known_set)
    fraction_unknown_ids = float(unknown_count) / n_rows

    # Duplicate (id, payload) pairs
    if raw_payloads and len(raw_payloads) == n_rows:
        pair_counts = collections.Counter(zip(raw_ids, raw_payloads))
        duplicate_count = sum(c - 1 for c in pair_counts.values() if c > 1)
        duplicate_payload_share = float(duplicate_count) / n_rows
    else:
        duplicate_payload_share = 0.0

    global_stats = np.array([
        n_unique_ids,
        entropy,
        top_id_share,
        fraction_unknown_ids,
        duplicate_payload_share,
    ], dtype=np.float32)

    # 3. Per-known-ID statistics: count, mean u16(b0, b1), std u16(b0, b1)
    per_id_stats: List[float] = []

    # Map raw_ids to indices in window
    id_indices: Dict[int, List[int]] = collections.defaultdict(list)
    for idx, arb_id in enumerate(raw_ids):
        id_indices[arb_id].append(idx)

    for kid in known_ids:
        idxs = id_indices.get(kid, [])
        cnt = float(len(idxs))

        if cnt > 0:
            # Extract u16 from normalized b0 (col 2) and b1 (col 3)
            # b0_val = int(round(feats[idx, 2] * 255.0))
            # b1_val = int(round(feats[idx, 3] * 255.0))
            u16_vals = [
                float(int(round(feats[idx, 2] * 255.0)) * 256 + int(round(feats[idx, 3] * 255.0)))
                for idx in idxs
            ]
            mean_u16 = float(np.mean(u16_vals))
            std_u16 = float(np.std(u16_vals)) if len(u16_vals) > 1 else 0.0
        else:
            mean_u16 = 0.0
            std_u16 = 0.0

        per_id_stats.extend([cnt, mean_u16, std_u16])

    per_id_vec = np.array(per_id_stats, dtype=np.float32)

    # Concatenate all components
    return np.concatenate([mean_vec, std_vec, min_vec, max_vec, global_stats, per_id_vec])


# ---------------------------------------------------------------------------
# SlidingWindower (Streaming Online Window Creation)
# ---------------------------------------------------------------------------

class SlidingWindower:
    """Sliding window accumulator for streaming feature vectors."""

    def __init__(
        self,
        size: int = 64,
        stride: int = 16,
        attack_window_fraction: Optional[float] = None,
        cfg: Optional[Any] = None,
    ):
        if cfg is None:
            cfg = load_config()
        self.cfg = cfg

        feat_cfg = getattr(cfg, "features", {})
        self.size = int(getattr(feat_cfg, "window_size", size))
        self.stride = int(getattr(feat_cfg, "window_stride", stride))

        train_cfg = getattr(cfg, "training", {})
        if attack_window_fraction is None:
            attack_window_fraction = float(getattr(train_cfg, "attack_window_fraction", 0.35))
        self.attack_window_fraction = float(attack_window_fraction)

        self.reset()

    def reset(self) -> None:
        """Clear all buffered messages."""
        self._vectors: List[np.ndarray] = []
        self._timestamps: List[float] = []
        self._labels: List[str] = []
        self._subtypes: List[str] = []
        self._raw_ids: List[int] = []
        self._raw_payloads: List[bytes] = []

    def push(
        self,
        vec: np.ndarray,
        ts: float,
        label: str = "Normal",
        subtype: str = "",
        raw_id: int = 0,
        raw_payload: bytes = b"",
    ) -> Optional[Window]:
        """Push a message vector and return a Window when size is reached."""
        self._vectors.append(vec)
        self._timestamps.append(ts)
        self._labels.append(label)
        self._subtypes.append(subtype)
        self._raw_ids.append(raw_id)
        self._raw_payloads.append(raw_payload)

        if len(self._vectors) >= self.size:
            w_feats = np.array(self._vectors[:self.size], dtype=np.float32)
            w_labels = list(self._labels[:self.size])
            w_subtypes = list(self._subtypes[:self.size])
            w_ids = list(self._raw_ids[:self.size])
            w_payloads = list(self._raw_payloads[:self.size])
            t_start = self._timestamps[0]
            t_end = self._timestamps[self.size - 1]

            # Aggregate label per CONTRACT rule
            w_label, w_subtype = aggregate_window_label(w_labels, w_subtypes, self.attack_window_fraction)

            window = Window(
                features=w_feats,
                t_start=t_start,
                t_end=t_end,
                labels=w_labels,
                subtypes=w_subtypes,
                raw_ids=w_ids,
                raw_payloads=w_payloads,
                label=w_label,
                subtype=w_subtype,
            )

            # Advance by stride
            self._vectors = self._vectors[self.stride:]
            self._timestamps = self._timestamps[self.stride:]
            self._labels = self._labels[self.stride:]
            self._subtypes = self._subtypes[self.stride:]
            self._raw_ids = self._raw_ids[self.stride:]
            self._raw_payloads = self._raw_payloads[self.stride:]

            return window

        return None


# ---------------------------------------------------------------------------
# Offline Batch Window Creation
# ---------------------------------------------------------------------------

def aggregate_window_label(
    labels: Sequence[str],
    subtypes: Sequence[str],
    attack_fraction_threshold: float = 0.35,
) -> Tuple[str, str]:
    """Determine window-level label and subtype from constituent message labels.

    Rule: If the attack fraction is >= attack_fraction_threshold, label is the
    majority attack label (and majority subtype); otherwise 'Normal'.
    """
    total = len(labels)
    if total == 0:
        return ("Normal", "")

    attack_labels = [l for l in labels if l != "Normal"]
    if (len(attack_labels) / total) >= attack_fraction_threshold:
        majority_label = collections.Counter(attack_labels).most_common(1)[0][0]
        # Majority subtype among attack frames
        attack_subtypes = [s for s, l in zip(subtypes, labels) if l != "Normal" and s]
        if attack_subtypes:
            majority_subtype = collections.Counter(attack_subtypes).most_common(1)[0][0]
        else:
            majority_subtype = ""
        return (majority_label, majority_subtype)

    return ("Normal", "")


def make_windows(
    feature_matrix: np.ndarray,
    timestamps: np.ndarray,
    labels: Optional[List[str]] = None,
    subtypes: Optional[List[str]] = None,
    raw_ids: Optional[List[int]] = None,
    raw_payloads: Optional[List[bytes]] = None,
    size: int = 64,
    stride: int = 16,
    attack_window_fraction: float = 0.35,
) -> List[Window]:
    """Batch-create Windows from arrays for an entire recorded run."""
    n_samples = feature_matrix.shape[0]
    windows: List[Window] = []

    if n_samples < size:
        return windows

    if labels is None:
        labels = ["Normal"] * n_samples
    if subtypes is None:
        subtypes = [""] * n_samples
    if raw_ids is None:
        raw_ids = [0] * n_samples
    if raw_payloads is None:
        raw_payloads = [b"\x00" * 8] * n_samples

    for start_idx in range(0, n_samples - size + 1, stride):
        end_idx = start_idx + size

        w_feats = feature_matrix[start_idx:end_idx]
        w_labels = labels[start_idx:end_idx]
        w_subtypes = subtypes[start_idx:end_idx]
        w_ids = raw_ids[start_idx:end_idx]
        w_payloads = raw_payloads[start_idx:end_idx]

        w_label, w_subtype = aggregate_window_label(w_labels, w_subtypes, attack_window_fraction)

        windows.append(Window(
            features=w_feats,
            t_start=float(timestamps[start_idx]),
            t_end=float(timestamps[end_idx - 1]),
            labels=w_labels,
            subtypes=w_subtypes,
            raw_ids=w_ids,
            raw_payloads=w_payloads,
            label=w_label,
            subtype=w_subtype,
        ))

    return windows

