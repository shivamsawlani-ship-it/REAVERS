"""
CAVShield Data Preprocessing.

Provides defensive ingestion and parsing for CAN frames, DataFrames (such as the
Car Hacking: Attack and Defense Challenge 2020 dataset), and V2X messages.
Adheres strictly to the rule: missing, delayed, or malformed input never raises.
"""

from __future__ import annotations

import re
from typing import Any, Dict, Optional, Union

import numpy as np
import pandas as pd

from src.v2x_simulator import V2XMessage


# ---------------------------------------------------------------------------
# CAN Message Parsing
# ---------------------------------------------------------------------------

def parse_can_message(msg: Any) -> Optional[Dict[str, Any]]:
    """Defensively parse a single CAN message from an object, dict, or tuple.

    Parameters
    ----------
    msg : Any
        python-can Message, CANFrame, dict, or tuple.

    Returns
    -------
    dict or None
        Normalized dict {'ts': float, 'arb_id': int, 'dlc': int, 'data': bytes,
        'label': str, 'subtype': str} or None if input cannot be parsed.
    """
    if msg is None:
        return None

    try:
        # 1. Dict input
        if isinstance(msg, dict):
            ts = float(msg.get("ts", msg.get("timestamp", 0.0)))
            raw_id = msg.get("arb_id", msg.get("arbitration_id", msg.get("id")))
            raw_data = msg.get("data", msg.get("payload", b""))
            label = str(msg.get("label", "Normal"))
            subtype = str(msg.get("subtype", ""))

        # 2. Object with attributes (CANFrame, can.Message, etc.)
        elif hasattr(msg, "arbitration_id") or hasattr(msg, "arb_id"):
            raw_id = getattr(msg, "arb_id", getattr(msg, "arbitration_id", None))
            ts = float(getattr(msg, "timestamp", 0.0) or 0.0)
            raw_data = getattr(msg, "data", b"")
            label = str(getattr(msg, "label", "Normal"))
            subtype = str(getattr(msg, "subtype", ""))

        # 3. Tuple / List: (ts, arb_id, dlc, data, [label], [subtype])
        elif isinstance(msg, (tuple, list)) and len(msg) >= 4:
            ts = float(msg[0])
            raw_id = msg[1]
            raw_data = msg[3]
            label = str(msg[4]) if len(msg) > 4 else "Normal"
            subtype = str(msg[5]) if len(msg) > 5 else ""

        else:
            return None

        # Parse arbitration ID (handles int, hex strings like '0x101' or '0101')
        arb_id = _parse_int_id(raw_id)
        if arb_id is None:
            return None

        # Parse payload data to 8 bytes
        data_bytes = _parse_payload_bytes(raw_data)
        if data_bytes is None:
            return None

        return {
            "ts": ts,
            "arb_id": arb_id,
            "dlc": len(data_bytes),
            "data": data_bytes,
            "label": label,
            "subtype": subtype,
        }

    except Exception:
        return None


# ---------------------------------------------------------------------------
# DataFrame Cleaning
# ---------------------------------------------------------------------------

def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Clean, standardize, and validate a CAN message DataFrame.

    Handles column name variants (such as HCRL 2020 format), converts hex IDs,
    pads payloads to 8 bytes, sorts by timestamp, and drops unrecoverable rows.
    Never raises an exception on malformed inputs.

    Parameters
    ----------
    df : pd.DataFrame
        Raw input DataFrame.

    Returns
    -------
    pd.DataFrame
        Standardized DataFrame with columns:
        ['ts', 'arb_id', 'dlc', 'data', 'label', 'subtype']
    """
    if df is None or not isinstance(df, pd.DataFrame) or df.empty:
        return pd.DataFrame(columns=["ts", "arb_id", "dlc", "data", "label", "subtype"])

    try:
        df = df.copy()

        # Map column names
        col_map = _resolve_column_names(df.columns)

        if "ts" not in col_map or "arb_id" not in col_map or "data" not in col_map:
            return pd.DataFrame(columns=["ts", "arb_id", "dlc", "data", "label", "subtype"])

        # Extract columns
        ts_col = df[col_map["ts"]]
        id_col = df[col_map["arb_id"]]
        data_col = df[col_map["data"]]

        dlc_col = df[col_map["dlc"]] if "dlc" in col_map else 8
        label_col = df[col_map["label"]] if "label" in col_map else "Normal"
        subtype_col = df[col_map["subtype"]] if "subtype" in col_map else ""

        # Vectorized / series transformations
        clean_ts = pd.to_numeric(ts_col, errors="coerce")
        clean_ids = id_col.apply(_parse_int_id)
        clean_data = data_col.apply(_parse_payload_bytes)

        # Build clean frame
        result = pd.DataFrame({
            "ts": clean_ts,
            "arb_id": clean_ids,
            "data": clean_data,
            "label": label_col.fillna("Normal").astype(str),
            "subtype": subtype_col.fillna("").astype(str),
        })

        # Drop invalid rows
        result = result.dropna(subset=["ts", "arb_id", "data"])
        if result.empty:
            return pd.DataFrame(columns=["ts", "arb_id", "dlc", "data", "label", "subtype"])

        result["arb_id"] = result["arb_id"].astype(int)
        result["dlc"] = result["data"].apply(len).astype(int)

        # Standardize labels from HCRL or other sources if present
        result["label"] = result["label"].apply(_normalize_label)
        result["subtype"] = result["subtype"].apply(_normalize_subtype)

        # Sort by timestamp
        result = result.sort_values(by="ts").reset_index(drop=True)

        return result[["ts", "arb_id", "dlc", "data", "label", "subtype"]]

    except Exception:
        return pd.DataFrame(columns=["ts", "arb_id", "dlc", "data", "label", "subtype"])


# ---------------------------------------------------------------------------
# V2X Defensive Parsing
# ---------------------------------------------------------------------------

def parse_v2x(data: Any) -> Optional[V2XMessage]:
    """Defensively parse a V2XMessage from dict, json string, or V2XMessage object.

    Never raises on malformed or invalid input.
    """
    if data is None:
        return None
    if isinstance(data, V2XMessage):
        return data
    if isinstance(data, str):
        return V2XMessage.from_json(data)
    if isinstance(data, dict):
        return V2XMessage.from_dict(data)
    return None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _parse_int_id(val: Any) -> Optional[int]:
    """Parse arbitration ID from int, float, or hex/decimal string."""
    if val is None or pd.isna(val):
        return None
    if isinstance(val, (int, np.integer)):
        return int(val)
    if isinstance(val, (float, np.floating)):
        return int(val) if not np.isnan(val) else None

    val_str = str(val).strip()
    if not val_str:
        return None

    try:
        # Check if formatted with 0x prefix
        if val_str.lower().startswith("0x"):
            return int(val_str, 16)
        # Check if contains hex characters [a-fA-F]
        if any(c in "abcdefABCDEF" for c in val_str):
            return int(val_str, 16)
        # Try decimal first, then hex
        try:
            return int(val_str, 10)
        except ValueError:
            return int(val_str, 16)
    except Exception:
        return None


def _parse_payload_bytes(val: Any) -> Optional[bytes]:
    """Parse and pad CAN payload to exactly 8 bytes."""
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return None

    try:
        if isinstance(val, (bytes, bytearray)):
            b = bytes(val)
        elif isinstance(val, str):
            val_clean = val.strip().replace(" ", "").replace("0x", "")
            # Odd length string
            if len(val_clean) % 2 != 0:
                val_clean = "0" + val_clean
            b = bytes.fromhex(val_clean)
        elif isinstance(val, (list, tuple)):
            b = bytes(int(x) & 0xFF for x in val)
        else:
            return None

        # Pad to 8 bytes if shorter, truncate to 8 if longer
        if len(b) < 8:
            b = b.ljust(8, b"\x00")
        elif len(b) > 8:
            b = b[:8]

        return b
    except Exception:
        return None


def _resolve_column_names(cols: Any) -> Dict[str, str]:
    """Map arbitrary column header variants to standard column names."""
    resolved = {}
    lower_map = {str(c).strip().lower(): c for c in cols}

    # Timestamp
    for candidate in ["ts", "timestamp", "time", "t", "timestamp(s)"]:
        if candidate in lower_map:
            resolved["ts"] = lower_map[candidate]
            break

    # Arb ID
    for candidate in ["arb_id", "arbitration_id", "id", "can_id", "arbitration id", "can id"]:
        if candidate in lower_map:
            resolved["arb_id"] = lower_map[candidate]
            break

    # DLC
    for candidate in ["dlc", "length", "len", "data_length", "data length"]:
        if candidate in lower_map:
            resolved["dlc"] = lower_map[candidate]
            break

    # Data
    for candidate in ["data", "payload", "bytes", "data[0-7]"]:
        if candidate in lower_map:
            resolved["data"] = lower_map[candidate]
            break

    # Label / Class
    for candidate in ["label", "class", "target", "attack", "ground_truth"]:
        if candidate in lower_map:
            resolved["label"] = lower_map[candidate]
            break

    # Subtype / SubClass
    for candidate in ["subtype", "subclass", "sub_class", "attack_type", "type"]:
        if candidate in lower_map:
            resolved["subtype"] = lower_map[candidate]
            break

    return resolved


def _normalize_label(val: Any) -> str:
    """Normalize label strings to CONTRACT attack classes."""
    s = str(val).strip().lower()
    if s in ["normal", "0", "benign"]:
        return "Normal"
    if s in ["flooding", "dos", "flood"]:
        return "Flooding"
    if s in ["fuzzy", "fuzzing", "fuzz"]:
        return "Fuzzing"
    if s in ["replay", "playback"]:
        return "Replay"
    if s in ["spoofing", "spoof", "rpm", "gear", "speed"]:
        return "Spoofing"
    return str(val).capitalize()


def _normalize_subtype(val: Any) -> str:
    """Normalize subtype strings."""
    s = str(val).strip().lower()
    if s in ["none", "nan", "normal", "0"]:
        return ""
    return s

