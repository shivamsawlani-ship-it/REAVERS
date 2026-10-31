"""
Unit tests for Part 4:
- src.preprocessing: parse_can_message, clean_dataframe, parse_v2x
- src.feature_engineering: CANFeatureExtractor (17 features, O(1), CONTRACT order), V2X feature extraction
- src.windowing: SlidingWindower (64x17, stride 16), window_stat_features, STAT_FEATURE_NAMES, label rule
- src.data_loader: load_hcrl, build_window_dataset, split_train_test_by_run
- src.engine: World offline stepping
"""

import io
from pathlib import Path
import random
import sys

import numpy as np
import pandas as pd
import pytest

# Ensure repository root is in python path
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src import load_config
from src.can_simulator import KNOWN_IDS, make_frame
from src.data_loader import build_window_dataset, load_hcrl, split_train_test_by_run
from src.engine import World
from src.feature_engineering import (
    FEATURE_NAMES,
    CANFeatureExtractor,
    extract_v2x_features,
)
from src.preprocessing import clean_dataframe, parse_can_message, parse_v2x
from src.v2x_simulator import SensorSnapshot, TrustStore, V2XMessage
from src.windowing import (
    STAT_FEATURE_NAMES,
    SlidingWindower,
    Window,
    aggregate_window_label,
    make_windows,
    window_stat_features,
)


# ---------------------------------------------------------------------------
# Preprocessing Tests
# ---------------------------------------------------------------------------

class TestPreprocessing:
    def test_parse_can_message_from_dict_and_frame(self):
        d = {"ts": 1.23, "arb_id": "0x101", "data": "02 BC 00 00 00 00 BE 07"}
        parsed = parse_can_message(d)
        assert parsed is not None
        assert parsed["ts"] == 1.23
        assert parsed["arb_id"] == 0x101
        assert parsed["dlc"] == 8
        assert len(parsed["data"]) == 8

        frame = make_frame(2.34, 0x102, b"\x01" * 8, label="Normal", subtype="")
        parsed_f = parse_can_message(frame)
        assert parsed_f is not None
        assert parsed_f["arb_id"] == 0x102
        assert parsed_f["ts"] == 2.34

    def test_parse_can_message_malformed_never_raises(self):
        assert parse_can_message(None) is None
        assert parse_can_message("bad string") is None
        assert parse_can_message({"arb_id": "invalid_hex"}) is None
        assert parse_can_message({}) is None

    def test_clean_dataframe_standardization(self):
        raw_csv = """Timestamp,Arbitration_ID,DLC,Data,Class,SubClass
0.001,0x101,8,01 02 03 04 05 06 07 08,Normal,None
0.002,102,8,0A 0B 0C 0D 0E 0F 10 11,Spoofing,spoof_brake
bad_row,xxx,8,00,Normal,None
0.003,0x000,8,00 00 00 00 00 00 00 00,Flooding,flood_bus
"""
        df_raw = pd.read_csv(io.StringIO(raw_csv))
        cleaned = clean_dataframe(df_raw)

        assert len(cleaned) == 3  # dropped malformed row
        assert list(cleaned.columns) == ["ts", "arb_id", "dlc", "data", "label", "subtype"]
        assert cleaned["arb_id"].iloc[0] == 0x101
        assert cleaned["arb_id"].iloc[1] == 102
        assert cleaned["arb_id"].iloc[2] == 0x000
        assert cleaned["label"].iloc[0] == "Normal"
        assert cleaned["label"].iloc[1] == "Spoofing"
        assert cleaned["label"].iloc[2] == "Flooding"

    def test_clean_dataframe_empty_or_none(self):
        empty = clean_dataframe(None)
        assert isinstance(empty, pd.DataFrame)
        assert empty.empty

    def test_parse_v2x(self):
        msg = V2XMessage("M1", "V-2041", 100.0, 3.6, 20.0, "BSM", 1.0)
        json_str = msg.to_json()

        assert parse_v2x(json_str) is not None
        assert parse_v2x(msg.to_dict()) is not None
        assert parse_v2x(msg) is not None
        assert parse_v2x("bad json") is None
        assert parse_v2x(None) is None


# ---------------------------------------------------------------------------
# Feature Engineering Tests
# ---------------------------------------------------------------------------

class TestFeatureEngineering:
    def test_feature_names_contract(self):
        assert len(FEATURE_NAMES) == 17
        expected = [
            "can_id_norm", "dlc", "b0", "b1", "b2", "b3", "b4", "b5", "b6", "b7",
            "dt_same_id", "dt_global", "freq_id", "payload_change_rate", "id_entropy",
            "repeated_payload", "safety_critical"
        ]
        assert FEATURE_NAMES == expected

    def test_can_feature_extractor_streaming(self):
        extractor = CANFeatureExtractor()
        data = b"\x01\x02\x03\x04\x05\x06\x07\x08"

        # First frame
        vec1 = extractor.extract(arb_id=0x101, data=data, ts=1.0, dlc=8)
        assert len(vec1) == 17
        assert vec1.dtype == np.float32
        assert abs(vec1[0] - (0x101 / 2047.0)) < 1e-4  # can_id_norm
        assert vec1[1] == 8.0  # dlc
        assert abs(vec1[2] - (1 / 255.0)) < 1e-4  # b0
        assert vec1[16] == 1.0  # 0x101 is safety-critical

        # Repeated payload on same ID
        vec2 = extractor.extract(arb_id=0x101, data=data, ts=1.02, dlc=8)
        assert vec2[15] == 1.0  # repeated_payload
        assert vec2[13] == 0.0  # payload_change_rate

        # Changed payload on same ID
        data_diff = b"\x01\x02\x03\x04\x05\x06\x07\x09"  # 1 bit flipped
        vec3 = extractor.extract(arb_id=0x101, data=data_diff, ts=1.04, dlc=8)
        assert vec3[15] == 0.0  # not repeated
        assert abs(vec3[13] - (1.0 / 64.0)) < 1e-4  # 1 bit / 64 bits

    def test_v2x_feature_extraction(self):
        ts = TrustStore()
        msg = V2XMessage("M1", "V-2041", 1020.0, 3.6, 20.0, "BSM", 1.0)
        ts.sign(msg)

        feats = extract_v2x_features(
            msg=msg,
            ego_x=1000.0,
            ego_y=0.0,
            ego_speed=20.0,
            sensor_snapshot=None,
            now=1.05,
            trust_store=ts,
        )

        assert "freshness" in feats
        assert "position_plausibility" in feats
        assert "speed_plausibility" in feats
        assert "trust_score" in feats
        assert "sensor_consistency" in feats
        assert feats["trust_score"] == 1.0
        assert feats["freshness"] > 0.9


# ---------------------------------------------------------------------------
# Windowing Tests
# ---------------------------------------------------------------------------

class TestWindowing:
    def test_sliding_windower_push_and_stride(self):
        windower = SlidingWindower(size=64, stride=16)

        # Push 63 vectors -> None
        for i in range(63):
            w = windower.push(np.zeros(17, dtype=np.float32), ts=float(i))
            assert w is None

        # 64th push -> returns Window
        w1 = windower.push(np.zeros(17, dtype=np.float32), ts=63.0)
        assert isinstance(w1, Window)
        assert w1.features.shape == (64, 17)
        assert w1.t_start == 0.0
        assert w1.t_end == 63.0

        # Next 15 pushes -> None
        for i in range(64, 79):
            w = windower.push(np.zeros(17, dtype=np.float32), ts=float(i))
            assert w is None

        # 16th push (index 79) -> returns second Window
        w2 = windower.push(np.zeros(17, dtype=np.float32), ts=79.0)
        assert isinstance(w2, Window)
        assert w2.t_start == 16.0  # advanced by stride 16
        assert w2.t_end == 79.0

    def test_window_label_aggregation_rule(self):
        # 10 out of 64 attack frames with threshold 0.15 = 15.6% -> triggers attack
        labels_atk = ["Flooding"] * 12 + ["Normal"] * 52
        subtypes_atk = ["flood_bus"] * 12 + [""] * 52
        lbl, sub = aggregate_window_label(labels_atk, subtypes_atk, attack_fraction_threshold=0.15)
        assert lbl == "Flooding"
        assert sub == "flood_bus"

        # 5 out of 64 attack frames with threshold 0.15 = 7.8% -> remains Normal
        labels_few = ["Flooding"] * 5 + ["Normal"] * 59
        subtypes_few = ["flood_bus"] * 5 + [""] * 59
        lbl_n, sub_n = aggregate_window_label(labels_few, subtypes_few, attack_fraction_threshold=0.15)
        assert lbl_n == "Normal"
        assert sub_n == ""

    def test_window_stat_features_shape(self):
        w = Window(
            features=np.ones((64, 17), dtype=np.float32),
            t_start=0.0,
            t_end=1.0,
            raw_ids=[0x101] * 64,
            raw_payloads=[b"\x00" * 8] * 64,
        )
        stat_vec = window_stat_features(w, known_ids=KNOWN_IDS)
        names = STAT_FEATURE_NAMES(known_ids=KNOWN_IDS)

        # 68 (17 cols x 4 stats) + 5 global + 30 (10 IDs x 3 stats) = 103
        assert len(names) == 103
        assert stat_vec.shape == (103,)


# ---------------------------------------------------------------------------
# Data Loader & Engine Tests
# ---------------------------------------------------------------------------

class TestDataLoaderAndEngine:
    def test_world_step(self):
        world = World()
        can_frames, v2x_msgs = world.step(now=0.01, dt=0.005)
        assert isinstance(can_frames, list)
        assert isinstance(v2x_msgs, list)

    def test_build_window_dataset_and_split(self):
        # Create minimal synthetic dataframe
        data_rows = []
        for run_id in ["run_01", "run_02"]:
            for i in range(100):
                data_rows.append({
                    "ts": i * 0.005,
                    "arb_id": 0x101,
                    "dlc": 8,
                    "data": b"\x00" * 8,
                    "label": "Normal",
                    "subtype": "",
                    "run_id": run_id,
                })
        df = pd.DataFrame(data_rows)

        X_seq, X_stat, y, subtypes, run_ids, t_ends = build_window_dataset(df, known_ids=KNOWN_IDS)
        assert len(X_seq) > 0
        assert X_seq.shape[1:] == (64, 17)
        assert X_stat.shape[1] == 103
        assert len(y) == len(X_seq)

        train_mask, test_mask = split_train_test_by_run(run_ids, test_split_pct=50.0)
        assert np.any(train_mask)
        assert np.any(test_mask)
        assert not np.any(train_mask & test_mask)  # mutually exclusive

