You are a senior automotive-cybersecurity + embedded-systems + ML engineer.
Project: CAVShield, an AI-driven, explainable cyber-physical intrusion detection and fail-safe system for Connected Autonomous Vehicles. It runs 100% locally on a Windows laptop, Python 3.11/3.12, no GPU, no real hardware, and works offline.

HARD RULES
- Output COMPLETE, RUNNABLE code for every file. No pseudocode, no ellipsis placeholders, no TODO stubs.
- Modular, typed, docstrings on non-trivial logic. Never crash on malformed input.
- No hardcoded thresholds: everything tunable lives in config/config.yaml.
- Windows-safe: pathlib, no bash-only commands, no fork-based multiprocessing assumptions.
- Always run code and tests with the project virtual-environment Python, never a global Python.
- Before starting a task, READ the files already in the workspace that the task depends on. Never re-invent or rename an interface that already exists.
- After coding, run the acceptance commands yourself, fix failures, and show the final output.
- Do not modify files outside the current part unless a bug requires it, and say so if you do.
- Do not rename anything in the CONTRACT below. Other parts depend on these names.

REPO LAYOUT (exact)
cavshield/ README.md requirements.txt run_dashboard.py run_simulation.py config/config.yaml
data/{raw,processed,sample}/  output/{models,metrics,logs,figures}/  reference/cavshield_ui_reference.html
tests/{test_pipeline.py,test_dashboard_api.py}
src/ data_loader.py preprocessing.py feature_engineering.py windowing.py can_simulator.py v2x_simulator.py attack_injector.py vehicle_digital_twin.py engine.py
  models/ lstm_autoencoder.py attack_classifier.py train_autoencoder.py train_classifier.py export_onnx.py
  detection/ anomaly_detector.py attack_classifier.py rule_engine.py physics_engine.py v2x_validator.py decision_engine.py
  failsafe/ failsafe_manager.py
  dashboard/ app.py static/index.html static/app.css static/app.js
  evaluation/ evaluate_models.py generate_report.py
  edge/ onnx_inference.py
(engine.py is an allowed extra: it wires everything together. Every folder has __init__.py.)

CONTRACT
1. CAN frame layout (all 10 IDs, DLC 8): bytes 0-5 = signals, byte 6 = XOR of bytes 0-5 (checksum), byte 7 low nibble = alive counter (0-15, +1 per frame per ID). This mimics AUTOSAR E2E and gives a real replay/freshness signal.
2. DBC (id, name, sender, period, safety-critical, criticality weight):
   0x100 ENGINE_RPM ECU_ENGINE 20ms no 0.6 | 0x101 VEHICLE_SPEED ECU_ENGINE 20ms yes 0.8
   0x102 BRAKE_STATUS ECU_BRAKE 20ms yes 0.9 | 0x103 STEERING_ANGLE ECU_STEERING 20ms yes 0.8
   0x104 GEAR_POSITION ECU_GEAR 100ms yes 0.8 | 0x105 WHEEL_SPEED SENSOR_WHEEL 20ms no 0.6
   0x106 IMU_DATA SENSOR_IMU 10ms no 0.6 | 0x200 BRAKE_COMMAND ECU_ADAS 20ms yes 1.0
   0x201 ACCEL_COMMAND ECU_ADAS 20ms yes 1.0 | 0x202 ADAS_COMMAND ECU_ADAS 50ms yes 1.0
   Total about 480 msg/s. Unknown IDs have criticality 0.5.
3. Signal encodings (big-endian): rpm u16 + throttle u8; speed u16 = km/h*100; brake status: pedal u8 = percent*2, brake_active u8, decel u16 = m/s2*100; steering s16 = deg*10; gear u8 (0=P,1=R,2=N,3=D); wheels 4x u8 in 0.75 km/h units; IMU ax s16 and yaw s16 = value*100; 0x200 cmd_decel u16 = value*100 + source u8 (0 DRIVER, 1 AEB, 2 ACC); 0x201 cmd_accel s16 = value*100 + source; 0x202 flags lka, acc, aeb, set_speed_kmh.
4. Dataclasses:
   CANFrame(timestamp, arb_id, dlc, data bytes, sender, msg_type, label default Normal, subtype default empty). label/subtype are GROUND TRUTH for evaluation only. NO detector may read them.
   V2XMessage(msg_id, sender_id, x, y, speed, msg_type, timestamp, heading, signature, seq, obstacle dict or None, claimed_decel, pos_accuracy_m, label, subtype); HMAC-SHA256 trust token via a TrustStore; message types BSM, COLLISION_WARNING, EMERGENCY_BRAKING, ROAD_HAZARD.
   Evidence(source in rule/physics/v2x, name, severity 0..1, detail text, hint in Flooding/Fuzzing/Replay/Spoofing or empty)
   Decision(level IntEnum LOW=0 MEDIUM=1 HIGH=2 CRITICAL=3, score, attack_type, confidence, explanation text, action text, components dict)
5. Attack classes: Normal, Flooding, Fuzzing, Replay, Spoofing (plus Unknown anomaly as an output when confidence is low). Spoof variants (subtype): spoof_brake, spoof_accel, spoof_speed, spoof_rpm, spoof_gear. V2X attacks: v2x_phantom, v2x_fake_ebrake, v2x_gps.
6. Per-CAN-message feature vector (17 floats, this order): can_id_norm, dlc, b0..b7 (divided by 255), dt_same_id, dt_global, freq_id, payload_change_rate, id_entropy (rolling 64 ids), repeated_payload, safety_critical. Counter and checksum are deliberately NOT AI features (the AI stays generic so it works on the real HCRL dataset; rules own the protocol checks).
7. Windows: size 64, stride 16, keep first/last timestamp for latency. Window-stat vector = mean/std/min/max of the 17 columns + global stats (n_unique_ids, entropy, top-id share, fraction unknown ids, duplicate (id,payload) pairs) + per-known-ID [count, mean u16(b0,b1), std u16(b0,b1)]. known_ids is a parameter (for the real dataset: top-N IDs of normal data).
8. Architecture rule: the IDS is an INLINE GATEWAY. A frame is only applied to the vehicle plant (digital twin) if the gateway forwards it. Blocking = not forwarding. Threads: world thread (physics + ECUs + attacks -> bus), pipeline thread (reads bus, fast path: rules + physics + risk + failsafe, microseconds), AI worker thread (windows -> anomaly + classifier, milliseconds, updates a shared AIContext). Never run AI inline in the per-frame path.
9. Risk fusion (per frame): det = max(rule severity, physics severity) + 0.05 per extra distinct evidence (cap 1); ai_conf = mean(anomaly_norm, attack_prob) if fresh (under 0.8 s) else 0; if det > 0: score = w_det*det + w_ai*ai_conf (0.65/0.35), else AI-only: level LOW per frame (a window-level MEDIUM alert is raised separately); score = score * (0.75 + 0.25*criticality). Levels: under 0.25 LOW, under 0.50 MEDIUM, under 0.72 HIGH, else CRITICAL. CONFIDENCE GATE: CRITICAL only if det >= 0.7 AND ai_conf >= 0.60 AND criticality >= 0.8; otherwise cap at HIGH. AI alone never exceeds MEDIUM. Frame action: LOW forward; MEDIUM forward + ALERT; HIGH drop frame / rate-limit / quarantine ID; CRITICAL block + mode change (command IDs -> SAFE MODE, status IDs -> DEGRADED). Modes: NORMAL, ALERT, DEGRADED, SAFE MODE, with timed auto-recovery one step at a time.
10. Config sections: simulation, can, v2x, sensors, features, rules, physics, risk, failsafe, ai, training, attacks, dashboard (host, port 8000), paths. Risk flags use_ai / use_can_rules / use_v2x / use_physics exist so the ablation study can switch components off.
11. SNAPSHOT SCHEMA: engine.snapshot() returns a JSON-serialisable dict built from COPIES under a lock and never blocks the pipeline. Any value not yet available is null (the UI shows a dash). Never invent numbers.
   - t: engine uptime seconds
   - mode: NORMAL | ALERT | DEGRADED | SAFE MODE
   - protection: bool
   - backend: {bus: text, ai: text}
   - banner: {level: LOW|MEDIUM|HIGH|CRITICAL, headline: text, driver_action: text}
   - active_attacks: list of attack names
   - twin: {speed_kmh, rpm, pedal, brake_active, decel, applied_decel, steering_deg, gear, aeb, unsafe (text), unsafe_seconds, perceived: {speed_kmh, rpm, pedal, steering_deg, gear}}
   - threat: {attack_type, confidence, level, explanation, ai_available, anomaly, attack_prob, probs: {class: value}}
   - scores: last 300 of {t, anomaly, attack_prob, level}; attack_markers: list of engine times when attacks started
   - can: newest first, last 12 of {ts, id (like 0x200), dlc, data (hex text), sender, verdict: normal|suspicious|blocked, reason}
   - v2x: newest first, last 10 of {ts, sender, type, trust, status: accepted|rejected, reason}
   - failsafe: {rejected_commands: {id: count}, rate_limited: [ids], quarantined: [ids], rejected_v2x: count, mode_history: [{t, mode}], actions: last 8 texts, incident_id}
   - incidents: last 10 of {id, ts, attack_type, level, target, confidence, latency_ms, count}
   - metrics: {accuracy, precision, recall, f1, fpr (0..1 or null), detection_latency_ms: {mean, p95, max}, failsafe_latency_ms: {mean, p95, max}, inference_ms: {mean, p95}, simulator_only: true, counts: {tp, fp, tn, fn}}
   - demo: {running, step, caption}