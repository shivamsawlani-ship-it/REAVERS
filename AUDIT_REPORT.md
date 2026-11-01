# CAVShield Comprehensive Quality Assurance & Automotive Cybersecurity Audit Report

**Date of Audit**: November 1, 2026  
**Auditor Role**: Independent QA Lead & Automotive-Security Reviewer  
**Audit Scope**: Static code analysis, automated test suite, training/ONNX pipelines, runtime behavior, cyber-physical attack matrix, latency benchmarking, API/UI security, and system robustness.  
**Target Environment**: Python 3.12.7 (`cavshield/.venv`), Windows 11 Host, Headless Simulation & FastAPI Dashboard.

---

## Section A: Executive Verdict

### Verdict: **DEMO-READY WITH WORKAROUNDS**

The CAVShield system demonstrates an impressive, architecturally sound dual-plane cyber-physical defense architecture. The inline gateway fast-path successfully blocks malicious actuator commands (e.g., dropping `spoof_brake` in under 0.13 seconds and preserving digital twin cruise speed) while permitting legitimate panic braking and radar-verified AEB. The FastAPI backend, WebSockets, Plotly telemetry streaming, and the 8-step guided demo timeline execute smoothly when operated within supported paths.

However, several critical discrepancies, an empirical race condition crash, ground-truth label leakage in the fail-safe monitor, an automated unit test failure, and marketing claim exaggerations prevent an unqualified "Demo-Ready" verdict without strict live-demo workarounds.

### Top 3 Operational & Technical Risks

1. **AI Worker Thread Race Condition (`deque mutated during iteration`)**:
   In `src/engine.py:1172`, the background AI worker appends to `self.score_history` without acquiring `self._snapshot_lock`. During concurrent `snapshot()` generation (called by `/ws` at 4 Hz and `/api/snapshot`), `list(self.score_history)` at line 1484 intermittently crashes the thread with `RuntimeError: deque mutated during iteration`. This caused an unhandled crash during the `spoof_rpm` attack test.
2. **Ground-Truth Label Leakage in Production Fail-Safe Guard**:
   In `src/failsafe/failsafe_manager.py:519`, the starvation watchdog checks `if is_cmd and frame.label == "Normal":`. In real automotive ECUs or production CAN captures, `frame.label` does not exist. Relying on ground-truth evaluation metadata inside the operational safety critical path violates basic cybersecurity design rules and will crash on external frame streams.
3. **Incomplete System Reset & Stale Threat Persistence**:
   `src/engine.py:reset()` fails to clear `self.ai_context` and fails to drain `self.ai_queue`. When a presenter clicks "Reset" after an attack, queued attack windows continue to process in the background, overwriting the UI banner with high threat scores, suspicious classification, and false alarms while the car is supposedly in clean "NORMAL" cruise.

---

## Section B: Issue Table Sorted by Severity

| ID | Severity | Area | What is Wrong | Evidence (File:Line / Command & Output) | Impact on Demo / Judges | Suggested Fix | Effort |
|---|---|---|---|---|---|---|---|
| **ISS-01** | **Blocker** | Concurrency / Engine | `score_history` deque mutated during iteration without mutex protection | `src/engine.py:1172` vs `src/engine.py:1484`<br/>Observed crash in `phase3_attack_matrix.py`: `RuntimeError: deque mutated during iteration` | Crash in WebSocket loop or snapshot API during live attack demonstration | Wrap `score_history.append()` in `_ai_worker_loop` under `with self._snapshot_lock:` | **S** |
| **ISS-02** | **Blocker** | Cybersecurity / Safety | Ground-truth metadata leakage in starvation monitor | `src/failsafe/failsafe_manager.py:519`<br/>`if is_cmd and frame.label == "Normal":` | Disqualifies automotive cybersecurity integrity; crashes on real/external CAN frames | Check decoded signal plausibility or rate instead of testing ground-truth label attribute | **S** |
| **ISS-03** | **High** | Test Suite | Unit test `test_headless_short_soak_and_recovery` deterministically fails | `pytest tests/test_lockout_and_recovery.py:174`<br/>Failed: `assert time.time() - t_stop < 3.5` (Actual recovery time: 4.8s - 5.0s) | Running `pytest` in front of judges results in a visible test failure | Update test assertion timeout to 5.5s or drain input bus queue before starting quiet timer | **S** |
| **ISS-04** | **High** | Lifecycle / Reset | `engine.reset()` does not clear `ai_context` or drain `ai_queue` | `src/engine.py:530-556`<br/>`test_reset_during_attack` failed: `ai_context.anomaly` retained pre-reset attack scores | UI displays high risk banner and threat scores immediately after pressing "Reset" | Reinitialize `self.ai_context = AIContext(available=...)` and empty `self.ai_queue` in `reset()` | **S** |
| **ISS-05** | **High** | Architecture / Claims | Hardcoded fast-path latency constants in code and documentation | `src/edge/onnx_inference.py:345`: `"end_to_end_pipeline_fast_path_us": 35.3`<br/>`src/engine.py:694, 721`: `detection_latency_ms=0.03`<br/>Measured p50: **61.3 µs**, mean: **73.6 µs** | Judges reviewing code will flag hardcoded benchmark numbers as faked metrics | Replace static dictionary constant with dynamic measurement from `MetricsTracker` | **S** |
| **ISS-06** | **High** | Storage / Memory | Unbounded file growth in incident logging; 1.2 GB disk usage accumulated | `output/logs/incidents_archive.jsonl.bak`: **1.19 GB**<br/>`output/logs/incidents.jsonl`: **50.4 MB**<br/>`output/logs/incidents.db`: **24.7 MB** | Rapidly fills disk during soak/flood tests, degrading filesystem I/O | Implement a rotating log handler with `maxBytes=20MB` and `backupCount=3` | **M** |
| **ISS-07** | **Medium** | Memory Safety | Three unbounded collections in data ingestion pipeline | `src/can_simulator.py:396` (`queue.Queue()` without maxsize)<br/>`src/engine.py:430` (`v2x_queue = queue.Queue()`)<br/>`src/feature_engineering.py:165` (`collections.deque()` without maxlen) | Potential Out-Of-Memory (OOM) memory leak under sustained flood attacks | Supply `maxsize=1000` to queues and `maxlen=200` to feature extraction deques | **S** |
| **ISS-08** | **Medium** | Frontend / XSS | Frontend uses `innerHTML` with dynamic server data | `src/dashboard/static/app.js:141, 194, 212, 230, 245, 276, 313, 335, 345`<br/>`el.innerHTML = ...` | Minor XSS vulnerability if sender IDs or threat strings contain unescaped HTML | Use `textContent` or `document.createElement()` for text fields | **M** |
| **ISS-09** | **Medium** | Model Runtime | Engine defaults to IsolationForest rather than promoted ONNX INT8 LSTM | `config.yaml:70`: `ai.anomaly_backend: "iforest"`<br/>PyTorch LSTM: 11.96ms; IsolationForest: 17.62ms; ONNX INT8: 1.11ms | Presentation deck promises sub-millisecond deep edge ONNX acceleration | Change default backend to `"onnx_int8"` once parity verification is complete | **S** |
| **ISS-10** | **Low** | CAN Driver | `KeyError: 'cavshield'` thrown by python-can VirtualBus on shutdown | `VirtualBus was not properly shut down`<br/>`KeyError: 'cavshield'` in `can/interfaces/virtual.py:163` | Noisy stderr traceback printed when stopping test scripts or exiting Python | Add explicit try/except around `bus.shutdown()` in `CAVShieldEngine.stop()` | **S** |
| **ISS-11** | **Low** | Documentation | Dead documentation links and inaccurate library claims | `README.md:10` links to non-existent `LICENSE`<br/>`README.md:231` claims Chart.js (code uses Plotly)<br/>`reference/cavshield_ui_reference.html` missing | Dilutes project professionalism when audited by evaluators | Add MIT `LICENSE`, update README to cite Plotly, and restore UI reference asset | **S** |

---

## Section C: Claims Versus Reality

| Claim Source | Claimed Item | Actual Code / Empirical Finding | Status |
|---|---|---|---|
| **README.md:19 & PITCH.md:53** | Inline gateway fast path operates in **under 35.3 microseconds** | Empirical measurement over 100k frames: **p50 = 61.3 µs**, **mean = 73.61 µs**, **p99 = 220.3 µs**. Value `35.3` is hardcoded in `src/edge/onnx_inference.py:345`. | **REFUTED / HARDCODED** |
| **PITCH.md:50** | Detection Accuracy is **99.64%** | Window-level classifier achieves 99.99%; however, offline frame-level fast path is **93.59%** accuracy and **84.08%** precision (`output/metrics/evaluation_frame_metrics.json`). | **PARTIALLY SUPPORTED** |
| **PITCH.md:51** | Safety-Critical Attack Recall is **100.00%** | Verified: 8,052 of 8,052 safety-critical attack frames blocked. Attack matrix confirms 100% of tested attacks detected. | **SUPPORTED** |
| **PITCH.md:55** | False Positive Rate is **0.42%** | Offline frame-level FPR is **9.43%** (13,583 normal frames dropped during replay). Window-level anomaly detector FPR is **0.04%**. Normal driving event false alarms: 10,307/hr. | **REFUTED (Frame Level)** |
| **README.md:87 & PITCH.md:67** | ONNX INT8 quantization achieves **0.72 ms** CPU latency and ** sub-millisecond** execution | Benchmarked using `onnx_benchmark`: PyTorch CPU = **11.96 ms**, ONNX FP32 = **1.27 ms**, ONNX INT8 = **1.11 ms**. INT8 reduces size to 134.9 KB (3.36x compression). | **SUPPORTED (Size)** / **OFF BY 0.39ms (Latency)** |
| **README.md:27** | Asynchronous deep LSTM-AE runs in live pipeline | Default `config.yaml` runs `iforest` (IsolationForest), taking **17.62 ms** (p50). ONNX model exists but is not selected by default. | **MISCONFIGURED** |
| **README.md:131** | Seamless support for real HCRL 2020 Car-Hacking Dataset | Code supports loading HCRL CSVs, but directory `data/raw/` is empty. Models were trained entirely on synthetic data. | **UNVERIFIED (No Raw Data)** |
| **README.md:231** | Interactive real-time Chart.js visualizer | Dashboard uses **Plotly.js** (`src/dashboard/static/plotly.min.js`), not Chart.js. | **INACCURATE DOCS** |
| **README.md:10** | Open source licensed under Apache / MIT `[LICENSE](LICENSE)` | File `LICENSE` does not exist in repository root. | **MISSING FILE** |

---

## Section D: Attack Matrix, Latency Profile & Metrics Comparison

### 1. Cyber-Physical Attack Matrix (Headless & Twin Actuation)

Tested empirically with 8.0s attack duration under Protection ON vs Protection OFF, followed by recovery observation:

| Attack Scenario | Target Arb ID / Entity | Protection ON Detection Time | Reported Threat Label | Max Twin Mode | Twin Speed Delta (Protection ON) | Twin Speed Delta (Protection OFF) | Actuator Impact Confirmed (Prot OFF) | Quiet Recovery Time |
|---|---|---|---|---|---|---|---|---|
| **`flood`** | Multiple / Bus | **0.108 s** | `Flooding` | NORMAL | 8.71 km/h (cruise steady) | 7.11 km/h | Yes (Bus saturated, latency spike) | **0.25 s** |
| **`fuzz`** | Randomized payloads | **0.118 s** | `Fuzzing` | DEGRADED | 8.42 km/h (isolated) | **72.00 km/h** | **Yes (Twin halted to 0 km/h)** | **2.70 s** |
| **`replay`** | 0x101, 0x102, 0x200 | **0.103 s** | `Replay` | DEGRADED | 8.81 km/h (isolated) | 8.71 km/h | Yes (Stale telemetry accepted) | **0.20 s** |
| **`spoof_brake`** | 0x200 (BRAKE_CMD) | **0.124 s** | `Spoofing (Brake)` | NORMAL | 8.11 km/h (cruise steady) | **72.00 km/h** | **Yes (Hostile E-Brake stops twin)** | **0.20 s** |
| **`spoof_accel`** | 0x201 (ACCEL_CMD) | **0.101 s** | `Spoofing (Accel)` | NORMAL | 7.54 km/h (cruise steady) | **+12.03 km/h** | **Yes (Unintended acceleration)** | **0.20 s** |
| **`spoof_speed`** | 0x101 (VEH_SPEED) | **0.103 s** | `Spoofing (Speed)` | DEGRADED | 8.12 km/h | 8.61 km/h | Yes (Sensor desync) | **2.46 s** |
| **`spoof_rpm`** | 0x100 (ENG_RPM) | *Crashed* | *Uncaught Exception* | — | — | — | *Crash: deque mutated during iteration* | — |
| **`spoof_gear`** | 0x104 (GEAR_POS) | **0.105 s** | `Spoofing (Gear)` | NORMAL | 4.94 km/h | 6.33 km/h | Yes (Plausibility violation) | **1.16 s** |
| **`v2x_phantom`** | V2X Beacon (V-9999) | **0.123 s** | `V2X Phantom Obstacle` | NORMAL | 7.16 km/h | **-29.66 km/h** | **Yes (Phantom braking applied)** | **0.45 s** |
| **`v2x_fake_ebrake`**| V2X Emergency Braking | **0.105 s** | `V2X Fake E-Brake` | NORMAL | 6.96 km/h | **-29.36 km/h** | **Yes (Ghost vehicle slows ego)** | **0.22 s** |
| **`v2x_gps`** | V2X Kinematic BSM | **0.105 s** | `V2X GPS Drift` | NORMAL | 7.56 km/h | 7.07 km/h | Yes (Invalid beacon location) | **0.25 s** |

*Key Takeaway*: In 10 of 10 completed attack scenarios, CAVShield detects malicious injection in **~0.10 to 0.12 seconds** and recovers to clean baseline in **< 2.7 seconds**. In `spoof_brake`, protection successfully prevents unintended vehicle stop (retaining 72 km/h cruise), whereas disabling protection causes complete vehicle halt (speed delta: 72.0 km/h).

---

### 2. Latency Benchmarks (Empirically Measured vs Claimed)

Measured using `time.perf_counter_ns` across **100,000 frames** (50,000 clean frames + 50,000 flood frames):

| Latency Metric | Claimed (README/Deck) | Empirical p50 | Empirical p95 | Empirical p99 | Empirical Max | Compliance |
|---|---|---|---|---|---|---|
| **Clean Gateway Fast-Path** | 35.3 µs | **61.3 µs** | **114.5 µs** | **220.3 µs** | 23.96 ms | **FAIL (< 50 µs target)** |
| **Flood Gateway Fast-Path** | < 50.0 µs | **2.2 µs** | **86.9 µs** | **159.7 µs** | 82.88 ms | **PASS (p50)** / **FAIL (p99)** |
| **Snapshot Generation** | < 5.0 ms | **0.21 ms** | **0.36 ms** | **0.92 ms** | 1.43 ms | **PASS** (Superb determinism) |
| **AI Inference (IsolationForest)**| ~1.0 ms | **17.62 ms** | **30.00 ms** | **43.73 ms** | 51.79 ms | **FAIL (17x slower than claim)** |
| **AI Inference (ONNX INT8)** | 0.72 ms | **1.11 ms** | **1.35 ms** | **1.52 ms** | 3.84 ms | **CLOSE (~0.39 ms delta)** |

---

### 3. Cybersecurity Detection Metrics Comparison

| Evaluation Level | Target Unit | Total Samples | Accuracy | Precision | Recall | FPR | Key Observation |
|---|---|---|---|---|---|---|---|
| **Window Level (Classifier)** | 64-frame Window | 9,836 windows | **99.99%** | **100.0%** | **100.0%** | **0.00%** | LightGBM perfectly separates synthetically generated attack windows. |
| **Window Level (Anomaly)** | 64-frame Window | 9,836 windows | **76.25%** | **99.80%** | **35.35%** | **0.04%** | Conservative threshold suppresses false alarms (FPR 0.04%) at expense of recall. |
| **Frame Level Fast-Path** | Single CAN Frame | 215,965 frames | **93.59%** | **84.08%** | **99.64%** | **9.43%** | 13,583 normal frames dropped after flood attacks due to rate-limiting cooldown. |
| **Event Level (Attack Episodes)**| Per Episode | 27,421 episodes| — | — | **49.79%** | — | `fuzz_random` detected with 99.76% recall; `flood_bus` had 0% episode recall. |
| **Normal Driving Segments** | Normal Hours | 0.42 hours | — | — | — | **10,308/hr**| Frame-level rate-limiter alerts fire 4,371 times across 25 minutes of normal driving replay. |

---

## Section E: What Passed (Safe Demo Features)

1. **15-Minute Headless Soak Test**:
   Ran 900.0 virtual seconds (432,003 frames) in 100.6 seconds wall-clock time with zero active attacks. Result: **0 false alarms, 0 dropped frames, vehicle speed steady at 72.0 km/h, final mode NORMAL**.
2. **Slow AI Starvation Decoupling**:
   Tested with AI worker artificially delayed to 2.0 seconds per window. Fast-path processed 57,599 frames with zero backlog accumulation and zero false positives, proving that the inline fast-path is strictly decoupled from AI inference.
3. **Hard Negatives (Legitimate Maneuvers Not Blocked)**:
   - *Driver Panic Braking*: 100% pedal depression and 8.0 m/s² deceleration forwarded cleanly without entering DEGRADED or dropping commands.
   - *Legitimate Radar Obstacle AEB*: ADAS automated brake command forwarded; digital twin braked smoothly.
   - *Benign Traffic Bursts*: Scheduler jitter (multiple frames within < 0.2 ms) did not trigger token-bucket flood drops.
4. **FastAPI Endpoints & Real-time WebSockets**:
   - `GET /api/health`, `GET /api/snapshot`, `POST /api/attack/{name}/start`, `POST /api/attack/{name}/stop`, `POST /api/scenario/real_obstacle`, `POST /api/protection`, and `POST /api/reset` all passed with 200 OK.
   - Malformed payloads rejected with `422 Unprocessable Entity`; unknown attack names rejected with `404 Not Found`.
   - 50 rapid-fire API snapshot calls completed in 380 ms without error.
   - WebSocket `/ws` streams continuous 4 Hz snapshots matching the CONTRACT schema.
5. **Security & Deployment Posture**:
   - UDP command port 47474 is bound strictly to `127.0.0.1` (no external socket exposure).
   - Zero hardcoded OS-specific paths (`C:\`, `/tmp`, `/dev/`).
   - Dashboard operates 100% offline (bundled Plotly.js, zero CDN scripts).
   - Graceful fallback when model directories are missing.

---

## Section F: Top 10 Fixes & Safe Demo Path

### Top 10 Recommended Fixes (Ranked by Severity & Effort)

1. **Mutex Lock for `score_history`** (`src/engine.py:1172`):
   Wrap `self.score_history.append(...)` in `with self._snapshot_lock:` to prevent concurrent iteration crashes during snapshots.
2. **Remove Ground-Truth Leakage** (`src/failsafe/failsafe_manager.py:519`):
   Replace `frame.label == "Normal"` with internal signal consistency or drop the label check entirely.
3. **Fix Test Recovery Timeout** (`tests/test_lockout_and_recovery.py:174`):
   Change timeout condition from `assert time.time() - t_stop < 3.5` to `< 5.5` to accommodate queue drain plus the 2.5s recovery quiet timer.
4. **Complete Engine Reset** (`src/engine.py:530`):
   In `engine.reset()`, add `self.ai_context = AIContext(available=...)` and empty `self.ai_queue` so pre-reset attack scores do not pollute subsequent normal runs.
5. **Implement Incident Log Rotation** (`src/failsafe/failsafe_manager.py`):
   Cap `output/logs/incidents.jsonl` and `incidents.db` to 10,000 entries or 20 MB with rollover to prevent multi-gigabyte disk exhaustion.
6. **Harmonize Latency Claims in Docs** (`README.md`, `PITCH.md`, `src/edge/onnx_inference.py:345`):
   Replace hardcoded `35.3 µs` claim with actual measured numbers: `~60 µs p50 gateway latency` and `1.1 ms ONNX INT8 inference`.
7. **Switch Default AI Backend to ONNX** (`config.yaml:70`):
   Change `ai.anomaly_backend: "iforest"` to `"onnx_int8"` to match the advertised sub-millisecond edge architecture.
8. **Add Max Limits to Containers** (`src/engine.py:430`, `src/feature_engineering.py:165`):
   Set `maxsize=1000` on `self.v2x_queue` and `maxlen=200` on `_rate_deques_by_id` to prevent memory leaks during prolonged flood runs.
9. **Eliminate Frontend `innerHTML` Usage** (`src/dashboard/static/app.js`):
   Replace `innerHTML` assignments in incident tables and telemetry cards with `textContent` or sanitized DOM node creation.
10. **Clean Up Python-CAN Bus Shutdown** (`src/engine.py:560`):
    Wrap `self.gateway_bus.shutdown()` in a `try...except (KeyError, Exception): pass` block to eliminate the noisy `KeyError: 'cavshield'` traceback on exit.

---

### Step-by-Step Safe Demo Path for Judges

To guarantee a 100% stable, flawless live presentation without triggering the identified edge-case bugs:

1. **Clean Start**:
   Open a terminal and start the server:
   ```powershell
   & ".\.venv\Scripts\python.exe" run_dashboard.py
   ```
   Open `http://127.0.0.1:8000` in Google Chrome.

2. **Step 1 — Baseline Cruise (Normal Operation)**:
   - Point out the clean telemetry: Speed steady at 72 km/h, RPM at ~2,760, Mode `NORMAL`, Threat Level `NORMAL`.
   - Point out that fast-path enforcement latency is under 0.1 ms and snapshot latency is ~0.3 ms.

3. **Step 2 — Actuator Spoofing (`spoof_brake`) with Protection ON**:
   - In the Attacks dropdown, select **Spoof Brake (0x200)** and click **Start Attack**.
   - Show that the banner turns **CRITICAL / HIGH (Spoofing)**.
   - Point out the Digital Twin telemetry: **Vehicle speed remains at 72 km/h cruise!** The inline gateway dropped the malicious brake command.
   - Click **Stop Attack**. Show that within **0.25 seconds**, mode automatically returns to **NORMAL**.

4. **Step 3 — Actuator Spoofing with Protection OFF (The Contrast)**:
   - Toggle **Protection to OFF** (yellow warning appears).
   - Trigger **Spoof Brake (0x200)** again.
   - Show that hostile packets reach the twin: **Vehicle speed drops to 0 km/h (Emergency Stop)**.
   - Stop the attack and toggle **Protection back ON**. Click **Reset System**.
   - *(Workaround)* Wait 3 seconds after reset for the AI sliding window buffer to flush clean.

5. **Step 4 — V2X Sensor Cross-Check (`v2x_phantom`)**:
   - Trigger **V2X Phantom Obstacle**.
   - Show that the V2X validator compares claimed hazard coordinates with Radar/LiDAR perceptions.
   - Show that the rogue hazard is rejected (`status: rejected`, reason: `No corroborating radar obstacle`). Speed is preserved.

6. **Step 5 — Bus Flooding Resilience (`flood`)**:
   - Trigger **Flooding Attack** (500+ Hz).
   - Show that the token-bucket rate limiter sheds the burst in 2.2 µs per frame without starving normal ECUs.
   - Stop attack.

*(Caution during demo)*: Do **not** trigger `spoof_rpm` while simultaneously clicking rapidly between dashboard tabs to avoid the `score_history` race condition. Do not run continuous flood for more than 10 minutes to avoid large log accumulation.

