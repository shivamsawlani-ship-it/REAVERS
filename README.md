# CAVShield

AI-driven, explainable cyber-physical intrusion detection and fail-safe system
for Connected Autonomous Vehicles.

## Windows Setup

```powershell
# 1. Create a virtual environment (Python 3.11 or 3.12)
python -m venv .venv

# 2. Activate
.venv\Scripts\Activate.ps1        # PowerShell
# or
.venv\Scripts\activate.bat        # CMD

# 3. Install dependencies
pip install -r requirements.txt
```

## Running

```powershell
# Start the simulation CLI runner
python run_simulation.py

# Launch the SOC Dashboard Console (runs at http://127.0.0.1:8000)
python run_dashboard.py
```

### Dashboard CLI Options

```powershell
# Custom port and headless (no browser launch)
python run_dashboard.py --port 8000 --no-browser

# Automatically trigger the 8-step guided demonstration on start
python run_dashboard.py --scenario demo
```

## Real-Time SOC Dashboard

The dashboard provides a pixel-faithful, responsive Security Operations Center (SOC) console for vehicle monitoring, threat inspection, forensic review, and interactive cyber-physical attack simulation.

### Architecture
- **In-Process Integration**: Runs FastAPI and `CAVShieldEngine` within the same Python process, eliminating cross-process serialization overhead for fast path execution.
- **100% Offline**: Serves the bundled Plotly library locally from the Python environment (`/static/plotly.min.js`); zero CDN or external internet dependencies.
- **Live WebSocket Streaming**: `/ws` emits atomic snapshots every 250 ms (4 Hz) matching CONTRACT Item 11.
- **Defensive UI**: Zero-framework vanilla JavaScript with strictly safe DOM manipulation (`textContent` & node creation) preventing XSS vulnerabilities.

### Key API Endpoints

| Endpoint | Method | Description |
|---|---|---|
| `/` | `GET` | SOC Console Single-Page Application (HTML) |
| `/static/plotly.min.js` | `GET` | Offline Plotly library bundle |
| `/api/health` | `GET` | System operational status and uptime |
| `/api/snapshot` | `GET` | Complete atomic snapshot (CONTRACT Item 11 schema) |
| `/ws` | `WebSocket` | Real-time 4 Hz streaming snapshot updates |
| `/api/attack/{name}/start` | `POST` | Inject cyber or V2X attack (`flood`, `fuzz`, `replay`, `spoof_brake`, etc.) |
| `/api/attack/{name}/stop` | `POST` | Halt an active attack scenario |
| `/api/attack/stop_all` | `POST` | Terminate all active attack scenarios |
| `/api/protection` | `POST` | Toggle gateway fail-safe enforcement (`enabled: true/false`) |
| `/api/scenario/real_obstacle` | `POST` | Spawn a legitimate obstacle in ego path (`distance`, `lifetime`) |
| `/api/reset` | `POST` | Reset plant twin, buffers, detection states, and counters |
| `/api/demo/start` | `POST` | Run automated 8-step guided interactive scenario |
| `/api/demo/stop` | `POST` | Stop running guided demonstration |
| `/api/incidents/export` | `GET` | Download persistent JSONL forensic audit log |


