"""
CAVShield Dashboard Application Runner.

Entry point for starting the cyber-physical CAVShield engine and
serving the real-time dark SOC web dashboard.

Usage:
    python run_dashboard.py
    python run_dashboard.py --port 8000 --no-browser
    python run_dashboard.py --scenario demo
"""

from __future__ import annotations

import argparse
import logging
import signal
import sys
import threading
import time
import webbrowser
from typing import Optional

import uvicorn

from src import load_config
from src.dashboard.app import create_app
from src.engine import CAVShieldEngine

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("cavshield.dashboard_runner")


def open_browser_delayed(url: str, delay_seconds: float = 1.0) -> None:
    """Launch the user's default browser after a short startup delay."""
    def _target() -> None:
        time.sleep(delay_seconds)
        logger.info("Opening dashboard in default web browser: %s", url)
        webbrowser.open(url)

    thread = threading.Thread(target=_target, name="BrowserLauncher", daemon=True)
    thread.start()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="CAVShield Real-Time SOC Dashboard Console"
    )
    parser.add_argument(
        "--host",
        type=str,
        default=None,
        help="Host address to bind (default: from config/config.yaml or 127.0.0.1)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=None,
        help="Port number to bind (default: from config/config.yaml or 8000)",
    )
    parser.add_argument(
        "--scenario",
        type=str,
        default=None,
        help="Optional attack scenario or 'demo' to auto-run on startup",
    )
    parser.add_argument(
        "--no-browser",
        action="store_true",
        help="Do not automatically launch web browser on startup",
    )
    args = parser.parse_args()

    # Load configuration
    cfg = load_config()
    dash_cfg = getattr(cfg, "dashboard", {})
    host = args.host or str(getattr(dash_cfg, "host", "127.0.0.1"))
    port = args.port or int(getattr(dash_cfg, "port", 8000))

    logger.info("Initializing CAVShield cyber-physical simulation engine...")
    engine = CAVShieldEngine(cfg=cfg)
    engine.start()

    # Build FastAPI application sharing the exact same in-memory engine
    app = create_app(engine=engine)
    demo_mgr = getattr(app.state, "demo_mgr", None)

    # Auto-run optional initial scenario
    if args.scenario:
        if args.scenario.lower() == "demo":
            if demo_mgr:
                logger.info("Auto-starting guided 8-step demonstration scenario...")
                demo_mgr.start()
        else:
            logger.info("Auto-injecting requested startup scenario: %s", args.scenario)
            engine.inject(args.scenario)

    server_url = f"http://{host}:{port}"
    if not args.no_browser:
        open_browser_delayed(server_url, delay_seconds=1.0)

    # Graceful shutdown handler
    def _signal_handler(sig, frame):
        logger.info("Received termination signal (%s). Shutting down...", sig)
        if demo_mgr:
            demo_mgr.stop()
        engine.stop()
        sys.exit(0)

    signal.signal(signal.SIGINT, _signal_handler)
    if hasattr(signal, "SIGTERM"):
        signal.signal(signal.SIGTERM, _signal_handler)

    try:
        logger.info("Starting uvicorn server at %s", server_url)
        uvicorn.run(app, host=host, port=port, log_level="info")
    except (KeyboardInterrupt, SystemExit):
        logger.info("Interrupted. Cleaning up resources...")
    finally:
        if demo_mgr:
            demo_mgr.stop()
        engine.stop()
        logger.info("CAVShield engine stopped cleanly.")


if __name__ == "__main__":
    main()

