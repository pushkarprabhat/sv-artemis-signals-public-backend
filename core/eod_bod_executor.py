
"""
Minimal EOD/BOD executor shim for the public-backend.

This file provides a lightweight `EODBODExecutor` implementation with a
`start_scheduler()` method that writes a heartbeat file periodically. The
real, full-featured implementation lives in the private repo; this shim keeps
the public backend healthy for local development and E2E tests.

For Shivaansh & Shivaansh — every small fix helps build the future.
"""

from __future__ import annotations

import logging
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Dict

try:
    from config import BASE_DIR
except Exception:
    BASE_DIR = Path(__file__).resolve().parents[2]

logger = logging.getLogger("artemis.eodshim")


class EODBODExecutor:
    """Lightweight executor exposing `start_scheduler`, `run_eod`, and `run_bod`.

    - `start_scheduler()` starts a background thread that updates
      `universe/metadata/scheduler_heartbeat.txt` every 60 seconds.
    - `run_eod()` and `run_bod()` are safe no-op stubs that return success
      placeholders for E2E tests.
    """

    def __init__(self):
        self.logger = logger
        self._scheduler_thread = None
        self._stop_event = threading.Event()
        self.heartbeat_path = Path(BASE_DIR) / "universe" / "metadata" / "scheduler_heartbeat.txt"
        self.heartbeat_path.parent.mkdir(parents=True, exist_ok=True)

    def start_scheduler(self) -> None:
        """Start a background thread to write heartbeat periodically.

        Safe to call multiple times; only one thread will run.
        """
        if self._scheduler_thread and self._scheduler_thread.is_alive():
            self.logger.info("[EODSHIM] Scheduler already running")
            return

        def _loop():
            self.logger.info("[EODSHIM] Scheduler thread started")
            while not self._stop_event.is_set():
                try:
                    ts = datetime.utcnow().isoformat() + "Z"
                    with open(self.heartbeat_path, "w", encoding="utf-8") as f:
                        f.write(f"{ts} OK\n")
                    self.logger.debug(f"[EODSHIM] Wrote heartbeat: {ts}")
                except Exception as e:
                    self.logger.error(f"[EODSHIM] Failed to write heartbeat: {e}")
                # Sleep 60 seconds between heartbeats
                time.sleep(60)

        t = threading.Thread(target=_loop, name="eod_shim_scheduler", daemon=True)
        self._scheduler_thread = t
        t.start()

    def stop_scheduler(self) -> None:
        """Stop the background heartbeat thread (if running)."""
        if self._scheduler_thread and self._scheduler_thread.is_alive():
            self._stop_event.set()
            self._scheduler_thread.join(timeout=5)
            self.logger.info("[EODSHIM] Scheduler stopped")

    def run_eod(self, force: bool = False) -> Dict:
        self.logger.info("[EODSHIM] run_eod called (public shim)")
        return {"status": "success", "message": "EOD run simulated by public shim"}

    def run_bod(self, force: bool = False) -> Dict:
        self.logger.info("[EODSHIM] run_bod called (public shim)")
        return {"status": "success", "message": "BOD run simulated by public shim"}


if __name__ == "__main__":
    print("EOD/BOD shim test")
    e = EODBODExecutor()
    e.start_scheduler()
    print("Heartbeat written to:", e.heartbeat_path)
