from datetime import datetime
import json
import os
from pathlib import Path
from typing import Optional, Dict, Any

# Log path: project-root/logs/failed_instruments_detailed.jsonl
log_file = Path(__file__).resolve().parents[2] / "logs" / "failed_instruments_detailed.jsonl"


def _ensure_log_dir() -> None:
    try:
        log_file.parent.mkdir(parents=True, exist_ok=True)
    except Exception:
        # best-effort: avoid raising during logging
        pass


def record_failure(symbol: str, exchange: str, reason: str, details: Optional[Dict[str, Any]] = None) -> None:
    """Append a best-effort JSONL record describing a failure related to an instrument.

    This function intentionally swallows errors to avoid bringing down callers in
    production. Keep records small and serializable.
    """
    try:
        _ensure_log_dir()
        entry = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "symbol": symbol,
            "exchange": exchange,
            "reason": reason,
            "details": details or {},
        }
        with open(log_file, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(entry, ensure_ascii=False) + "\n")
    except Exception:
        # swallow; best-effort only
        return


def log_failure(*args, **kwargs) -> None:
    """Compatibility alias: older modules import `log_failure`.

    Delegates to `record_failure` while accepting both positional and keyword
    forms for minimal friction.
    """
    try:
        # If called with (entry_dict,) treat it as a single payload
        if len(args) == 1 and isinstance(args[0], dict):
            payload = args[0]
            record_failure(
                payload.get("symbol", ""),
                payload.get("exchange", ""),
                payload.get("reason", ""),
                payload.get("details"),
            )
            return

        # Otherwise forward positional/keyword args to record_failure
        record_failure(*args, **kwargs)
    except Exception:
        # keep compatibility wrapper best-effort
        return
