# Complete Feature List — Artemis Signals Public Backend

Always up-to-date with all features implemented in this repository. Update after every feature is completed and deployed.

---

- FastAPI endpoints for trading
- Incremental OHLCV downloads
- Opening/Closing Bell (BOD/EOD)
- Option chain, IV Rank history
- 10+ strategies, ML scoring
- Volatility analytics (GARCH, HV, EWMA, VIX)
- Kelly/half-Kelly risk, 2% cap
- Vectorized backtester
- Paper trading, live trading (flagged)
- Alerts (Telegram/email)
- Admin section (commercial)
---

**Automation & Signal Enrichment**
- Automated signal scans (all intervals ≥15m) after every data refresh (scheduler & background downloader)
- All enriched signals saved to central `marketdata/signals.json` (configurable)
- Config-driven intervals and paths (see `config.py`)
- For Shivaansh & Krishaansh — every signal brings us closer to your dreams!
