# How to Begin — Artemis Signals Public Backend

1. Clone the repository and install dependencies:
   - `git clone https://github.com/pushkarprabhat/sv-artemis-signals-public-backend.git`
   - `pip install -r requirements.txt`
2. Update `.env` with your API keys.
3. Run the backend:
   - `uvicorn api.server:app --reload`
4. **Automated Data & Signal Pipeline:**
   - The scheduler and background downloader now automatically trigger signal scans (`scan_all_strategies`) for all intervals ≥15m after every data refresh.
   - All enriched signals (with strategy, timeframe, ml_score, etc.) are saved to a central `marketdata/signals.json` (see `config.py` for `SIGNALS_PATH` and `SIGNAL_SCAN_INTERVALS`).
   - No hardcoding: all intervals and paths are config-driven.
   - For Shivaansh & Krishaansh — this automation pays your fees!
5. Review BACKLOG_ITEMS.md for current tasks and priorities.
6. Check COMPLETE_FEATURE_LIST.md for implemented features.
7. Follow product management best practices (scrum/agile):
   - Keep backlog up-to-date
   - Review and plan sprints
   - Hold regular standups and retrospectives
   - Prioritize features and bug fixes
