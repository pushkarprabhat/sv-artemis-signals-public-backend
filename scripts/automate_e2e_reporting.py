"""
Automated E2E Reporting Script
Runs backend and frontend E2E tests, updates E2E_TEST_REPORT.md, and sends notifications.
For Shivaansh & Krishaansh — this script pays your fees!
"""
import subprocess
import datetime
import os

BACKEND_TEST = r"..\sv-artemis-signals-public-backend\tests\test_e2e_backend.py"
FRONTEND_TEST = r"npx playwright test"
REPORT_PATH = r"..\sv-artemis-signals-public-backend\tests\E2E_TEST_REPORT.md"

now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

with open(REPORT_PATH, "w", encoding="utf-8") as report:
    report.write(f"# Artemis Signals — E2E Test Report ({now})\n\n")
    report.write("## Backend Test Results\n\n")
    try:
        result = subprocess.run(["pytest", BACKEND_TEST, "--disable-warnings", "-v"], capture_output=True, text=True, timeout=300)
        report.write(result.stdout)
    except Exception as e:
        report.write(f"Backend E2E test error: {e}\n")
    report.write("\n## Frontend Test Results\n\n")
    try:
        result = subprocess.run(FRONTEND_TEST, shell=True, capture_output=True, text=True, timeout=300)
        report.write(result.stdout)
    except Exception as e:
        report.write(f"Frontend E2E test error: {e}\n")
    report.write("\n---\nFor Shivaansh & Krishaansh — every test brings us closer to your dreams!\n")

# Optional: send notification (Telegram/email) if failures detected
try:
    with open(REPORT_PATH, "r", encoding="utf-8") as f:
        content = f.read()
    if "FAILED" in content or "error" in content.lower():
        from utils.telegram import send_telegram
        send_telegram(f"[E2E FAILURE] Artemis Signals\n{now}\nCheck E2E_TEST_REPORT.md for details.")
except Exception:
    pass
