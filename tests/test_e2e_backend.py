# Artemis Signals E2E Backend Test
# For Shivaansh & Krishaansh — this test pays your fees!
import requests
import os

API_URL = os.getenv("BACKEND_URL", "http://localhost:8000")

# Health check
def test_health():
    r = requests.get(f"{API_URL}/health")
    assert r.status_code == 200
    assert r.json().get("status") == "ok"

# Onboarding
def test_onboarding():
    r = requests.post(f"{API_URL}/signup", json={"email": "testuser@artemis.com", "password": "testpass"})
    assert r.status_code == 200
    assert "user_id" in r.json()

# Billing
def test_billing():
    r = requests.get(f"{API_URL}/billing/status", headers={"Authorization": "Bearer testtoken"})
    assert r.status_code in [200, 403]

# Alerts
def test_alerts():
    r = requests.get(f"{API_URL}/alerts/recent", headers={"Authorization": "Bearer testtoken"})
    assert r.status_code == 200
    assert isinstance(r.json(), list)

# Universe data
def test_universe():
    r = requests.get(f"{API_URL}/universe")
    assert r.status_code == 200
    assert "data" in r.json()


# Entry Signal
def test_entry_signal():
    r = requests.get(f"{API_URL}/api/v1/signal/entry")
    assert r.status_code == 200
    data = r.json()
    assert data["status"] == "ok"
    assert "signal" in data

# Exit Signal
def test_exit_signal():
    r = requests.get(f"{API_URL}/api/v1/signal/exit")
    assert r.status_code == 200
    data = r.json()
    assert data["status"] == "ok"
    assert "signal" in data

# Paper Trading
def test_paper_trading():
    r = requests.get(f"{API_URL}/api/v1/papertrading")
    assert r.status_code == 200
    data = r.json()
    assert data["status"] == "ok"

# Marketdata Download
def test_marketdata_download():
    r = requests.get(f"{API_URL}/api/v1/data/download")
    assert r.status_code == 200
    data = r.json()
    assert data["status"] == "ok"

# Marketdata Refresh
def test_marketdata_refresh():
    r = requests.get(f"{API_URL}/api/v1/data/refresh")
    assert r.status_code == 200
    data = r.json()
    assert data["status"] == "ok"

# EOD Job
def test_eod_job():
    r = requests.get(f"{API_URL}/api/v1/eod")
    assert r.status_code == 200
    data = r.json()
    assert data["status"] in ["success", "skipped"]

# BOD Job
def test_bod_job():
    r = requests.get(f"{API_URL}/api/v1/bod")
    assert r.status_code == 200
    data = r.json()
    assert data["status"] in ["success", "skipped"]
