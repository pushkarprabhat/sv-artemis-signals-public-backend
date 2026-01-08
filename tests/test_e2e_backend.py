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

# Signals
def test_signals():
    r = requests.get(f"{API_URL}/signals")
    assert r.status_code == 200
    assert isinstance(r.json(), list)

# EOD job
def test_eod_job():
    r = requests.get(f"{API_URL}/eod/status")
    assert r.status_code == 200
    assert "last_run" in r.json()

# Debt tracker
def test_debt_tracker():
    r = requests.get(f"{API_URL}/debt")
    assert r.status_code == 200
    assert "debt" in r.json()

# Motivational message
def test_motivation():
    r = requests.get(f"{API_URL}/motivation")
    assert r.status_code == 200
    assert "Shivaansh" in r.text or "Krishaansh" in r.text
