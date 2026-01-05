# Artemis Signals Backend Health Alert Script
# Notifies via email if /health endpoint reports degraded status

import requests
import smtplib
from email.mime.text import MIMEText
import time

HEALTH_URL = "http://localhost:8000/health"  # Update if needed
CHECK_INTERVAL = 300  # seconds
ALERT_EMAIL = "your-alert@example.com"
FROM_EMAIL = "artemis-alerts@example.com"
SMTP_SERVER = "smtp.example.com"
SMTP_PORT = 587
SMTP_USER = "smtp-user"
SMTP_PASS = "smtp-pass"


from typing import Any
def send_alert(subject: str, body: str) -> None:
    msg = MIMEText(str(body))
    msg["Subject"] = subject
    msg["From"] = FROM_EMAIL
    msg["To"] = ALERT_EMAIL
    with smtplib.SMTP(SMTP_SERVER, SMTP_PORT) as server:
        server.starttls()
        server.login(SMTP_USER, SMTP_PASS)
        server.sendmail(FROM_EMAIL, [ALERT_EMAIL], msg.as_string())


def check_health():
    try:
        resp = requests.get(HEALTH_URL, timeout=10)
        data = resp.json()
        if data.get("status") != "ok":
            send_alert(
                "[ALERT] Artemis Backend Health Degraded",
                f"Health check failed: {data}"
            )
    except Exception as e:
        send_alert(
            "[ALERT] Artemis Backend Health Check Error",
            f"Exception during health check: {e}"
        )


def main():
    while True:
        check_health()
        time.sleep(CHECK_INTERVAL)

if __name__ == "__main__":
    main()
