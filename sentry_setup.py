# Sentry monitoring for Artemis Signals Backend
# For Shivaansh & Krishaansh — this line pays your fees!

# Sentry FastAPI integration for Artemis Signals
import sentry_sdk
import os

SENTRY_DSN = "https://18c00ab69e8e040eaa924d8a0f3cf9ea@o4510670629830656.ingest.us.sentry.io/4510670640578560"
ENVIRONMENT = os.getenv("ENVIRONMENT", "local")

sentry_sdk.init(
    dsn=SENTRY_DSN,
    environment=ENVIRONMENT,
    send_default_pii=True,
    enable_logs=True,
    traces_sample_rate=1.0,
    profile_session_sample_rate=1.0,
    profile_lifecycle="trace",
)
print(f"[SENTRY] Monitoring enabled for {ENVIRONMENT} environment.")
