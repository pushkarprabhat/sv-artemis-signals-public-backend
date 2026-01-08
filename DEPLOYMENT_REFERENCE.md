# Artemis Signals Deployment Reference

## 1. Sample .env Files (Local, UAT, Prod)

### .env.local
```
# Local Development
DATABASE_URL=postgresql://artemis_local:localpass@localhost:5432/artemis_db
SECRET_KEY=dev-secret-key-please-change
TELEGRAM_TOKEN=dev-telegram-token
EMAIL_HOST=smtp.mailtrap.io
EMAIL_PORT=2525
EMAIL_USER=dev-email-user
EMAIL_PASS=dev-email-pass
SENTRY_DSN=
KITE_API_KEY=your_kite_api_key
KITE_API_SECRET=your_kite_api_secret
LIVE_TRADING_ENABLED=False
DOMAIN=localhost
FRONTEND_URL=http://localhost:5173
BACKEND_URL=http://localhost:8000
ADMIN_EMAIL=admin@localhost
SMTP_TLS=True
LOG_LEVEL=DEBUG
BACKUP_PATH=./backups
SESSION_COOKIE_SECURE=False
ALLOWED_HOSTS=localhost,127.0.0.1
RBAC_ENABLED=True
EOD_TASK_TIME=17:50
MAX_CONCURRENT_TRADES=10
MAX_TRADE_RISK_PCT=2
HALF_KELLY_ENABLED=True
PAPER_TRADING_DAYS=60
```

### .env.uat
```
# UAT/Staging
DATABASE_URL=postgresql://artemis_uat:uatpass@uat-db-host:5432/artemis_db
SECRET_KEY=uat-secret-key-please-change
TELEGRAM_TOKEN=uat-telegram-token
EMAIL_HOST=smtp.sendgrid.net
EMAIL_PORT=587
EMAIL_USER=uat-email-user
EMAIL_PASS=uat-email-pass
SENTRY_DSN=your-uat-sentry-dsn
KITE_API_KEY=your_kite_api_key
KITE_API_SECRET=your_kite_api_secret
LIVE_TRADING_ENABLED=False
DOMAIN=uat.artemis-signals.com
FRONTEND_URL=https://uat.artemis-signals.com
BACKEND_URL=https://api-uat.artemis-signals.com
ADMIN_EMAIL=admin@uat.artemis-signals.com
SMTP_TLS=True
LOG_LEVEL=INFO
BACKUP_PATH=/var/backups/artemis
SESSION_COOKIE_SECURE=True
ALLOWED_HOSTS=uat.artemis-signals.com,api-uat.artemis-signals.com
RBAC_ENABLED=True
EOD_TASK_TIME=17:50
MAX_CONCURRENT_TRADES=10
MAX_TRADE_RISK_PCT=2
HALF_KELLY_ENABLED=True
PAPER_TRADING_DAYS=60
```

### .env.prod
```
# Production
DATABASE_URL=postgresql://artemis_prod:prodpass@prod-db-host:5432/artemis_db
SECRET_KEY=prod-secret-key-rotate-before-go-live
TELEGRAM_TOKEN=prod-telegram-token
EMAIL_HOST=smtp.sendgrid.net
EMAIL_PORT=587
EMAIL_USER=prod-email-user
EMAIL_PASS=prod-email-pass
SENTRY_DSN=your-prod-sentry-dsn
KITE_API_KEY=your_kite_api_key
KITE_API_SECRET=your_kite_api_secret
LIVE_TRADING_ENABLED=True
DOMAIN=artemis-signals.com
FRONTEND_URL=https://artemis-signals.com
BACKEND_URL=https://api.artemis-signals.com
ADMIN_EMAIL=admin@artemis-signals.com
SMTP_TLS=True
LOG_LEVEL=WARNING
BACKUP_PATH=/var/backups/artemis
SESSION_COOKIE_SECURE=True
ALLOWED_HOSTS=artemis-signals.com,api.artemis-signals.com
RBAC_ENABLED=True
EOD_TASK_TIME=17:50
MAX_CONCURRENT_TRADES=10
MAX_TRADE_RISK_PCT=2
HALF_KELLY_ENABLED=True
PAPER_TRADING_DAYS=60
```

---

## 2. Config Key Explanations

- **DATABASE_URL**: Full DB URI, use strong passwords and unique DB per env.
- **SECRET_KEY**: Unique per env, rotate before go-live.
- **TELEGRAM_TOKEN**: For alerts, set or disable gracefully.
- **EMAIL_HOST/PORT/USER/PASS**: SMTP config for notifications.
- **SENTRY_DSN**: Sentry error tracking, unique per env.
- **KITE_API_KEY/SECRET**: Trading API, keep secure.
- **LIVE_TRADING_ENABLED**: Always False for dev/UAT, True for prod after paper trading.
- **DOMAIN/FRONTEND_URL/BACKEND_URL**: Used for CORS, links, and host validation.
- **ADMIN_EMAIL**: For system alerts and admin notifications.
- **SMTP_TLS**: Enforce TLS for email.
- **LOG_LEVEL**: DEBUG for dev, INFO for UAT, WARNING for prod.
- **BACKUP_PATH**: Secure backup location, offsite if possible.
- **SESSION_COOKIE_SECURE**: True for prod/UAT.
- **ALLOWED_HOSTS**: Restrict to known domains/subdomains.
- **RBAC_ENABLED**: Role-based access control.
- **EOD_TASK_TIME**: End-of-day automation time.
- **MAX_CONCURRENT_TRADES/MAX_TRADE_RISK_PCT/HALF_KELLY_ENABLED/PAPER_TRADING_DAYS**: Risk management.

---

## 3. SIGNALS_PATH Import Error Fix

- Ensure `SIGNALS_PATH` is defined in `config/__init__.py` or imported from the correct config file.
- Example for `config/__init__.py`:
  ```python
  import os
  SIGNALS_PATH = os.getenv("SIGNALS_PATH", "./signals")
  # ...other config...
  ```
- In `main.py`, use:
  ```python
  from config import SIGNALS_PATH
  ```
- If you use a config class, import as:
  ```python
  from config import Config
  signals_path = Config.SIGNALS_PATH
  ```

---

## 4. Deployment Readiness Tasks

- [ ] Place `.env.local`, `.env.uat`, `.env.prod` in backend and frontend root folders.
- [ ] Patch `config.py` to load all keys from `.env` (use python-dotenv if needed).
- [ ] Validate all keys are present and correct for each environment.
- [ ] Rotate all secrets before go-live.
- [ ] Test all notification flows (email, Telegram).
- [ ] Ensure Sentry, log rotation, and backups are working.
- [ ] Validate domains, subdomains, SSL, and CORS.
- [ ] Run E2E tests for endpoints, onboarding, billing, health checks.
- [ ] Polish admin guide and documentation.
- [ ] Confirm motivational messages for Shivaansh & Krishaansh are present.

---

## 5. Quick Troubleshooting

- **ImportError**: Check config file path and key definition.
- **ModuleNotFoundError**: Install missing packages (`pip install sentry-sdk` etc.).
- **Env Key Missing**: Add to `.env` and reload app.
- **Email/Telegram not working**: Check credentials and network/firewall.

---

## 6. For Shivaansh & Krishaansh — this checklist pays your fees!

---

# Artemis Signals — Deployment Reference

This file is your master guide for robust, production-ready deployment. Refer to it for every environment, config, and readiness task. Every line here is a step toward your dreams!
