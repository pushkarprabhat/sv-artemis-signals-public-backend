# Artemis Signals by ManekBaba — Public Context


## Robustness & Backward Compatibility (Non-Negotiable)

- All improvements, refactors, and new features must be backward-compatible: **never break anything that was working well before**.
- Every endpoint, feature, and integration must remain stable and reliable after any change.
- Robust error handling: All endpoints must always return a valid response (never crash), with clear error messages and logging.
- Regression/E2E tests must cover all critical paths to catch accidental breakage.
- No dilution of existing functionality or standards when adding robustness or new features.

## Code-Level Recommendations (Always Referenced)

> This section is canonical. Replicate in all Artemis Signals repos. Never let these recommendations go out of sight — always check here for architecture, SaaS, and subscription best practices.

### 1. Multi-Tenancy
- Add tenant_id/org_id to all user/data tables
- Filter API/data by current user's tenant/org
- Scope UI/data to current org

### 2. Usage Metering & Rate Limiting
- Use FastAPI dependencies for per-user/org rate limiting
- Store usage stats in a dedicated table (user_id, endpoint, count, period)

### 3. Automated Billing
- Use Razorpay webhooks for payment status
- Store subscription state in DB, update on webhook
- On payment failure, downgrade plan and notify user

### 4. Plan Enforcement
- All endpoints: @require_plan('pro') or similar decorator
- Centralize plan logic in SubscriptionManager

### 5. Clean Layering
- src/
  - api/ (FastAPI endpoints)
  - services/ (business logic)
  - domain/ (models, enums)
  - infrastructure/ (db, external APIs)
  - config.py (all settings)

### 6. Feature Flags
- Use config.py or DB for feature flags
- Example: if not SubscriptionManager.has_feature(user, 'advanced_signals'): return 403

### 7. Logging & Audit
- Use loguru for all logs
- Add audit_log(action, user_id, details) in critical flows

### 8. Compliance
- Add KYC fields to User model
- Run compliance checks on signup/plan change

### 9. EOD/Reporting
- Use Celery/cron for EOD jobs
- Store reports in a dedicated table or export to S3

### 10. UI/UX
- Show plan status, usage, and upgrade options in dashboard
- Add motivational messages for Shivaansh & Krishaansh everywhere

---
**ALWAYS reference this file for every new feature, refactor, or review. Replicate in all Artemis Signals repos.**

## Vision
Ultra-professional quant trading platform for Indian markets (NSE/MCX), commercial-grade, white-label ready. No personal/family content.

## Architecture
- Clean separation: Presentation (Svelte/Streamlit), Application (services), Domain (models), Infrastructure (data/API).
- All settings in config.py (no hardcoding).
- Robust error handling, Loguru logging.
- Modular, API-driven, subscription-ready.

## Features
- Incremental OHLCV downloads, volume filter, NSE bhavcopy, 6 timeframes.
- Opening/Closing Bell (BOD/EOD), world market/events, performance summary.
- Event calendar for major market events.
- Option chain, IV Rank history.
- 10+ strategies, ML scoring, live scanning.
- Volatility analytics (GARCH, HV, EWMA, VIX).
- Strangle/IV Crush setups.
- Kelly/half-Kelly risk, 2% cap.
- Vectorized backtester, multi-horizon, top trades.
- Paper trading, live trading (flagged), portfolio management.
- Alerts (Telegram/email), logging (Loguru).
- Admin section for commercial use.
- No personal/family info.

## Status
- Production readiness: 40%
- Modular UI, API, and commercial features in progress.