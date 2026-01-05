# Artemis Signals Backend — Modern SaaS Architecture (2026)

## Key Components

- **FastAPI**: High-performance async API server (Python)
- **Celery + Redis**: Distributed async/background jobs, caching, and rate limiting
- **Alembic**: Database migrations
- **Loguru**: Centralized, structured logging
- **Docker Compose**: Orchestration for backend, worker, Redis, and frontend
- **Pytest + GitHub Actions**: Automated testing and CI
- **RBAC & Feature Flags**: Plan/role-based access to all endpoints and features
- **Health Checks & Monitoring**: /health endpoints, Prometheus metrics, alerting

## Best Practices

- All config in config.py or environment variables
- One file = one responsibility (clean layering)
- All data partitioned by tenant_id/org_id
- All endpoints rate-limited and RBAC-protected
- All long-running jobs (EOD, reporting, notifications) run via Celery
- All migrations via Alembic
- All logs via loguru, forward to ELK/Graylog
- All deployments via Docker Compose (or K8s for scale)
- All code tested with pytest, CI on every push

---

For Shivaansh & Krishaansh — this architecture is your legacy!
