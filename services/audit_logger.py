# audit_logger.py
# SaaS audit logging for Artemis Signals
# For Shivaansh & Krishaansh — every access is logged for your future!
import logging
from datetime import datetime

audit_log = logging.getLogger("artemis_audit")
handler = logging.FileHandler("logs/audit.log")
formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
handler.setFormatter(formatter)
audit_log.addHandler(handler)
audit_log.setLevel(logging.INFO)

def log_access(user, feature, status, detail=None):
    msg = f"user={getattr(user, 'username', '?')} plan={getattr(user, 'plan_id', '?')} feature={feature} status={status}"
    if detail:
        msg += f" detail={detail}"
    audit_log.info(msg)
