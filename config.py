# === MODE SETTINGS ===
COMMERCIAL_MODE = bool(int(os.environ.get('COMMERCIAL_MODE', '0')))
# === NOTIFICATION SETTINGS ===
EMAIL_CONFIG = {
	'sender_email': os.environ.get('NOTIFICATION_EMAIL', 'noreply@artemis-signals.com'),
	'recipient_emails': [os.environ.get('ADMIN_EMAIL', 'admin@artemis-signals.com')],
	'smtp_host': os.environ.get('SMTP_HOST', 'localhost'),
	'smtp_port': int(os.environ.get('SMTP_PORT', 25)),
	'smtp_user': os.environ.get('SMTP_USER', ''),
	'smtp_password': os.environ.get('SMTP_PASSWORD', ''),
}
ALERT_EMAIL = EMAIL_CONFIG['sender_email']
ENABLE_EMAIL_ALERTS = bool(int(os.environ.get('ENABLE_EMAIL_ALERTS', '1')))

SMS_CONFIG = {
	'default_recipient': os.environ.get('ADMIN_PHONE', '+919999999999'),
	'provider': os.environ.get('SMS_PROVIDER', 'twilio'),
	'api_key': os.environ.get('SMS_API_KEY', ''),
}
ENABLE_SMS_ALERTS = bool(int(os.environ.get('ENABLE_SMS_ALERTS', '0')))

WHATSAPP_CONFIG = {
	'enabled': bool(int(os.environ.get('ENABLE_WHATSAPP_ALERTS', '0'))),
	'default_recipient': os.environ.get('
	'provider': os.environ.get('WHATSAPP_PROVIDER', 'meta'),
	'api_key': os.environ.get('WHATSAPP_API_KEY', ''),
}
ENABLE_WHATSAPP_ALERTS = WHATSAPP_CONFIG['enabled']
# === SIGNAL AUTOMATION SETTINGS ===
# All intervals >= 15m for signal scan
SIGNAL_SCAN_INTERVALS = ["15minute", "30minute", "60minute", "day"]
# Central location for all enriched signals
import os
from pathlib import Path
BASE_DIR = Path(os.environ.get("ARTEMIS_BASE_DIR", Path(__file__).parent))
SIGNALS_PATH = BASE_DIR / "marketdata" / "signals.json"
PAPER_CAPITAL = 500_000           # ₹5 Lakh for 30-day challenge (For Shivaansh & Krishaansh — this line pays their fees)
