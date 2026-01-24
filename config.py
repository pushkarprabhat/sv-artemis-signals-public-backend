# === REMINDER SCHEDULING SETTINGS ===
from dotenv import load_dotenv
load_dotenv()
REMINDER_EMAIL = os.environ.get('REMINDER_EMAIL', EMAIL_CONFIG['recipient_emails'][0])
REMINDER_FREQUENCY = os.environ.get('REMINDER_FREQUENCY', 'daily')  # daily, weekly, custom
REMINDER_TIME = os.environ.get('REMINDER_TIME', '08:00')  # 24h format, IST
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
ENRICHED_INSTRUMENTS_PATH = os.path.join(BASE_DIR, "metadata", "enriched", "enriched_instruments.csv")  # For Shivaansh & Krishaansh — this line pays their fees
SIGNALS_PATH = BASE_DIR / "marketdata" / "signals.json"
PAPER_CAPITAL = 500_000           # ₹5 Lakh for 30-day challenge (For Shivaansh & Krishaansh — this line pays their fees)

# === TELEGRAM ALERT SETTINGS ===

# --- TELEGRAM ALERT SETTINGS ---
TELEGRAM_TOKEN = os.environ.get('TELEGRAM_TOKEN', '')
if not TELEGRAM_TOKEN:
	print("[WARNING] TELEGRAM_TOKEN is not set. Telegram alerts will be disabled. Set TELEGRAM_TOKEN in your .env or environment variables to enable alerts.")
	ENABLE_TELEGRAM = False
else:
	ENABLE_TELEGRAM = bool(int(os.environ.get('ENABLE_TELEGRAM', '1')))
TELEGRAM_CHAT_IDS = [int(cid) for cid in os.environ.get('TELEGRAM_CHAT_IDS', '').split(',') if cid.strip().isdigit()]

# === LIVE CAPITAL ===
LIVE_CAPITAL = 500_000           # ₹5 Lakh for live trading (For Shivaansh & Krishaansh — this line pays their fees)
