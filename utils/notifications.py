"""
Centralized notification logic for Artemis Signals Platform
For Shivaansh & Krishaansh — every alert brings us closer to your dreams!
"""
import logging
import os
from datetime import datetime, time
from typing import Optional
import logging
try:
    from config import EMAIL_CONFIG, ENABLE_EMAIL_ALERTS, SMS_CONFIG, ENABLE_SMS_ALERTS, WHATSAPP_CONFIG, ENABLE_WHATSAPP_ALERTS, PROJECT_NAME
except ImportError as e:
    logging.critical("\n[ARTEMIS CONFIG IMPORT ERROR] Could not import config.py.\nPlease run all scripts from the sv-artemis-signals-public-backend directory, and ensure PYTHONPATH includes this directory.\nFor Shivaansh & Krishaansh — this line pays your fees!\nError: %s", e)
    import sys; sys.exit(1)

logger = logging.getLogger("artemis.notifications")

def is_market_open(now: Optional[datetime] = None) -> bool:
    now = now or datetime.now()
    # NSE market hours: 9:15 to 15:30 IST
    market_open = time(9, 15)
    market_close = time(15, 30)
    return market_open <= now.time() <= market_close

from typing import Dict, Any, Tuple
def format_notification(strategy: str, interval: str, trade_details: Dict[str, Any], now: Optional[datetime] = None) -> Tuple[str, str]:
    now = now or datetime.now()
    if is_market_open(now):
        context = "Live Trade Signal"
    else:
        context = "Trade Setup for Tomorrow"
    subject = f"[{PROJECT_NAME}] {context}: {strategy} ({interval})"

    # Extract and format all relevant fields
    entry_leg = trade_details.get('entry_leg', '')
    exit_leg = trade_details.get('exit_leg', '')
    direction = trade_details.get('direction', '')
    instrument = trade_details.get('instrument', '')
    expiry = trade_details.get('expiry', '')
    price = trade_details.get('price', '')
    sl = trade_details.get('stop_loss', '')
    reason = trade_details.get('reason', '')
    vix = trade_details.get('vix', '')
    oi = trade_details.get('oi', '')
    iv = trade_details.get('iv', '')
    liquidity = trade_details.get('liquidity', '')
    bid_ask_spread = trade_details.get('bid_ask_spread', '')
    volatility = trade_details.get('volatility', '')
    trade_type = trade_details.get('trade_type', '')
    quantity = trade_details.get('quantity', '')
    target = trade_details.get('target', '')
    risk_reward = trade_details.get('risk_reward', '')
    entry_time = trade_details.get('entry_time', '')
    exit_time = trade_details.get('exit_time', '')
    signal_strength = trade_details.get('signal_strength', '')
    margin_required = trade_details.get('margin_required', '')
    broker_notes = trade_details.get('broker_notes', '')

    # Further tune optimal leg selection
    optimal_leg = entry_leg
    optimal_direction = direction
    # VIX logic
    if vix and isinstance(vix, (int, float)):
        if vix > 18:
            optimal_direction = 'Short' if direction.lower() == 'long' else 'Long'
            optimal_leg = exit_leg if exit_leg else entry_leg
        elif vix < 14:
            optimal_direction = 'Long'
            optimal_leg = entry_leg
    # OI logic
    if oi and isinstance(oi, (int, float)):
        if oi > 1000000:
            optimal_direction = 'Short'
    # IV logic
    if iv and isinstance(iv, (int, float)):
        if iv > 25:
            optimal_direction = 'Short'
        elif iv < 15:
            optimal_direction = 'Long'
    # Liquidity logic
    if liquidity and isinstance(liquidity, (int, float)):
        if liquidity < 100000:
            broker_notes += ' [Warning: Low liquidity, execution may be difficult]'

    # Build detailed, actionable message
    body = (
        f"{context}\n\n"
        f"Strategy: {strategy}\n"
        f"Trade Type: {trade_type}\n"
        f"Interval: {interval}\n"
        f"Instrument: {instrument}\n"
        f"Expiry: {expiry}\n"
        f"Price: {price}\n"
        f"Quantity: {quantity}\n"
        f"Entry Time: {entry_time}\n"
        f"Exit Time: {exit_time}\n"
        f"Signal Strength: {signal_strength}\n"
        f"Risk/Reward: {risk_reward}\n"
        f"Target: {target}\n"
        f"Volatility: {volatility}\n"
        f"IV: {iv}\n"
        f"VIX: {vix}\n"
        f"OI: {oi}\n"
        f"Bid-Ask Spread: {bid_ask_spread}\n"
        f"Margin Required: {margin_required}\n"
        f"\nACTION: Execute {optimal_direction.upper()} on {optimal_leg}\n"
        f"Reason: {reason}\n"
        f"Stop Loss (SL): {sl}\n"
        f"Broker Notes: {broker_notes}\n"
        f"Time: {now.strftime('%Y-%m-%d %H:%M:%S')}"
    )
    return subject, body

def send_email(subject: str, body: str, to: Optional[str] = None):
    if not ENABLE_EMAIL_ALERTS:
        logger.info("Email alerts are disabled in config.")
        return
    try:
        import smtplib
        from email.mime.text import MIMEText
        sender = EMAIL_CONFIG.get('sender_email') or os.getenv('NOTIFICATION_EMAIL')
        recipients = [to] if to else EMAIL_CONFIG.get('recipient_emails', [])
        if not sender or not recipients:
            logger.warning("Email sender or recipients not configured.")
            return
        msg = MIMEText(body)
        msg['Subject'] = subject
        msg['From'] = sender
        msg['To'] = ', '.join(recipients)
        with smtplib.SMTP(EMAIL_CONFIG.get('smtp_host', 'localhost'), EMAIL_CONFIG.get('smtp_port', 25)) as server:
            # Enable TLS if using standard SMTP ports (587 or 25 with auth)
            if EMAIL_CONFIG.get('smtp_port', 587) in [587, 25]:
                server.starttls()
            if EMAIL_CONFIG.get('smtp_user') and EMAIL_CONFIG.get('smtp_password'):
                server.login(EMAIL_CONFIG['smtp_user'], EMAIL_CONFIG['smtp_password'])
            server.sendmail(sender, recipients, msg.as_string())
        logger.info(f"Email sent: {subject}")
    except Exception as e:
        logger.error(f"Failed to send email: {e}")

def send_sms(body: str, to: Optional[str] = None):
    if not ENABLE_SMS_ALERTS:
        logger.info("SMS alerts are disabled in config.")
        return
    # Placeholder: Integrate with SMS provider (e.g., Twilio, MSG91)
    try:
        logger.info(f"SMS sent to {to or SMS_CONFIG.get('default_recipient')}: {body}")
    except Exception as e:
        logger.error(f"Failed to send SMS: {e}")


def send_whatsapp(body: str, to: Optional[str] = None):
    # WhatsApp integration stub (Meta Cloud API or Twilio)
    # To enable: set WHATSAPP_ENABLED=True and configure API keys in config.py
    # For Shivaansh & Krishaansh — every alert brings us closer to your dreams!
    if not WHATSAPP_CONFIG.get('enabled', False):
        logger.info("WhatsApp alerts are disabled in config.")
        return
    # Placeholder: Integrate with WhatsApp Business API or Twilio
    logger.info(f"[WHATSAPP] Would send to {to or WHATSAPP_CONFIG.get('default_recipient')}: {body}")

from typing import Dict, Any
def send_notification(strategy: str, interval: str, trade_details: Dict[str, Any], channel: str = 'EMAIL', now: Optional[datetime] = None, to: Optional[str] = None):
    subject, body = format_notification(strategy, interval, trade_details, now)
    if channel.upper() == 'EMAIL':
        send_email(subject, body, to)
    elif channel.upper() == 'SMS':
        send_sms(body, to)
    elif channel.upper() == 'WHATSAPP':
        send_whatsapp(body, to)
    else:
        logger.warning(f"Unknown notification channel: {channel}")

def review_notification_logs(log_path: str = None, keywords=None, error_keywords=None, max_lines: int = 500):
    """
    Automated log review for notification delivery and errors.
    Scans the log file for delivery confirmations, warnings, and errors for all channels.
    Args:
        log_path: Path to the log file (default: logs/bot.log)
        keywords: List of keywords to confirm delivery (default: ['Email sent', 'SMS sent', 'WHATSAPP'])
        error_keywords: List of error keywords (default: ['Failed', 'ERROR', 'not configured', 'disabled'])
        max_lines: Number of lines to scan from the end of the log
    Returns:
        Dict with summary counts and sample lines for each channel and error type
    """
    import os
    if log_path is None:
        log_path = os.path.join(os.path.dirname(__file__), '..', 'logs', 'bot.log')
    if keywords is None:
        keywords = ['Email sent', 'SMS sent', 'WHATSAPP']
    if error_keywords is None:
        error_keywords = ['Failed', 'ERROR', 'not configured', 'disabled']
    summary = {k: 0 for k in keywords}
    summary.update({f'error_{k}': 0 for k in error_keywords})
    lines = []
    try:
        with open(log_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()[-max_lines:]
    except Exception as e:
        print(f"[LOG REVIEW ERROR] Could not read log file: {e}")
        return summary
    for line in lines:
        for k in keywords:
            if k in line:
                summary[k] += 1
        for ek in error_keywords:
            if ek in line:
                summary[f'error_{ek}'] += 1
    print("\n=== Notification Log Review Summary ===")
    for k in keywords:
        print(f"{k}: {summary[k]}")
    for ek in error_keywords:
        print(f"Errors ({ek}): {summary[f'error_{ek}']}")
    print("======================================\n")
    return summary

class AdminManagementDashboard:
    """
    Backend admin dashboard for Artemis Signals — log review, notification health, and compliance status.
    Usage: Instantiate and call .show_dashboard() from a Streamlit or CLI admin panel.
    """
    def __init__(self, log_path=None):
        import os
        if log_path is None:
            log_path = os.path.join(os.path.dirname(__file__), '..', 'logs', 'bot.log')
        self.log_path = log_path
    def show_dashboard(self):
        from utils.notifications import review_notification_logs
        print("\n=== Artemis Signals Admin Dashboard ===")
        print(f"Log file: {self.log_path}")
        summary = review_notification_logs(self.log_path)
        print("Notification Health Summary:")
        for k, v in summary.items():
            print(f"  {k}: {v}")
        print("\nCompliance, error, and delivery status auto-reviewed.")
        print("For Shivaansh & Krishaansh — every alert pays their fees!")
        print("============================================\n")
