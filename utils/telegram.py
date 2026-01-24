# utils/telegram.py — TELEGRAM ALERTS FOR TRADE EXECUTION
# Send real-time notifications when trades execute or profit milestones hit
# Professional notification system for real-time trade updates


import requests
from config import TELEGRAM_TOKEN, TELEGRAM_CHAT_IDS, ENABLE_TELEGRAM
from utils.logger import logger
from datetime import datetime

# --- STARTUP CHECK: Warn if TELEGRAM_TOKEN is missing ---
if not TELEGRAM_TOKEN:
    logger.warning("[TELEGRAM] TELEGRAM_TOKEN is not set. Telegram alerts will be disabled.\nFor Shivaansh & Krishaansh — this line pays your fees!")

def send_telegram(message: str, error: bool = False, chat_ids: list = None) -> bool:
    """Send Telegram message to configured chat(s)
    
    Args:
        message: Message text to send
        error: If True, message will NOT be sent (errors only logged, not broadcast)
        chat_ids: Optional list of chat IDs to send to (uses config default if None)
    
    Returns:
        True if sent successfully to at least one recipient, False otherwise
    """
    # Do NOT send errors to Telegram - only log them
    if error:
        logger.info(f"Error suppressed from Telegram (logged only): {message}")
        return False
    
    # Check if Telegram is enabled and configured
    if not ENABLE_TELEGRAM:
        return False
    
    if not TELEGRAM_TOKEN:
        logger.debug("Telegram not configured (missing token)")
        return False
    
    # Use provided chat IDs or fallback to config
    recipients = chat_ids or TELEGRAM_CHAT_IDS
    if not recipients:
        logger.debug("Telegram not configured (no chat IDs)")
        return False
    
    try:
        # Add timestamp and emoji (no error prefix)
        timestamp = datetime.now().strftime("%H:%M:%S")
        prefix = "📢 "
        full_message = f"{prefix}[Artemis] [{timestamp}] {message}"
        
        url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
        
        success_count = 0
        for chat_id in recipients:
            try:
                payload = {"chat_id": chat_id, "text": full_message}
                response = requests.post(url, data=payload, timeout=5)
                
                if response.status_code == 200:
                    logger.debug(f"[OK] Telegram sent to {chat_id}: {message[:50]}...")
                    success_count += 1
                else:
                    logger.warning(f"Telegram API error for {chat_id}: {response.status_code}")
            except Exception as e:
                logger.warning(f"Failed to send to {chat_id}: {e}")
        
        return success_count > 0
            
    except requests.exceptions.Timeout:
        logger.warning("Telegram request timeout")
        return False
    except Exception as e:
        logger.warning(f"Failed to send Telegram: {e}")
        return False
        return False

def send_entry_signal(symbol: str, strategy: str, price: float, confidence: float = None, 
                      kelly_pct: float = None, additional_info: str = None) -> bool:
    """Send ENTRY signal to Telegram
    
    Args:
        symbol: Trading symbol/pair (e.g., "INFY-TCS", "NIFTY")
        strategy: Strategy name (e.g., "Pair Trading", "Strangle")
        price: Entry price
        confidence: Confidence score 0-100 (optional)
        kelly_pct: Kelly-based position sizing percentage (optional)
        additional_info: Extra details to include (optional)
    
    Returns:
        True if sent successfully, False otherwise
    """
    msg = f"🎯 ENTRY SIGNAL\n"
    msg += f"Symbol: {symbol}\n"
    msg += f"Strategy: {strategy}\n"
    msg += f"Entry Price: ₹{price:.2f}"
    
    if confidence is not None:
        msg += f"\nConfidence: {confidence:.0f}%"
    if kelly_pct is not None:
        msg += f"\nPosition Size: {kelly_pct:.2f}%"
    if additional_info:
        msg += f"\nDetails: {additional_info}"
    
    return send_telegram(msg)


def send_exit_signal(symbol: str, entry_price: float, exit_price: float, 
                     profit: float = None, reason: str = None) -> bool:
    """Send EXIT signal to Telegram
    
    Args:
        symbol: Trading symbol/pair
        entry_price: Entry price
        exit_price: Exit price
        profit: Profit/loss amount in rupees (optional)
        reason: Exit reason (e.g., "Profit Target", "Stop Loss", "Time Exit")
    
    Returns:
        True if sent successfully, False otherwise
    """
    msg = f"✅ EXIT SIGNAL\n"
    msg += f"Symbol: {symbol}\n"
    msg += f"Entry: ₹{entry_price:.2f} → Exit: ₹{exit_price:.2f}"
    
    if profit is not None:
        emoji = "🟢" if profit >= 0 else "🔴"
        msg += f"\n{emoji} P&L: ₹{profit:,.2f}"
        if profit != 0:
            pct = (profit / entry_price) * 100 if entry_price != 0 else 0
            msg += f" ({pct:+.2f}%)"
    
    if reason:
        msg += f"\nReason: {reason}"
    
    return send_telegram(msg)


def send_profit_milestone(amount: float, strategy: str = "Pairs"):
    """Send celebration message when profit milestone hit
    
    Args:
        amount: Profit amount in rupees
        strategy: Strategy name (default "Pairs")
    """
    emoji = "🎉"
    if amount > 500000:
        emoji = "🚀"
    if amount > 500000:
        emoji = "💎"
    
    msg = f"{emoji} PROFIT MILESTONE: {strategy} strategy earned ₹{amount:,.0f}!"
    send_telegram(msg)

# Example usage (uncomment to test):
# send_telegram("IV Rank 92% → Sold NIFTY strangle → ₹2,48,000 profit in 24 hours")
# send_trade_alert("INFY-TCS", "ENTRY", 2500.50, kelly_size=2.0)
# send_profit_milestone(250000, "Strangle")