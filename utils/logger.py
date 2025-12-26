"""Logging helper using Python stdlib logging.

Provides a module-level `logger` compatible with basic logging calls.
Supports console and rotating file handlers and optional Telegram alerts.
"""
import logging
import logging.handlers
import sys
from pathlib import Path
from config import LOG_LEVEL, LOG_TO_FILE, LOG_DIR, ENABLE_TELEGRAM


def _get_level(level_name: str):
	try:
		return getattr(logging, level_name.upper())
	except Exception:
		return logging.INFO


class CarriageReturnFormatter(logging.Formatter):
	"""Custom formatter that adds carriage return before ERROR and WARNING messages"""
	def format(self, record: logging.LogRecord) -> str:
		msg = super().format(record)
		# Add carriage return before ERROR and WARNING to separate them visually
		if record.levelno >= logging.WARNING:  # WARNING=30, ERROR=40
			msg = "\n" + msg
		return msg


LOGGER_NAME = 'Artemis Signals'
logger = logging.getLogger(LOGGER_NAME)
logger.setLevel(_get_level(LOG_LEVEL))
logger.propagate = False

# remove existing handlers if reloaded
for h in list(logger.handlers):
	logger.removeHandler(h)

# Console handler
ch = logging.StreamHandler(sys.stderr)
ch.setLevel(_get_level(LOG_LEVEL))
fmt = CarriageReturnFormatter("%(asctime)s %(levelname)-8s [%(name)s] %(message)s", "%Y-%m-%d %H:%M:%S")
ch.setFormatter(fmt)
logger.addHandler(ch)

# File handler
if LOG_TO_FILE:
	try:
		Path(LOG_DIR).mkdir(parents=True, exist_ok=True)
		# Use RotatingFileHandler to keep only last 10 MB
		fh = logging.handlers.RotatingFileHandler(
			str(Path(LOG_DIR) / 'bot.log'),
			maxBytes=10 * 1024 * 1024,  # 10 MB limit
			backupCount=5,
			encoding='utf-8'
		)
		fh.setLevel(_get_level(LOG_LEVEL))
		fh.setFormatter(CarriageReturnFormatter("%(asctime)s %(levelname)-8s [%(name)s] %(message)s", "%Y-%m-%d %H:%M:%S"))
		logger.addHandler(fh)
	except Exception:
		logger.exception('Failed to create file handler for logs')


class TelegramHandler(logging.Handler):
	def __init__(self, level=logging.ERROR):
		super().__init__(level)

	def emit(self, record: logging.LogRecord) -> None:
		try:
			msg = self.format(record)
			# lazy import to avoid circular imports
			from utils.telegram import send_telegram
			send_telegram(msg)
		except Exception:
			# never let logging raise
			pass


def configure_telegram_alerts(level: str = 'ERROR'):
	if ENABLE_TELEGRAM:
		th = TelegramHandler(_get_level(level))
		th.setFormatter(logging.Formatter("%(asctime)s %(levelname)-8s %(message)s", "%Y-%m-%d %H:%M:%S"))
		logger.addHandler(th)


# configure telegram if enabled
configure_telegram_alerts()
