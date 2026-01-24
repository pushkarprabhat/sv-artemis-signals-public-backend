import time
import requests

# Import the telegram module from utils
from utils import telegram as tg

# Ensure telegram module uses test-friendly settings
try:
    tg.ENABLE_TELEGRAM = True
    tg.TELEGRAM_TOKEN = "DUMMY_TOKEN"
    tg.TELEGRAM_CHAT_IDS = [111111, 222222, 333333]
except Exception:
    pass

# Monkeypatch requests.post to avoid real network calls
orig_post = requests.post
class DummyResp:
    def __init__(self, code=200):
        self.status_code = code

def fake_post(url, data=None, timeout=5):
    print(f"FAKE_POST -> url={url} chat_id={data.get('chat_id')} text_preview={data.get('text')[:40]!r}")
    return DummyResp(200)

requests.post = fake_post

print("Call 1 (should send):", tg.send_telegram("Duplicate test message", error=False))
print("Call 2 (should be suppressed):", tg.send_telegram("Duplicate test message", error=False))

print("Sleeping past TTL (31s)...")
time.sleep(31)
print("Call 3 after TTL (should send again):", tg.send_telegram("Duplicate test message", error=False))

# Restore requests.post
requests.post = orig_post

print("Test completed")
