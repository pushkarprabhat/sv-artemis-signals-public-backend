import sys
import os
import time

# Ensure project root is on sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from core.eod_bod_executor import EODBODExecutor


if __name__ == '__main__':
    e = EODBODExecutor()
    e.start_scheduler()
    time.sleep(1)
    try:
        with open(e.heartbeat_path, 'r', encoding='utf-8') as f:
            print(f.read())
    except Exception as ex:
        print('Heartbeat read failed:', ex)
