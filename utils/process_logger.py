"""
Process Logging Module - Track EOD and BOD Process Execution
File: utils/process_logger.py
"""

import json
from datetime import datetime
from pathlib import Path
import pytz
from config import BASE_DIR
from utils.logger import logger

class ProcessLogger:
    """Log and monitor EOD/BOD process execution"""
    
    LOG_FILE = BASE_DIR / 'logs' / 'processes.json'
    
    @classmethod
    def log_process_start(cls, process_name: str) -> dict:
        """Log start of a process (EOD, BOD, etc)"""
        ist = pytz.timezone('Asia/Kolkata')
        start_time = datetime.now(ist)
        
        log_entry = {
            'process': process_name,
            'start_time': start_time.isoformat(),
            'start_timestamp': start_time.timestamp(),
            'status': 'IN_PROGRESS',
            'end_time': None,
            'duration_seconds': None,
            'success': None,
            'error': None
        }
        
        logger.info(f"Starting {process_name} at {start_time}")
        return log_entry
    
    @classmethod
    def log_process_complete(cls, process_name: str, success: bool = True, error: str = None) -> dict:
        """Log completion of a process"""
        ist = pytz.timezone('Asia/Kolkata')
        end_time = datetime.now(ist)
        
        # Load existing logs
        logs = cls._load_logs()
        
        # Find the matching process
        for log in logs:
            if log['process'] == process_name and log['status'] == 'IN_PROGRESS':
                log['status'] = 'COMPLETED' if success else 'FAILED'
                log['end_time'] = end_time.isoformat()
                log['success'] = success
                log['error'] = error
                
                # Calculate duration
                if log['start_timestamp']:
                    log['duration_seconds'] = end_time.timestamp() - log['start_timestamp']
                
                logger.info(f"Completed {process_name} at {end_time} - Status: {'SUCCESS' if success else 'FAILED'}")
                
                # Save updated logs
                cls._save_logs(logs)
                return log
        
        logger.error(f"No matching in-progress process found for {process_name}")
        return None
    
    @classmethod
    def get_last_process(cls, process_name: str) -> dict:
        """Get last execution of a specific process"""
        logs = cls._load_logs()
        
        # Get all logs for this process, sorted by timestamp (newest first)
        process_logs = [log for log in logs if log['process'] == process_name]
        if process_logs:
            return process_logs[-1]  # Return most recent
        return None
    
    @classmethod
    def get_all_processes(cls) -> list:
        """Get all process logs"""
        return cls._load_logs()
    
    @classmethod
    def check_process_status(cls, process_name: str) -> dict:
        """Check if a process completed successfully"""
        last_log = cls.get_last_process(process_name)
        
        if not last_log:
            return {
                'process': process_name,
                'status': 'NEVER_RUN',
                'last_run': None,
                'success': None
            }
        
        return {
            'process': process_name,
            'status': last_log.get('status'),
            'last_run': last_log.get('end_time'),
            'success': last_log.get('success'),
            'duration_seconds': last_log.get('duration_seconds'),
            'start_time': last_log.get('start_time'),
            'error': last_log.get('error')
        }
    
    @classmethod
    def _load_logs(cls) -> list:
        """Load all process logs from file"""
        if cls.LOG_FILE.exists():
            try:
                with open(cls.LOG_FILE, 'r') as f:
                    return json.load(f)
            except:
                return []
        return []
    
    @classmethod
    def _save_logs(cls, logs: list):
        """Save process logs to file"""
        cls.LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
        try:
            with open(cls.LOG_FILE, 'w') as f:
                json.dump(logs, f, indent=2)
        except Exception as e:
            logger.error(f"Error saving process logs: {e}")


def get_eod_status() -> dict:
    """Quick function to check EOD process status"""
    return ProcessLogger.check_process_status('EOD')


def get_bod_status() -> dict:
    """Quick function to check BOD process status"""
    return ProcessLogger.check_process_status('BOD')


if __name__ == '__main__':
    # Example usage
    print("\n=== EOD Process Status ===")
    eod = ProcessLogger.check_process_status('EOD')
    print(f"Status: {eod['status']}")
    print(f"Last Run: {eod['last_run']}")
    print(f"Success: {eod['success']}")
    if eod.get('duration_seconds'):
        print(f"Duration: {eod['duration_seconds']:.1f} seconds")
    
    print("\n=== BOD Process Status ===")
    bod = ProcessLogger.check_process_status('BOD')
    print(f"Status: {bod['status']}")
    print(f"Last Run: {bod['last_run']}")
    print(f"Success: {bod['success']}")
    if bod.get('duration_seconds'):
        print(f"Duration: {bod['duration_seconds']:.1f} seconds")
    
    print("\n=== All Recent Processes ===")
    for proc in ProcessLogger.get_all_processes()[-5:]:
        print(f"{proc['process']}: {proc['status']} at {proc.get('end_time', 'N/A')}")
