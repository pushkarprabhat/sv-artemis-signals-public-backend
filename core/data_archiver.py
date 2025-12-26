# core/data_archiver.py — Data Retention and Archival Management System
# Implements 1-year retention policy with automatic archival and purge mechanism
# Thread-safe singleton for persistent data lifecycle management

import os
import shutil
import gzip
import json
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
from enum import Enum
import threading
import pandas as pd
from utils.logger import logger

class ArchivalStatus(Enum):
    """Data archival status"""
    ACTIVE = "active"
    ARCHIVED = "archived"
    PURGED = "purged"
    PENDING = "pending"

class DataFile:
    """Metadata about a data file"""
    def __init__(self, file_path: Path):
        self.file_path = file_path
        self.filename = file_path.name
        self.created_date = datetime.fromtimestamp(file_path.stat().st_ctime)
        self.modified_date = datetime.fromtimestamp(file_path.stat().st_mtime)
        self.file_size_bytes = file_path.stat().st_size
        self.status = ArchivalStatus.ACTIVE
        
        # Extract metadata from filename (e.g., SYMBOL.parquet or SYMBOL.csv)
        self.symbol = self.filename.rsplit('.', 1)[0] if '.' in self.filename else self.filename
    
    def days_old(self) -> int:
        """Get age of file in days"""
        return (datetime.now() - self.modified_date).days
    
    def to_dict(self) -> Dict:
        return {
            "file_path": str(self.file_path),
            "filename": self.filename,
            "symbol": self.symbol,
            "created_date": self.created_date.isoformat(),
            "modified_date": self.modified_date.isoformat(),
            "file_size_bytes": self.file_size_bytes,
            "file_size_mb": round(self.file_size_bytes / (1024 * 1024), 2),
            "age_days": self.days_old(),
            "status": self.status.value
        }

class ArchivalPolicy:
    """Data retention and archival policy"""
    def __init__(self):
        self.active_retention_days = 365  # Keep 1 year active
        self.archive_retention_days = 1095  # Keep archives for 3 years
        self.purge_after_days = 1095  # Delete after 3 years
        self.auto_archive = True
        self.archive_compression = "gzip"  # gzip, bz2, or none
        self.archive_schedule = "monthly"  # daily, weekly, monthly
        self.archive_on_size_mb = 100  # Archive interval when file reaches this size
    
    def should_archive(self, file: DataFile) -> bool:
        """Check if file should be archived"""
        if not self.auto_archive:
            return False
        return file.days_old() > self.active_retention_days
    
    def should_purge(self, file: DataFile) -> bool:
        """Check if archived file should be purged"""
        if file.status != ArchivalStatus.ARCHIVED:
            return False
        return file.days_old() > self.purge_after_days
    
    def to_dict(self) -> Dict:
        return {
            "active_retention_days": self.active_retention_days,
            "archive_retention_days": self.archive_retention_days,
            "purge_after_days": self.purge_after_days,
            "auto_archive": self.auto_archive,
            "archive_compression": self.archive_compression,
            "archive_schedule": self.archive_schedule,
            "archive_on_size_mb": self.archive_on_size_mb
        }

class DataArchiver:
    """
    Manage data lifecycle with archival and purge mechanisms
    
    Features:
    - 1-year active data retention policy
    - Automatic archival of older data
    - Configurable compression (gzip, bz2)
    - Selective purge mechanism
    - Progress tracking and reporting
    - File metadata persistence
    - Scheduled archival operations
    - Multi-interval support (15min, 30min, 60min, day, week)
    """
    
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        if hasattr(self, "_initialized"):
            return
        
        self._initialized = True
        self._lock = threading.Lock()
        
        self.base_data_dir = Path("marketdata")
        self.archive_dir = self.base_data_dir / "archives"
        self.state_file = self.base_data_dir / ".archiver_state.json"
        
        # Create archive directory
        self.archive_dir.mkdir(parents=True, exist_ok=True)
        
        self.policy = ArchivalPolicy()
        self.file_metadata: Dict[str, DataFile] = {}
        self.archival_history: List[Dict] = []
        
        self._load_state()
        logger.info(f"[DataArchiver] Initialized with archive dir: {self.archive_dir}")
    
    def _load_state(self):
        """Load persistent archival state from disk"""
        try:
            if self.state_file.exists():
                with open(self.state_file, 'r') as f:
                    state = json.load(f)
                
                self.policy.active_retention_days = state.get("policy", {}).get("active_retention_days", 365)
                self.policy.archive_retention_days = state.get("policy", {}).get("archive_retention_days", 1095)
                self.policy.purge_after_days = state.get("policy", {}).get("purge_after_days", 1095)
                self.policy.auto_archive = state.get("policy", {}).get("auto_archive", True)
                
                self.archival_history = state.get("history", [])
                
                logger.debug("Loaded persistent archiver state")
        
        except Exception as e:
            logger.error(f"Error loading archiver state: {e}")
    
    def _save_state(self):
        """Persist archival state to disk"""
        try:
            state = {
                "policy": self.policy.to_dict(),
                "history": self.archival_history[-100:],  # Keep last 100 operations
                "last_saved": datetime.now().isoformat()
            }
            
            with open(self.state_file, 'w') as f:
                json.dump(state, f, indent=2)
        
        except Exception as e:
            logger.error(f"Error saving archiver state: {e}")
    
    def scan_data_directory(self, interval: str = None) -> Dict[str, List[DataFile]]:
        """
        Scan data directory for archivable files
        
        Args:
            interval: Specific interval to scan (e.g., '15minute'), or None for all
        
        Returns:
            Dict mapping interval -> list of DataFile objects
        """
        scan_results = {}
        
        # Intervals to scan
        intervals = [interval] if interval else ["5minute", "15minute", "30minute", "60minute", "120minute", "180minute", "240minute", "day", "week", "month"]
        
        for tf in intervals:
            interval_dir = self.base_data_dir / tf
            
            if not interval_dir.exists():
                continue
            
            files = []
            try:
                for file_path in interval_dir.glob("*.parquet"):
                    if file_path.is_file():
                        data_file = DataFile(file_path)
                        files.append(data_file)
                
                if files:
                    scan_results[tf] = files
                    logger.debug(f"Scanned {tf}: {len(files)} files")
            
            except Exception as e:
                logger.error(f"Error scanning {interval_dir}: {e}")
        
        return scan_results
    
    def get_archivable_files(self, interval: str = None) -> List[DataFile]:
        """Get list of files that should be archived based on policy"""
        scan_results = self.scan_data_directory(interval)
        archivable = []
        
        for tf, files in scan_results.items():
            for data_file in files:
                if self.policy.should_archive(data_file):
                    archivable.append(data_file)
        
        return archivable
    
    def archive_file(self, file: DataFile, compress: bool = True) -> Tuple[bool, Optional[str]]:
        """
        Archive a single file
        
        Args:
            file: DataFile object to archive
            compress: Whether to compress the file
        
        Returns:
            Tuple of (success, error_message)
        """
        try:
            with self._lock:
                # Create archive subdirectory structure based on date
                archive_subdir = self.archive_dir / file.created_date.strftime("%Y/%m")
                archive_subdir.mkdir(parents=True, exist_ok=True)
                
                # Determine compression extension
                compress_ext = f".{self.policy.archive_compression}" if compress else ""
                archive_filename = f"{file.filename}{compress_ext}"
                archive_path = archive_subdir / archive_filename
                
                # Compress and archive
                if compress and self.policy.archive_compression == "gzip":
                    with open(file.file_path, 'rb') as f_in:
                        with gzip.open(archive_path, 'wb') as f_out:
                            shutil.copyfileobj(f_in, f_out)
                else:
                    shutil.copy2(file.file_path, archive_path)
                
                # Update metadata
                file.status = ArchivalStatus.ARCHIVED
                
                # Log operation
                operation = {
                    "type": "archive",
                    "timestamp": datetime.now().isoformat(),
                    "source_file": str(file.file_path),
                    "archive_file": str(archive_path),
                    "compressed": compress,
                    "original_size_bytes": file.file_size_bytes,
                    "archive_size_bytes": archive_path.stat().st_size if archive_path.exists() else 0
                }
                self.archival_history.append(operation)
                
                logger.info(f"Archived: {file.filename} -> {archive_path}")
                return True, None
        
        except Exception as e:
            error_msg = f"Failed to archive {file.filename}: {e}"
            logger.error(error_msg)
            return False, error_msg
    
    def purge_file(self, file: DataFile) -> Tuple[bool, Optional[str]]:
        """
        Purge an archived file permanently
        
        Args:
            file: DataFile object to purge
        
        Returns:
            Tuple of (success, error_message)
        """
        try:
            with self._lock:
                if not file.file_path.exists():
                    logger.warning(f"File not found for purge: {file.file_path}")
                    return True, None
                
                file.file_path.unlink()
                file.status = ArchivalStatus.PURGED
                
                # Log operation
                operation = {
                    "type": "purge",
                    "timestamp": datetime.now().isoformat(),
                    "file": str(file.file_path),
                    "file_size_bytes": file.file_size_bytes
                }
                self.archival_history.append(operation)
                
                logger.info(f"Purged: {file.filename}")
                return True, None
        
        except Exception as e:
            error_msg = f"Failed to purge {file.filename}: {e}"
            logger.error(error_msg)
            return False, error_msg
    
    def run_archival_cycle(self, interval: str = None, dry_run: bool = False) -> Dict:
        """
        Run complete archival cycle: archive old files, then purge archived files
        
        Args:
            interval: Specific interval to process, or None for all
            dry_run: If True, only report what would be done
        
        Returns:
            Dict with results and statistics
        """
        results = {
            "timestamp": datetime.now().isoformat(),
            "dry_run": dry_run,
            "archived_count": 0,
            "archived_size_bytes": 0,
            "purged_count": 0,
            "purged_size_bytes": 0,
            "errors": []
        }
        
        # Phase 1: Archive old files
        archivable = self.get_archivable_files(interval)
        
        for data_file in archivable:
            if dry_run:
                logger.info(f"[DRY RUN] Would archive: {data_file.filename} (age: {data_file.days_old()} days)")
                results["archived_count"] += 1
                results["archived_size_bytes"] += data_file.file_size_bytes
            else:
                success, error = self.archive_file(data_file)
                if success:
                    results["archived_count"] += 1
                    results["archived_size_bytes"] += data_file.file_size_bytes
                else:
                    results["errors"].append(error)
        
        # Phase 2: Purge archived files that exceed retention
        scan_results = self.scan_data_directory(interval)
        
        for tf, files in scan_results.items():
            for data_file in files:
                if self.policy.should_purge(data_file):
                    if dry_run:
                        logger.info(f"[DRY RUN] Would purge: {data_file.filename} (archived {data_file.days_old()} days ago)")
                        results["purged_count"] += 1
                        results["purged_size_bytes"] += data_file.file_size_bytes
                    else:
                        success, error = self.purge_file(data_file)
                        if success:
                            results["purged_count"] += 1
                            results["purged_size_bytes"] += data_file.file_size_bytes
                        else:
                            results["errors"].append(error)
        
        # Convert to MB for readability
        results["archived_size_mb"] = round(results["archived_size_bytes"] / (1024 * 1024), 2)
        results["purged_size_mb"] = round(results["purged_size_bytes"] / (1024 * 1024), 2)
        
        self._save_state()
        logger.info(f"Archival cycle complete: Archived {results['archived_count']}, Purged {results['purged_count']}")
        
        return results
    
    def get_storage_summary(self) -> Dict:
        """Get summary of storage usage"""
        summary = {
            "active_data_bytes": 0,
            "archived_data_bytes": 0,
            "total_data_bytes": 0,
            "active_file_count": 0,
            "archived_file_count": 0,
            "by_interval": {}
        }
        
        # Active data
        scan_results = self.scan_data_directory()
        for tf, files in scan_results.items():
            tf_data = {
                "file_count": len(files),
                "total_size_bytes": sum(f.file_size_bytes for f in files),
                "total_size_mb": round(sum(f.file_size_bytes for f in files) / (1024 * 1024), 2)
            }
            summary["by_interval"][tf] = tf_data
            summary["active_data_bytes"] += tf_data["total_size_bytes"]
            summary["active_file_count"] += tf_data["file_count"]
        
        # Archived data
        if self.archive_dir.exists():
            for file_path in self.archive_dir.rglob("*"):
                if file_path.is_file():
                    summary["archived_data_bytes"] += file_path.stat().st_size
                    summary["archived_file_count"] += 1
        
        summary["total_data_bytes"] = summary["active_data_bytes"] + summary["archived_data_bytes"]
        summary["active_data_mb"] = round(summary["active_data_bytes"] / (1024 * 1024), 2)
        summary["archived_data_mb"] = round(summary["archived_data_bytes"] / (1024 * 1024), 2)
        summary["total_data_mb"] = round(summary["total_data_bytes"] / (1024 * 1024), 2)
        
        return summary
    
    def restore_file(self, archive_file_path: Path) -> Tuple[bool, Optional[str]]:
        """
        Restore an archived file back to active storage
        
        Args:
            archive_file_path: Path to archived file
        
        Returns:
            Tuple of (success, error_message)
        """
        try:
            with self._lock:
                # Determine original location from archive structure or metadata
                # For now, restore to original timeframe directory
                filename = archive_file_path.name
                
                # Remove compression extension if present
                if filename.endswith(f".{self.policy.archive_compression}"):
                    filename = filename.rsplit(f".{self.policy.archive_compression}", 1)[0]
                
                # Try to determine interval from archive path (YYYY/MM structure)
                # This is a fallback - could also store in metadata
                restore_path = self.base_data_dir / "day" / filename
                
                # Decompress if needed
                if archive_file_path.name.endswith(f".{self.policy.archive_compression}"):
                    with gzip.open(archive_file_path, 'rb') as f_in:
                        with open(restore_path, 'wb') as f_out:
                            shutil.copyfileobj(f_in, f_out)
                else:
                    shutil.copy2(archive_file_path, restore_path)
                
                logger.info(f"Restored: {archive_file_path.name} -> {restore_path}")
                return True, None
        
        except Exception as e:
            error_msg = f"Failed to restore {archive_file_path.name}: {e}"
            logger.error(error_msg)
            return False, error_msg
    
    def set_policy(self, **kwargs):
        """Update archival policy settings"""
        allowed_fields = [
            "active_retention_days",
            "archive_retention_days",
            "purge_after_days",
            "auto_archive",
            "archive_compression",
            "archive_schedule",
            "archive_on_size_mb"
        ]
        
        for key, value in kwargs.items():
            if key in allowed_fields:
                setattr(self.policy, key, value)
                logger.info(f"Updated policy: {key} = {value}")
        
        self._save_state()


def get_data_archiver() -> DataArchiver:
    """Get singleton instance of DataArchiver"""
    return DataArchiver()
