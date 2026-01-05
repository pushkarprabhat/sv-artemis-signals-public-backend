from datetime import datetime
from sqlalchemy import Column, Integer, String, Text, DateTime, Float, JSON, Boolean, ForeignKey, create_engine, Index
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship, sessionmaker

# ============================================================================
# DATABASE SETUP
# ============================================================================

Base = declarative_base()

# --- SAAS USER MODEL ---
class User(Base):
    __tablename__ = 'users'
    id = Column(Integer, primary_key=True, autoincrement=True)
    tenant_id = Column(String(64), nullable=False, index=True)  # Multi-tenancy: org/tenant
    username = Column(String(64), unique=True, nullable=False)
    email = Column(String(128), unique=True, nullable=False)
    plan_id = Column(String(32), nullable=False)
    plan_expiry = Column(DateTime)
    is_active = Column(Boolean, default=True)
    razorpay_subscription_id = Column(String(64), nullable=True)
    hashed_password = Column(String(128), nullable=False)  # Secure password for JWT
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
"""
Database Models for Universe Data Management

Stores:
- Universe snapshots (periodic exports from Kite)
- Universe logs (refresh history and operations)
- Universe anomalies (detected changes, new/delisted symbols)
"""

# ============================================================================
# MODELS
# ============================================================================

class UniverseSnapshot(Base):
    """
    Periodic snapshot of universe from Kite API
    
    Stores complete state of all instruments at a point in time
    """
    __tablename__ = 'universe_snapshots'

    id = Column(Integer, primary_key=True, autoincrement=True)
    tenant_id = Column(String(64), nullable=False, index=True)

    # Snapshot metadata
    total_count = Column(Integer, nullable=False)  # Total instruments in snapshot
    segments = Column(Text, nullable=True)  # JSON: segment -> count mapping
    
    # Stored data
    instruments_json = Column(Text, nullable=False)  # JSON array of instruments
    checksum = Column(String(64), nullable=False)  # SHA-256 checksum for verification
    
    # Anomaly info
    trigger = Column(String(50), nullable=False)  # manual, scheduled, anomaly_detected, cache_expired, startup
    anomalies_count = Column(Integer, default=0)  # Number of anomalies detected
    
    # Timestamps
    created_at = Column(DateTime, default=datetime.now, nullable=False)
    timestamp = Column(DateTime, default=datetime.now, nullable=False, index=True)
    
    # Relationships
    anomalies = relationship("UniverseAnomaly", back_populates="snapshot", cascade="all, delete-orphan")
    logs = relationship("UniverseLog", back_populates="snapshot")

    def __repr__(self):
        return (
            f"<UniverseSnapshot "
            f"id={self.id}, "
            f"count={self.total_count}, "
            f"anomalies={self.anomalies_count}, "
            f"trigger={self.trigger}, "
            f"time={self.timestamp.strftime('%Y-%m-%d %H:%M:%S')}>"
        )


class UniverseAnomaly(Base):
    """
    Detected anomaly in universe data
    
    Examples:
    - New symbol added
    - Symbol delisted
    - Symbol properties changed
    - Data corruption detected
    """
    __tablename__ = 'universe_anomalies'

    id = Column(Integer, primary_key=True, autoincrement=True)
    tenant_id = Column(String(64), nullable=False, index=True)
    
    # Reference to snapshot
    snapshot_id = Column(Integer, ForeignKey('universe_snapshots.id'), nullable=False, index=True)
    
    # Anomaly type: new_symbol, delisted_symbol, modified_symbol, invalid_ohlcv, data_corruption
    type = Column(String(50), nullable=False, index=True)
    
    # Details (JSON format, structure varies by type)
    details = Column(Text, nullable=False)  # JSON string with anomaly details
    
    # Severity and flags
    severity = Column(String(20), default='medium')  # low, medium, high, critical
    flagged_for_review = Column(Boolean, default=False)
    
    # Timestamps
    created_at = Column(DateTime, default=datetime.now, nullable=False)
    timestamp = Column(DateTime, default=datetime.now, nullable=False, index=True)
    
    # Relationships
    snapshot = relationship("UniverseSnapshot", back_populates="anomalies")

    def __repr__(self):
        return (
            f"<UniverseAnomaly "
            f"id={self.id}, "
            f"type={self.type}, "
            f"snapshot={self.snapshot_id}, "
            f"severity={self.severity}>"
        )


class UniverseLog(Base):
    """
    Audit log for universe operations
    
    Tracks all refresh operations, downloads, and modifications
    """
    __tablename__ = 'universe_logs'

    id = Column(Integer, primary_key=True, autoincrement=True)
    tenant_id = Column(String(64), nullable=False, index=True)
    
    # Reference to snapshot (nullable for system-level logs)
    snapshot_id = Column(Integer, ForeignKey('universe_snapshots.id'), nullable=True)
    
    # Operation type
    operation = Column(String(50), nullable=False, index=True)  
    # Examples: refresh, download_csv, download_json, anomaly_detection, db_store, user_action
    
    # Operation result
    status = Column(String(20), nullable=False)  # success, failure, partial
    message = Column(Text, nullable=True)  # Result message
    error_details = Column(Text, nullable=True)  # If failed, error details
    
    # Statistics from operation
    stats = Column(Text, nullable=True)  # JSON: operation-specific statistics
    
    # Trigger information
    trigger = Column(String(50), nullable=True)  # Why this operation was run
    
    # Timing
    duration_ms = Column(Integer, nullable=True)  # Operation duration in milliseconds
    
    # Timestamps
    created_at = Column(DateTime, default=datetime.now, nullable=False)
    started_at = Column(DateTime, nullable=True)
    completed_at = Column(DateTime, nullable=True)
    timestamp = Column(DateTime, default=datetime.now, nullable=False, index=True)
    
    # Relationships
    snapshot = relationship("UniverseSnapshot", back_populates="logs")

    def __repr__(self):
        return (
            f"<UniverseLog "
            f"id={self.id}, "
            f"op={self.operation}, "
            f"status={self.status}, "
            f"time={self.timestamp.strftime('%Y-%m-%d %H:%M:%S')}>"
        )


class UniverseChangeLog(Base):
    """
    Detailed log of changes detected between snapshots
    
    Examples:
    - RELIANCE added to NSE
    - TCS delisted from NSE
    - INFY token changed from 12345 to 12346
    """
    __tablename__ = 'universe_change_logs'

    id = Column(Integer, primary_key=True, autoincrement=True)
    tenant_id = Column(String(64), nullable=False, index=True)
    
    # Change information
    symbol = Column(String(50), nullable=False, index=True)  # Trading symbol
    change_type = Column(String(50), nullable=False)  # added, removed, modified, unchanged
    
    # Before and after state
    previous_state = Column(Text, nullable=True)  # JSON of previous state
    current_state = Column(Text, nullable=True)  # JSON of current state
    
    # What changed
    changed_fields = Column(Text, nullable=True)  # JSON list of field names that changed
    
    # Snapshots involved
    from_snapshot_id = Column(Integer, ForeignKey('universe_snapshots.id'), nullable=True)
    to_snapshot_id = Column(Integer, ForeignKey('universe_snapshots.id'), nullable=True)
    
    # Timestamps
    created_at = Column(DateTime, default=datetime.now, nullable=False)
    timestamp = Column(DateTime, default=datetime.now, nullable=False, index=True)

    def __repr__(self):
        return (
            f"<UniverseChangeLog "
            f"id={self.id}, "
            f"symbol={self.symbol}, "
            f"change={self.change_type}, "
            f"time={self.timestamp.strftime('%Y-%m-%d %H:%M:%S')}>"
        )


class UniverseStatistics(Base):
    """
    Aggregated statistics about universe
    
    Updated after each refresh for quick querying
    """
    __tablename__ = 'universe_statistics'

    id = Column(Integer, primary_key=True, autoincrement=True)
    tenant_id = Column(String(64), nullable=False, index=True)
    
    # Timestamp of the snapshot these stats are for
    snapshot_id = Column(Integer, ForeignKey('universe_snapshots.id'), nullable=False, unique=True)
    
    # Segment counts
    nse_count = Column(Integer, default=0)
    nfo_count = Column(Integer, default=0)
    bse_count = Column(Integer, default=0)
    mcx_count = Column(Integer, default=0)
    
    # Instrument type counts (for NSE equity)
    equity_count = Column(Integer, default=0)
    index_count = Column(Integer, default=0)
    
    # Derivative type counts (for NFO)
    futures_count = Column(Integer, default=0)
    options_count = Column(Integer, default=0)
    
    # Quality metrics
    symbols_with_metadata = Column(Integer, default=0)
    symbols_without_metadata = Column(Integer, default=0)
    corrupted_entries = Column(Integer, default=0)
    
    # Change metrics
    new_symbols = Column(Integer, default=0)
    delisted_symbols = Column(Integer, default=0)
    modified_symbols = Column(Integer, default=0)
    
    # Timestamps
    created_at = Column(DateTime, default=datetime.now, nullable=False)
    timestamp = Column(DateTime, default=datetime.now, nullable=False, index=True)

    def __repr__(self):
        return (
            f"<UniverseStatistics "
            f"id={self.id}, "
            f"total={self.nse_count + self.nfo_count + self.bse_count + self.mcx_count}>"
        )


# ============================================================================
# PHASE 1 MVP: Quality, Corporate Actions, Aggregation Models
# ============================================================================

class DataAudit(Base):
    """
    Quality audit record for downloaded OHLCV data
    
    Tracks validation results, completeness, accuracy scores
    """
    __tablename__ = 'data_audits'
    id = Column(Integer, primary_key=True)
    tenant_id = Column(String(64), nullable=False, index=True)
    
    symbol = Column(String(20), nullable=False, index=True)
    timeframe = Column(String(20), nullable=False, index=True)
    instrument_token = Column(String(50), index=True)
    
    # Quality metrics (percentage)
    completeness_pct = Column(Float, default=0.0)
    accuracy_score = Column(Float, default=0.0)
    anomaly_score = Column(Float, default=0.0)
    validation_score = Column(Float, default=0.0, index=True)
    
    # Data details
    total_candles = Column(Integer, default=0)
    gap_count = Column(Integer, default=0)
    gap_details = Column(JSON, default=dict)  # {date: gap_size}
    
    # Validation hash
    checksum = Column(String(64), unique=True)
    
    # Status: pass, warn, quarantine
    status = Column(String(20), default='pending')
    notes = Column(Text, default='')
    
    # Timestamps
    audit_timestamp = Column(DateTime, default=datetime.utcnow, nullable=False, index=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)


class PendingOHLCVRecord(Base):
    """
    Quarantine table for data that failed validation
    
    Records pending retry with exponential backoff
    """
    __tablename__ = 'pending_ohlcv'
    id = Column(Integer, primary_key=True)
    tenant_id = Column(String(64), nullable=False, index=True)
    
    symbol = Column(String(20), nullable=False, index=True)
    timeframe = Column(String(20), nullable=False, index=True)
    instrument_token = Column(String(50))
    
    # Retry tracking
    retry_count = Column(Integer, default=0, index=True)
    max_retries = Column(Integer, default=3)
    last_error = Column(Text, default='')
    last_validation_score = Column(Float, default=0.0)
    
    # Next retry timing
    created_ts = Column(DateTime, default=datetime.utcnow, nullable=False, index=True)
    next_retry_ts = Column(DateTime, nullable=True)
    updated_ts = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)


class CorporateActionLog(Base):
    """
    Log of corporate actions (dividends, splits, bonus)
    
    Tracks what adjustments have been applied to each symbol
    """
    __tablename__ = 'corporate_action_logs'
    id = Column(Integer, primary_key=True)
    tenant_id = Column(String(64), nullable=False, index=True)
    
    symbol = Column(String(20), nullable=False, index=True)
    
    # Action type: dividend, split, bonus
    action_type = Column(String(20), nullable=False)
    action_date = Column(DateTime, nullable=False, index=True)
    
    # Details
    value = Column(Float)  # Dividend amount or split ratio
    ex_date = Column(DateTime)
    record_date = Column(DateTime)
    payment_date = Column(DateTime)
    
    # Applied flag
    applied = Column(Boolean, default=False)
    notes = Column(Text)
    
    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)


class AdjustmentLog(Base):
    """
    Log of price adjustments applied to OHLCV data
    
    Records backward adjustments for corporate actions
    """
    __tablename__ = 'adjustment_logs'
    id = Column(Integer, primary_key=True)
    tenant_id = Column(String(64), nullable=False, index=True)
    
    symbol = Column(String(20), nullable=False, index=True)
    timeframe = Column(String(20), nullable=False, index=True)
    
    # Reason for adjustment
    adjustment_reason = Column(String(50), nullable=False)  # div, split, bonus
    adjustment_ratio = Column(Float)  # The adjustment factor applied
    applied_date = Column(DateTime, nullable=False, index=True)
    
    # Details
    rows_affected = Column(Integer)  # How many rows were adjusted
    adjustment_details = Column(JSON)  # {date: ratio_applied}
    
    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow)
