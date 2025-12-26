"""
core/audit_models.py — Data Audit and Quality Models

Stores:
- Data quality audit records
- Pending OHLCV records (quarantine)
- Corporate actions and adjustments
"""

from datetime import datetime
from sqlalchemy import Column, Integer, String, Text, DateTime, Float, JSON, Boolean, ForeignKey, Numeric
from sqlalchemy.orm import relationship
from core.database import Base


class DataAudit(Base):
    """Quality audit records for downloaded OHLCV data"""
    __tablename__ = 'data_audits'

    id = Column(Integer, primary_key=True, autoincrement=True)
    
    # Reference info
    symbol = Column(String(20), nullable=False, index=True)
    timeframe = Column(String(10), nullable=False, index=True)
    instrument_token = Column(String(20), index=True)
    batch_id = Column(String(50), index=True)
    
    # Quality metrics
    validation_score = Column(Float, default=0.0)
    completeness = Column(Float, default=0.0)  # % coverage
    accuracy = Column(Float, default=0.0)  # OHLC constraint validation
    anomalies = Column(Float, default=0.0)  # Outlier score
    
    # Status and audit trail
    status = Column(String(20), default='pending')  # pending, pass, warn, quarantine
    is_quarantined = Column(Boolean, default=False)
    quarantine_reason = Column(Text)
    
    # Data integrity
    row_count = Column(Integer)
    missing_count = Column(Integer, default=0)
    duplicate_count = Column(Integer, default=0)
    gap_count = Column(Integer, default=0)
    checksum = Column(String(64))  # SHA256
    
    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    audit_date = Column(DateTime, default=datetime.utcnow, index=True)
    
    def __repr__(self):
        return f"<DataAudit {self.symbol} {self.timeframe} score={self.validation_score} status={self.status}>"


class PendingOHLCVRecord(Base):
    """Quarantine table for OHLCV records that failed quality checks"""
    __tablename__ = 'pending_ohlcv_records'

    id = Column(Integer, primary_key=True, autoincrement=True)
    
    # Reference info
    symbol = Column(String(20), nullable=False, index=True)
    timeframe = Column(String(10), nullable=False)
    instrument_token = Column(String(20))
    batch_id = Column(String(50), index=True)
    
    # OHLCV data
    date = Column(DateTime, nullable=False, index=True)
    open = Column(Numeric(18, 6), nullable=False)
    high = Column(Numeric(18, 6), nullable=False)
    low = Column(Numeric(18, 6), nullable=False)
    close = Column(Numeric(18, 6), nullable=False)
    volume = Column(Numeric(20, 0))
    
    # Quarantine metadata
    reason = Column(Text)  # Why was it quarantined?
    issue_type = Column(String(50))  # invalid_ohlc, outlier, missing_volume, etc.
    severity = Column(String(20))  # low, medium, high
    
    # Resolution
    is_resolved = Column(Boolean, default=False)
    resolution_notes = Column(Text)
    resolved_at = Column(DateTime)
    
    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    def __repr__(self):
        return f"<PendingOHLCV {self.symbol} {self.date} issue={self.issue_type}>"


class CorporateActionLog(Base):
    """Corporate actions (dividends, splits, etc.) that affect OHLCV"""
    __tablename__ = 'corporate_action_logs'

    id = Column(Integer, primary_key=True, autoincrement=True)
    
    # Reference info
    symbol = Column(String(20), nullable=False, index=True)
    instrument_token = Column(String(20))
    
    # Corporate action details
    action_type = Column(String(50), nullable=False)  # dividend, split, bonus, rights, etc.
    ex_date = Column(DateTime, nullable=False, index=True)
    record_date = Column(DateTime)
    payment_date = Column(DateTime)
    
    # Action parameters
    ratio = Column(Float)  # For splits (e.g., 1:2 = 0.5)
    amount = Column(Numeric(18, 6))  # For dividends (absolute amount)
    amount_type = Column(String(20))  # absolute, percentage
    
    # Status
    is_applied = Column(Boolean, default=False)
    applied_at = Column(DateTime)
    
    # Source and metadata
    source = Column(String(50))  # NSE, BSE, corporate website, etc.
    notes = Column(Text)
    
    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    def __repr__(self):
        return f"<CorporateAction {self.symbol} {self.action_type} ex={self.ex_date.date()}>"


class AdjustmentLog(Base):
    """Log of all OHLCV adjustments made (corporate actions, corrections, etc.)"""
    __tablename__ = 'adjustment_logs'

    id = Column(Integer, primary_key=True, autoincrement=True)
    
    # Reference info
    symbol = Column(String(20), nullable=False, index=True)
    timeframe = Column(String(10), nullable=False)
    instrument_token = Column(String(20))
    
    # Adjustment details
    adjustment_type = Column(String(50), nullable=False)  # dividend, split, bonus, manual_correction
    affected_rows = Column(Integer)  # How many OHLCV records were affected
    from_date = Column(DateTime, nullable=False)
    to_date = Column(DateTime, nullable=False)
    
    # Adjustment formula/parameters
    adjustment_factor = Column(Numeric(18, 6))  # e.g., 0.98 for 2% dividend
    adjustment_formula = Column(Text)  # Human-readable description
    
    # Impact metrics
    avg_price_change_pct = Column(Float)
    volume_change_pct = Column(Float)
    
    # Status and approval
    status = Column(String(20), default='applied')  # pending, applied, reversed, rejected
    applied_at = Column(DateTime, default=datetime.utcnow)
    reversed_at = Column(DateTime)
    reverse_reason = Column(Text)
    
    # Audit trail
    applied_by = Column(String(100))  # User or system that applied adjustment
    reviewed_by = Column(String(100))
    reviewed_at = Column(DateTime)
    
    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    def __repr__(self):
        return f"<Adjustment {self.symbol} {self.adjustment_type} rows={self.affected_rows}>"
