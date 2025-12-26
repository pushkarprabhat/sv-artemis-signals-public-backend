"""
ARTEMIS Status Bar - Real-time process tracking and status display
Integrates with Streamlit session state for persistent status updates

Features:
- Process status tracking (IDLE, PROCESSING, COMPLETED, ERROR, WARNING)
- Real-time progress with visual progress bar
- Scanning statistics (symbols scanned, pairs found, signals detected)
- Trade execution tracking (entry, exit, P&L)
- Session state persistence and history
- ARTEMIS theme integration
"""

import streamlit as st
from typing import Optional, List, Dict, Any
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum


class ProcessStatus(Enum):
    """Process status states"""
    IDLE = "idle"
    PROCESSING = "processing"
    COMPLETED = "completed"
    ERROR = "error"
    WARNING = "warning"


@dataclass
class ScanningStats:
    """Scanning operation statistics"""
    symbols_scanned: int = 0
    pairs_analyzed: int = 0
    signals_found: int = 0
    buy_signals: int = 0
    sell_signals: int = 0
    processing_time: float = 0.0  # seconds


@dataclass
class ProcessUpdate:
    """Represents a single process update"""
    timestamp: datetime
    status: ProcessStatus
    message: str
    progress: float = 0.0  # 0-100
    details: Optional[str] = None
    scanning_stats: Optional[ScanningStats] = None


class ARTEMISStatusBar:
    """Main status bar for ARTEMIS system"""
    def __init__(self):
        if 'artemis_status' not in st.session_state:
            st.session_state.artemis_status = []
    
    def update(self, message: str, status: str = "info", progress: float = 0.0):
        """Update status bar with new message"""
        st.session_state.artemis_status.append({
            'timestamp': datetime.now(),
            'message': message,
            'status': status,
            'progress': progress
        })

# Alias for compatibility
PhoenixStatusBar = ARTEMISStatusBar


class ArtemisStatusBar:
    """
    ARTEMIS-themed status bar for displaying real-time process updates
    Uses Streamlit session state for persistence across reruns
    Supports scanning operations with detailed statistics
    """
    
    STATE_KEY = "artemis_status_bar"
    HISTORY_KEY = "artemis_status_history"
    SCANNING_STATS_KEY = "artemis_scanning_stats"
    
    # Status colors matching ARTEMIS theme
    STATUS_COLORS = {
        ProcessStatus.IDLE: "#999999",          # Gray
        ProcessStatus.PROCESSING: "#FFD700",    # Gold - active
        ProcessStatus.COMPLETED: "#4CAF50",     # Green
        ProcessStatus.ERROR: "#F44336",         # Red
        ProcessStatus.WARNING: "#FF9800",       # Orange
    }
    
    STATUS_ICONS = {
        ProcessStatus.IDLE: "⏸️",
        ProcessStatus.PROCESSING: "⚙️",
        ProcessStatus.COMPLETED: "✅",
        ProcessStatus.ERROR: "❌",
        ProcessStatus.WARNING: "⚠️",
    }
    
    @staticmethod
    def init():
        """Initialize status bar in session state"""
        if ArtemisStatusBar.STATE_KEY not in st.session_state:
            st.session_state[ArtemisStatusBar.STATE_KEY] = {
                "status": ProcessStatus.IDLE,
                "message": "Ready",
                "progress": 0.0,
                "details": None,
                "timestamp": datetime.now(),
            }
        if ArtemisStatusBar.HISTORY_KEY not in st.session_state:
            st.session_state[ArtemisStatusBar.HISTORY_KEY] = []
        if ArtemisStatusBar.SCANNING_STATS_KEY not in st.session_state:
            st.session_state[ArtemisStatusBar.SCANNING_STATS_KEY] = None
    
    @staticmethod
    def update(
        status: ProcessStatus,
        message: str,
        progress: float = 0.0,
        details: Optional[str] = None,
        scanning_stats: Optional[ScanningStats] = None
    ):
        """Update status bar with new state"""
        ArtemisStatusBar.init()
        
        state = st.session_state[ArtemisStatusBar.STATE_KEY]
        state["status"] = status
        state["message"] = message
        state["progress"] = max(0.0, min(100.0, progress))
        state["details"] = details
        state["timestamp"] = datetime.now()
        
        # Update scanning stats
        if scanning_stats:
            st.session_state[ArtemisStatusBar.SCANNING_STATS_KEY] = scanning_stats
        
        # Add to history
        update = ProcessUpdate(
            timestamp=state["timestamp"],
            status=status,
            message=message,
            progress=state["progress"],
            details=details,
            scanning_stats=scanning_stats
        )
        st.session_state[ArtemisStatusBar.HISTORY_KEY].append(update)
    
    @staticmethod
    def render():
        """Render the status bar with optional scanning statistics"""
        ArtemisStatusBar.init()
        
        state = st.session_state[ArtemisStatusBar.STATE_KEY]
        status = state["status"]
        message = state["message"]
        progress = state["progress"]
        details = state["details"]
        scanning_stats = st.session_state[ArtemisStatusBar.SCANNING_STATS_KEY]
        
        icon = ArtemisStatusBar.STATUS_ICONS.get(status, "•")
        color = ArtemisStatusBar.STATUS_COLORS.get(status, "#FFD700")
        
        # Build status bar with proper HTML
        with st.container(border=True):
            col1, col2, col3 = st.columns([1, 5, 1])
            
            with col1:
                st.markdown(f"<div style='font-size: 32px; text-align: center;'>{icon}</div>", unsafe_allow_html=True)
            
            with col2:
                st.markdown(f"<div style='color: {color}; font-weight: 600; font-size: 16px;'>{message}</div>", unsafe_allow_html=True)
                if details:
                    st.markdown(f"<div style='color: #999; font-size: 12px; margin-top: 4px;'>{details}</div>", unsafe_allow_html=True)
                
                # Progress bar
                if status == ProcessStatus.PROCESSING and progress > 0:
                    st.progress(progress / 100.0, text=f"{progress:.0f}%")
            
            with col3:
                st.markdown(f"<div style='color: #FFD700; font-size: 14px; font-weight: 600; text-align: right;'>{progress:.0f}%</div>", unsafe_allow_html=True)
        
        # Add scanning statistics if available
        if scanning_stats:
            with st.container():
                col1, col2, col3, col4, col5, col6 = st.columns(6)
                
                with col1:
                    st.metric("Symbols", scanning_stats.symbols_scanned)
                with col2:
                    st.metric("Pairs", scanning_stats.pairs_analyzed)
                with col3:
                    st.metric("Signals", scanning_stats.signals_found)
                with col4:
                    st.metric("Buy ↑", scanning_stats.buy_signals)
                with col5:
                    st.metric("Sell ↓", scanning_stats.sell_signals)
                with col6:
                    st.metric("Time", f"{scanning_stats.processing_time:.2f}s")
    
    @staticmethod
    def render_compact():
        """Render a compact version of the status bar"""
        ArtemisStatusBar.init()
        
        state = st.session_state[ArtemisStatusBar.STATE_KEY]
        status = state["status"]
        message = state["message"]
        
        icon = ArtemisStatusBar.STATUS_ICONS.get(status, "•")
        color = ArtemisStatusBar.STATUS_COLORS.get(status, "#FFD700")
        
        # Compact version (single line)
        st.markdown(
            f'<span style="color: {color}; font-weight: 500;">{icon} {message}</span>',
            unsafe_allow_html=True
        )
    
    @staticmethod
    def get_history() -> List[ProcessUpdate]:
        """Get update history"""
        ArtemisStatusBar.init()
        return st.session_state[ArtemisStatusBar.HISTORY_KEY]
    
    @staticmethod
    def clear_history():
        """Clear update history"""
        if ArtemisStatusBar.HISTORY_KEY in st.session_state:
            st.session_state[ArtemisStatusBar.HISTORY_KEY] = []
    
    @staticmethod
    def reset():
        """Reset status bar to idle state"""
        ArtemisStatusBar.update(
            ProcessStatus.IDLE,
            "Ready",
            progress=0.0,
            details=None
        )
    
    @staticmethod
    def show_error(message: str, details: Optional[str] = None):
        """Show error state"""
        ArtemisStatusBar.update(
            ProcessStatus.ERROR,
            message,
            progress=0.0,
            details=details
        )
    
    @staticmethod
    def show_warning(message: str, details: Optional[str] = None):
        """Show warning state"""
        ArtemisStatusBar.update(
            ProcessStatus.WARNING,
            message,
            progress=0.0,
            details=details
        )
    
    @staticmethod
    def show_processing(message: str, progress: float = 0.0, details: Optional[str] = None):
        """Show processing state"""
        ArtemisStatusBar.update(
            ProcessStatus.PROCESSING,
            message,
            progress=progress,
            details=details
        )
    
    @staticmethod
    def show_success(message: str = "Process completed successfully", details: Optional[str] = None, scanning_stats: Optional[ScanningStats] = None):
        """Show success state"""
        ArtemisStatusBar.update(
            ProcessStatus.COMPLETED,
            message,
            progress=100.0,
            details=details,
            scanning_stats=scanning_stats
        )
    
    @staticmethod
    def show_scanning(
        message: str,
        progress: float = 0.0,
        details: Optional[str] = None,
        symbols_scanned: int = 0,
        pairs_analyzed: int = 0,
        signals_found: int = 0,
        buy_signals: int = 0,
        sell_signals: int = 0,
        processing_time: float = 0.0
    ):
        """Show scanning state with statistics"""
        scanning_stats = ScanningStats(
            symbols_scanned=symbols_scanned,
            pairs_analyzed=pairs_analyzed,
            signals_found=signals_found,
            buy_signals=buy_signals,
            sell_signals=sell_signals,
            processing_time=processing_time
        )
        ArtemisStatusBar.update(
            ProcessStatus.PROCESSING,
            message,
            progress=progress,
            details=details,
            scanning_stats=scanning_stats
        )


    @staticmethod
    def update_status(progress: float, message: str, is_error: bool = False, details: Optional[str] = None):
        """Compatibility method for update_status calls"""
        status = ProcessStatus.ERROR if is_error else ProcessStatus.PROCESSING
        if progress >= 100 and not is_error:
            status = ProcessStatus.COMPLETED
            
        ArtemisStatusBar.update(
            status,
            message,
            progress=progress,
            details=details
        )

def create_status_context(process_name: str):
    """
    Context manager for status tracking during a process
    
    Usage:
        with create_status_context("Analysing pairs"):
            # Your code here
            ArtemisStatusBar.show_processing("Analysing pairs", progress=50, details="Processing pair 5/10")
    """
    
    class StatusContext:
        def __enter__(self):
            ArtemisStatusBar.show_processing(process_name, progress=0)
            return self
        
        def __exit__(self, exc_type, exc_val, exc_tb):
            if exc_type:
                ArtemisStatusBar.show_error(
                    f"{process_name} failed",
                    details=str(exc_val) if exc_val else None
                )
            else:
                ArtemisStatusBar.show_success(f"{process_name} completed")
    
    return StatusContext()
