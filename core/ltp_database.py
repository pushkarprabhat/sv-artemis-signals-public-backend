"""
LTP Database — Persistent storage for Last Traded Prices
Professional price tracking: Maintain accurate price history across sessions
"""

import sqlite3
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import pandas as pd
from threading import Lock

logger = logging.getLogger(__name__)

# Database path
DB_PATH = Path(__file__).parent.parent / "data" / "ltp_database.db"
DB_PATH.parent.mkdir(parents=True, exist_ok=True)

# Thread-safe database access
_db_lock = Lock()


class LTPDatabase:
    """
    Persistent storage for Last Traded Prices with historical tracking.
    
    Features:
    - Store current LTP for all instruments
    - Track LTP history for analysis
    - Fast retrieval by symbol
    - Automatic cleanup of old data
    - Thread-safe operations
    """
    
    def __init__(self, db_path: str = None):
        """Initialize LTP database."""
        self.db_path = db_path or str(DB_PATH)
        self._initialize_database()
        logger.info(f"✅ LTP Database initialized: {self.db_path}")
    
    def _initialize_database(self):
        """Create database tables if they don't exist."""
        with _db_lock:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Current LTP table (one row per symbol)
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS current_ltp (
                    symbol TEXT PRIMARY KEY,
                    instrument_token INTEGER,
                    last_price REAL NOT NULL,
                    volume INTEGER DEFAULT 0,
                    change_percent REAL DEFAULT 0.0,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                    exchange TEXT DEFAULT 'NSE'
                )
            """)
            
            # Historical LTP table (time series)
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS ltp_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol TEXT NOT NULL,
                    instrument_token INTEGER,
                    last_price REAL NOT NULL,
                    volume INTEGER DEFAULT 0,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                    exchange TEXT DEFAULT 'NSE'
                )
            """)
            
            # Create indexes for faster queries
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_symbol 
                ON current_ltp(symbol)
            """)
            
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_history_symbol_time 
                ON ltp_history(symbol, timestamp DESC)
            """)
            
            conn.commit()
            conn.close()
    
    def update_ltp(self, symbol: str, last_price: float, 
                   instrument_token: int = None, volume: int = 0,
                   change_percent: float = 0.0, exchange: str = 'NSE'):
        """
        Update LTP for a symbol (upsert into current_ltp, insert into history).
        
        Args:
            symbol: Symbol name (e.g., 'NIFTY', 'RELIANCE')
            last_price: Current last traded price
            instrument_token: Instrument token from Zerodha
            volume: Trading volume
            change_percent: % change from previous close
            exchange: Exchange name (NSE/BSE/MCX)
        """
        try:
            with _db_lock:
                conn = sqlite3.connect(self.db_path)
                cursor = conn.cursor()
                
                # Update current LTP (upsert)
                cursor.execute("""
                    INSERT INTO current_ltp 
                    (symbol, instrument_token, last_price, volume, change_percent, 
                     timestamp, exchange)
                    VALUES (?, ?, ?, ?, ?, CURRENT_TIMESTAMP, ?)
                    ON CONFLICT(symbol) DO UPDATE SET
                        last_price = excluded.last_price,
                        volume = excluded.volume,
                        change_percent = excluded.change_percent,
                        timestamp = CURRENT_TIMESTAMP
                """, (symbol, instrument_token, last_price, volume, 
                      change_percent, exchange))
                
                # Insert into history
                cursor.execute("""
                    INSERT INTO ltp_history 
                    (symbol, instrument_token, last_price, volume, timestamp, exchange)
                    VALUES (?, ?, ?, ?, CURRENT_TIMESTAMP, ?)
                """, (symbol, instrument_token, last_price, volume, exchange))
                
                conn.commit()
                conn.close()
                
        except Exception as e:
            logger.error(f"Error updating LTP for {symbol}: {e}")
    
    def get_ltp(self, symbol: str) -> Optional[Dict]:
        """
        Get current LTP for a symbol.
        
        Args:
            symbol: Symbol name
            
        Returns:
            Dict with LTP data or None if not found
        """
        try:
            with _db_lock:
                conn = sqlite3.connect(self.db_path)
                cursor = conn.cursor()
                
                cursor.execute("""
                    SELECT symbol, instrument_token, last_price, volume, 
                           change_percent, timestamp, exchange
                    FROM current_ltp
                    WHERE symbol = ?
                """, (symbol,))
                
                row = cursor.fetchone()
                conn.close()
                
                if row:
                    return {
                        'symbol': row[0],
                        'instrument_token': row[1],
                        'last_price': row[2],
                        'volume': row[3],
                        'change_percent': row[4],
                        'timestamp': row[5],
                        'exchange': row[6]
                    }
                return None
                
        except Exception as e:
            logger.error(f"Error fetching LTP for {symbol}: {e}")
            return None
    
    def get_all_ltp(self) -> pd.DataFrame:
        """
        Get all current LTP data as DataFrame.
        
        Returns:
            DataFrame with all LTP data
        """
        try:
            with _db_lock:
                conn = sqlite3.connect(self.db_path)
                df = pd.read_sql_query("""
                    SELECT symbol, last_price, volume, change_percent, 
                           timestamp, exchange
                    FROM current_ltp
                    ORDER BY symbol
                """, conn)
                conn.close()
                return df
                
        except Exception as e:
            logger.error(f"Error fetching all LTP: {e}")
            return pd.DataFrame()
    
    def get_ltp_history(self, symbol: str, 
                        limit: int = 100) -> pd.DataFrame:
        """
        Get LTP history for a symbol.
        
        Args:
            symbol: Symbol name
            limit: Max number of records to return
            
        Returns:
            DataFrame with historical LTP data
        """
        try:
            with _db_lock:
                conn = sqlite3.connect(self.db_path)
                df = pd.read_sql_query("""
                    SELECT symbol, last_price, volume, timestamp, exchange
                    FROM ltp_history
                    WHERE symbol = ?
                    ORDER BY timestamp DESC
                    LIMIT ?
                """, conn, params=(symbol, limit))
                conn.close()
                return df
                
        except Exception as e:
            logger.error(f"Error fetching LTP history for {symbol}: {e}")
            return pd.DataFrame()
    
    def bulk_update_ltp(self, ltp_data: Dict[str, Dict]):
        """
        Bulk update LTP for multiple symbols.
        
        Args:
            ltp_data: Dict mapping symbol to LTP data dict
                     e.g., {'NIFTY': {'last_price': 21500, 'volume': 1000000, ...}}
        """
        try:
            with _db_lock:
                conn = sqlite3.connect(self.db_path)
                cursor = conn.cursor()
                
                for symbol, data in ltp_data.items():
                    last_price = data.get('last_price', 0.0)
                    instrument_token = data.get('instrument_token')
                    volume = data.get('volume', 0)
                    change_percent = data.get('change_percent', 0.0)
                    exchange = data.get('exchange', 'NSE')
                    
                    # Update current LTP
                    cursor.execute("""
                        INSERT INTO current_ltp 
                        (symbol, instrument_token, last_price, volume, 
                         change_percent, timestamp, exchange)
                        VALUES (?, ?, ?, ?, ?, CURRENT_TIMESTAMP, ?)
                        ON CONFLICT(symbol) DO UPDATE SET
                            last_price = excluded.last_price,
                            volume = excluded.volume,
                            change_percent = excluded.change_percent,
                            timestamp = CURRENT_TIMESTAMP
                    """, (symbol, instrument_token, last_price, volume,
                          change_percent, exchange))
                    
                    # Insert into history
                    cursor.execute("""
                        INSERT INTO ltp_history 
                        (symbol, instrument_token, last_price, volume, 
                         timestamp, exchange)
                        VALUES (?, ?, ?, ?, CURRENT_TIMESTAMP, ?)
                    """, (symbol, instrument_token, last_price, volume, exchange))
                
                conn.commit()
                conn.close()
                logger.info(f"✅ Bulk updated LTP for {len(ltp_data)} symbols")
                
        except Exception as e:
            logger.error(f"Error in bulk LTP update: {e}")
    
    def cleanup_old_history(self, days_to_keep: int = 30):
        """
        Remove LTP history older than specified days.
        
        Args:
            days_to_keep: Number of days to retain
        """
        try:
            with _db_lock:
                conn = sqlite3.connect(self.db_path)
                cursor = conn.cursor()
                
                cursor.execute("""
                    DELETE FROM ltp_history
                    WHERE timestamp < datetime('now', '-' || ? || ' days')
                """, (days_to_keep,))
                
                deleted = cursor.rowcount
                conn.commit()
                conn.close()
                
                if deleted > 0:
                    logger.info(f"🗑️ Cleaned up {deleted} old LTP records")
                    
        except Exception as e:
            logger.error(f"Error cleaning up old LTP history: {e}")
    
    def get_summary_stats(self) -> Dict:
        """
        Get summary statistics about LTP database.
        
        Returns:
            Dict with stats (total symbols, last update time, etc.)
        """
        try:
            with _db_lock:
                conn = sqlite3.connect(self.db_path)
                cursor = conn.cursor()
                
                # Total symbols
                cursor.execute("SELECT COUNT(*) FROM current_ltp")
                total_symbols = cursor.fetchone()[0]
                
                # Last update time
                cursor.execute("""
                    SELECT MAX(timestamp) FROM current_ltp
                """)
                last_update = cursor.fetchone()[0]
                
                # Total history records
                cursor.execute("SELECT COUNT(*) FROM ltp_history")
                total_history = cursor.fetchone()[0]
                
                conn.close()
                
                return {
                    'total_symbols': total_symbols,
                    'last_update': last_update,
                    'total_history_records': total_history
                }
                
        except Exception as e:
            logger.error(f"Error fetching summary stats: {e}")
            return {}


# Global instance
_ltp_db = None

def get_ltp_database() -> LTPDatabase:
    """Get global LTP database instance (singleton)."""
    global _ltp_db
    if _ltp_db is None:
        _ltp_db = LTPDatabase()
    return _ltp_db
