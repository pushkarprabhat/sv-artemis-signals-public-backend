# core/security_classifier.py
"""
Two-Tier Security Classification System
Tier 1: Market Segment (EQUITY, FO, CURRENCY, COMMODITY, DEBT, INDICES)
Tier 2: Instrument Type (STOCK, ETF, INDEX, FUTURE, OPTION, etc.)

Determines download strategy based on market segment and instrument type.
Handles special cases like 705MP32-SG (no daily data available).
"""

from enum import Enum
from typing import Tuple, Dict, Optional
from dataclasses import dataclass
import logging
from pathlib import Path
import pandas as pd

logger = logging.getLogger(__name__)


class MarketSegment(Enum):
    """Tier 1: Market Segment Classification"""
    EQUITY = "equity"              # Cash market equities
    FO = "fo"                      # Futures & Options
    CURRENCY = "currency"          # Forex
    COMMODITY = "commodity"        # Commodity derivatives
    DEBT = "debt"                  # Bonds, Government Securities
    INDICES = "indices"            # Market benchmarks
    UNKNOWN = "unknown"            # Unable to classify


class InstrumentType(Enum):
    """Tier 2: Instrument Type Classification within each segment"""
    
    # EQUITY Segment
    STOCK = "stock"                # Regular equity shares (YES - download all intervals)
    ETF = "etf"                    # Exchange-traded funds (YES - if liquid)
    INDEX = "index"                # Equity indices (CONDITIONAL - for correlation)
    UNIT = "unit"                  # Mutual fund units (NO - skip)
    
    # FO Segment
    FUTURE = "future"              # Index/Stock/Commodity futures (YES - all intervals)
    OPTION = "option"              # Index/Stock options (YES - all intervals)
    RIGHTS = "rights"              # Rights issues (CONDITIONAL)
    
    # CURRENCY Segment
    CURRENCY_PAIR = "currency_pair"  # USD-INR, EUR-INR (YES - if liquid)
    
    # COMMODITY Segment
    COMMODITY_FUTURE = "commodity_future"  # Gold, Silver, Crude Oil (YES)
    
    # DEBT Segment
    BOND = "bond"                  # Government securities, Corporate bonds (CONDITIONAL)
    
    # Special Cases
    COMPOSITE_INDEX = "composite_index"  # VIVIDM-like composites (NO - skip)
    ILLIQUID = "illiquid"          # Low volume, high spread (NO - skip)
    DELISTED = "delisted"          # Removed from universe (NO - archive only)
    UNKNOWN = "unknown"            # Unable to determine


@dataclass
class SecurityMetadata:
    """Metadata for classified security"""
    symbol: str
    market_segment: MarketSegment
    instrument_type: InstrumentType
    confidence_level: float = 0.0  # 0.0 to 1.0
    has_daily_data: bool = False
    has_intraday_data: bool = False
    liquidity_status: str = "unknown"  # "high", "medium", "low"
    reason: str = ""  # Why classified this way
    recommendations: Dict = None  # Download strategy
    

    def __post_init__(self):
        if self.recommendations is None:
            self.recommendations = {}


class SecurityClassifier:
    """
    Two-tier security classifier:
    1. Determine Market Segment (EQUITY, FO, CURRENCY, COMMODITY, DEBT, INDICES)
    2. Determine Instrument Type (STOCK, ETF, INDEX, FUTURE, OPTION, etc.)
    3. Return handling strategy based on classification
    """
    
    def __init__(self, data_dir: Path = None, kite=None):
        self.data_dir = data_dir or Path(__file__).parent.parent / "marketdata" / "NSE"
        self.kite = kite
        self.known_indices = {
            "NIFTY50", "NIFTY100", "NIFTY200", "NIFTY500",
            "SENSEX", "BANKEX", "NIFTY_BANK", "NIFTY_IT",
            "NIFTY_PHARMA", "NIFTY_AUTO", "NIFTY_FMCG",
            "NIFTYNXT50", "MIDCAP50", "SMALLCAP50",
            "VIVIDM", "VIVID_INDEX"  # Add VIVID-like indices here
        }
        
    def classify(self, symbol: str, exchange: str = "NSE", 
                 instrument_data: Dict = None) -> SecurityMetadata:
        """
        Classify security using two-tier system
        
        Args:
            symbol: Security symbol
            exchange: NSE or BSE
            instrument_data: Data from Zerodha instruments API (optional)
        
        Returns:
            SecurityMetadata with segment, type, and strategy
        """
        logger.debug(f"[CLASSIFY] Starting classification for {symbol}")
        
        # Try to classify using instrument_data first (most reliable)
        if instrument_data:
            return self._classify_from_instrument_data(symbol, exchange, instrument_data)
        
        # Fallback: Use heuristics
        return self._classify_from_heuristics(symbol, exchange)
    
    def _classify_from_instrument_data(self, symbol: str, exchange: str, 
                                       data: Dict) -> SecurityMetadata:
        """Classify using Zerodha instrument data"""
        
        # Extract market segment from Zerodha
        segment = data.get("segment", "NSE").upper()
        instrument_type_raw = data.get("instrument_type", "").upper()
        expiry = data.get("expiry", None)
        
        # Determine Tier 1: Market Segment
        if segment == "NSE":
            market_segment = self._get_market_segment_nse(instrument_type_raw, symbol)
        elif segment == "BSE":
            market_segment = self._get_market_segment_bse(instrument_type_raw, symbol)
        elif segment == "NFO":  # NSE F&O
            market_segment = MarketSegment.FO
        elif segment == "MCX" or segment == "NCDEX":
            market_segment = MarketSegment.COMMODITY
        else:
            market_segment = MarketSegment.UNKNOWN
        
        # Determine Tier 2: Instrument Type
        instrument_type = self._get_instrument_type(
            market_segment, instrument_type_raw, symbol, expiry
        )
        
        # Check data availability
        has_daily = self._check_daily_data(symbol)
        has_intraday = self._check_intraday_data(symbol) if has_daily else False
        
        # Get liquidity status
        liquidity = self._estimate_liquidity(data)
        
        # Build metadata
        metadata = SecurityMetadata(
            symbol=symbol,
            market_segment=market_segment,
            instrument_type=instrument_type,
            confidence_level=0.95,  # High confidence when using instrument data
            has_daily_data=has_daily,
            has_intraday_data=has_intraday,
            liquidity_status=liquidity,
            reason=f"Classified from Zerodha data: {instrument_type_raw}"
        )
        
        # Get download strategy
        metadata.recommendations = self._get_download_strategy(metadata)
        
        logger.debug(f"[CLASSIFY] {symbol}: {market_segment.value}/{instrument_type.value}")
        return metadata
    
    def _classify_from_heuristics(self, symbol: str, exchange: str) -> SecurityMetadata:
        """Classify using symbol patterns and heuristics"""
        
        # Check if it's a known index
        if symbol.upper() in self.known_indices:
            metadata = SecurityMetadata(
                symbol=symbol,
                market_segment=MarketSegment.INDICES,
                instrument_type=InstrumentType.INDEX,
                confidence_level=0.90,
                reason="Known index symbol"
            )
            metadata.recommendations = self._get_download_strategy(metadata)
            return metadata
        
        # Check for derivative-like patterns
        if "-" in symbol:  # 705MP32-SG pattern
            metadata = SecurityMetadata(
                symbol=symbol,
                market_segment=MarketSegment.FO,
                instrument_type=InstrumentType.UNKNOWN,
                confidence_level=0.50,
                has_daily_data=False,  # These often don't have daily
                reason="Derivative-like symbol pattern (contains hyphen)"
            )
            metadata.recommendations = self._get_download_strategy(metadata)
            return metadata
        
        # Check if has daily data
        has_daily = self._check_daily_data(symbol)
        has_intraday = self._check_intraday_data(symbol) if has_daily else False
        
        # Default to STOCK if in NSE and has data
        if has_daily:
            instrument_type = InstrumentType.STOCK
            market_segment = MarketSegment.EQUITY
            confidence = 0.70
        else:
            instrument_type = InstrumentType.UNKNOWN
            market_segment = MarketSegment.UNKNOWN
            confidence = 0.30
        
        metadata = SecurityMetadata(
            symbol=symbol,
            market_segment=market_segment,
            instrument_type=instrument_type,
            confidence_level=confidence,
            has_daily_data=has_daily,
            has_intraday_data=has_intraday,
            reason="Classified from heuristics (no instrument data available)"
        )
        
        metadata.recommendations = self._get_download_strategy(metadata)
        return metadata
    
    def _get_market_segment_nse(self, instrument_type: str, symbol: str) -> MarketSegment:
        """Determine market segment for NSE securities"""
        if symbol.upper() in self.known_indices:
            return MarketSegment.INDICES
        elif instrument_type in ["STOCK", "ETF"]:
            return MarketSegment.EQUITY
        else:
            return MarketSegment.EQUITY  # Default for NSE cash market
    
    def _get_market_segment_bse(self, instrument_type: str, symbol: str) -> MarketSegment:
        """Determine market segment for BSE securities"""
        if instrument_type in ["STOCK", "ETF"]:
            return MarketSegment.EQUITY
        else:
            return MarketSegment.EQUITY
    
    def _get_instrument_type(self, segment: MarketSegment, type_raw: str,
                            symbol: str, expiry: Optional[str]) -> InstrumentType:
        """Determine instrument type based on segment and raw type"""
        
        type_upper = type_raw.upper()
        
        # FO Segment
        if segment == MarketSegment.FO:
            if "FUT" in type_upper or "FUTURE" in type_upper:
                return InstrumentType.FUTURE
            elif "OPT" in type_upper or "OPTION" in type_upper:
                return InstrumentType.OPTION
            else:
                return InstrumentType.UNKNOWN
        
        # EQUITY Segment
        elif segment == MarketSegment.EQUITY:
            if "ETF" in type_upper:
                return InstrumentType.ETF
            elif symbol.upper() in self.known_indices:
                return InstrumentType.INDEX
            else:
                return InstrumentType.STOCK
        
        # CURRENCY Segment
        elif segment == MarketSegment.CURRENCY:
            return InstrumentType.CURRENCY_PAIR
        
        # COMMODITY Segment
        elif segment == MarketSegment.COMMODITY:
            if "FUT" in type_upper:
                return InstrumentType.COMMODITY_FUTURE
            else:
                return InstrumentType.UNKNOWN
        
        # INDICES Segment
        elif segment == MarketSegment.INDICES:
            if "COMPOSITE" in type_upper:
                return InstrumentType.COMPOSITE_INDEX
            else:
                return InstrumentType.INDEX
        
        # DEBT Segment
        elif segment == MarketSegment.DEBT:
            return InstrumentType.BOND
        
        else:
            return InstrumentType.UNKNOWN
    
    def _check_daily_data(self, symbol: str) -> bool:
        """Check if daily data file exists"""
        daily_file = self.data_dir / "day" / f"{symbol}.parquet"
        return daily_file.exists()
    
    def _check_intraday_data(self, symbol: str) -> bool:
        """Check if intraday (5min or 3min) data exists"""
        five_min = self.data_dir / "5minute" / f"{symbol}.parquet"
        three_min = self.data_dir / "3minute" / f"{symbol}.parquet"
        return five_min.exists() or three_min.exists()
    
    def _estimate_liquidity(self, instrument_data: Dict) -> str:
        """Estimate liquidity from instrument data"""
        # This could use volume, bid-ask spread, etc.
        # For now, simple heuristic
        volume = instrument_data.get("volume", 0)
        
        if volume > 10_000_000:  # > 10M shares
            return "high"
        elif volume > 1_000_000:  # > 1M shares
            return "medium"
        else:
            return "low"
    
    def _get_download_strategy(self, metadata: SecurityMetadata) -> Dict:
        """
        Determine download strategy based on classification
        
        Decision matrix:
        ┌─────────────────────────┬──────────────────┬─────────────────┐
        │ Segment/Type            │ Has Daily Data   │ Action          │
        ├─────────────────────────┼──────────────────┼─────────────────┤
        │ EQUITY/STOCK (liquid)   │ YES              │ ALL intervals   │
        │ EQUITY/STOCK (illiquid) │ YES              │ Daily only      │
        │ EQUITY/ETF              │ YES              │ ALL intervals   │
        │ EQUITY/INDEX            │ YES              │ Daily + Corr.   │
        │ FO/FUTURE               │ YES              │ ALL intervals   │
        │ FO/OPTION               │ YES              │ ALL intervals   │
        │ CURRENCY/PAIR           │ YES              │ ALL intervals   │
        │ Any/Any                 │ NO               │ SKIP (archive)  │
        │ Any/COMPOSITE_INDEX     │ ANY              │ SKIP            │
        │ Any/DELISTED            │ ANY              │ Archive only    │
        └─────────────────────────┴──────────────────┴─────────────────┘
        """
        
        strategy = {
            "should_download": False,
            "intervals": [],
            "reason": "",
            "priority": 0  # Lower = lower priority
        }
        
        # Case 1: Check known problematic types FIRST (before checking data)
        if metadata.instrument_type == InstrumentType.COMPOSITE_INDEX:
            strategy["reason"] = "Composite index - skip individual download"
            strategy["should_download"] = False
            strategy["priority"] = 0
            return strategy
        
        if metadata.instrument_type == InstrumentType.DELISTED:
            strategy["reason"] = "Delisted security - archive only"
            strategy["should_download"] = False
            strategy["priority"] = 0
            return strategy
        
        if metadata.instrument_type == InstrumentType.ILLIQUID:
            strategy["reason"] = "Illiquid - insufficient volume"
            strategy["should_download"] = False
            strategy["priority"] = 0
            return strategy
        
        # Case 2: No daily data available
        if not metadata.has_daily_data:
            strategy["reason"] = "No daily data foundation (check if security has volume)"
            strategy["should_download"] = False
            strategy["priority"] = 0
            
            # Special handling for derivatives
            if metadata.market_segment == MarketSegment.FO:
                strategy["reason"] += " | FO contract may be expired/non-trading"
            
            return strategy
        
        # Case 3: Liquid tradeable securities (with daily data)
        if metadata.market_segment == MarketSegment.EQUITY:
            if metadata.instrument_type in [InstrumentType.STOCK, InstrumentType.ETF]:
                if metadata.liquidity_status == "high":
                    strategy["should_download"] = True
                    strategy["intervals"] = ["day", "5minute", "3minute"]
                    strategy["reason"] = "Equity stock/ETF with good liquidity"
                    strategy["priority"] = 10
                elif metadata.liquidity_status == "medium":
                    strategy["should_download"] = True
                    strategy["intervals"] = ["day", "5minute"]
                    strategy["reason"] = "Equity stock/ETF with medium liquidity"
                    strategy["priority"] = 7
                else:
                    strategy["should_download"] = True
                    strategy["intervals"] = ["day"]
                    strategy["reason"] = "Equity stock/ETF - download daily only"
                    strategy["priority"] = 4
            
            elif metadata.instrument_type == InstrumentType.INDEX:
                strategy["should_download"] = True
                strategy["intervals"] = ["day"]
                strategy["reason"] = "Index - download daily for correlation analysis"
                strategy["priority"] = 3
        
        # Case 3b: Indices segment (NIFTY50, SENSEX, etc.)
        elif metadata.market_segment == MarketSegment.INDICES:
            if metadata.instrument_type == InstrumentType.INDEX:
                strategy["should_download"] = True
                strategy["intervals"] = ["day"]
                strategy["reason"] = "Market index - download daily for correlation"
                strategy["priority"] = 2  # Lower priority than equity indices
        
        # Case 4: Derivatives
        elif metadata.market_segment == MarketSegment.FO:
            if metadata.instrument_type in [InstrumentType.FUTURE, InstrumentType.OPTION]:
                strategy["should_download"] = True
                strategy["intervals"] = ["day", "5minute"]  # FO usually has 5min, limited 3min
                strategy["reason"] = "Derivative - download day + 5min"
                strategy["priority"] = 8
        
        # Case 5: Currency
        elif metadata.market_segment == MarketSegment.CURRENCY:
            strategy["should_download"] = True
            strategy["intervals"] = ["day", "5minute"]
            strategy["reason"] = "Currency pair - download day + 5min"
            strategy["priority"] = 6
        
        # Case 6: Commodity
        elif metadata.market_segment == MarketSegment.COMMODITY:
            strategy["should_download"] = True
            strategy["intervals"] = ["day"]
            strategy["reason"] = "Commodity - download daily"
            strategy["priority"] = 5
        
        else:
            strategy["should_download"] = False
            strategy["reason"] = f"Unknown segment/type combination"
            strategy["priority"] = 0
        
        return strategy


# Export for use in other modules
def get_security_classifier(data_dir: Path = None, kite=None) -> SecurityClassifier:
    """Factory function to create classifier instance"""
    return SecurityClassifier(data_dir=data_dir, kite=kite)
