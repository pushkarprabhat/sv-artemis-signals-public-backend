"""
NIFTY-BANKNIFTY Pair Trading Strategy

Specialized strategy for trading the NIFTY50-BANKNIFTY index pair.
This strategy exploits the correlation between these two major indices
through multiple hedging and spread trading approaches.

Key Concepts:
- NIFTY50: Broad market index (50 stocks across all sectors)
- BANKNIFTY: Financial sector index (12 major banks)
- Correlation: Typically 0.70-0.85 (strong positive)
- Hedging: Long NIFTY, short BANKNIFTY for beta-hedged portfolio

Trading Approaches:
1. Statistical Arbitrage: Trade mean-reversion in spread
2. Index Pair Hedging: Long index, short sector exposure
3. Constituent Hedging: Long index, short largest constituents
4. Correlation-based: Trade when correlation deviates from norm
5. Volatility Spread: Long high-IV index, short low-IV index
"""

import pandas as pd
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional
from enum import Enum
from datetime import datetime
import json


class PairTradingStrategy(Enum):
    """Available NIFTY-BANKNIFTY strategies"""
    STAT_ARB = "statistical_arbitrage"           # Mean-reversion trading on spread
    INDEX_HEDGE = "index_hedging"               # Long broad, short sector
    CONSTITUENT_HEDGE = "constituent_hedging"  # Long index, short constituent
    CORRELATION = "correlation_deviation"      # Trade correlation breakdowns
    VOLATILITY = "volatility_spread"            # Long IV, short IV
    RATIO_TRADE = "ratio_trading"              # Trade value ratios


@dataclass
class PairSpreadMetrics:
    """Metrics for the NIFTY-BANKNIFTY spread"""
    nifty_price: float
    banknifty_price: float
    spread: float                    # NIFTY - BANKNIFTY (or ratio)
    spread_zscore: float            # Z-score of spread (std dev from mean)
    correlation: float              # Current correlation
    correlation_deviation: float    # vs 30-day MA
    nifty_volatility: float
    banknifty_volatility: float
    volatility_ratio: float         # BANKNIFTY_vol / NIFTY_vol
    liquidity_spread_nifty: float   # Bid-ask spread %
    liquidity_spread_banknifty: float
    timestamp: datetime


@dataclass
class PairTradingSignal:
    """Trading signal for NIFTY-BANKNIFTY pair"""
    strategy: PairTradingStrategy
    action: str                     # BUY, SELL, CLOSE, HOLD
    long_instrument: str            # NIFTY50 or constituent
    short_instrument: str           # BANKNIFTY or constituent
    long_quantity: int
    short_quantity: int
    entry_price_long: float
    entry_price_short: float
    stop_loss: float
    take_profit: float
    risk_reward_ratio: float
    confidence: float               # 0-100
    reason: str
    metrics: PairSpreadMetrics
    timestamp: datetime


@dataclass
class PortfolioConstituent:
    """Constituent details with weight"""
    symbol: str
    sector: str
    weight: float          # Portfolio weight %
    market_cap: float      # In billions
    liquidity_tier: str    # Tier 1-4


class NiftyBankniftyStrategy:
    """
    NIFTY-BANKNIFTY pair trading implementation
    
    This strategy exploits:
    - Spread mean-reversion
    - Index-sector hedging
    - Correlation dynamics
    - Volatility ratios
    - Constituent hedging
    """
    
    def __init__(self, index_weights_file: str = "data/index_weights.json"):
        """
        Initialize strategy with index weights
        
        Args:
            index_weights_file: Path to JSON with index constituent weights
        """
        self.index_weights = self._load_index_weights(index_weights_file)
        self.nifty_constituents = self.index_weights.get('NIFTY50', {}).get('constituents', [])
        self.banknifty_constituents = self.index_weights.get('BANKNIFTY', {}).get('constituents', [])
        
        # Strategy parameters
        self.stat_arb_zscore_threshold = 2.0  # Entry when spread is 2 SD from mean
        self.correlation_deviation_threshold = 0.05
        self.volatility_ratio_threshold = 1.2
        self.hedge_ratio = 0.8  # Beta-adjusted hedge ratio
    
    def _load_index_weights(self, filepath: str) -> Dict:
        """Load index weights from JSON file"""
        try:
            with open(filepath, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            print(f"Warning: {filepath} not found")
            return {}
    
    # ========== STATISTICAL ARBITRAGE ==========
    
    def stat_arb_signal(self,
                       nifty_prices: pd.Series,
                       banknifty_prices: pd.Series,
                       spread_window: int = 20) -> Optional[PairTradingSignal]:
        """
        Statistical Arbitrage: Trade mean-reversion in spread
        
        Approach:
        1. Calculate spread ratio: NIFTY / BANKNIFTY
        2. Compute Z-score over rolling window
        3. Enter when Z-score > 2.0 (spread too wide)
        4. Exit when Z-score returns to mean
        
        Args:
            nifty_prices: NIFTY50 price series
            banknifty_prices: BANKNIFTY price series
            spread_window: Rolling window for mean/std calculation
        
        Returns:
            PairTradingSignal if condition met, else None
        """
        # Calculate spread as ratio
        spread = nifty_prices / banknifty_prices
        spread_mean = spread.rolling(window=spread_window).mean()
        spread_std = spread.rolling(window=spread_window).std()
        
        # Z-score
        zscore = (spread - spread_mean) / spread_std
        
        last_spread = spread.iloc[-1]
        last_zscore = zscore.iloc[-1]
        last_nifty = nifty_prices.iloc[-1]
        last_banknifty = banknifty_prices.iloc[-1]
        
        # Generate signal
        if last_zscore > self.stat_arb_zscore_threshold:
            # Spread too wide: Long BANKNIFTY, Short NIFTY
            action = "SELL"
            long_instrument = "BANKNIFTY"
            short_instrument = "NIFTY50"
            reason = f"Spread at {last_zscore:.2f} SD above mean (overbought)"
            
        elif last_zscore < -self.stat_arb_zscore_threshold:
            # Spread too narrow: Short BANKNIFTY, Long NIFTY
            action = "BUY"
            long_instrument = "NIFTY50"
            short_instrument = "BANKNIFTY"
            reason = f"Spread at {last_zscore:.2f} SD below mean (oversold)"
            
        else:
            return None
        
        # Calculate quantities (value-matched)
        nifty_qty = 1
        banknifty_qty = int(last_nifty / last_banknifty)
        
        # Calculate stops and targets
        stop_loss = abs(last_zscore) * 0.5  # Exit at 0.5 SD
        take_profit = 0.05  # Profit at 5% spread reversion
        
        metrics = PairSpreadMetrics(
            nifty_price=last_nifty,
            banknifty_price=last_banknifty,
            spread=last_spread,
            spread_zscore=last_zscore,
            correlation=nifty_prices.corr(banknifty_prices),
            correlation_deviation=0.0,
            nifty_volatility=nifty_prices.pct_change().std() * np.sqrt(252),
            banknifty_volatility=banknifty_prices.pct_change().std() * np.sqrt(252),
            volatility_ratio=banknifty_prices.pct_change().std() / nifty_prices.pct_change().std(),
            liquidity_spread_nifty=0.01,
            liquidity_spread_banknifty=0.01,
            timestamp=nifty_prices.index[-1]
        )
        
        return PairTradingSignal(
            strategy=PairTradingStrategy.STAT_ARB,
            action=action,
            long_instrument=long_instrument,
            short_instrument=short_instrument,
            long_quantity=nifty_qty if long_instrument == "NIFTY50" else banknifty_qty,
            short_quantity=nifty_qty if short_instrument == "NIFTY50" else banknifty_qty,
            entry_price_long=last_nifty if long_instrument == "NIFTY50" else last_banknifty,
            entry_price_short=last_banknifty if short_instrument == "BANKNIFTY" else last_nifty,
            stop_loss=stop_loss,
            take_profit=take_profit,
            risk_reward_ratio=take_profit / stop_loss,
            confidence=min(100, abs(last_zscore) * 30),
            reason=reason,
            metrics=metrics,
            timestamp=datetime.now()
        )
    
    # ========== INDEX HEDGING ==========
    
    def index_hedge_signal(self,
                          nifty_prices: pd.Series,
                          banknifty_prices: pd.Series,
                          banknifty_beta: float = 1.2) -> Optional[PairTradingSignal]:
        """
        Index Hedging: Long broad market, short sector exposure
        
        Approach:
        1. Long NIFTY50 for broad market exposure
        2. Short BANKNIFTY for financial sector hedging
        3. Use beta-adjusted quantities for delta-neutral portfolio
        
        Args:
            nifty_prices: NIFTY50 price series
            banknifty_prices: BANKNIFTY price series
            banknifty_beta: BANKNIFTY beta vs NIFTY50 (typically 1.1-1.3)
        
        Returns:
            Hedging signal
        """
        last_nifty = nifty_prices.iloc[-1]
        last_banknifty = banknifty_prices.iloc[-1]
        
        # Value-matched quantities with beta adjustment
        nifty_qty = 1
        banknifty_qty = int((last_nifty / last_banknifty) * banknifty_beta)
        
        # Calculate performance
        nifty_return = nifty_prices.pct_change().sum()
        banknifty_return = banknifty_prices.pct_change().sum()
        
        # Trigger if sectors outperform significantly
        if banknifty_return > nifty_return * 1.1:
            reason = "Financial sector outperforming - hedge with short BANKNIFTY"
            confidence = 60
        else:
            reason = "Sector in line with broad market - maintain hedge"
            confidence = 50
        
        metrics = PairSpreadMetrics(
            nifty_price=last_nifty,
            banknifty_price=last_banknifty,
            spread=last_nifty - last_banknifty,
            spread_zscore=0.0,
            correlation=nifty_prices.corr(banknifty_prices),
            correlation_deviation=0.0,
            nifty_volatility=nifty_prices.pct_change().std() * np.sqrt(252),
            banknifty_volatility=banknifty_prices.pct_change().std() * np.sqrt(252),
            volatility_ratio=banknifty_prices.pct_change().std() / nifty_prices.pct_change().std(),
            liquidity_spread_nifty=0.01,
            liquidity_spread_banknifty=0.01,
            timestamp=nifty_prices.index[-1]
        )
        
        return PairTradingSignal(
            strategy=PairTradingStrategy.INDEX_HEDGE,
            action="HOLD" if confidence == 50 else "BUY",
            long_instrument="NIFTY50",
            short_instrument="BANKNIFTY",
            long_quantity=nifty_qty,
            short_quantity=banknifty_qty,
            entry_price_long=last_nifty,
            entry_price_short=last_banknifty,
            stop_loss=0.02,  # 2% hedge stop-loss
            take_profit=0.05,
            risk_reward_ratio=2.5,
            confidence=confidence,
            reason=reason,
            metrics=metrics,
            timestamp=datetime.now()
        )
    
    # ========== CONSTITUENT HEDGING ==========
    
    def constituent_hedge_signal(self,
                                index_price: float,
                                constituent_price: float,
                                constituent_weight: float,
                                constituent_name: str) -> Optional[PairTradingSignal]:
        """
        Constituent Hedging: Long index, short largest constituent
        
        Approach:
        1. Long NIFTY50 (or index) for broad exposure
        2. Short largest weight constituent (e.g., RELIANCE at 11.2%)
        3. Hedges concentration risk
        
        Args:
            index_price: Index price (e.g., NIFTY50)
            constituent_price: Constituent stock price
            constituent_weight: Weight in index (e.g., 0.112 for 11.2%)
            constituent_name: Stock symbol (e.g., 'RELIANCE')
        
        Returns:
            Hedging signal
        """
        # Quantity for hedge
        index_qty = 1
        constituent_qty = int((index_price / constituent_price) * constituent_weight * 1.2)
        
        reason = f"Hedge concentration risk: Long NIFTY50, short {constituent_name} ({constituent_weight*100:.1f}% weight)"
        
        metrics = PairSpreadMetrics(
            nifty_price=index_price,
            banknifty_price=constituent_price,
            spread=index_price - constituent_price,
            spread_zscore=0.0,
            correlation=0.85,
            correlation_deviation=0.0,
            nifty_volatility=0.15,
            banknifty_volatility=0.18,
            volatility_ratio=1.2,
            liquidity_spread_nifty=0.01,
            liquidity_spread_banknifty=0.02,
            timestamp=datetime.now()
        )
        
        return PairTradingSignal(
            strategy=PairTradingStrategy.CONSTITUENT_HEDGE,
            action="BUY",
            long_instrument="NIFTY50",
            short_instrument=constituent_name,
            long_quantity=index_qty,
            short_quantity=constituent_qty,
            entry_price_long=index_price,
            entry_price_short=constituent_price,
            stop_loss=0.03,
            take_profit=0.04,
            risk_reward_ratio=1.33,
            confidence=70,
            reason=reason,
            metrics=metrics,
            timestamp=datetime.now()
        )
    
    # ========== CORRELATION-BASED TRADING ==========
    
    def correlation_deviation_signal(self,
                                    nifty_prices: pd.Series,
                                    banknifty_prices: pd.Series,
                                    correlation_window: int = 30) -> Optional[PairTradingSignal]:
        """
        Correlation Deviation: Trade breakdown in index relationship
        
        Approach:
        1. Calculate rolling correlation (typically 0.70-0.85)
        2. Identify deviation when correlation breaks down
        3. Trade mean-reversion in correlation
        
        Args:
            nifty_prices: NIFTY50 series
            banknifty_prices: BANKNIFTY series
            correlation_window: Window for correlation calculation
        
        Returns:
            Trading signal
        """
        correlation = nifty_prices.rolling(window=correlation_window).corr(banknifty_prices)
        correlation_ma = correlation.rolling(window=10).mean()
        
        last_corr = correlation.iloc[-1]
        corr_mean = correlation_ma.iloc[-1]
        corr_deviation = last_corr - corr_mean
        
        if corr_deviation < -self.correlation_deviation_threshold:
            # Correlation weaker than expected: They might converge
            # Trade in direction of weaker performer
            nifty_return = nifty_prices.pct_change().sum()
            banknifty_return = banknifty_prices.pct_change().sum()
            
            if nifty_return < banknifty_return:
                action = "BUY"
                reason = "Correlation broken: NIFTY underperforming - buy for convergence"
            else:
                action = "SELL"
                reason = "Correlation broken: BANKNIFTY underperforming - short for convergence"
            
            confidence = 65
        else:
            return None
        
        last_nifty = nifty_prices.iloc[-1]
        last_banknifty = banknifty_prices.iloc[-1]
        
        metrics = PairSpreadMetrics(
            nifty_price=last_nifty,
            banknifty_price=last_banknifty,
            spread=last_nifty - last_banknifty,
            spread_zscore=0.0,
            correlation=last_corr,
            correlation_deviation=corr_deviation,
            nifty_volatility=nifty_prices.pct_change().std() * np.sqrt(252),
            banknifty_volatility=banknifty_prices.pct_change().std() * np.sqrt(252),
            volatility_ratio=banknifty_prices.pct_change().std() / nifty_prices.pct_change().std(),
            liquidity_spread_nifty=0.01,
            liquidity_spread_banknifty=0.01,
            timestamp=nifty_prices.index[-1]
        )
        
        return PairTradingSignal(
            strategy=PairTradingStrategy.CORRELATION,
            action=action,
            long_instrument="NIFTY50" if action == "BUY" else "BANKNIFTY",
            short_instrument="BANKNIFTY" if action == "BUY" else "NIFTY50",
            long_quantity=1,
            short_quantity=1,
            entry_price_long=last_nifty if action == "BUY" else last_banknifty,
            entry_price_short=last_banknifty if action == "BUY" else last_nifty,
            stop_loss=0.02,
            take_profit=0.03,
            risk_reward_ratio=1.5,
            confidence=confidence,
            reason=reason,
            metrics=metrics,
            timestamp=datetime.now()
        )
    
    # ========== VOLATILITY SPREAD TRADING ==========
    
    def volatility_spread_signal(self,
                                nifty_prices: pd.Series,
                                banknifty_prices: pd.Series,
                                vol_window: int = 20) -> Optional[PairTradingSignal]:
        """
        Volatility Spread: Trade when vol ratio deviates
        
        Long high-IV index, short low-IV index
        
        Args:
            nifty_prices: NIFTY50 series
            banknifty_prices: BANKNIFTY series
            vol_window: Window for volatility calculation
        
        Returns:
            Trading signal
        """
        nifty_vol = nifty_prices.pct_change().rolling(window=vol_window).std() * np.sqrt(252)
        banknifty_vol = banknifty_prices.pct_change().rolling(window=vol_window).std() * np.sqrt(252)
        
        last_nifty_vol = nifty_vol.iloc[-1]
        last_banknifty_vol = banknifty_vol.iloc[-1]
        vol_ratio = last_banknifty_vol / last_nifty_vol if last_nifty_vol > 0 else 1.0
        
        if vol_ratio > self.volatility_ratio_threshold:
            # BANKNIFTY more volatile: Long NIFTY (stable), short BANKNIFTY (volatile)
            action = "SELL"
            reason = f"Vol ratio {vol_ratio:.2f}: Short high-vol BANKNIFTY"
        elif vol_ratio < 1/self.volatility_ratio_threshold:
            # NIFTY more volatile: Long BANKNIFTY, short NIFTY
            action = "BUY"
            reason = f"Vol ratio {vol_ratio:.2f}: Long low-vol BANKNIFTY"
        else:
            return None
        
        last_nifty = nifty_prices.iloc[-1]
        last_banknifty = banknifty_prices.iloc[-1]
        
        metrics = PairSpreadMetrics(
            nifty_price=last_nifty,
            banknifty_price=last_banknifty,
            spread=last_nifty - last_banknifty,
            spread_zscore=0.0,
            correlation=nifty_prices.corr(banknifty_prices),
            correlation_deviation=0.0,
            nifty_volatility=last_nifty_vol,
            banknifty_volatility=last_banknifty_vol,
            volatility_ratio=vol_ratio,
            liquidity_spread_nifty=0.01,
            liquidity_spread_banknifty=0.01,
            timestamp=nifty_prices.index[-1]
        )
        
        return PairTradingSignal(
            strategy=PairTradingStrategy.VOLATILITY,
            action=action,
            long_instrument="NIFTY50" if action == "SELL" else "BANKNIFTY",
            short_instrument="BANKNIFTY" if action == "SELL" else "NIFTY50",
            long_quantity=1,
            short_quantity=1,
            entry_price_long=last_nifty if action == "SELL" else last_banknifty,
            entry_price_short=last_banknifty if action == "SELL" else last_nifty,
            stop_loss=0.02,
            take_profit=0.04,
            risk_reward_ratio=2.0,
            confidence=60,
            reason=reason,
            metrics=metrics,
            timestamp=datetime.now()
        )
    
    def get_all_signals(self,
                       nifty_df: pd.DataFrame,
                       banknifty_df: pd.DataFrame) -> List[PairTradingSignal]:
        """
        Get all available signals from all strategies
        
        Args:
            nifty_df: DataFrame with NIFTY50 OHLCV
            banknifty_df: DataFrame with BANKNIFTY OHLCV
        
        Returns:
            List of trading signals
        """
        signals = []
        
        # Stat Arb
        stat_arb = self.stat_arb_signal(nifty_df['close'], banknifty_df['close'])
        if stat_arb:
            signals.append(stat_arb)
        
        # Index Hedge
        hedge = self.index_hedge_signal(nifty_df['close'], banknifty_df['close'])
        if hedge:
            signals.append(hedge)
        
        # Correlation
        corr = self.correlation_deviation_signal(nifty_df['close'], banknifty_df['close'])
        if corr:
            signals.append(corr)
        
        # Volatility
        vol = self.volatility_spread_signal(nifty_df['close'], banknifty_df['close'])
        if vol:
            signals.append(vol)
        
        return signals
    
    def get_largest_constituents(self, index_name: str = "NIFTY50", 
                                count: int = 5) -> List[PortfolioConstituent]:
        """Get largest constituents by weight for hedging"""
        constituents_data = self.index_weights.get(index_name, {}).get('constituents', [])
        constituents = []
        for const in constituents_data[:count]:
            constituents.append(PortfolioConstituent(
                symbol=const.get('symbol'),
                sector=const.get('sector'),
                weight=const.get('weight'),
                market_cap=const.get('market_cap_b'),
                liquidity_tier=const.get('liquidity_tier')
            ))
        return constituents
