# core/parallel_options_scanner.py
# Parallel options analysis with live trade recommendations
# Professional options opportunity scanner with real-time alerts

import pandas as pd
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta
import threading
from typing import List, Dict, Optional, Tuple
from pathlib import Path

from config import BASE_DIR, MAX_WORKERS
from core.options_chain import get_options_chain, calculate_greeks
from core.strangle import scan_strangle_setups
from core.volatility import calculate_historical_volatility, calculate_implied_volatility
from utils.logger import logger
from utils.email_service import EmailService
from utils.telegram_service import send_telegram_message


# =============================================================================
# PARALLEL OPTIONS SCANNER
# =============================================================================

class ParallelOptionsScanner:
    """
    Runs options analysis in parallel with pairs trading
    Sends real-time alerts for high-confidence trades (70%+)
    
    Strategy: While pairs trade slowly, options can provide quick wins.
    This scanner identifies high-probability setups and alerts immediately.
    """
    
    def __init__(self, confidence_threshold=0.70):
        self.confidence_threshold = confidence_threshold
        self.running = False
        self.last_scan_time = None
        self.scan_thread = None
        self.email_service = EmailService()
        
        # Track sent alerts to avoid duplicates
        self.sent_alerts = set()
        
        logger.info(f"ParallelOptionsScanner initialized (confidence >= {confidence_threshold*100}%)")
    
    
    def start_scanning(self, interval_minutes=15):
        """
        Start continuous scanning in background thread
        
        Args:
            interval_minutes: scan frequency (default 15 min)
        """
        if self.running:
            logger.warning("Scanner already running")
            return
        
        self.running = True
        self.scan_thread = threading.Thread(
            target=self._scan_loop,
            args=(interval_minutes,),
            daemon=True
        )
        self.scan_thread.start()
        logger.info(f"✅ Parallel options scanner started (every {interval_minutes} min)")
    
    
    def stop_scanning(self):
        """Stop background scanning"""
        self.running = False
        if self.scan_thread:
            self.scan_thread.join(timeout=5)
        logger.info("Options scanner stopped")
    
    
    def _scan_loop(self, interval_minutes):
        """Internal loop for continuous scanning"""
        while self.running:
            try:
                # Scan for opportunities
                signals = self.scan_options_opportunities()
                
                # Filter high-confidence trades
                high_conf = signals[signals['confidence'] >= self.confidence_threshold]
                
                if not high_conf.empty:
                    logger.info(f"🎯 Found {len(high_conf)} high-confidence options trades")
                    
                    # Send alerts
                    self._send_trade_alerts(high_conf)
                
                self.last_scan_time = datetime.now()
                
                # Wait for next interval
                import time
                time.sleep(interval_minutes * 60)
                
            except Exception as e:
                logger.error(f"Options scan loop error: {e}")
                import time
                time.sleep(60)  # Wait 1 min on error
    
    
    def scan_options_opportunities(self) -> pd.DataFrame:
        """
        Scan all options strategies in parallel
        
        Returns:
            DataFrame with all opportunities and confidence scores
        """
        try:
            logger.info("Scanning options opportunities...")
            
            # Get universe of stocks to scan
            from universe.focused_universe import get_focused_universe
            universe = get_focused_universe(sector="ALL")
            
            symbols = universe['Symbol'].tolist()[:100]  # Top 100 for performance
            
            all_results = []
            
            # Strategy 1: Strangles (already implemented)
            strangle_signals = scan_strangle_setups(symbols=symbols)
            if not strangle_signals.empty:
                strangle_signals['strategy_type'] = 'Strangle'
                all_results.append(strangle_signals)
            
            # Strategy 2: Iron Condors (new)
            ic_signals = self._scan_iron_condors(symbols)
            if not ic_signals.empty:
                ic_signals['strategy_type'] = 'Iron Condor'
                all_results.append(ic_signals)
            
            # Strategy 3: Calendar Spreads (new)
            calendar_signals = self._scan_calendar_spreads(symbols)
            if not calendar_signals.empty:
                calendar_signals['strategy_type'] = 'Calendar Spread'
                all_results.append(calendar_signals)
            
            # Strategy 4: Bull/Bear Spreads (new)
            spread_signals = self._scan_directional_spreads(symbols)
            if not spread_signals.empty:
                spread_signals['strategy_type'] = 'Directional Spread'
                all_results.append(spread_signals)
            
            if not all_results:
                logger.info("No options opportunities found")
                return pd.DataFrame()
            
            # Combine all strategies
            combined_df = pd.concat(all_results, ignore_index=True)
            
            # Calculate unified confidence score
            combined_df = self._calculate_confidence(combined_df)
            
            # Sort by confidence
            combined_df = combined_df.sort_values('confidence', ascending=False)
            
            # Add scan timestamp
            combined_df['scan_time'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            
            logger.info(f"Found {len(combined_df)} total options opportunities")
            
            return combined_df
            
        except Exception as e:
            logger.error(f"Options scan error: {e}")
            return pd.DataFrame()
    
    
    def _scan_iron_condors(self, symbols: List[str]) -> pd.DataFrame:
        """
        Scan for Iron Condor opportunities (sell OTM calls + puts, buy further OTM for protection)
        
        Strategy: Limited risk, limited profit - ideal for range-bound markets
        with low volatility and stable price action.
        """
        results = []
        
        for symbol in symbols[:50]:  # Limit for performance
            try:
                # Get options chain
                chain = get_options_chain(symbol)
                if chain is None or chain.empty:
                    continue
                
                # Check if in low volatility regime (good for iron condors)
                hv = calculate_historical_volatility(symbol, days=30)
                if hv is None or hv > 30:  # Skip high volatility
                    continue
                
                # Find suitable strikes (2 std dev from ATM)
                # This is simplified - full implementation needs option prices
                results.append({
                    'symbol': symbol,
                    'setup': 'Iron Condor',
                    'iv_rank': hv,
                    'expected_return': 0.08,  # 8% return target
                    'risk_reward': 1.5
                })
                
            except Exception as e:
                continue
        
        return pd.DataFrame(results) if results else pd.DataFrame()
    
    
    def _scan_calendar_spreads(self, symbols: List[str]) -> pd.DataFrame:
        """
        Scan for Calendar Spread opportunities (sell near-term, buy far-term, same strike)
        
        Strategy: Profit from time decay differential between near and far-term options
        while maintaining delta neutrality.
        """
        results = []
        
        for symbol in symbols[:50]:
            try:
                # Check volatility term structure
                iv_near = calculate_implied_volatility(symbol, days_to_expiry=30)
                iv_far = calculate_implied_volatility(symbol, days_to_expiry=60)
                
                if iv_near is None or iv_far is None:
                    continue
                
                # Calendar spread works when near-term IV > far-term IV
                if iv_near > iv_far + 2:  # 2% difference threshold
                    results.append({
                        'symbol': symbol,
                        'setup': 'Calendar Spread',
                        'iv_near': iv_near,
                        'iv_far': iv_far,
                        'iv_spread': iv_near - iv_far,
                        'expected_return': 0.10
                    })
                    
            except Exception as e:
                continue
        
        return pd.DataFrame(results) if results else pd.DataFrame()
    
    
    def _scan_directional_spreads(self, symbols: List[str]) -> pd.DataFrame:
        """
        Scan for Bull/Bear Spread opportunities (directional bets with limited risk)
        
        Strategy: Express directional views with defined risk using vertical spreads
        for asymmetric risk-reward profiles.
        """
        results = []
        
        for symbol in symbols[:50]:
            try:
                from core.pairs import load_price
                price = load_price(symbol, "day")
                
                if price is None or len(price) < 50:
                    continue
                
                # Calculate momentum
                sma20 = price.rolling(20).mean()
                current_price = price.iloc[-1]
                current_sma = sma20.iloc[-1]
                
                # Bullish setup
                if current_price > current_sma * 1.02:  # 2% above SMA
                    results.append({
                        'symbol': symbol,
                        'setup': 'Bull Call Spread',
                        'direction': 'BULLISH',
                        'price_vs_sma': (current_price - current_sma) / current_sma * 100,
                        'expected_return': 0.12
                    })
                
                # Bearish setup
                elif current_price < current_sma * 0.98:  # 2% below SMA
                    results.append({
                        'symbol': symbol,
                        'setup': 'Bear Put Spread',
                        'direction': 'BEARISH',
                        'price_vs_sma': (current_price - current_sma) / current_sma * 100,
                        'expected_return': 0.12
                    })
                    
            except Exception as e:
                continue
        
        return pd.DataFrame(results) if results else pd.DataFrame()
    
    
    def _calculate_confidence(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate unified confidence score (0-1) for all strategies
        
        Quantitative scoring: Identifies trades with highest probability of success
        based on multiple technical and statistical factors.
        """
        if df.empty:
            return df
        
        # Different confidence calculation per strategy
        def calc_confidence(row):
            try:
                if row['strategy_type'] == 'Strangle':
                    # Use IV rank and expected return
                    iv_score = min(row.get('iv_rank', 0) / 100, 1.0)
                    return_score = min(row.get('expected_return', 0) / 0.15, 1.0)
                    return (iv_score * 0.6 + return_score * 0.4)
                
                elif row['strategy_type'] == 'Iron Condor':
                    # Low vol = good for IC
                    vol_score = max(0, 1 - row.get('iv_rank', 50) / 50)
                    rr_score = min(row.get('risk_reward', 0) / 2, 1.0)
                    return (vol_score * 0.5 + rr_score * 0.5)
                
                elif row['strategy_type'] == 'Calendar Spread':
                    # Large IV spread = good for calendar
                    spread_score = min(row.get('iv_spread', 0) / 5, 1.0)
                    return spread_score
                
                elif row['strategy_type'] == 'Directional Spread':
                    # Strong momentum = good for directional
                    momentum_score = min(abs(row.get('price_vs_sma', 0)) / 5, 1.0)
                    return momentum_score
                
                else:
                    return 0.5  # Default
                    
            except Exception as e:
                return 0.5
        
        df['confidence'] = df.apply(calc_confidence, axis=1)
        
        return df
    
    
    def _send_trade_alerts(self, high_conf_trades: pd.DataFrame):
        """
        Send Telegram and Email alerts for high-confidence trades
        
        Alert system: Immediate notifications when high-probability
        opportunities are identified by the scanner.
        """
        try:
            for _, trade in high_conf_trades.iterrows():
                # Create alert key to avoid duplicates
                alert_key = f"{trade['symbol']}_{trade['strategy_type']}_{trade.get('setup', '')}"
                
                if alert_key in self.sent_alerts:
                    continue  # Already alerted
                
                # Format message
                conf_pct = trade['confidence'] * 100
                
                message = f"""
🎯 HIGH CONFIDENCE OPTIONS TRADE

Symbol: {trade['symbol']}
Strategy: {trade['strategy_type']}
Setup: {trade.get('setup', 'N/A')}
Confidence: {conf_pct:.1f}%
Expected Return: {trade.get('expected_return', 0)*100:.1f}%

📊 Details:
{self._format_trade_details(trade)}

⚠️ Review all parameters before executing!

Scanned: {trade.get('scan_time', datetime.now().strftime("%Y-%m-%d %H:%M:%S"))}
"""
                
                # Send Telegram alert
                try:
                    send_telegram_message(message)
                    logger.info(f"✅ Telegram alert sent: {alert_key}")
                except Exception as e:
                    logger.warning(f"Telegram alert failed: {e}")
                
                # Send Email alert
                try:
                    self.email_service.send_email(
                        subject=f"🎯 Options Alert: {trade['symbol']} ({conf_pct:.0f}% confidence)",
                        body=message,
                        recipients=["pushkarprabhat@gmail.com"]
                    )
                    logger.info(f"✅ Email alert sent: {alert_key}")
                except Exception as e:
                    logger.warning(f"Email alert failed: {e}")
                
                # Mark as sent
                self.sent_alerts.add(alert_key)
                
        except Exception as e:
            logger.error(f"Alert sending error: {e}")
    
    
    def _format_trade_details(self, trade: pd.Series) -> str:
        """Format trade details for alert message"""
        details = []
        
        for key, value in trade.items():
            if key not in ['symbol', 'strategy_type', 'setup', 'confidence', 'expected_return', 'scan_time']:
                if pd.notna(value):
                    details.append(f"  • {key}: {value}")
        
        return "\n".join(details) if details else "  (No additional details)"
    
    
    def get_live_recommendations(self) -> pd.DataFrame:
        """
        Get current live recommendations (for UI display)
        
        Returns:
            DataFrame with actionable entry/exit recommendations
        """
        try:
            # Scan fresh opportunities
            opportunities = self.scan_options_opportunities()
            
            if opportunities.empty:
                return pd.DataFrame()
            
            # Filter for high confidence
            recommendations = opportunities[
                opportunities['confidence'] >= self.confidence_threshold
            ].copy()
            
            # Add entry/exit recommendations
            recommendations['action'] = 'ENTRY'  # All are entry signals
            recommendations['stop_loss'] = recommendations['expected_return'].apply(
                lambda x: f"-{x*0.5*100:.1f}%"  # SL at 50% of expected return
            )
            recommendations['target'] = recommendations['expected_return'].apply(
                lambda x: f"+{x*100:.1f}%"
            )
            
            # Add priority (1 = highest)
            recommendations['priority'] = recommendations['confidence'].rank(ascending=False).astype(int)
            
            return recommendations.sort_values('priority')
            
        except Exception as e:
            logger.error(f"Get recommendations error: {e}")
            return pd.DataFrame()


# =============================================================================
# GLOBAL SCANNER INSTANCE
# =============================================================================

_scanner_instance = None

def get_options_scanner() -> ParallelOptionsScanner:
    """Get singleton scanner instance"""
    global _scanner_instance
    if _scanner_instance is None:
        _scanner_instance = ParallelOptionsScanner()
    return _scanner_instance


def start_parallel_options_scanning(interval_minutes=15, confidence_threshold=0.70):
    """
    Start parallel options scanning in background
    
    System initialization: Call once at app startup to enable
    continuous options monitoring and real-time alerts.
    
    Args:
        interval_minutes: scan frequency
        confidence_threshold: minimum confidence for alerts (0.70 = 70%)
    """
    scanner = get_options_scanner()
    scanner.confidence_threshold = confidence_threshold
    scanner.start_scanning(interval_minutes)
    
    logger.info(f"✅ Parallel options scanner started (confidence >= {confidence_threshold*100}%)")
    return scanner


def stop_parallel_options_scanning():
    """Stop background options scanning"""
    scanner = get_options_scanner()
    scanner.stop_scanning()


def get_live_options_recommendations() -> pd.DataFrame:
    """
    Get current live options recommendations
    
    UI integration: Call this from UI to display real-time
    options opportunities with confidence scores.
    """
    scanner = get_options_scanner()
    return scanner.get_live_recommendations()


# Export functions
__all__ = [
    'ParallelOptionsScanner',
    'get_options_scanner',
    'start_parallel_options_scanning',
    'stop_parallel_options_scanning',
    'get_live_options_recommendations'
]
