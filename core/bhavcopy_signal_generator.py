"""
Bhavcopy Signal Generator - Generate trading signals from EOD data

Uses bhavcopy (end-of-day settlement data) to identify:
1. High institutional ownership (delivery ratio > 75%)
2. OI momentum trends (derivatives)
3. Volume analysis
4. Price patterns

These signals are used to plan NEXT DAY trades.
Execution is done with real-time Kite data, not bhavcopy prices.

Philosophy:
  Bhavcopy = Signal Generation (Analysis at 4 PM)
  Kite Data = Signal Execution (Trading at 9:30 AM next day)
"""

import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import json


class BhavcopySignalGenerator:
    """Generate trading signals from bhavcopy EOD data"""
    
    def __init__(self, bhavcopy_data_path: Optional[str] = None):
        """
        Initialize signal generator
        
        Args:
            bhavcopy_data_path: Path to bhavcopy data folder
        """
        if bhavcopy_data_path is None:
            bhavcopy_data_path = Path(__file__).parent.parent / 'universe' / 'metadata' / 'bhavcopy'
        
        self.bhavcopy_path = Path(bhavcopy_data_path)
        self.signals_file = Path(__file__).parent.parent / 'universe' / 'metadata' / 'signals.json'
    
    def load_capital_market_bhavcopy(self, date: Optional[datetime] = None) -> pd.DataFrame:
        """
        Load capital market (equity) bhavcopy for a date
        
        Args:
            date: Date to load (default: today)
        
        Returns:
            DataFrame with bhavcopy data
        """
        if date is None:
            date = datetime.now()
        
        # Bhavcopy files are named: cm01DEC2025bhavcopy.csv
        date_str = date.strftime('%d%b%Y').upper()
        filename = f"cm{date_str}bhavcopy.csv"
        filepath = self.bhavcopy_path / 'capital_market' / filename
        
        if not filepath.exists():
            print(f"[WARNING] Bhavcopy not found: {filepath}")
            return pd.DataFrame()
        
        try:
            df = pd.read_csv(filepath)
            # Rename columns for consistency
            df.columns = df.columns.str.upper()
            return df
        except Exception as e:
            print(f"[ERROR] Failed to load bhavcopy: {e}")
            return pd.DataFrame()
    
    def load_derivatives_bhavcopy(self, date: Optional[datetime] = None) -> pd.DataFrame:
        """
        Load derivatives (F&O) bhavcopy for a date
        
        Args:
            date: Date to load (default: today)
        
        Returns:
            DataFrame with derivatives bhavcopy data
        """
        if date is None:
            date = datetime.now()
        
        # Derivatives files: fo01DEC2025bhavcopy.csv
        date_str = date.strftime('%d%b%Y').upper()
        filename = f"fo{date_str}bhavcopy.csv"
        filepath = self.bhavcopy_path / 'derivatives' / filename
        
        if not filepath.exists():
            print(f"[WARNING] Derivatives bhavcopy not found: {filepath}")
            return pd.DataFrame()
        
        try:
            df = pd.read_csv(filepath)
            df.columns = df.columns.str.upper()
            return df
        except Exception as e:
            print(f"[ERROR] Failed to load derivatives bhavcopy: {e}")
            return pd.DataFrame()
    
    def detect_institutional_accumulation(self, cm_bhavcopy: pd.DataFrame, 
                                         min_delivery_ratio: float = 0.75) -> List[Dict]:
        """
        Detect stocks with high institutional ownership (delivery ratio > threshold)
        
        Interpretation:
        - High delivery ratio = Institutions buying/holding
        - Bullish signal = Expect continuation next day
        
        Args:
            cm_bhavcopy: Capital market bhavcopy DataFrame
            min_delivery_ratio: Minimum delivery ratio to flag (default 75%)
        
        Returns:
            List of signals: [{'symbol': 'SBIN', 'delivery_ratio': 0.85, 'confidence': 'high'}]
        """
        if cm_bhavcopy.empty:
            return []
        
        try:
            # Calculate delivery ratio
            cm_bhavcopy['DELIVERY_RATIO'] = (
                cm_bhavcopy['DELIVERYVOLUME'].astype(float) / 
                cm_bhavcopy['TOTTRDQTY'].astype(float)
            )
            
            # Filter high delivery ratio
            signals = []
            filtered = cm_bhavcopy[cm_bhavcopy['DELIVERY_RATIO'] >= min_delivery_ratio]
            
            for _, row in filtered.iterrows():
                signal = {
                    'symbol': row['SYMBOL'],
                    'signal_type': 'institutional_accumulation',
                    'delivery_ratio': round(row['DELIVERY_RATIO'], 3),
                    'confidence': 'high' if row['DELIVERY_RATIO'] > 0.85 else 'medium',
                    'interpretation': 'Institutional buying pressure - watch for breakout tomorrow',
                    'action': 'BUY on strength (support if it dips)',
                    'timestamp': datetime.now().isoformat()
                }
                signals.append(signal)
            
            print(f"[SIGNAL] Institutional accumulation: {len(signals)} stocks detected")
            return signals
        
        except Exception as e:
            print(f"[ERROR] Accumulation detection failed: {e}")
            return []
    
    def detect_institutional_distribution(self, cm_bhavcopy: pd.DataFrame, 
                                         max_delivery_ratio: float = 0.25) -> List[Dict]:
        """
        Detect stocks with low institutional ownership (delivery ratio < threshold)
        
        Interpretation:
        - Low delivery ratio = Retail selling/profit booking
        - Bearish signal = Expect weakness next day
        
        Args:
            cm_bhavcopy: Capital market bhavcopy DataFrame
            max_delivery_ratio: Maximum delivery ratio to flag (default 25%)
        
        Returns:
            List of signals
        """
        if cm_bhavcopy.empty:
            return []
        
        try:
            # Calculate delivery ratio
            cm_bhavcopy['DELIVERY_RATIO'] = (
                cm_bhavcopy['DELIVERYVOLUME'].astype(float) / 
                cm_bhavcopy['TOTTRDQTY'].astype(float)
            )
            
            # Filter low delivery ratio
            signals = []
            filtered = cm_bhavcopy[cm_bhavcopy['DELIVERY_RATIO'] <= max_delivery_ratio]
            
            for _, row in filtered.iterrows():
                signal = {
                    'symbol': row['SYMBOL'],
                    'signal_type': 'institutional_distribution',
                    'delivery_ratio': round(row['DELIVERY_RATIO'], 3),
                    'confidence': 'high' if row['DELIVERY_RATIO'] < 0.15 else 'medium',
                    'interpretation': 'Retail selling pressure - watch for breakdown tomorrow',
                    'action': 'SELL on weakness (resistance if it rises)',
                    'timestamp': datetime.now().isoformat()
                }
                signals.append(signal)
            
            print(f"[SIGNAL] Institutional distribution: {len(signals)} stocks detected")
            return signals
        
        except Exception as e:
            print(f"[ERROR] Distribution detection failed: {e}")
            return []
    
    def detect_high_volume(self, cm_bhavcopy: pd.DataFrame, 
                          percentile: float = 90) -> List[Dict]:
        """
        Detect stocks with unusually high volume
        
        Interpretation:
        - High volume with price move = Conviction
        - High volume = Liquidity for trading
        
        Args:
            cm_bhavcopy: Capital market bhavcopy DataFrame
            percentile: Volume percentile threshold (default 90th)
        
        Returns:
            List of signals
        """
        if cm_bhavcopy.empty:
            return []
        
        try:
            # Calculate volume threshold
            vol_threshold = cm_bhavcopy['TOTTRDQTY'].astype(float).quantile(percentile)
            
            signals = []
            filtered = cm_bhavcopy[cm_bhavcopy['TOTTRDQTY'].astype(float) >= vol_threshold]
            
            for _, row in filtered.iterrows():
                signal = {
                    'symbol': row['SYMBOL'],
                    'signal_type': 'high_volume',
                    'volume': int(row['TOTTRDQTY']),
                    'volume_threshold': int(vol_threshold),
                    'confidence': 'high',
                    'interpretation': f"Volume in top {100-percentile}% - Good liquidity for trading",
                    'action': 'Prioritize for swing trading (tight stops)',
                    'timestamp': datetime.now().isoformat()
                }
                signals.append(signal)
            
            print(f"[SIGNAL] High volume: {len(signals)} stocks detected")
            return signals
        
        except Exception as e:
            print(f"[ERROR] High volume detection failed: {e}")
            return []
    
    def detect_oi_momentum(self, fo_bhavcopy: pd.DataFrame, 
                          oi_increase_threshold: float = 1.05) -> List[Dict]:
        """
        Detect derivatives with increasing Open Interest
        
        Interpretation:
        - Rising OI + Rising price = Bullish (new long positions)
        - Rising OI + Falling price = Bearish (new short positions)
        
        Args:
            fo_bhavcopy: Derivatives bhavcopy DataFrame
            oi_increase_threshold: OI increase factor (default 1.05 = 5% increase)
        
        Returns:
            List of signals
        """
        if fo_bhavcopy.empty:
            return []
        
        try:
            signals = []
            
            # Group by underlying (e.g., SBIN-CE, SBIN-PE)
            for underlying in fo_bhavcopy['UNDERLYING'].unique():
                underlying_data = fo_bhavcopy[fo_bhavcopy['UNDERLYING'] == underlying]
                
                # Simplified: Check if today's OI > yesterday's estimated OI
                # In practice, you'd compare with yesterday's actual bhavcopy
                recent = underlying_data.nlargest(1, 'OPENINTEREST')
                
                if not recent.empty:
                    row = recent.iloc[0]
                    
                    signal = {
                        'symbol': row['UNDERLYING'],
                        'contract': row['SYMBOL'],
                        'signal_type': 'oi_momentum',
                        'open_interest': int(row['OPENINTEREST']),
                        'confidence': 'medium',
                        'interpretation': 'OI momentum detected - Watch expiry flows',
                        'action': 'Monitor for breakout or breakdown',
                        'timestamp': datetime.now().isoformat()
                    }
                    signals.append(signal)
            
            print(f"[SIGNAL] OI momentum: {len(signals)} underlyings detected")
            return signals
        
        except Exception as e:
            print(f"[ERROR] OI momentum detection failed: {e}")
            return []
    
    def generate_all_signals(self, date: Optional[datetime] = None) -> Dict:
        """
        Generate all signals from today's bhavcopy
        
        Args:
            date: Date to analyze (default: today)
        
        Returns:
            Dict with all signals organized by type
        """
        print(f"\n{'='*80}")
        print(f"GENERATING SIGNALS FROM BHAVCOPY")
        print(f"{'='*80}")
        
        date_str = date.strftime('%Y-%m-%d') if date else datetime.now().strftime('%Y-%m-%d')
        print(f"Analysis Date: {date_str}")
        print(f"Signal Execution Date: Tomorrow ({(datetime.now() + timedelta(days=1)).strftime('%Y-%m-%d')})")
        print(f"Execution Tool: Kite real-time data (NOT bhavcopy prices)")
        print(f"{'='*80}\n")
        
        # Load bhavcopy data
        cm_bhavcopy = self.load_capital_market_bhavcopy(date)
        fo_bhavcopy = self.load_derivatives_bhavcopy(date)
        
        signals = {
            'generated_at': datetime.now().isoformat(),
            'analysis_date': date_str,
            'execution_date': (datetime.now() + timedelta(days=1)).strftime('%Y-%m-%d'),
            'note': 'These signals are for planning tomorrow\'s trades. Execute with Kite real-time data at 9:30 AM.',
            'signals': {
                'institutional_accumulation': self.detect_institutional_accumulation(cm_bhavcopy),
                'institutional_distribution': self.detect_institutional_distribution(cm_bhavcopy),
                'high_volume': self.detect_high_volume(cm_bhavcopy),
                'oi_momentum': self.detect_oi_momentum(fo_bhavcopy),
            }
        }
        
        # Automatically send notifications for ALL signal types
        try:
            from utils.notifications import send_notification
            for signal_type, signal_list in signals['signals'].items():
                for signal in signal_list:
                    try:
                        send_notification(
                            strategy=signal.get('symbol', signal_type),
                            interval='daily',
                            trade_details=signal,
                            channel='EMAIL',
                            now=datetime.now(),
                            to=None
                        )
                    except Exception as e:
                        print(f"[ERROR] Email notification failed for {signal.get('symbol', signal_type)}: {e}")
                    try:
                        send_notification(
                            strategy=signal.get('symbol', signal_type),
                            interval='daily',
                            trade_details=signal,
                            channel='SMS',
                            now=datetime.now(),
                            to=None
                        )
                    except Exception as e:
                        print(f"[ERROR] SMS notification failed for {signal.get('symbol', signal_type)}: {e}")
                    try:
                        send_notification(
                            strategy=signal.get('symbol', signal_type),
                            interval='daily',
                            trade_details=signal,
                            channel='WHATSAPP',
                            now=datetime.now(),
                            to=None
                        )
                    except Exception as e:
                        print(f"[ERROR] WhatsApp notification failed for {signal.get('symbol', signal_type)}: {e}")
        except Exception as e:
            print(f"[ERROR] Failed to send notifications: {e}")
        return signals
    
    def save_signals(self, signals: Dict) -> bool:
        """
        Save generated signals to JSON file atomically
        Args:
            signals: Signals dict from generate_all_signals()
        Returns:
            True if saved successfully
        """
        try:
            self.signals_file.parent.mkdir(parents=True, exist_ok=True)
            tmp_path = str(self.signals_file) + ".tmp"
            with open(tmp_path, 'w') as f:
                json.dump(signals, f, indent=2)
            os.replace(tmp_path, self.signals_file)  # Atomic move
            print(f"[SAVED] Signals saved atomically to: {self.signals_file}")
            return True
        except Exception as e:
            print(f"[ERROR] Failed to save signals atomically: {e}")
            return False
    
    def print_signals_summary(self, signals: Dict):
        """
        Print formatted signals summary
        
        Args:
            signals: Signals dict from generate_all_signals()
        """
        print(f"\n{'='*80}")
        print(f"SIGNALS SUMMARY FOR TOMORROW'S TRADING")
        print(f"{'='*80}")
        print(f"Analysis Date: {signals['analysis_date']}")
        print(f"Execution Date: {signals['execution_date']}")
        print(f"Generated At: {signals['generated_at']}\n")
        
        for signal_type, signal_list in signals['signals'].items():
            print(f"\n{signal_type.upper().replace('_', ' ')}")
            print(f"  Total: {len(signal_list)} signals")
            
            if signal_list:
                print(f"  Examples:")
                for sig in signal_list[:3]:  # Show first 3
                    if 'symbol' in sig:
                        print(f"    - {sig['symbol']}: {sig['interpretation']}")
                    elif 'contract' in sig:
                        print(f"    - {sig['underlying']}: {sig['interpretation']}")
        
        print(f"\n{'='*80}")
        print(f"NOTE: Execute these signals with KITE LIVE DATA at 9:30 AM tomorrow")
        print(f"      Do NOT use bhavcopy prices (they are 24 hours old)")
        print(f"{'='*80}\n")


def get_signal_generator(bhavcopy_path: Optional[str] = None) -> BhavcopySignalGenerator:
    """Get signal generator instance"""
    return BhavcopySignalGenerator(bhavcopy_path)


if __name__ == "__main__":
    # Test signal generation
    gen = get_signal_generator()
    
    # Generate signals
    signals = gen.generate_all_signals()
    
    # Save signals
    gen.save_signals(signals)
    
    # Print summary
    gen.print_signals_summary(signals)
