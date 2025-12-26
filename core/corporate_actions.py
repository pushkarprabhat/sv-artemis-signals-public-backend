"""
Corporate Actions Handler
Manages stock splits, dividends, bonus issues, and their impact on historical prices

Key Corporate Actions:
1. Stock Split (e.g., 1:2 split means 1 share becomes 2 shares, price halves)
2. Dividend (cash/stock distribution, typically reduces closing price)
3. Bonus Issue (free shares, stock split equivalent, affects all prices)
4. Rights Issue (shareholders buy more at discount, minimal price adjustment)
5. Buyback (company buys own shares, no direct price adjustment needed)
6. Merger/Demerger (significant structural change, affects base price)

Impact on Price Data:
- Stock Split (1:2): All historical prices HALVED, volumes DOUBLED
- Bonus (1:1): All historical prices HALVED, volumes DOUBLED  
- Dividend: Close price reduced by dividend amount on ex-date
- Adjustment: Forward adjustment recommended (adjust old prices down)
"""

import pandas as pd
from datetime import datetime, date
from typing import Dict, List, Tuple, Optional
from pathlib import Path
from utils.logger import logger
import json


class CorporateAction:
    """Single corporate action record"""
    
    def __init__(self, 
                 symbol: str,
                 action_type: str,  # 'split', 'dividend', 'bonus', 'rights'
                 action_date: date,
                 ratio: str,  # e.g., "1:2" for split, "1.5" for dividend per share
                 description: str = ""):
        self.symbol = symbol
        self.action_type = action_type
        self.action_date = action_date
        self.ratio = ratio  # "old:new" for splits/bonus, amount for dividends
        self.description = description
    
    def to_dict(self):
        return {
            'symbol': self.symbol,
            'action_type': self.action_type,
            'action_date': str(self.action_date),
            'ratio': self.ratio,
            'description': self.description
        }


class CorporateActionsManager:
    """Manages corporate actions and price adjustments"""
    
    def __init__(self, data_dir: Path = None):
        self.data_dir = data_dir or Path("marketdata/corporate_actions")
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.actions_file = self.data_dir / "actions.json"
        self.checksums_file = self.data_dir / "checksums.json"
        self.actions: Dict[str, List[CorporateAction]] = self._load_actions()
        self.checksums: Dict[str, str] = self._load_checksums()
    
    def _load_actions(self) -> Dict[str, List[CorporateAction]]:
        """Load corporate actions from file"""
        if not self.actions_file.exists():
            return {}
        
        try:
            with open(self.actions_file, 'r') as f:
                data = json.load(f)
            
            actions = {}
            for symbol, action_list in data.items():
                actions[symbol] = [
                    CorporateAction(
                        symbol=action['symbol'],
                        action_type=action['action_type'],
                        action_date=datetime.strptime(action['action_date'], '%Y-%m-%d').date(),
                        ratio=action['ratio'],
                        description=action.get('description', '')
                    )
                    for action in action_list
                ]
            return actions
        except Exception as e:
            logger.warning(f"Could not load corporate actions: {e}")
            return {}
    
    def _load_checksums(self) -> Dict[str, str]:
        """Load file checksums for integrity verification"""
        if not self.checksums_file.exists():
            return {}
        
        try:
            with open(self.checksums_file, 'r') as f:
                return json.load(f)
        except:
            return {}
    
    def record_action(self, action: CorporateAction):
        """Record a new corporate action"""
        symbol = action.symbol
        if symbol not in self.actions:
            self.actions[symbol] = []
        
        self.actions[symbol].append(action)
        self.actions[symbol].sort(key=lambda x: x.action_date)
        self._save_actions()
        logger.info(f"[CORP-ACTION] Recorded {action.action_type} for {symbol} on {action.action_date}")
    
    def _save_actions(self):
        """Save corporate actions to file"""
        data = {
            symbol: [action.to_dict() for action in action_list]
            for symbol, action_list in self.actions.items()
        }
        
        with open(self.actions_file, 'w') as f:
            json.dump(data, f, indent=2)
    
    def get_actions_for_symbol(self, symbol: str, before_date: date = None) -> List[CorporateAction]:
        """Get all corporate actions for a symbol"""
        actions = self.actions.get(symbol, [])
        
        if before_date:
            actions = [a for a in actions if a.action_date <= before_date]
        
        return sorted(actions, key=lambda x: x.action_date)
    
    def adjust_price_data(self, 
                         df: pd.DataFrame, 
                         symbol: str,
                         adjustment_method: str = 'forward') -> pd.DataFrame:
        """
        Adjust historical prices for corporate actions
        
        Args:
            df: DataFrame with OHLCV data (must have 'date' column)
            symbol: Stock symbol
            adjustment_method: 'forward' (adjust old prices down) or 'backward' (adjust new prices up)
        
        Returns:
            Adjusted DataFrame
        
        Example:
            Stock split 1:2 on 2024-01-15
            - Forward: All prices before 2024-01-15 are halved
            - Backward: All prices after 2024-01-15 are doubled
        """
        df = df.copy()
        actions = self.get_actions_for_symbol(symbol)
        
        if not actions:
            return df
        
        # Convert date column to datetime if needed
        if df['date'].dtype == 'object':
            df['date'] = pd.to_datetime(df['date'])
        
        # Apply adjustments in chronological order
        for action in actions:
            action_dt = pd.to_datetime(action.action_date)
            
            if action.action_type == 'split':
                # Parse ratio: "1:2" means 1 old share = 2 new shares
                old_ratio, new_ratio = map(int, action.ratio.split(':'))
                adjustment_factor = old_ratio / new_ratio  # e.g., 1/2 = 0.5
                
                if adjustment_method == 'forward':
                    # Adjust prices BEFORE the split down
                    mask = df['date'] < action_dt
                    df.loc[mask, ['open', 'high', 'low', 'close']] *= adjustment_factor
                    df.loc[mask, 'volume'] /= adjustment_factor
                else:  # backward
                    # Adjust prices AFTER the split up
                    mask = df['date'] >= action_dt
                    df.loc[mask, ['open', 'high', 'low', 'close']] /= adjustment_factor
                    df.loc[mask, 'volume'] *= adjustment_factor
                
                logger.info(f"[ADJUST] {symbol}: Applied {action.ratio} split at {action.action_date} ({adjustment_method})")
            
            elif action.action_type == 'bonus':
                # Bonus 1:1 is equivalent to 1:2 split
                old_ratio, new_ratio = map(int, action.ratio.split(':'))
                adjustment_factor = old_ratio / new_ratio
                
                if adjustment_method == 'forward':
                    mask = df['date'] < action_dt
                    df.loc[mask, ['open', 'high', 'low', 'close']] *= adjustment_factor
                    df.loc[mask, 'volume'] /= adjustment_factor
                else:
                    mask = df['date'] >= action_dt
                    df.loc[mask, ['open', 'high', 'low', 'close']] /= adjustment_factor
                    df.loc[mask, 'volume'] *= adjustment_factor
                
                logger.info(f"[ADJUST] {symbol}: Applied {action.ratio} bonus at {action.action_date} ({adjustment_method})")
            
            elif action.action_type == 'dividend':
                # Dividend reduces closing price by dividend amount on ex-date
                # Typically: Close' = Close - Dividend
                # But this is tricky - we only adjust if we have the dividend amount
                try:
                    dividend_per_share = float(action.ratio)
                    mask = df['date'] >= action_dt
                    df.loc[mask, 'close'] = df.loc[mask, 'close'] - dividend_per_share
                    
                    # Also adjust high/low if dividend > price drop
                    df.loc[mask, 'high'] = df.loc[mask, 'high'] - dividend_per_share
                    df.loc[mask, 'low'] = df.loc[mask, 'low'] - dividend_per_share
                    # Open may not change (typically)
                    
                    logger.info(f"[ADJUST] {symbol}: Applied ₹{dividend_per_share} dividend at {action.action_date}")
                except ValueError:
                    logger.warning(f"Could not parse dividend amount: {action.ratio}")
        
        return df
    
    def compute_file_checksum(self, file_path: Path) -> str:
        """
        Compute SHA256 checksum of file for integrity verification
        Ensures source data hasn't been modified
        """
        import hashlib
        
        sha256_hash = hashlib.sha256()
        with open(file_path, "rb") as f:
            for byte_block in iter(lambda: f.read(4096), b""):
                sha256_hash.update(byte_block)
        
        return sha256_hash.hexdigest()
    
    def record_file_checksum(self, symbol: str, timeframe: str, file_path: Path):
        """Record checksum of downloaded file to detect future modifications"""
        checksum = self.compute_file_checksum(file_path)
        key = f"{symbol}_{timeframe}"
        self.checksums[key] = {
            'checksum': checksum,
            'timestamp': datetime.now().isoformat(),
            'file_path': str(file_path)
        }
        self._save_checksums()
        logger.debug(f"[CHECKSUM] Recorded for {symbol}_{timeframe}: {checksum[:8]}...")
    
    def verify_file_integrity(self, symbol: str, timeframe: str, file_path: Path) -> bool:
        """Verify file hasn't been modified since last download"""
        key = f"{symbol}_{timeframe}"
        
        if key not in self.checksums:
            logger.debug(f"[CHECKSUM] No baseline for {symbol}_{timeframe} - first time")
            return True
        
        stored = self.checksums[key]
        current_checksum = self.compute_file_checksum(file_path)
        
        if current_checksum == stored['checksum']:
            logger.debug(f"[CHECKSUM] {symbol}_{timeframe} verified - data unchanged")
            return True
        else:
            logger.warning(f"[CHECKSUM] {symbol}_{timeframe} MISMATCH - data may have changed!")
            logger.warning(f"  Expected: {stored['checksum']}")
            logger.warning(f"  Got: {current_checksum}")
            return False
    
    def _save_checksums(self):
        """Save checksums to file"""
        with open(self.checksums_file, 'w') as f:
            json.dump(self.checksums, f, indent=2)


# Singleton instance
_instance = None

def get_corporate_actions_manager() -> CorporateActionsManager:
    """Get singleton instance of corporate actions manager"""
    global _instance
    if _instance is None:
        _instance = CorporateActionsManager()
    return _instance
