"""
utils/nse_derivatives_manager.py - Download and manage NSE derivatives data
Fetches all derivatives from NSE and maintains universe with derivative status indicators
"""

import requests
import pandas as pd
from pathlib import Path
from datetime import datetime
import json
import logging
from typing import Dict, List, Tuple, Set
from functools import lru_cache

logger = logging.getLogger(__name__)

# NSE derivatives URL
NSE_DERIVATIVES_URL = "https://www.nseindia.com/static/products-services/equity-derivatives-products"

class NSEDerivativesManager:
    """Manage NSE derivatives universe and classification"""
    
    def __init__(self, cache_dir: Path = Path("marketdata/derivatives")):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        self.derivatives_file = self.cache_dir / "nse_derivatives_universe.json"
        self.derivatives_list_file = self.cache_dir / "derivatives_list.csv"
        self.classification_file = self.cache_dir / "symbol_classification.json"
        
        # Load on init
        self.derivatives_universe = self._load_derivatives_universe()
        self.symbol_classification = self._load_classification()
    
    def download_nse_derivatives_list(self) -> pd.DataFrame:
        """
        Download derivatives list from NSE website
        Returns DataFrame with symbol, type, expiry info
        """
        try:
            logger.info("Downloading NSE derivatives list...")
            
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            }
            
            # Fetch derivatives page
            response = requests.get(NSE_DERIVATIVES_URL, headers=headers, timeout=10)
            response.raise_for_status()
            
            # Parse HTML - extract table data
            # NSE provides derivatives data in specific format
            # This would typically be in a table on the page
            
            logger.info("Parsing NSE derivatives data...")
            
            # For now, return structured format
            # In production, parse the actual HTML/API response
            derivatives_data = self._parse_nse_derivatives_response(response.text)
            
            df = pd.DataFrame(derivatives_data)
            
            # Save to cache
            df.to_csv(self.derivatives_list_file, index=False)
            logger.info(f"Saved {len(df)} derivatives to {self.derivatives_list_file}")
            
            return df
            
        except Exception as e:
            logger.error(f"Error downloading NSE derivatives: {e}")
            # Return cached version if available
            if self.derivatives_list_file.exists():
                logger.info("Using cached derivatives list")
                return pd.read_csv(self.derivatives_list_file)
            return pd.DataFrame()
    
    def _parse_nse_derivatives_response(self, html_content: str) -> List[Dict]:
        """Parse NSE derivatives HTML response"""
        try:
            from bs4 import BeautifulSoup
            soup = BeautifulSoup(html_content, 'html.parser')
            
            derivatives = []
            
            # Find tables with derivative data
            tables = soup.find_all('table')
            
            for table in tables:
                rows = table.find_all('tr')
                for row in rows[1:]:  # Skip header
                    cols = row.find_all('td')
                    if len(cols) >= 3:
                        try:
                            derivatives.append({
                                'symbol': cols[0].text.strip(),
                                'instrument_type': cols[1].text.strip(),
                                'underlying': cols[2].text.strip() if len(cols) > 2 else '',
                                'expiry_date': cols[3].text.strip() if len(cols) > 3 else '',
                            })
                        except:
                            continue
            
            return derivatives
            
        except ImportError:
            logger.warning("BeautifulSoup not available, returning empty list")
            return []
        except Exception as e:
            logger.error(f"Error parsing derivatives: {e}")
            return []
    
    def build_comprehensive_universe(self) -> Dict[str, Dict]:
        """
        Build comprehensive universe with derivative classification
        Returns: {symbol: {type, has_options, has_futures, has_stock, sector, ...}}
        """
        logger.info("Building comprehensive universe with derivatives...")
        
        universe = {}
        
        # Get derivatives list
        derivatives_df = self.download_nse_derivatives_list()
        
        if derivatives_df.empty:
            logger.warning("No derivatives data available")
            return universe
        
        # Group by underlying symbol
        by_symbol = derivatives_df.groupby('underlying')
        
        for symbol, group in by_symbol:
            symbol_clean = symbol.strip().upper()
            
            # Analyze instrument types
            instrument_types = set(group['instrument_type'].str.upper().str.strip())
            
            universe[symbol_clean] = {
                'symbol': symbol_clean,
                'has_options': bool('OPTION' in instrument_types or 'CE' in instrument_types or 'PE' in instrument_types),
                'has_futures': bool('FUTURE' in instrument_types),
                'has_stock': True,  # Assume all are stocks unless index
                'is_index': self._is_index(symbol_clean),
                'derivative_status': self._get_derivative_status(
                    has_options=bool('OPTION' in instrument_types or 'CE' in instrument_types),
                    has_futures=bool('FUTURE' in instrument_types)
                ),
                'data_download_enabled': True,
                'last_updated': datetime.now().isoformat(),
            }
        
        # Save classification
        self._save_classification(universe)
        
        logger.info(f"Built universe with {len(universe)} symbols")
        return universe
    
    def _is_index(self, symbol: str) -> bool:
        """Check if symbol is an index"""
        index_patterns = ['NIFTY', 'SENSEX', 'BANKEX', 'VIX', 'INDEX']
        return any(pattern in symbol.upper() for pattern in index_patterns)
    
    def _get_derivative_status(self, has_options: bool, has_futures: bool) -> str:
        """Get derivative status string"""
        if has_options and has_futures:
            return "Stock+Options+Futures"
        elif has_options:
            return "Stock+Options"
        elif has_futures:
            return "Stock+Futures"
        else:
            return "Stock-Only"
    
    def get_symbol_derivative_status(self, symbol: str) -> str:
        """Get derivative status for a symbol"""
        symbol = symbol.strip().upper()
        
        if symbol in self.symbol_classification:
            return self.symbol_classification[symbol].get('derivative_status', 'Unknown')
        
        return "Unknown"
    
    def get_symbols_by_derivative_status(self, status: str) -> List[str]:
        """Get all symbols with specific derivative status"""
        status_upper = status.upper()
        return [
            sym for sym, data in self.symbol_classification.items()
            if data.get('derivative_status', '').upper() == status_upper
        ]
    
    def classify_recommendation(self, symbol: str, strategy: str) -> Dict:
        """
        Classify a recommendation with derivative and strategy information
        
        Returns: {
            symbol, strategy, derivative_status, is_index, market_segment,
            recommendation_details
        }
        """
        symbol = symbol.strip().upper()
        
        if symbol not in self.symbol_classification:
            return {
                'symbol': symbol,
                'status': 'Not in universe',
                'derivative_status': 'Unknown',
            }
        
        classification = self.symbol_classification[symbol]
        
        return {
            'symbol': symbol,
            'strategy': strategy,
            'derivative_status': classification['derivative_status'],
            'has_options': classification['has_options'],
            'has_futures': classification['has_futures'],
            'is_index': classification['is_index'],
            'market_segment': 'Index' if classification['is_index'] else 'Stock',
            'tradeable': True,
            'data_available': classification['data_download_enabled'],
        }
    
    def _load_derivatives_universe(self) -> Dict:
        """Load cached derivatives universe"""
        if self.derivatives_file.exists():
            try:
                with open(self.derivatives_file, 'r') as f:
                    return json.load(f)
            except:
                return {}
        return {}
    
    def _load_classification(self) -> Dict:
        """Load cached classification"""
        if self.classification_file.exists():
            try:
                with open(self.classification_file, 'r') as f:
                    return json.load(f)
            except:
                return {}
        return {}
    
    def _save_classification(self, classification: Dict):
        """Save classification to file"""
        try:
            with open(self.classification_file, 'w') as f:
                json.dump(classification, f, indent=2, default=str)
            logger.info(f"Saved classification to {self.classification_file}")
        except Exception as e:
            logger.error(f"Error saving classification: {e}")
    
    def get_nifty500_with_derivatives(self) -> pd.DataFrame:
        """
        Get NIFTY500 constituents with their derivative status
        Returns DataFrame with symbol, name, derivative_status, has_options, has_futures
        """
        logger.info("Loading NIFTY500 constituents with derivatives...")
        
        # Load NIFTY500 from local cache
        nifty500_file = Path("marketdata") / "nifty500_constituents.csv"
        
        if not nifty500_file.exists():
            logger.warning(f"NIFTY500 file not found: {nifty500_file}")
            # Create empty dataframe
            return pd.DataFrame(columns=['symbol', 'derivative_status', 'has_options', 'has_futures'])
        
        try:
            nifty500_df = pd.read_csv(nifty500_file)
            
            # Add derivative status
            nifty500_df['derivative_status'] = nifty500_df['symbol'].apply(
                self.get_symbol_derivative_status
            )
            
            # Add derivative flags
            nifty500_df['has_options'] = nifty500_df['symbol'].apply(
                lambda s: self.symbol_classification.get(s.upper(), {}).get('has_options', False)
            )
            nifty500_df['has_futures'] = nifty500_df['symbol'].apply(
                lambda s: self.symbol_classification.get(s.upper(), {}).get('has_futures', False)
            )
            
            logger.info(f"Loaded {len(nifty500_df)} NIFTY500 constituents")
            return nifty500_df
            
        except Exception as e:
            logger.error(f"Error loading NIFTY500: {e}")
            return pd.DataFrame()
    
    def get_statistics(self) -> Dict:
        """Get universe statistics"""
        if not self.symbol_classification:
            return {}
        
        classification = self.symbol_classification
        
        stats = {
            'total_symbols': len(classification),
            'stock_plus_options_plus_futures': len([
                s for s in classification.values()
                if s.get('derivative_status') == 'Stock+Options+Futures'
            ]),
            'stock_plus_options': len([
                s for s in classification.values()
                if s.get('derivative_status') == 'Stock+Options'
            ]),
            'stock_plus_futures': len([
                s for s in classification.values()
                if s.get('derivative_status') == 'Stock+Futures'
            ]),
            'stock_only': len([
                s for s in classification.values()
                if s.get('derivative_status') == 'Stock-Only'
            ]),
            'indices': len([
                s for s in classification.values()
                if s.get('is_index', False)
            ]),
        }
        
        return stats


# Global instance for easy access
_manager_instance = None

def get_derivatives_manager() -> NSEDerivativesManager:
    """Get or create global derivatives manager"""
    global _manager_instance
    if _manager_instance is None:
        _manager_instance = NSEDerivativesManager()
    return _manager_instance


# Convenience functions
def get_symbol_status(symbol: str) -> str:
    """Get derivative status for symbol"""
    return get_derivatives_manager().get_symbol_derivative_status(symbol)

def classify_trade_recommendation(symbol: str, strategy: str) -> Dict:
    """Classify a trade recommendation with full details"""
    return get_derivatives_manager().classify_recommendation(symbol, strategy)

def get_nifty500_universe() -> pd.DataFrame:
    """Get NIFTY500 with derivative information"""
    return get_derivatives_manager().get_nifty500_with_derivatives()

def build_universe():
    """Build comprehensive universe from NSE derivatives"""
    return get_derivatives_manager().build_comprehensive_universe()

def get_universe_stats() -> Dict:
    """Get universe statistics"""
    return get_derivatives_manager().get_statistics()
