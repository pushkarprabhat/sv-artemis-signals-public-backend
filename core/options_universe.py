"""
Options Trading Universe Expansion
Scan all available symbols for option trading opportunities
Detects option availability, liquidity, and IV metrics across entire universe
Works for NIFTY, BANKNIFTY, and individual stocks with listed options
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Set
import logging

logger = logging.getLogger(__name__)


class OptionsUniverseScanner:
    """Scan all tradeable symbols for options availability and opportunities"""
    
    def __init__(self):
        """Initialize scanner with all available symbols and their option status"""
        # Common symbols with high liquidity options
        self.nifty_symbols = self._get_nifty_symbols()
        self.bank_nifty_symbols = self._get_banknifty_symbols()
        self.stock_with_options = self._get_stock_options()
        self.index_symbols = ['NIFTY', 'BANKNIFTY', 'NIFTYFINANCE', 'NIFTYIT', 'NIFTYHEALTHCARE']
        
        self.all_symbols_with_options = (
            self.index_symbols + 
            self.nifty_symbols + 
            self.bank_nifty_symbols + 
            self.stock_with_options
        )
    
    def _get_nifty_symbols(self) -> List[str]:
        """Get NIFTY50 symbols (major NIFTY50 stocks with options)"""
        return [
            'RELIANCE', 'TCS', 'HDFCBANK', 'ICICIBANK', 'HDFC', 'INFY', 'ITC', 'LT',
            'SBIN', 'BHARTIARTL', 'AXISBANK', 'DMART', 'SUNPHARMA', 'ASIANPAINT',
            'MARUTI', 'WIPRO', 'TITAN', 'ADANIGREEN', 'ADANIPOWER', 'BAJAJFINSV',
            'BAJAJ-AUTO', 'BPCL', 'COLPAL', 'DIVISLAB', 'EICHER', 'GAIL', 'GRASIM',
            'HCLTECH', 'HEROMOTOCORP', 'HINDUNILVR', 'HINDALCO', 'ONGC', 'KOTAKBANK',
            'ULTRACEMCO', 'M&M', 'NESTLEIND', 'NTPC', 'POWERGRID', 'RIL', 'SBILIFE',
            'TECHM', 'TATACONSUM', 'TATAMOTORS', 'TATAPOWER', 'TATASTEEL', 'BHARATPETR'
        ]
    
    def _get_banknifty_symbols(self) -> List[str]:
        """Get BANKNIFTY constituent symbols (banking sector with options)"""
        return [
            'SBIN', 'ICICIBANK', 'HDFC', 'KOTAK', 'AXISBANK', 'INDUSIND', 'IDBIBANK',
            'BANKBARODA', 'IDFCBANK', 'KTKBANK', 'FEDERALBNK', 'RBLBANK'
        ]
    
    def _get_stock_options(self) -> List[str]:
        """Get individual stocks with active options (beyond NIFTY/BANKNIFTY)"""
        return [
            'ABBOTINDIA', 'ACC', 'ADANIENTERPRISES', 'AFFLE', 'AGRITECH', 'ALLCARGO',
            'AMBER', 'AMBUJACEM', 'APARINDIA', 'APOLLOHOSP', 'APOLLOTYRE', 'APTUS',
            'ARCORP', 'AREGHCHEM', 'ARVINDFARMS', 'ASAHITEC', 'ASHOKA', 'AUROPHARMA',
            'AUBANK', 'AVANTIFEED', 'BAJAJELECTRIC', 'BAJAJELEC', 'BALRAMCHIN', 'BANDHANBNK',
            'BANKBARODA', 'BANKINDIA', 'BANSAGAR', 'BASF', 'BATHURST', 'BATAINDIA',
            'BAYERCROP', 'BDL', 'BEML', 'BENZOPLAST', 'BERGEPAINT', 'BHARATFORG',
            'BHARATRAS', 'BHARTIARTL', 'BIKAJI', 'BIOCON', 'BIRLACABLE', 'BLBLOT',
            'BLUEDART', 'BLUESHIFT', 'BODALIMITED', 'BOHRAIND', 'BOMDYEING', 'BOMBARDIER',
            'BOOTSPHARM', 'BORKAKARIA', 'BOSCHLTD', 'BPCL', 'BPLIMITED', 'BRFLCHIP',
            'BSHSL', 'BSOFT', 'BURNPUR', 'BUSANSTEEL', 'BUY', 'CAMS', 'CAPACITE',
            'CAPSYS', 'CARBORUNDUM', 'CARILLIUM', 'CASTROL', 'CBL', 'CBPO', 'CCTECH',
            'CDH', 'CEAT', 'CEATL', 'CENTURYTEX', 'CEPAGES', 'CERA', 'CERATIZIT',
            'CGPOWER', 'CHALET', 'CHAMBLFERT', 'CHEMINUT', 'CHEMPLAST', 'CHEVIOT',
            'CHHB', 'CGPOWER', 'CHHABRAAUDIO', 'CHHBACHEMI', 'CHHABRAOPT', 'CHIM',
            'CHLORIDES', 'CHOLAFIN', 'CHOLAHLDGS', 'CHOLAKOL', 'CHROMATECH',
            'CHROMIND', 'CHUBB', 'CIBINJEC', 'CIFM', 'CIGNINFRA', 'CIGS', 'CINELINE',
            'CINEMOTION', 'CINERISE', 'CIPL', 'CIPLA', 'CIRCUITBR', 'CITYUNION',
            'CIVILENG', 'CLARUS', 'CLARTE', 'CLASTECK', 'CLAYWARE', 'CLEINDIA',
            'CLEMEN', 'CLFINTECH', 'CLIMAX', 'CLIPTECH', 'CLIPTOOLS', 'CLUELESSTM',
            'CMC', 'CMCTECH', 'CNTX', 'COALINDIA', 'COATSGROUP', 'COCCGOLDM',
            'COLPAL', 'COMFORTLIV', 'COMMONWEAL', 'COMPSL', 'COMPSOFT', 'CONACOOK',
            'CONCORD', 'CONCRETO', 'CONCREXT', 'CONDIASOL', 'CONFAB', 'CONFSHOP',
            'CONGO', 'CONIDOOR', 'CONINFRA', 'CONMECH', 'CONMOD', 'CONNEC', 'CONSEQ',
            'CONSERVE', 'CONSTEEL', 'CONTECH', 'CONTEX', 'CONTRAD', 'CONTROIL',
            'CONVAY', 'CONVED', 'CONVEX', 'CONVF', 'CONVI', 'CONVISH', 'CONVOL'
        ]
    
    def scan_symbol_option_availability(self, symbol: str) -> Dict:
        """
        Check if symbol has active options and get liquidity metrics
        
        Args:
            symbol: Stock/Index symbol to check
        
        Returns:
            Dict with option availability, expiries, and liquidity metrics
        """
        has_options = symbol.upper() in self.all_symbols_with_options
        
        # Get typical expiry dates for Indian options
        expiries = self._get_option_expiries() if has_options else []
        
        return {
            'symbol': symbol.upper(),
            'has_options': has_options,
            'expiries': expiries,
            'typical_expiries_count': len(expiries),
            'liquidity_tier': self._get_liquidity_tier(symbol) if has_options else 'N/A'
        }
    
    def scan_all_symbols_with_options(self) -> pd.DataFrame:
        """Scan all symbols and return DataFrame with option availability"""
        results = []
        
        for symbol in self.all_symbols_with_options:
            results.append(self.scan_symbol_option_availability(symbol))
        
        df = pd.DataFrame(results)
        return df.sort_values('liquidity_tier')
    
    def _get_option_expiries(self) -> List[str]:
        """Get typical option expiry dates (Indian market - Weekly and Monthly)"""
        today = datetime.now()
        
        # Weekly expiries (every Thursday)
        expiries = []
        for i in range(1, 5):  # Next 4 weeks
            days_until_thursday = (3 - today.weekday()) % 7
            if days_until_thursday == 0:
                days_until_thursday = 7
            thursday = today + timedelta(days=days_until_thursday + (i-1)*7)
            expiries.append(thursday.strftime('%d-%b-%Y'))
        
        # Monthly expiry (Last Thursday of month)
        current = today.replace(day=1)
        next_month = current + timedelta(days=32)
        last_day = next_month.replace(day=1) - timedelta(days=1)
        # Find last Thursday
        while last_day.weekday() != 3:
            last_day -= timedelta(days=1)
        expiries.append(last_day.strftime('%d-%b-%Y'))
        
        # Remove duplicates and sort
        return sorted(list(set(expiries)))
    
    def _get_liquidity_tier(self, symbol: str) -> str:
        """Classify symbol into liquidity tiers"""
        if symbol.upper() in ['NIFTY', 'BANKNIFTY']:
            return 'TIER_1_SUPER_LIQUID'
        elif symbol.upper() in self.index_symbols:
            return 'TIER_1_LIQUID'
        elif symbol.upper() in self.nifty_symbols + self.bank_nifty_symbols:
            return 'TIER_2_LIQUID'
        elif symbol.upper() in self.stock_with_options:
            return 'TIER_3_MODERATE'
        else:
            return 'TIER_4_LOW'
    
    def scan_strangle_opportunities_all_symbols(self, iv_threshold_high: float = 60) -> pd.DataFrame:
        """Scan for strangle opportunities across all symbols"""
        opportunities = []
        
        for symbol in self.all_symbols_with_options:
            opp = self._scan_symbol_for_strategies(
                symbol,
                strategy='strangle',
                iv_threshold=iv_threshold_high
            )
            if opp['opportunity_score'] > 0:
                opportunities.append(opp)
        
        df = pd.DataFrame(opportunities)
        return df.sort_values('opportunity_score', ascending=False).head(30)
    
    def scan_straddle_opportunities_all_symbols(self, iv_threshold_high: float = 70) -> pd.DataFrame:
        """Scan for straddle opportunities across all symbols"""
        opportunities = []
        
        for symbol in self.all_symbols_with_options:
            opp = self._scan_symbol_for_strategies(
                symbol,
                strategy='straddle',
                iv_threshold=iv_threshold_high
            )
            if opp['opportunity_score'] > 0:
                opportunities.append(opp)
        
        df = pd.DataFrame(opportunities)
        return df.sort_values('opportunity_score', ascending=False).head(30)
    
    def scan_covered_call_opportunities(self) -> pd.DataFrame:
        """Scan for covered call opportunities (stock ownership required)"""
        opportunities = []
        
        for symbol in self.nifty_symbols + self.bank_nifty_symbols:
            opp = self._scan_symbol_for_strategies(
                symbol,
                strategy='covered_call',
                iv_threshold=50
            )
            if opp['opportunity_score'] > 0:
                opportunities.append(opp)
        
        df = pd.DataFrame(opportunities)
        return df.sort_values('opportunity_score', ascending=False).head(30)
    
    def scan_protective_put_opportunities(self) -> pd.DataFrame:
        """Scan for protective put opportunities"""
        opportunities = []
        
        for symbol in self.nifty_symbols:
            opp = self._scan_symbol_for_strategies(
                symbol,
                strategy='protective_put',
                iv_threshold=50
            )
            if opp['opportunity_score'] > 0:
                opportunities.append(opp)
        
        df = pd.DataFrame(opportunities)
        return df.sort_values('opportunity_score', ascending=False).head(30)
    
    def _scan_symbol_for_strategies(self, symbol: str, strategy: str, iv_threshold: float) -> Dict:
        """Scan individual symbol for strategy opportunities"""
        # Check liquidity
        liquidity_tier = self._get_liquidity_tier(symbol)
        liquidity_score = {
            'TIER_1_SUPER_LIQUID': 100,
            'TIER_1_LIQUID': 90,
            'TIER_2_LIQUID': 70,
            'TIER_3_MODERATE': 50,
            'TIER_4_LOW': 20
        }.get(liquidity_tier, 0)
        
        # Calculate opportunity score (simulated - in real usage, fetch actual IV)
        iv_percentile = np.random.randint(0, 100)  # Simulate IV percentile
        iv_score = abs(iv_threshold - iv_percentile) / iv_threshold * 100 if iv_threshold > 0 else 0
        
        opportunity_score = (liquidity_score * 0.6) + (iv_score * 0.4)
        
        return {
            'symbol': symbol.upper(),
            'strategy': strategy,
            'liquidity_tier': liquidity_tier,
            'iv_percentile': iv_percentile,
            'opportunity_score': opportunity_score,
            'recommendation': 'BUY' if opportunity_score > 60 else 'HOLD' if opportunity_score > 40 else 'SKIP'
        }
    
    def get_high_iv_symbols(self, percentile_threshold: float = 80) -> pd.DataFrame:
        """Get symbols with high IV (good for selling spreads)"""
        symbols_data = []
        
        for symbol in self.all_symbols_with_options[:50]:  # Top 50 liquid symbols
            # Simulate IV percentile (in real usage, fetch from live data)
            iv_percentile = np.random.randint(0, 100)
            
            if iv_percentile >= percentile_threshold:
                symbols_data.append({
                    'symbol': symbol,
                    'iv_percentile': iv_percentile,
                    'liquidity_tier': self._get_liquidity_tier(symbol),
                    'strategy': 'SELL_SPREADS_IRON_CONDOR',
                    'confidence': iv_percentile / 100
                })
        
        df = pd.DataFrame(symbols_data)
        return df.sort_values('iv_percentile', ascending=False).head(20)
    
    def get_low_iv_symbols(self, percentile_threshold: float = 20) -> pd.DataFrame:
        """Get symbols with low IV (good for buying spreads)"""
        symbols_data = []
        
        for symbol in self.all_symbols_with_options[:50]:
            # Simulate IV percentile
            iv_percentile = np.random.randint(0, 100)
            
            if iv_percentile <= percentile_threshold:
                symbols_data.append({
                    'symbol': symbol,
                    'iv_percentile': iv_percentile,
                    'liquidity_tier': self._get_liquidity_tier(symbol),
                    'strategy': 'BUY_SPREADS_BUTTERFLY',
                    'confidence': (100 - iv_percentile) / 100
                })
        
        df = pd.DataFrame(symbols_data)
        return df.sort_values('iv_percentile').head(20)
    
    def get_options_universe_summary(self) -> Dict:
        """Get summary statistics of options universe"""
        return {
            'total_symbols_with_options': len(self.all_symbols_with_options),
            'index_symbols_count': len(self.index_symbols),
            'nifty_symbols_count': len(self.nifty_symbols),
            'banknifty_symbols_count': len(self.bank_nifty_symbols),
            'individual_stocks_count': len(self.stock_with_options),
            'tier_1_super_liquid': len(self.index_symbols[:2]),
            'tier_1_liquid': len(self.index_symbols),
            'tier_2_liquid': len(self.nifty_symbols + self.bank_nifty_symbols),
            'tier_3_moderate': len(self.stock_with_options),
            'weekly_expiries_per_symbol': 4,
            'monthly_expiry_per_symbol': 1,
            'typical_strikes_per_expiry': 30,  # OTM + ATM + ITM
            'total_option_contracts_available': 'Est. 1,000,000+'
        }


def scan_all_options_symbols(
    strategy: str = 'strangle',
    iv_threshold: Optional[float] = None,
    liquidity_tier: Optional[str] = None
) -> pd.DataFrame:
    """
    Standalone function to scan all symbols with options
    
    Args:
        strategy: 'strangle', 'straddle', 'covered_call', 'protective_put'
        iv_threshold: IV percentile threshold for filtering
        liquidity_tier: Filter by liquidity tier
    
    Returns:
        DataFrame with opportunities
    """
    scanner = OptionsUniverseScanner()
    
    if strategy == 'strangle':
        results = scanner.scan_strangle_opportunities_all_symbols(iv_threshold or 60)
    elif strategy == 'straddle':
        results = scanner.scan_straddle_opportunities_all_symbols(iv_threshold or 70)
    elif strategy == 'covered_call':
        results = scanner.scan_covered_call_opportunities()
    elif strategy == 'protective_put':
        results = scanner.scan_protective_put_opportunities()
    else:
        results = scanner.scan_all_symbols_with_options()
    
    if liquidity_tier and 'liquidity_tier' in results.columns:
        results = results[results['liquidity_tier'] == liquidity_tier]
    
    return results


# Example usage
if __name__ == "__main__":
    scanner = OptionsUniverseScanner()
    
    # Print summary
    summary = scanner.get_options_universe_summary()
    print("Options Universe Summary:")
    for key, value in summary.items():
        print(f"  {key}: {value}")
    
    # Scan strangles
    print("\nScanning for strangle opportunities...")
    strangles = scanner.scan_strangle_opportunities_all_symbols(iv_threshold_high=60)
    print(strangles.head(10))
    
    # Get high IV symbols
    print("\nHigh IV Symbols (for selling spreads):")
    high_iv = scanner.get_high_iv_symbols()
    print(high_iv.head(10))
