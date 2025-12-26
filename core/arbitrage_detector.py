# core/arbitrage_detector.py — Spot-Futures Arbitrage Detector
# Detects: Spot-futures mispricing opportunities
# Strategies: Cash-and-carry, reverse cash-and-carry, conversion/reversal
# Features: Real-time basis calculation, opportunity scoring, execution support

import pandas as pd
import numpy as np
import datetime as dt
from pathlib import Path
from config import BASE_DIR
from utils.logger import logger
import utils.helpers


class ArbitrageDetector:
    """Detects spot-futures arbitrage opportunities"""
    
    def __init__(self):
        self.kite = utils.helpers.kite
        self.base_dir = BASE_DIR / 'data' / 'arbitrage'
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self._cache_dir = self.base_dir / 'cache'
        self._cache_dir.mkdir(parents=True, exist_ok=True)
    
    def get_spot_price(self, symbol):
        """Get current spot price for a symbol
        
        Args:
            symbol: Stock symbol (e.g., 'RELIANCE', 'TCS')
        
        Returns:
            float: Current spot price or None
        """
        try:
            # Get instrument tokens
            eq_instruments = self.kite.instruments('NSE')
            
            spot_token = None
            for instr in eq_instruments:
                if instr['tradingsymbol'] == symbol:
                    spot_token = instr['instrument_token']
                    break
            
            if not spot_token:
                logger.warning(f"[ARB] Spot instrument not found: {symbol}")
                return None
            
            # Fetch LTP
            quote = self.kite.quote(spot_token)
            if spot_token in quote:
                return quote[spot_token]['last_price']
            
            return None
        
        except Exception as e:
            logger.error(f"[ARB] Error fetching spot price for {symbol}: {e}")
            return None
    
    def get_futures_price(self, symbol):
        """Get current futures price for a symbol
        
        Args:
            symbol: Stock symbol or index name
        
        Returns:
            float: Current futures price or None
        """
        try:
            # Get NFO instruments
            nfo_instruments = self.kite.instruments('NFO')
            
            # Find active futures contract
            futures_token = None
            for instr in nfo_instruments:
                if (instr.get('name') == symbol and 
                    'FUT' in instr['tradingsymbol']):
                    futures_token = instr['instrument_token']
                    break
            
            if not futures_token:
                logger.warning(f"[ARB] Futures instrument not found: {symbol}")
                return None
            
            # Fetch LTP
            quote = self.kite.quote(futures_token)
            if futures_token in quote:
                return quote[futures_token]['last_price']
            
            return None
        
        except Exception as e:
            logger.error(f"[ARB] Error fetching futures price for {symbol}: {e}")
            return None
    
    def calculate_basis(self, spot_price, futures_price, days_to_expiry=30,
                       cost_of_carry=0.08, dividend_yield=0.0):
        """Calculate futures basis and fair value
        
        Args:
            spot_price: Current spot price
            futures_price: Current futures price
            days_to_expiry: Days to futures expiry
            cost_of_carry: Annual carry cost (interest rate)
            dividend_yield: Annual dividend yield
        
        Returns:
            dict: {basis, basis_pct, fair_value, mispricing}
        """
        if not spot_price or not futures_price:
            return None
        
        # Time to expiry in years
        T = days_to_expiry / 365
        
        # Fair value: F = S * e^((r - q) * T)
        # where r = cost of carry (interest), q = dividend yield
        rate = cost_of_carry - dividend_yield
        fair_value = spot_price * np.exp(rate * T)
        
        # Basis = Futures - Spot
        basis = futures_price - spot_price
        basis_pct = (basis / spot_price) * 100
        
        # Theoretical basis
        theoretical_basis = fair_value - spot_price
        
        # Mispricing
        mispricing = futures_price - fair_value
        mispricing_pct = (mispricing / fair_value) * 100
        
        return {
            'spot_price': spot_price,
            'futures_price': futures_price,
            'basis': basis,
            'basis_pct': basis_pct,
            'fair_value': fair_value,
            'theoretical_basis': theoretical_basis,
            'mispricing': mispricing,
            'mispricing_pct': mispricing_pct,
            'days_to_expiry': days_to_expiry,
            'opportunity': 'cash_and_carry' if mispricing > 0 else 'reverse_carry' if mispricing < 0 else 'none',
            'opportunity_strength': abs(mispricing_pct)
        }
    
    def find_mispriced_futures(self, symbols, min_mispricing_pct=0.5,
                              days_to_expiry=30, cost_of_carry=0.08):
        """Find mispriced futures across multiple symbols
        
        Args:
            symbols: List of symbols to check
            min_mispricing_pct: Minimum mispricing % to flag
            days_to_expiry: Days to expiry
            cost_of_carry: Annual interest rate
        
        Returns:
            DataFrame: [symbol, spot, futures, mispricing%, type]
        """
        logger.info(f"[ARB] Scanning {len(symbols)} symbols for arbitrage")
        
        opportunities = []
        
        for symbol in symbols:
            try:
                spot = self.get_spot_price(symbol)
                futures = self.get_futures_price(symbol)
                
                if not spot or not futures:
                    continue
                
                basis_data = self.calculate_basis(spot, futures, days_to_expiry, cost_of_carry)
                
                if (basis_data and 
                    abs(basis_data['mispricing_pct']) >= min_mispricing_pct):
                    
                    opportunities.append({
                        'symbol': symbol,
                        'spot_price': spot,
                        'futures_price': futures,
                        'fair_value': basis_data['fair_value'],
                        'mispricing': basis_data['mispricing'],
                        'mispricing_pct': basis_data['mispricing_pct'],
                        'opportunity_type': basis_data['opportunity'],
                        'strength': basis_data['opportunity_strength'],
                        'timestamp': dt.datetime.now()
                    })
            
            except Exception as e:
                logger.warning(f"[ARB] Error processing {symbol}: {e}")
        
        if opportunities:
            df = pd.DataFrame(opportunities)
            df = df.sort_values('strength', ascending=False)
            logger.info(f"[ARB] Found {len(df)} opportunities")
            return df
        
        logger.info("[ARB] No opportunities found above threshold")
        return pd.DataFrame()
    
    def calculate_arbitrage_pnl(self, symbol, spot_price, futures_price,
                               position_size=1, holding_days=5,
                               borrow_cost=0.08, transaction_cost_pct=0.005):
        """Calculate P&L for an arbitrage position
        
        Args:
            symbol: Symbol name
            spot_price: Entry spot price
            futures_price: Entry futures price
            position_size: Lot size (contract size)
            holding_days: Days to hold position
            borrow_cost: Annual borrowing cost
            transaction_cost_pct: Transaction cost as % of trade value
        
        Returns:
            dict: P&L calculation details
        """
        # Entry costs
        spot_cost = spot_price * position_size
        entry_transaction_cost = spot_cost * transaction_cost_pct
        total_entry_cost = spot_cost + entry_transaction_cost
        
        # Holding cost (interest)
        time_period = holding_days / 365
        holding_interest = total_entry_cost * borrow_cost * time_period
        
        # Exit transaction cost
        exit_transaction_cost = total_entry_cost * transaction_cost_pct
        
        # Total cost
        total_cost = entry_transaction_cost + holding_interest + exit_transaction_cost
        
        # Gross profit (futures - spot)
        gross_profit = (futures_price - spot_price) * position_size
        
        # Net profit
        net_profit = gross_profit - total_cost
        net_profit_pct = (net_profit / total_entry_cost) * 100
        
        # Annualized return
        annualized_return = net_profit_pct * (365 / holding_days)
        
        return {
            'symbol': symbol,
            'position_size': position_size,
            'spot_price': spot_price,
            'futures_price': futures_price,
            'entry_spot_cost': spot_cost,
            'entry_transaction_cost': entry_transaction_cost,
            'holding_interest': holding_interest,
            'exit_transaction_cost': exit_transaction_cost,
            'total_costs': total_cost,
            'gross_profit': gross_profit,
            'net_profit': net_profit,
            'net_profit_pct': net_profit_pct,
            'annualized_return': annualized_return,
            'holding_days': holding_days,
            'profitable': net_profit > 0
        }
    
    def detect_conversion_reversal(self, symbol, days_to_expiry=30):
        """Detect conversion (buy stock, sell call, buy put) or reversal opportunities
        
        For this, we'd need options data. Placeholder for future implementation.
        
        Args:
            symbol: Stock symbol
            days_to_expiry: Days to expiry
        
        Returns:
            dict: Conversion/reversal opportunity data
        """
        logger.info(f"[ARB] Analyzing conversion/reversal for {symbol}")
        
        return {
            'note': 'Requires options data - implement after options module ready'
        }
    
    def detect_calendar_spreads(self, symbol):
        """Detect calendar spread opportunities (near month vs far month)
        
        For this, we'd need multiple expiry futures data.
        
        Args:
            symbol: Futures symbol
        
        Returns:
            dict: Calendar spread opportunity data
        """
        logger.info(f"[ARB] Analyzing calendar spreads for {symbol}")
        
        return {
            'note': 'Requires multiple expiry data - implement after completing downloads'
        }
    
    def save_opportunity(self, opportunity_data, opportunity_type='spot_futures'):
        """Save identified opportunity to cache
        
        Args:
            opportunity_data: Opportunity details (dict)
            opportunity_type: Type of opportunity
        
        Returns:
            Path: Where opportunity was saved
        """
        timestamp = dt.datetime.now().strftime('%Y%m%d_%H%M%S')
        file_path = self._cache_dir / f"{opportunity_type}_{timestamp}.parquet"
        
        df = pd.DataFrame([opportunity_data])
        df.to_parquet(file_path, index=False)
        
        logger.info(f"[ARB] Saved opportunity: {file_path}")
        return file_path
    
    def rank_opportunities(self, opportunities_df, min_return_pct=0.5):
        """Rank arbitrage opportunities by potential return
        
        Args:
            opportunities_df: DataFrame with opportunity data
            min_return_pct: Minimum return % threshold
        
        Returns:
            DataFrame: Ranked opportunities
        """
        if opportunities_df.empty:
            return opportunities_df
        
        df = opportunities_df.copy()
        
        # Filter by minimum return
        df = df[df['strength'] >= min_return_pct]
        
        # Rank by opportunity strength
        df = df.sort_values('strength', ascending=False)
        df['rank'] = range(1, len(df) + 1)
        
        return df
    
    def get_opportunity_summary(self, opportunities_df):
        """Generate summary statistics for identified opportunities
        
        Args:
            opportunities_df: DataFrame with opportunity data
        
        Returns:
            dict: Summary statistics
        """
        if opportunities_df.empty:
            return {'total_opportunities': 0}
        
        cash_and_carry = (opportunities_df['opportunity_type'] == 'cash_and_carry').sum()
        reverse_carry = (opportunities_df['opportunity_type'] == 'reverse_carry').sum()
        
        return {
            'total_opportunities': len(opportunities_df),
            'cash_and_carry': cash_and_carry,
            'reverse_carry': reverse_carry,
            'avg_strength': opportunities_df['strength'].mean(),
            'max_strength': opportunities_df['strength'].max(),
            'min_strength': opportunities_df['strength'].min(),
            'top_symbol': opportunities_df.iloc[0]['symbol'] if len(opportunities_df) > 0 else None,
            'top_strength': opportunities_df.iloc[0]['strength'] if len(opportunities_df) > 0 else 0
        }
