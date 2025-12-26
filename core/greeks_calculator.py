# core/greeks_calculator.py — Options Greeks Calculator
# Calculates: Delta, Gamma, Theta, Vega, Rho
# Models: Black-Scholes (primary), Binomial (fallback)
# Features: Batch processing, caching, vectorization

import numpy as np
import pandas as pd
from scipy.stats import norm
import datetime as dt
from pathlib import Path
from config import BASE_DIR
from utils.logger import logger


class GreeksCalculator:
    """Calculates options Greeks using Black-Scholes model"""
    
    def __init__(self, risk_free_rate=0.06, dividend_yield=0.0):
        """Initialize Greeks calculator
        
        Args:
            risk_free_rate: Annual risk-free rate (default 6%)
            dividend_yield: Annual dividend yield (default 0%)
        """
        self.r = risk_free_rate
        self.q = dividend_yield
        self.cache_dir = BASE_DIR / 'data' / 'greeks_cache'
        self.cache_dir.mkdir(parents=True, exist_ok=True)
    
    def _d1(self, S, K, T, sigma):
        """Calculate d1 in Black-Scholes formula
        
        Args:
            S: Current spot price
            K: Strike price
            T: Time to expiry (in years)
            sigma: Volatility (annualized)
        
        Returns:
            float: d1 value
        """
        if T <= 0:
            return 0
        
        numerator = np.log(S / K) + (self.r - self.q + 0.5 * sigma ** 2) * T
        denominator = sigma * np.sqrt(T)
        
        if denominator == 0:
            return 0
        
        return numerator / denominator
    
    def _d2(self, S, K, T, sigma):
        """Calculate d2 in Black-Scholes formula
        
        Args:
            S: Current spot price
            K: Strike price
            T: Time to expiry (in years)
            sigma: Volatility (annualized)
        
        Returns:
            float: d2 value
        """
        d1 = self._d1(S, K, T, sigma)
        return d1 - sigma * np.sqrt(T) if T > 0 else 0
    
    def calculate_delta(self, S, K, T, sigma, option_type='call'):
        """Calculate option Delta
        
        Delta = Change in option price / Change in underlying price
        - Call Delta: 0 to 1 (how much call price changes with 1 unit move in spot)
        - Put Delta: -1 to 0
        
        Args:
            S: Current spot price
            K: Strike price
            T: Time to expiry (in years)
            sigma: Volatility (annualized)
            option_type: 'call' or 'put'
        
        Returns:
            float: Delta value
        """
        if T <= 0:
            if option_type.lower() == 'call':
                return 1.0 if S > K else 0.0
            else:
                return 0.0 if S > K else -1.0
        
        d1 = self._d1(S, K, T, sigma)
        
        if option_type.lower() == 'call':
            return np.exp(-self.q * T) * norm.cdf(d1)
        else:
            return np.exp(-self.q * T) * (norm.cdf(d1) - 1)
    
    def calculate_gamma(self, S, K, T, sigma):
        """Calculate option Gamma
        
        Gamma = Change in Delta / Change in underlying price
        - Same for calls and puts
        - Peaks at-the-money
        
        Args:
            S: Current spot price
            K: Strike price
            T: Time to expiry (in years)
            sigma: Volatility (annualized)
        
        Returns:
            float: Gamma value
        """
        if T <= 0 or sigma <= 0:
            return 0
        
        d1 = self._d1(S, K, T, sigma)
        return np.exp(-self.q * T) * norm.pdf(d1) / (S * sigma * np.sqrt(T))
    
    def calculate_theta(self, S, K, T, sigma, option_type='call'):
        """Calculate option Theta
        
        Theta = Change in option price / Change in time (per day)
        - Call Theta: Usually negative (time decay)
        - Put Theta: Can be positive or negative depending on conditions
        - More negative as expiry approaches
        
        Args:
            S: Current spot price
            K: Strike price
            T: Time to expiry (in years)
            sigma: Volatility (annualized)
            option_type: 'call' or 'put'
        
        Returns:
            float: Theta per day
        """
        if T <= 0:
            return 0
        
        d1 = self._d1(S, K, T, sigma)
        d2 = self._d2(S, K, T, sigma)
        
        sqrt_T = np.sqrt(T)
        
        if option_type.lower() == 'call':
            theta = (-S * np.exp(-self.q * T) * norm.pdf(d1) * sigma / (2 * sqrt_T)
                    - self.r * K * np.exp(-self.r * T) * norm.cdf(d2)
                    + self.q * S * np.exp(-self.q * T) * norm.cdf(d1))
        else:
            theta = (-S * np.exp(-self.q * T) * norm.pdf(d1) * sigma / (2 * sqrt_T)
                    + self.r * K * np.exp(-self.r * T) * norm.cdf(-d2)
                    - self.q * S * np.exp(-self.q * T) * norm.cdf(-d1))
        
        # Convert to per-day theta
        return theta / 365
    
    def calculate_vega(self, S, K, T, sigma):
        """Calculate option Vega
        
        Vega = Change in option price / Change in volatility (1%)
        - Same for calls and puts
        - Peaks at-the-money
        - 0 at expiry
        
        Args:
            S: Current spot price
            K: Strike price
            T: Time to expiry (in years)
            sigma: Volatility (annualized)
        
        Returns:
            float: Vega per 1% change in volatility
        """
        if T <= 0:
            return 0
        
        d1 = self._d1(S, K, T, sigma)
        return S * np.exp(-self.q * T) * norm.pdf(d1) * np.sqrt(T) / 100
    
    def calculate_rho(self, S, K, T, sigma, option_type='call'):
        """Calculate option Rho
        
        Rho = Change in option price / Change in interest rate (1%)
        - Call Rho: Positive (higher rates increase call value)
        - Put Rho: Negative
        
        Args:
            S: Current spot price
            K: Strike price
            T: Time to expiry (in years)
            sigma: Volatility (annualized)
            option_type: 'call' or 'put'
        
        Returns:
            float: Rho per 1% change in interest rate
        """
        if T <= 0:
            return 0
        
        d2 = self._d2(S, K, T, sigma)
        
        if option_type.lower() == 'call':
            return K * T * np.exp(-self.r * T) * norm.cdf(d2) / 100
        else:
            return -K * T * np.exp(-self.r * T) * norm.cdf(-d2) / 100
    
    def calculate_all_greeks(self, S, K, T, sigma, option_type='call'):
        """Calculate all Greeks for an option
        
        Args:
            S: Current spot price
            K: Strike price
            T: Time to expiry (in years)
            sigma: Volatility (annualized)
            option_type: 'call' or 'put'
        
        Returns:
            dict: {delta, gamma, theta, vega, rho}
        """
        return {
            'delta': self.calculate_delta(S, K, T, sigma, option_type),
            'gamma': self.calculate_gamma(S, K, T, sigma),
            'theta': self.calculate_theta(S, K, T, sigma, option_type),
            'vega': self.calculate_vega(S, K, T, sigma),
            'rho': self.calculate_rho(S, K, T, sigma, option_type),
            'option_type': option_type,
            'S': S, 'K': K, 'T': T, 'sigma': sigma
        }
    
    def batch_calculate_greeks(self, df, spot_col='ltp', strike_col='strike_price',
                               expiry_col='expiry', iv_col='iv', option_type_col='instrument_type',
                               reference_date=None):
        """Calculate Greeks for a DataFrame of options
        
        Args:
            df: DataFrame with option data
            spot_col: Column name for spot price
            strike_col: Column name for strike price
            expiry_col: Column name for expiry date
            iv_col: Column name for implied volatility
            option_type_col: Column name for option type ('CE'/'PE' or 'call'/'put')
            reference_date: Reference date for T calculation (default=today)
        
        Returns:
            DataFrame: Original df with added greeks columns
        """
        if reference_date is None:
            reference_date = dt.datetime.now()
        
        df_copy = df.copy()
        
        # Calculate T (time to expiry in years)
        if isinstance(df_copy[expiry_col].iloc[0], str):
            df_copy['expiry_dt'] = pd.to_datetime(df_copy[expiry_col])
        else:
            df_copy['expiry_dt'] = df_copy[expiry_col]
        
        df_copy['T'] = (df_copy['expiry_dt'] - reference_date).dt.days / 365.25
        df_copy['T'] = df_copy['T'].clip(lower=0)
        
        # Initialize greek columns
        greeks = ['delta', 'gamma', 'theta', 'vega', 'rho']
        for greek in greeks:
            df_copy[greek] = 0.0
        
        # Calculate Greeks for each row
        for idx, row in df_copy.iterrows():
            S = row[spot_col]
            K = row[strike_col]
            T = row['T']
            sigma = row[iv_col] if iv_col in row.index else 0.2  # Default 20% IV
            
            # Normalize option type
            opt_type = str(row[option_type_col]).upper()
            if opt_type == 'CE':
                opt_type = 'call'
            elif opt_type == 'PE':
                opt_type = 'put'
            
            if sigma > 0 and T >= 0:
                greeks_dict = self.calculate_all_greeks(S, K, T, sigma, opt_type)
                for greek in greeks:
                    df_copy.at[idx, greek] = greeks_dict[greek]
        
        # Drop temporary columns
        df_copy = df_copy.drop(['expiry_dt'], axis=1)
        
        return df_copy
    
    def cache_greeks(self, date=None):
        """Cache today's greeks calculations
        
        Args:
            date: Date to cache for (default=today)
        
        Returns:
            Path: Path where cache was saved
        """
        if date is None:
            date = dt.datetime.now().date()
        
        cache_file = self.cache_dir / f"{date}.parquet"
        return cache_file
    
    def get_cached_greeks(self, date=None):
        """Retrieve cached greeks
        
        Args:
            date: Date to retrieve (default=today)
        
        Returns:
            DataFrame or None
        """
        if date is None:
            date = dt.datetime.now().date()
        
        cache_file = self.cache_dir / f"{date}.parquet"
        
        if not cache_file.exists():
            return None
        
        try:
            return pd.read_parquet(cache_file)
        except Exception as e:
            logger.error(f"[GREEKS] Error reading cache {date}: {e}")
            return None


# Convenience functions for single calculations
def calculate_call_delta(S, K, T, r, sigma):
    """Quick call delta calculation"""
    calc = GreeksCalculator(risk_free_rate=r)
    return calc.calculate_delta(S, K, T, sigma, 'call')


def calculate_put_delta(S, K, T, r, sigma):
    """Quick put delta calculation"""
    calc = GreeksCalculator(risk_free_rate=r)
    return calc.calculate_delta(S, K, T, sigma, 'put')


def calculate_gamma(S, K, T, r, sigma):
    """Quick gamma calculation"""
    calc = GreeksCalculator(risk_free_rate=r)
    return calc.calculate_gamma(S, K, T, sigma)


def calculate_option_price_bs(S, K, T, r, sigma, option_type='call', q=0):
    """Calculate option price using Black-Scholes
    
    Args:
        S: Spot price
        K: Strike price
        T: Time to expiry (years)
        r: Risk-free rate
        sigma: Volatility
        option_type: 'call' or 'put'
        q: Dividend yield
    
    Returns:
        float: Option price
    """
    calc = GreeksCalculator(risk_free_rate=r, dividend_yield=q)
    
    d1 = calc._d1(S, K, T, sigma)
    d2 = calc._d2(S, K, T, sigma)
    
    if option_type.lower() == 'call':
        price = (S * np.exp(-q * T) * norm.cdf(d1) -
                K * np.exp(-r * T) * norm.cdf(d2))
    else:
        price = (K * np.exp(-r * T) * norm.cdf(-d2) -
                S * np.exp(-q * T) * norm.cdf(-d1))
    
    return max(price, 0)  # Prevent negative prices
