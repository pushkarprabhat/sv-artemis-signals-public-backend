"""
core/forex_trading.py — FOREX/CURRENCY PAIR TRADING STRATEGIES
===============================================================
Mean reversion and momentum strategies for currency pairs:
- USDINR: US Dollar vs Indian Rupee (Primary)
- EURINR: Euro vs Indian Rupee
- GBPINR: British Pound vs Indian Rupee
- JPYINR: Japanese Yen vs Indian Rupee

Forex markets have unique characteristics:
- 24/5 liquidity with different sessions (Asian, European, American)
- Strong macroeconomic drivers (RBI, Fed, ECB decisions)
- Interest rate differentials crucial (carry trade)
- Central bank interventions common
- High correlation with equity markets (risk sentiment)
"""

import pandas as pd
import numpy as np
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Forex pair definitions
FOREX_PAIRS = {
    'USDINR': {
        'description': 'US Dollar vs Indian Rupee (Primary)',
        'quote_currency': 'INR',
        'base_currency': 'USD',
        'typical_daily_range': 0.30,  # Typical daily move in paise
        'trading_volume': 'Very High',
        'risk_factors': ['Fed policy', 'RBI decisions', 'US-India trade', 'Oil prices', 'Capital flows']
    },
    'EURINR': {
        'description': 'Euro vs Indian Rupee',
        'quote_currency': 'INR',
        'base_currency': 'EUR',
        'typical_daily_range': 0.40,
        'trading_volume': 'High',
        'risk_factors': ['ECB policy', 'RBI decisions', 'Eurozone data', 'Capital flows']
    },
    'GBPINR': {
        'description': 'British Pound vs Indian Rupee',
        'quote_currency': 'INR',
        'base_currency': 'GBP',
        'typical_daily_range': 0.50,
        'trading_volume': 'Medium-High',
        'risk_factors': ['BoE policy', 'RBI decisions', 'UK economic data', 'Brexit implications']
    },
    'JPYINR': {
        'description': 'Japanese Yen vs Indian Rupee',
        'quote_currency': 'INR',
        'base_currency': 'JPY',
        'typical_daily_range': 0.60,
        'trading_volume': 'Medium',
        'risk_factors': ['BoJ policy', 'RBI decisions', 'Carry trade flows', 'China tensions']
    },
}

# Forex-specific parameters
FOREX_CONFIG = {
    'USDINR': {'pip_value': 0.0001, 'standard_lot': 100000, 'min_lot': 1000},
    'EURINR': {'pip_value': 0.0001, 'standard_lot': 100000, 'min_lot': 1000},
    'GBPINR': {'pip_value': 0.0001, 'standard_lot': 100000, 'min_lot': 1000},
    'JPYINR': {'pip_value': 0.01, 'standard_lot': 100000, 'min_lot': 1000},
}


def calculate_ppp_fair_value(currency_pair, relative_inflation_rates, current_rate):
    """
    Calculate Purchasing Power Parity (PPP) fair value for currency pair.
    Used for mean reversion trades.
    
    Args:
        currency_pair: Pair name (e.g., 'USDINR')
        relative_inflation_rates: (rate_base_currency - rate_quote_currency) in %
        current_rate: Current exchange rate
    
    Returns:
        Fair value and deviation from PPP
    """
    try:
        # PPP formula: New Rate = Current Rate × (1 + Inflation_Base) / (1 + Inflation_Quote)
        ppp_rate = current_rate * (1 + relative_inflation_rates / 100)
        
        # Deviation from fair value (%)
        deviation = ((current_rate - ppp_rate) / ppp_rate) * 100
        
        return {
            'currency_pair': currency_pair,
            'current_rate': current_rate,
            'ppp_fair_value': ppp_rate,
            'deviation_pct': deviation,
            'signal': 'BUY' if deviation < -2 else ('SELL' if deviation > 2 else 'NEUTRAL'),
            'overvalued': current_rate > ppp_rate,
            'undervalued': current_rate < ppp_rate,
        }
    except Exception as e:
        print(f"Error in PPP calculation: {e}")
        return None


def calculate_interest_rate_differential(currency_pair, base_currency_rate, quote_currency_rate):
    """
    Calculate interest rate differential (carry trade driver).
    
    Args:
        currency_pair: Pair name (e.g., 'USDINR')
        base_currency_rate: Rate for base currency (USD, EUR, etc.)
        quote_currency_rate: Rate for quote currency (INR)
    
    Returns:
        Interest rate differential and carry trade implications
    """
    try:
        # Carry differential (positive = profitable carry)
        differential = quote_currency_rate - base_currency_rate
        
        # Annualized carry (per ₹1 lakh)
        carry_per_100k = differential * 1000  # Assuming ₹100,000 position
        
        return {
            'currency_pair': currency_pair,
            'base_rate': base_currency_rate,
            'quote_rate': quote_currency_rate,
            'differential': differential,
            'carry_annualized_per_100k': carry_per_100k,
            'is_positive_carry': differential > 0,
            'carry_signal': 'BUY' if differential > 1.5 else ('SELL' if differential < -1.5 else 'NEUTRAL'),
        }
    except Exception as e:
        print(f"Error in carry calculation: {e}")
        return None


def calculate_forex_momentum(price_data, lookback=20):
    """
    Calculate momentum for forex pair using rate of change.
    
    Args:
        price_data: Price series
        lookback: Number of periods for momentum calculation
    
    Returns:
        dict with momentum metrics
    """
    try:
        # Rate of change
        roc = ((price_data.iloc[-1] - price_data.iloc[-lookback]) / price_data.iloc[-lookback]) * 100
        
        # Momentum strength (returns per unit volatility)
        returns = price_data.pct_change()
        momentum_strength = returns.iloc[-lookback:].mean() / returns.iloc[-lookback:].std() if returns.iloc[-lookback:].std() > 0 else 0
        
        # Directional indicator
        ma_20 = price_data.iloc[-20:].mean()
        ma_50 = price_data.iloc[-50:].mean() if len(price_data) >= 50 else ma_20
        
        return {
            'roc': roc,
            'momentum_strength': momentum_strength,
            'signal': 'BUY' if roc > 0.5 else ('SELL' if roc < -0.5 else 'NEUTRAL'),
            'trend': 'UPTREND' if price_data.iloc[-1] > ma_20 > ma_50 else 'DOWNTREND',
            'ma_20': ma_20,
            'ma_50': ma_50,
        }
    except Exception as e:
        print(f"Error in momentum calculation: {e}")
        return None


def calculate_forex_mean_reversion(price_data, lookback=20):
    """
    Calculate mean reversion signal for forex pair.
    
    Args:
        price_data: Price series
        lookback: Number of periods for mean calculation
    
    Returns:
        dict with mean reversion metrics
    """
    try:
        # Calculate moving average and standard deviation
        ma = price_data.iloc[-lookback:].mean()
        std = price_data.iloc[-lookback:].std()
        
        # Z-score (how many std devs away from mean)
        z_score = (price_data.iloc[-1] - ma) / std if std > 0 else 0
        
        # Bollinger Bands
        upper_band = ma + (2 * std)
        lower_band = ma - (2 * std)
        
        return {
            'current_price': price_data.iloc[-1],
            'moving_average': ma,
            'std_deviation': std,
            'z_score': z_score,
            'upper_band': upper_band,
            'lower_band': lower_band,
            'signal': 'BUY' if z_score < -1.5 else ('SELL' if z_score > 1.5 else 'NEUTRAL'),
            'overextended': abs(z_score) > 2,
        }
    except Exception as e:
        print(f"Error in mean reversion calculation: {e}")
        return None


def scan_forex_pairs(forex_data_dict, strategy='momentum'):
    """
    Scan forex pairs for trading opportunities.
    
    Args:
        forex_data_dict: Dict of {pair_name: price_series}
        strategy: 'momentum' or 'mean_reversion'
    
    Returns:
        DataFrame with forex pairs and signals
    """
    results = []
    
    for pair_name, price_series in forex_data_dict.items():
        try:
            if strategy == 'momentum':
                analysis = calculate_forex_momentum(price_series)
            else:
                analysis = calculate_forex_mean_reversion(price_series)
            
            if analysis:
                analysis['Pair'] = pair_name
                analysis['Strategy'] = strategy
                results.append(analysis)
        except Exception as e:
            continue
    
    if results:
        df = pd.DataFrame(results)
        
        # Calculate ML score
        if strategy == 'momentum':
            df['ML_Score'] = abs(df['momentum_strength']) * 50
        else:
            df['ML_Score'] = abs(df['z_score']) * 30
        
        return df.sort_values('ML_Score', ascending=False)
    
    return pd.DataFrame()


def estimate_forex_capital_requirement(pair_name, quantity_units, current_rate):
    """
    Calculate capital requirement for forex trade.
    
    Args:
        pair_name: Currency pair name (e.g., 'USDINR')
        quantity_units: Number of units (in thousands)
        current_rate: Current exchange rate
    
    Returns:
        Capital requirement and margin requirement
    """
    try:
        config = FOREX_CONFIG.get(pair_name, {})
        
        # Notional value
        notional_value = quantity_units * 1000 * current_rate
        
        # Margin requirement (typically 2-5% for forex)
        margin_percent = 0.03  # 3% margin
        margin_required = notional_value * margin_percent
        
        return {
            'pair': pair_name,
            'quantity': quantity_units * 1000,
            'rate': current_rate,
            'notional_value': notional_value,
            'margin_required': margin_required,
            'capital_efficiency': notional_value / margin_required,
            'pip_value': config.get('pip_value', 0.0001),
        }
    except Exception as e:
        print(f"Error in capital calculation: {e}")
        return None


def identify_central_bank_risks(pair_name, upcoming_events):
    """
    Identify central bank event risks for forex pair.
    
    Args:
        pair_name: Currency pair name
        upcoming_events: List of upcoming event dates and types
    
    Returns:
        Risk assessment for the pair
    """
    pair_config = FOREX_PAIRS.get(pair_name, {})
    
    risk_factors = pair_config.get('risk_factors', [])
    
    # Check if any upcoming events match risk factors
    matching_risks = []
    for event in upcoming_events:
        if any(factor.lower() in event.lower() for factor in risk_factors):
            matching_risks.append(event)
    
    return {
        'pair': pair_name,
        'overall_risk_factors': risk_factors,
        'upcoming_event_risks': matching_risks,
        'high_risk': len(matching_risks) > 0,
        'recommended_action': 'REDUCE_POSITION' if len(matching_risks) > 1 else 'MONITOR',
    }


def get_forex_session_timing():
    """
    Get forex trading session timings (IST).
    
    Returns:
        dict with session information
    """
    return {
        'Asian_Session': {
            'start': '08:00 IST',
            'end': '12:30 IST',
            'primary_currencies': ['JPY', 'AUD', 'NZD'],
            'volatility': 'Low-Medium'
        },
        'European_Session': {
            'start': '12:30 IST',
            'end': '19:00 IST',
            'primary_currencies': ['EUR', 'GBP'],
            'volatility': 'High'
        },
        'American_Session': {
            'start': '19:00 IST',
            'end': '02:00 IST',
            'primary_currencies': ['USD'],
            'volatility': 'Very High'
        },
        'Overlaps': [
            'Asian-European: 12:30-12:45 IST (Medium volatility)',
            'European-American: 19:00-19:30 IST (Very High volatility)',
        ]
    }


if __name__ == '__main__':
    print("""
    ╔════════════════════════════════════════════════════════════╗
    ║        FOREX TRADING STRATEGIES MODULE                     ║
    ║   Mean Reversion & Momentum for Currency Pairs             ║
    ╚════════════════════════════════════════════════════════════╝
    
    Available Functions:
    • calculate_ppp_fair_value() - PPP-based valuation
    • calculate_interest_rate_differential() - Carry trade analysis
    • calculate_forex_momentum() - Momentum calculation
    • calculate_forex_mean_reversion() - Mean reversion signals
    • scan_forex_pairs() - Find tradeable currency pairs
    • estimate_forex_capital_requirement() - Calculate margin needed
    • identify_central_bank_risks() - RBI/Fed/ECB event risk
    • get_forex_session_timing() - Trading session info
    
    Supported Forex Pairs (INR-based):
    • USDINR: US Dollar vs Indian Rupee (Primary)
    • EURINR: Euro vs Indian Rupee
    • GBPINR: British Pound vs Indian Rupee
    • JPYINR: Japanese Yen vs Indian Rupee
    
    Key Drivers:
    - Interest rate differentials (carry trades)
    - PPP deviations (mean reversion)
    - Momentum/Trend following
    - Central bank policy decisions
    """)
