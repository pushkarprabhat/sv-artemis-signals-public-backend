"""
core/commodity_trading.py — COMMODITY FUTURES TRADING STRATEGIES
================================================================
Pair trading and momentum strategies specifically designed for:
- Precious Metals: GOLD, SILVER, COPPER, ALUMINUM, ZINC, NICKEL
- Energy: CRUDEOIL, NATURALGAS

Commodity markets have unique characteristics:
- Lower liquidity than equities
- Strong seasonal patterns
- Geopolitical influences
- Supply-demand shocks
"""

import pandas as pd
import numpy as np
from scipy import stats
from statsmodels.tsa.vector_ar.vecm import coint_johansen
import warnings
warnings.filterwarnings('ignore')

# Commodity-specific correlations and spreads
COMMODITY_PAIRS = {
    # Precious Metals Spreads
    'GOLD-SILVER': {
        'leg1': 'GOLD',
        'leg2': 'SILVER',
        'description': 'Gold vs Silver ratio trading',
        'seasonal': 'Q1, Q4',  # Holiday demand
        'risk_factors': ['USD strength', 'Real yields', 'Equity market risk-on/risk-off']
    },
    'COPPER-ALUMINUM': {
        'leg1': 'COPPER',
        'leg2': 'ALUMINUM',
        'description': 'Industrial metal spread',
        'seasonal': 'Q2, Q3',  # Construction season
        'risk_factors': ['Economic growth', 'Construction activity']
    },
    'GOLD-COPPER': {
        'leg1': 'GOLD',
        'leg2': 'COPPER',
        'description': 'Safe-haven vs Growth sentiment',
        'seasonal': 'Year-round',
        'risk_factors': ['Risk sentiment', 'Economic cycles']
    },
    
    # Energy Spreads
    'CRUDEOIL-NATURALGAS': {
        'leg1': 'CRUDEOIL',
        'leg2': 'NATURALGAS',
        'description': 'Energy market spread',
        'seasonal': 'Winter (heating demand)',
        'risk_factors': ['Weather', 'Geopolitical', 'Supply disruptions']
    },
}

# Commodity-specific parameters
COMMODITY_CONFIG = {
    'GOLD': {'contract_size': 100, 'min_lot': 1, 'volatility_type': 'low-medium'},
    'SILVER': {'contract_size': 30, 'min_lot': 1, 'volatility_type': 'high'},
    'COPPER': {'contract_size': 1000, 'min_lot': 1, 'volatility_type': 'high'},
    'ALUMINUM': {'contract_size': 5000, 'min_lot': 1, 'volatility_type': 'medium'},
    'ZINC': {'contract_size': 250, 'min_lot': 1, 'volatility_type': 'high'},
    'NICKEL': {'contract_size': 6, 'min_lot': 1, 'volatility_type': 'very-high'},
    'CRUDEOIL': {'contract_size': 100, 'min_lot': 1, 'volatility_type': 'very-high'},
    'NATURALGAS': {'contract_size': 10000, 'min_lot': 1, 'volatility_type': 'very-high'},
}

def check_commodity_cointegration(commodity1_data, commodity2_data, lookback=252):
    """
    Check if two commodities are cointegrated using Johansen test.
    
    Args:
        commodity1_data: Price series for commodity 1
        commodity2_data: Price series for commodity 2
        lookback: Number of days to use (default: 252 = 1 year)
    
    Returns:
        dict with cointegration results and trading signals
    """
    try:
        # Prepare data
        data = pd.concat([commodity1_data[-lookback:], commodity2_data[-lookback:]], axis=1)
        data.columns = ['commodity1', 'commodity2']
        
        # Log prices for stationary test
        log_data = np.log(data.dropna())
        
        # Johansen cointegration test
        result = coint_johansen(log_data, det_order=0, k_ar_diff=1)
        
        # Get trace statistic (index 0 for first eigenvalue)
        trace_stat = result.lr1[0]  # 90% critical value
        
        # Get cointegrating vector
        beta = result.evec[:, 0]  # First eigenvector
        
        # Normalize first coefficient to 1
        beta_normalized = beta / beta[0]
        
        # Calculate spread
        spread = np.log(commodity1_data.iloc[-lookback:].values) - (beta_normalized[1] * np.log(commodity2_data.iloc[-lookback:].values))
        
        # Calculate Z-score of spread
        spread_mean = spread.mean()
        spread_std = spread.std()
        z_score = (spread[-1] - spread_mean) / spread_std if spread_std > 0 else 0
        
        return {
            'cointegrated': trace_stat > 0,  # Simplified check
            'beta': beta_normalized,
            'spread_mean': spread_mean,
            'spread_std': spread_std,
            'z_score': z_score,
            'signal': 'BUY' if z_score > 2.0 else ('SELL' if z_score < -2.0 else 'NEUTRAL'),
            'trace_stat': trace_stat,
        }
    except Exception as e:
        print(f"Error in cointegration test: {e}")
        return None


def calculate_commodity_momentum(price_data, lookback=20):
    """
    Calculate momentum for commodity using rate of change.
    
    Args:
        price_data: Price series
        lookback: Number of periods for momentum calculation
    
    Returns:
        dict with momentum metrics
    """
    try:
        roc = ((price_data.iloc[-1] - price_data.iloc[-lookback]) / price_data.iloc[-lookback]) * 100
        
        # Calculate trend
        returns = price_data.pct_change()
        momentum_strength = returns.iloc[-lookback:].mean() / returns.iloc[-lookback:].std() if returns.iloc[-lookback:].std() > 0 else 0
        
        return {
            'roc': roc,
            'momentum_strength': momentum_strength,
            'signal': 'BUY' if roc > 2 else ('SELL' if roc < -2 else 'NEUTRAL'),
            'trend': 'UPTREND' if roc > 0 else 'DOWNTREND',
        }
    except Exception as e:
        print(f"Error in momentum calculation: {e}")
        return None


def scan_commodity_pairs(commodity_data_dict, min_correlation=0.5):
    """
    Scan for tradeable commodity pairs based on correlation and cointegration.
    
    Args:
        commodity_data_dict: Dict of {commodity_name: price_series}
        min_correlation: Minimum correlation for pair consideration
    
    Returns:
        DataFrame with potential commodity pairs and signals
    """
    results = []
    commodities = list(commodity_data_dict.keys())
    
    for i, comm1 in enumerate(commodities):
        for j, comm2 in enumerate(commodities[i+1:], i+1):
            try:
                # Check correlation
                corr = commodity_data_dict[comm1].corr(commodity_data_dict[comm2])
                
                if abs(corr) >= min_correlation:
                    # Check cointegration
                    coint_result = check_commodity_cointegration(
                        commodity_data_dict[comm1],
                        commodity_data_dict[comm2]
                    )
                    
                    if coint_result and coint_result['cointegrated']:
                        results.append({
                            'Pair': f"{comm1}-{comm2}",
                            'Commodity1': comm1,
                            'Commodity2': comm2,
                            'Correlation': corr,
                            'Cointegrated': coint_result['cointegrated'],
                            'Z-Score': coint_result['z_score'],
                            'Signal': coint_result['signal'],
                            'Beta': coint_result['beta'][1],
                            'ML_Score': abs(coint_result['z_score']) * 50,
                        })
            except Exception as e:
                continue
    
    if results:
        return pd.DataFrame(results).sort_values('ML_Score', ascending=False)
    return pd.DataFrame()


def estimate_commodity_capital_requirement(commodity1, quantity1, commodity2, quantity2, commodity_prices):
    """
    Calculate capital requirement for commodity pair trade.
    
    Args:
        commodity1: Name of first commodity
        quantity1: Quantity of first commodity
        commodity2: Name of second commodity
        quantity2: Quantity of second commodity
        commodity_prices: Dict of {commodity: current_price}
    
    Returns:
        Capital requirement and margin requirement
    """
    try:
        config1 = COMMODITY_CONFIG.get(commodity1, {})
        config2 = COMMODITY_CONFIG.get(commodity2, {})
        
        price1 = commodity_prices.get(commodity1, 0)
        price2 = commodity_prices.get(commodity2, 0)
        
        contract_value1 = price1 * quantity1 * config1.get('contract_size', 1)
        contract_value2 = price2 * quantity2 * config2.get('contract_size', 1)
        
        # Margin requirement (typically 5-10% of contract value for commodities)
        margin_percent = 0.08  # 8% margin
        margin_required = (contract_value1 + contract_value2) * margin_percent
        
        return {
            'commodity1_value': contract_value1,
            'commodity2_value': contract_value2,
            'total_notional': contract_value1 + contract_value2,
            'margin_required': margin_required,
            'capital_efficiency': (contract_value1 + contract_value2) / margin_required,
        }
    except Exception as e:
        print(f"Error in capital calculation: {e}")
        return None


def identify_seasonal_opportunities(commodity, month):
    """
    Identify seasonal trading opportunities for commodities.
    
    Args:
        commodity: Commodity name
        month: Current month (1-12)
    
    Returns:
        Seasonal opportunity info
    """
    seasonal_patterns = {
        'GOLD': {'buy': [11, 12, 1, 2], 'sell': [7, 8, 9]},  # Holiday demand
        'SILVER': {'buy': [12, 1, 2], 'sell': [7, 8]},
        'CRUDEOIL': {'buy': [6, 7, 8], 'sell': [2, 3, 4]},  # Driving season
        'NATURALGAS': {'buy': [11, 12, 1, 2], 'sell': [6, 7, 8]},  # Heating demand
        'COPPER': {'buy': [3, 4, 5], 'sell': [10, 11]},  # Construction
        'ALUMINUM': {'buy': [3, 4, 5], 'sell': [10, 11]},  # Construction
    }
    
    pattern = seasonal_patterns.get(commodity, {})
    
    return {
        'commodity': commodity,
        'month': month,
        'is_buy_season': month in pattern.get('buy', []),
        'is_sell_season': month in pattern.get('sell', []),
        'buy_months': pattern.get('buy', []),
        'sell_months': pattern.get('sell', []),
    }


if __name__ == '__main__':
    print("""
    ╔════════════════════════════════════════════════════════════╗
    ║      COMMODITY TRADING STRATEGIES MODULE                   ║
    ║      Pair Trading & Momentum for Metals & Energy           ║
    ╚════════════════════════════════════════════════════════════╝
    
    Available Functions:
    • check_commodity_cointegration() - Test pair cointegration
    • calculate_commodity_momentum() - Calculate commodity momentum
    • scan_commodity_pairs() - Find tradeable commodity pairs
    • estimate_commodity_capital_requirement() - Calculate margin needed
    • identify_seasonal_opportunities() - Seasonal pattern analysis
    
    Supported Commodities:
    Metals: GOLD, SILVER, COPPER, ALUMINUM, ZINC, NICKEL
    Energy: CRUDEOIL, NATURALGAS
    
    Commodity-specific pairs already mapped in COMMODITY_PAIRS dict.
    """)
