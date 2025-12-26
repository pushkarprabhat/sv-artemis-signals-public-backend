# core/volatility.py — COMPLETE INSTITUTIONAL VOLATILITY ENGINE
# GARCH(1,1) + Historical Volatility + India VIX + Full Explanation
# Professional quantitative volatility modeling framework

import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from config import BASE_DIR
from utils.logger import logger

try:
    from arch import arch_model
    HAS_ARCH = True
except ImportError:
    HAS_ARCH = False
    logger.warning("arch package not installed. GARCH will not be available. Install with: pip install arch")


# ============================================================================
# LIVE VIX & IV CRUSH DETECTION
# ============================================================================

def get_vix_level():
    """Fetch live INDIA VIX from NSE"""
    try:
        from utils.kite_worker import get_kite
        kite = get_kite()
        vix_quote = kite.quote('NSEIND:INDIA VIX')
        if 'INDIA VIX' in vix_quote and 'last_price' in vix_quote['INDIA VIX']:
            return vix_quote['INDIA VIX']['last_price']
    except Exception as e:
        logger.warning(f"Could not fetch VIX: {e}")
    return None

def is_iv_crush_setup(vix_level=None):
    """Check if current VIX conditions favor IV Crush"""
    if vix_level is None:
        vix_level = get_vix_level()
    
    if vix_level is None:
        return False, "VIX data unavailable"
    
    # IV Crush setup when VIX is elevated (>16) and showing weakness
    if vix_level > 16:
        return True, f"🚀 IV Crush Setup: VIX = {vix_level:.2f}"
    elif vix_level > 14:
        return True, f"⚡ Potential IV Crush: VIX = {vix_level:.2f}"
    else:
        return False, f"📊 Low Vol Environment: VIX = {vix_level:.2f}"

# ============================================================================
# HISTORICAL VOLATILITY & GARCH
# ============================================================================


def historical_volatility(returns: pd.Series, window: int = 20) -> float:
    """
    Historical Volatility (HV) — Realized volatility over past N days
    
    - HV20 = 20-day (most used by quants)
    - HV60 = 60-day  
    - HV252 = 1-year (annual volatility)
    
    Formula: σ = std(log_returns) * sqrt(252)
    
    Annualized using sqrt(252 trading days per year)
    
    With love: "HV remembers what happened. We learn from history to avoid repeating it."
    """
    try:
        if returns is None or len(returns) < window:
            return np.nan
        
        # Calculate log returns
        log_ret = np.log(1 + returns)
        
        # Compute rolling standard deviation and annualize
        vol = log_ret.rolling(window).std().iloc[-1]
        annual_vol = vol * np.sqrt(252)
        
        return round(annual_vol, 4)
    except Exception as e:
        logger.debug(f"Error calculating historical volatility: {e}")
        return np.nan


def garch_volatility(returns: pd.Series) -> float:
    """
    GARCH(1,1) — THE GOLD STANDARD OF VOLATILITY FORECASTING
    
    Used by BlackRock, Citadel, Jane Street, RBI, and professional traders worldwide
    
    Why GARCH is better than simple HV:
    • It remembers past volatility shocks (volatility clustering)
    • It predicts tomorrow's volatility, not just describes yesterday
    • It knows "volatility clusters" — high vol days come in groups like monsoon rains
    • Used for entry signals when vol is expected to spike
    
    Formula:
    σ²_t = α₀ + α₁ × ε²_{t-1} + β₁ × σ²_{t-1}
    
    Where:
    - α₀ = constant (long-run average volatility)
    - α₁ = weight on recent shocks (news impact)
    - β₁ = weight on past volatility (persistence)
    - ε_{t-1} = recent return surprise
    - σ²_{t-1} = yesterday's volatility
    
    We use GARCH to:
    • Avoid trading during volatility explosions (α₁ spike)
    • Sell options ONLY when GARCH is low but IV Rank is high → MAXIMUM EDGE
    • Protect capital during monsoon seasons (market panic)
    
    With love: "GARCH is like a father who remembers past storms. 
    He knows they come in clusters, so he prepares before the next rain."
    """
    if not HAS_ARCH:
        logger.warning("GARCH requires 'arch' package. Using historical volatility instead.")
        return historical_volatility(returns, window=20)
    
    try:
        if returns is None or len(returns) < 100:
            return np.nan
        
        # Remove any NaN values
        returns_clean = returns.dropna()
        if len(returns_clean) < 100:
            return np.nan
        
        # Scale returns for numerical stability (GARCH prefers percentage returns)
        scaled_returns = returns_clean * 100
        
        # Fit GARCH(1,1) model
        model = arch_model(scaled_returns, vol='Garch', p=1, q=1, dist='Normal')
        result = model.fit(disp='off', show_warning=False)
        
        # Forecast next day volatility
        forecast = result.forecast(horizon=1)
        vol_forecast_next_day = np.sqrt(forecast.variance.values[-1, 0])
        
        # Convert back to annualized volatility (undo scaling and annualize)
        annual_vol = (vol_forecast_next_day / 100) * np.sqrt(252)
        
        return round(annual_vol, 4)
    
    except Exception as e:
        logger.debug(f"GARCH calculation failed: {e}. Falling back to HV.")
        return historical_volatility(returns, window=20)


def ewma_volatility(returns: pd.Series, span: int = 20) -> float:
    """
    EWMA (Exponential Weighted Moving Average) Volatility
    
    Simpler than GARCH but still effective for day-to-day risk management
    
    Formula:
    σ_t = √[ λ × σ²_{t-1} + (1-λ) × r²_{t-1} ]
    
    Typically λ = 0.94 (RiskMetrics standard)
    
    Advantages:
    • Fast to calculate (no optimization needed)
    • Reacts quickly to recent market moves
    • Good for intraday risk monitoring
    
    Disadvantages:
    • Doesn't capture volatility clustering as well as GARCH
    • Weights all past data exponentially (recent more important)
    
    With love: "EWMA is like a quick check. It tells us today's weather,
    but GARCH warns us about tomorrow's monsoon."
    """
    try:
        if returns is None or len(returns) < span:
            return np.nan
        
        # Calculate exponential weights (span is half-life parameter)
        log_returns_sq = np.log(1 + returns) ** 2
        
        # Use pandas ewm (exponential weighted moving average)
        ewma_var = log_returns_sq.ewm(span=span).mean()
        ewma_vol = np.sqrt(ewma_var.iloc[-1])
        
        # Annualize
        annual_vol = ewma_vol * np.sqrt(252)
        
        return round(annual_vol, 4)
    except Exception as e:
        logger.debug(f"Error calculating EWMA volatility: {e}")
        return np.nan


def india_vix_signal(current_vix: float) -> str:
    """
    India VIX = Fear Gauge of Indian Equities Market
    
    Interpretation:
    - Below 15 → Extreme calm → BUY momentum strategies (complacency)
    - 15–25 → Normal → Use regular strategy sizing
    - 25–35 → High fear → SELL OPTIONS (IV Rank > 80%)
    - Above 35 → Panic → SELL STRANGLES AGGRESSIVELY (fear premium)
    
    With love: "When fear is high, we sell fear. When calm is here, we collect slow premium."
    """
    try:
        if current_vix > 35:
            return "🔴 MARKET IN PANIC (VIX > 35) — SELL STRANGLES NOW — Max premium available"
        elif current_vix > 25:
            return "🟠 High Fear (VIX 25-35) — Strong Sell Signal — IV Rank bias to sell"
        elif current_vix < 15:
            return "🟢 Extreme Calm (VIX < 15) — Buy Momentum — Consider strangle shorts risky"
        else:
            return "🟡 Normal Market (VIX 15-25) — Regular Strategy"
    except Exception as e:
        logger.debug(f"Error interpreting VIX: {e}")
        return "Unknown VIX reading"


def volatility_regime(hv: float, garch: float, ewma: float) -> str:
    """
    Classify volatility regime for strategy selection
    
    Args:
        hv: Historical volatility (20-day annualized)
        garch: GARCH forecast volatility
        ewma: EWMA volatility
    
    Returns:
        Regime string: "LOW", "NORMAL", "HIGH", "SPIKE"
    
    With love: "Know the market weather before you trade the rain."
    """
    try:
        avg_vol = np.nanmean([hv, garch, ewma])
        
        # Calculate regime thresholds based on 252-day average
        if garch > avg_vol * 1.5:
            return "SPIKE"  # Vol explosion expected
        elif avg_vol < 15:
            return "LOW"    # Calm market
        elif avg_vol < 25:
            return "NORMAL" # Typical regime
        else:
            return "HIGH"   # Elevated vol
    except:
        return "UNKNOWN"


def simulate_volatility_data(days: int = 500) -> pd.DataFrame:
    """
    Simulate historical price data for testing volatility calculations
    
    With love: "Dry runs teach us before real fire tests our courage."
    """
    try:
        dates = pd.date_range("2023-01-01", periods=days, freq='D')
        np.random.seed(42)
        
        # Simulate with volatility clustering (realistic)
        returns = []
        vol_state = 0.02
        for _ in range(days):
            # GARCH-like behavior: vol clustering
            shock = np.random.normal(0, vol_state)
            returns.append(shock)
            vol_state = np.sqrt(0.00001 + 0.05 * shock**2 + 0.94 * vol_state**2)
        
        price = 25000 * np.exp(np.cumsum(returns))
        
        df = pd.DataFrame({
            'date': dates,
            'open': price * np.random.uniform(0.99, 1.01, len(price)),
            'high': price * np.random.uniform(1.00, 1.03, len(price)),
            'low': price * np.random.uniform(0.97, 1.00, len(price)),
            'close': price,
            'volume': np.random.randint(100000, 1000000, len(price))
        })
        
        return df.set_index('date')
    except Exception as e:
        logger.error(f"Error simulating volatility data: {e}")
        return pd.DataFrame()


if __name__ == "__main__":
    # Test the volatility module
    print("Testing volatility calculations...")
    
    # Simulate data
    df = simulate_volatility_data()
    returns = df['close'].pct_change().dropna()
    
    # Calculate volatilities
    hv20 = historical_volatility(returns, window=20)
    hv60 = historical_volatility(returns, window=60)
    hv252 = historical_volatility(returns, window=252)
    garch_vol = garch_volatility(returns)
    ewma_vol = ewma_volatility(returns)
    
    print(f"HV20: {hv20:.4f}")
    print(f"HV60: {hv60:.4f}")
    print(f"HV252: {hv252:.4f}")
    print(f"GARCH: {garch_vol:.4f}")
    print(f"EWMA: {ewma_vol:.4f}")
    
    # Test regime
    regime = volatility_regime(hv20, garch_vol, ewma_vol)
    print(f"Volatility Regime: {regime}")
    
    # Test VIX signal (simulated)
    print(f"VIX Signal (20): {india_vix_signal(20)}")
    print(f"VIX Signal (35): {india_vix_signal(35)}")
