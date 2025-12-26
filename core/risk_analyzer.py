"""
Risk Measures & Analytics Module

Comprehensive risk metrics calculation:
- Value at Risk (VaR) - Parametric & Historical
- Conditional Value at Risk (CVaR/Expected Shortfall)
- Sharpe Ratio, Sortino Ratio, Calmar Ratio
- Maximum Drawdown & Drawdown Analysis
- Volatility Metrics (Historical, Exponential, GARCH)
- Correlation & Beta Analysis
- Tail Risk Metrics
"""

import numpy as np
import pandas as pd
from scipy.stats import norm, t
from scipy import optimize
from typing import Dict, Tuple, Optional, List
from dataclasses import dataclass
import warnings

warnings.filterwarnings('ignore')


@dataclass
class RiskMetrics:
    """Container for comprehensive risk metrics"""
    # Return metrics
    total_return: float
    annual_return: float
    
    # Volatility metrics
    daily_volatility: float
    annual_volatility: float
    
    # Risk-adjusted returns
    sharpe_ratio: float
    sortino_ratio: float
    calmar_ratio: float
    
    # Downside metrics
    max_drawdown: float
    avg_drawdown: float
    drawdown_duration: int
    
    # Tail risk
    var_95: float  # 95% confidence
    var_99: float  # 99% confidence
    cvar_95: float  # Conditional VaR
    cvar_99: float  # Conditional VaR
    
    # Distribution metrics
    skewness: float
    kurtosis: float
    
    # Other metrics
    win_rate: float
    profit_factor: float
    recovery_factor: float
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for display"""
        return {
            'Total Return': f"{self.total_return:.2%}",
            'Annual Return': f"{self.annual_return:.2%}",
            'Daily Volatility': f"{self.daily_volatility:.2%}",
            'Annual Volatility': f"{self.annual_volatility:.2%}",
            'Sharpe Ratio': f"{self.sharpe_ratio:.3f}",
            'Sortino Ratio': f"{self.sortino_ratio:.3f}",
            'Calmar Ratio': f"{self.calmar_ratio:.3f}",
            'Max Drawdown': f"{self.max_drawdown:.2%}",
            'Average Drawdown': f"{self.avg_drawdown:.2%}",
            'Drawdown Duration': f"{self.drawdown_duration} days",
            'VaR (95%)': f"{self.var_95:.2%}",
            'VaR (99%)': f"{self.var_99:.2%}",
            'CVaR (95%)': f"{self.cvar_95:.2%}",
            'CVaR (99%)': f"{self.cvar_99:.2%}",
            'Skewness': f"{self.skewness:.3f}",
            'Kurtosis': f"{self.kurtosis:.3f}",
            'Win Rate': f"{self.win_rate:.2%}",
            'Profit Factor': f"{self.profit_factor:.2f}",
            'Recovery Factor': f"{self.recovery_factor:.2f}",
        }


class RiskAnalyzer:
    """
    Comprehensive risk analysis engine
    """
    
    def __init__(self, 
                 returns: pd.Series,
                 prices: Optional[pd.Series] = None,
                 risk_free_rate: float = 0.06,
                 confidence_level: float = 0.95):
        """
        Initialize risk analyzer
        
        Args:
            returns: Series of returns (daily/weekly/monthly)
            prices: Series of prices (used for drawdown, alternative to returns)
            risk_free_rate: Annual risk-free rate
            confidence_level: For VaR calculation (e.g., 0.95 for 95%)
        """
        self.returns = returns
        self.prices = prices if prices is not None else (1 + returns).cumprod()
        self.risk_free_rate = risk_free_rate
        self.confidence_level = confidence_level
        
        # Detect frequency
        self.freq = self._detect_frequency()
    
    def _detect_frequency(self) -> int:
        """Detect return frequency (252=daily, 52=weekly, 12=monthly)"""
        if len(self.returns) > 1000:
            return 252
        elif len(self.returns) > 250:
            return 52
        else:
            return 12
    
    def calculate_all_metrics(self) -> RiskMetrics:
        """Calculate all risk metrics"""
        return RiskMetrics(
            total_return=self._total_return(),
            annual_return=self._annual_return(),
            daily_volatility=self.returns.std(),
            annual_volatility=self.returns.std() * np.sqrt(self.freq),
            sharpe_ratio=self._sharpe_ratio(),
            sortino_ratio=self._sortino_ratio(),
            calmar_ratio=self._calmar_ratio(),
            max_drawdown=self._max_drawdown(),
            avg_drawdown=self._avg_drawdown(),
            drawdown_duration=self._drawdown_duration(),
            var_95=self._var(0.95),
            var_99=self._var(0.99),
            cvar_95=self._cvar(0.95),
            cvar_99=self._cvar(0.99),
            skewness=self._skewness(),
            kurtosis=self._kurtosis(),
            win_rate=self._win_rate(),
            profit_factor=self._profit_factor(),
            recovery_factor=self._recovery_factor(),
        )
    
    # Return metrics
    def _total_return(self) -> float:
        """Total cumulative return"""
        return (1 + self.returns).prod() - 1
    
    def _annual_return(self) -> float:
        """Annualized return"""
        total_ret = self._total_return()
        years = len(self.returns) / self.freq
        return (1 + total_ret) ** (1 / years) - 1 if years > 0 else 0
    
    # Risk-adjusted return metrics
    def _sharpe_ratio(self) -> float:
        """
        Sharpe Ratio = (Portfolio Return - Risk-Free Rate) / Portfolio Volatility
        Measures excess return per unit of risk
        """
        annual_return = self._annual_return()
        annual_vol = self.returns.std() * np.sqrt(self.freq)
        
        if annual_vol == 0:
            return 0
        
        return (annual_return - self.risk_free_rate) / annual_vol
    
    def _sortino_ratio(self) -> float:
        """
        Sortino Ratio = (Portfolio Return - Risk-Free Rate) / Downside Deviation
        Like Sharpe but only penalizes downside volatility
        """
        annual_return = self._annual_return()
        
        # Downside deviation (only negative returns)
        downside_returns = self.returns[self.returns < 0]
        if len(downside_returns) == 0:
            downside_dev = 0
        else:
            downside_dev = downside_returns.std() * np.sqrt(self.freq)
        
        if downside_dev == 0:
            return float('inf') if annual_return >= self.risk_free_rate else float('-inf')
        
        return (annual_return - self.risk_free_rate) / downside_dev
    
    def _calmar_ratio(self) -> float:
        """
        Calmar Ratio = Annual Return / Max Drawdown
        Measures return relative to maximum loss
        """
        annual_return = self._annual_return()
        max_dd = abs(self._max_drawdown())
        
        if max_dd == 0:
            return 0
        
        return annual_return / max_dd
    
    # Drawdown metrics
    def _max_drawdown(self) -> float:
        """Maximum drawdown from peak"""
        cum_returns = (1 + self.returns).cumprod()
        running_max = cum_returns.expanding().max()
        drawdown = (cum_returns - running_max) / running_max
        return drawdown.min()
    
    def _avg_drawdown(self) -> float:
        """Average drawdown during drawdown periods"""
        cum_returns = (1 + self.returns).cumprod()
        running_max = cum_returns.expanding().max()
        drawdown = (cum_returns - running_max) / running_max
        
        drawdown_periods = drawdown[drawdown < 0]
        if len(drawdown_periods) == 0:
            return 0
        
        return drawdown_periods.mean()
    
    def _drawdown_duration(self) -> int:
        """Duration of maximum drawdown in days"""
        cum_returns = (1 + self.returns).cumprod()
        running_max = cum_returns.expanding().max()
        drawdown = (cum_returns - running_max) / running_max
        
        # Find longest consecutive drawdown period
        in_drawdown = (drawdown < 0).astype(int)
        
        if in_drawdown.sum() == 0:
            return 0
        
        # Count consecutive drawdown periods
        max_duration = 0
        current_duration = 0
        
        for val in in_drawdown:
            if val == 1:
                current_duration += 1
                max_duration = max(max_duration, current_duration)
            else:
                current_duration = 0
        
        return max_duration
    
    # Value at Risk metrics
    def _var(self, confidence: float = 0.95) -> float:
        """
        Value at Risk - maximum loss at given confidence level
        Uses historical method
        
        Args:
            confidence: Confidence level (e.g., 0.95 for 95%)
        """
        return np.percentile(self.returns, (1 - confidence) * 100)
    
    def _cvar(self, confidence: float = 0.95) -> float:
        """
        Conditional Value at Risk (Expected Shortfall)
        Average of returns worse than VaR
        
        Args:
            confidence: Confidence level (e.g., 0.95 for 95%)
        """
        var = self._var(confidence)
        return self.returns[self.returns <= var].mean()
    
    # Distribution metrics
    def _skewness(self) -> float:
        """Distribution skewness (3rd moment)"""
        from scipy.stats import skew
        return skew(self.returns)
    
    def _kurtosis(self) -> float:
        """Distribution kurtosis (4th moment)"""
        from scipy.stats import kurtosis
        return kurtosis(self.returns)
    
    # Trade-based metrics
    def _win_rate(self) -> float:
        """Percentage of profitable periods"""
        positive_returns = (self.returns > 0).sum()
        total_periods = len(self.returns)
        
        return positive_returns / total_periods if total_periods > 0 else 0
    
    def _profit_factor(self) -> float:
        """
        Profit Factor = Sum of Gains / Sum of Losses
        Values > 1.0 indicate profitable trading
        """
        gains = self.returns[self.returns > 0].sum()
        losses = abs(self.returns[self.returns < 0].sum())
        
        if losses == 0:
            return float('inf') if gains > 0 else 0
        
        return gains / losses
    
    def _recovery_factor(self) -> float:
        """
        Recovery Factor = Total Gain / Max Drawdown
        Higher is better (indicates ability to recover from losses)
        """
        total_gain = self._total_return()
        max_dd = abs(self._max_drawdown())
        
        if max_dd == 0:
            return 0
        
        return total_gain / max_dd


class BetaCalculator:
    """Calculate beta and correlation metrics"""
    
    @staticmethod
    def calculate_beta(asset_returns: pd.Series,
                      market_returns: pd.Series) -> float:
        """
        Calculate beta coefficient
        Beta = Covariance(Asset, Market) / Variance(Market)
        """
        covariance = asset_returns.cov(market_returns)
        market_variance = market_returns.var()
        
        if market_variance == 0:
            return 0
        
        return covariance / market_variance
    
    @staticmethod
    def calculate_alpha(asset_returns: pd.Series,
                       market_returns: pd.Series,
                       risk_free_rate: float = 0.06) -> float:
        """
        Calculate Jensen's alpha
        Alpha = Portfolio Return - [Risk-Free Rate + Beta * (Market Return - Risk-Free Rate)]
        """
        # Annualize returns
        freq = 252 if len(asset_returns) > 1000 else 52
        
        portfolio_return = (1 + asset_returns).prod() ** (freq / len(asset_returns)) - 1
        market_return = (1 + market_returns).prod() ** (freq / len(market_returns)) - 1
        
        beta = BetaCalculator.calculate_beta(asset_returns, market_returns)
        
        alpha = portfolio_return - (risk_free_rate + beta * (market_return - risk_free_rate))
        return alpha
    
    @staticmethod
    def calculate_correlation(series1: pd.Series,
                             series2: pd.Series) -> float:
        """Calculate Pearson correlation coefficient"""
        return series1.corr(series2)
    
    @staticmethod
    def correlation_matrix(returns_df: pd.DataFrame) -> pd.DataFrame:
        """Calculate correlation matrix for multiple assets"""
        return returns_df.corr()


class VolatilityCalculator:
    """Advanced volatility calculations"""
    
    @staticmethod
    def historical_volatility(returns: pd.Series,
                             window: int = 20) -> pd.Series:
        """Rolling historical volatility"""
        return returns.rolling(window).std()
    
    @staticmethod
    def exponential_volatility(returns: pd.Series,
                              span: int = 20) -> pd.Series:
        """Exponentially weighted volatility (EWMA)"""
        return returns.ewm(span=span).std()
    
    @staticmethod
    def garch_volatility(returns: pd.Series,
                        p: int = 1,
                        q: int = 1,
                        forecast_days: int = 1) -> float:
        """
        GARCH(p,q) volatility forecast
        Simple implementation using optimization
        """
        try:
            from arch import arch_model
            model = arch_model(returns * 100, vol='Garch', p=p, q=q)
            result = model.fit(disp='off')
            return result.forecast(horizon=forecast_days).variance.values[-1, -1] ** 0.5 / 100
        except:
            # Fallback to historical if arch not available
            return returns.std()
    
    @staticmethod
    def realized_volatility(high_freq_returns: pd.Series,
                           periods_per_day: int = 252) -> float:
        """Realized volatility from high-frequency returns"""
        return high_freq_returns.std() * np.sqrt(periods_per_day)


class CorrelationAnalyzer:
    """Analyze correlation dynamics"""
    
    @staticmethod
    def rolling_correlation(series1: pd.Series,
                           series2: pd.Series,
                           window: int = 50) -> pd.Series:
        """Rolling correlation between two series"""
        return series1.rolling(window).corr(series2)
    
    @staticmethod
    def correlation_breakdown(returns_df: pd.DataFrame,
                             lookback: int = 252) -> Dict[str, float]:
        """
        Correlation analysis at different periods
        """
        corr_full = returns_df.corr()
        corr_recent = returns_df.tail(lookback).corr()
        
        # Average correlation across assets
        values_full = corr_full.values[np.triu_indices_from(corr_full.values, k=1)]
        values_recent = corr_recent.values[np.triu_indices_from(corr_recent.values, k=1)]
        
        return {
            'correlation_full': values_full.mean(),
            'correlation_recent': values_recent.mean(),
            'correlation_change': values_recent.mean() - values_full.mean(),
        }
