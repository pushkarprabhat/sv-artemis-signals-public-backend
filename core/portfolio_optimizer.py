"""
Portfolio Optimization Module

Implements Modern Portfolio Theory (MPT) with:
- Efficient frontier calculation
- Sharpe ratio optimization
- Minimum variance portfolio
- Maximum diversification
- Risk parity allocation
- Monte Carlo simulation for portfolio performance

Supports:
- Individual stocks
- Pair trading strategies
- Multi-asset portfolios (stocks, indices, commodities, forex)
- Rebalancing strategies
"""

import numpy as np
import pandas as pd
from scipy.optimize import minimize, LinearConstraint, Bounds
from scipy.stats import norm
import warnings
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from enum import Enum

warnings.filterwarnings('ignore')


class OptimizationMethod(Enum):
    """Portfolio optimization methods"""
    MAX_SHARPE = "max_sharpe"           # Maximum Sharpe ratio
    MIN_VARIANCE = "min_variance"       # Minimum variance
    MAX_DIVERSIFICATION = "max_diversification"  # Maximum diversification ratio
    RISK_PARITY = "risk_parity"        # Risk parity weights
    INVERSE_VOLATILITY = "inverse_vol"  # Inverse volatility weighting
    EQUAL_WEIGHT = "equal_weight"       # Equal weight (1/n)


@dataclass
class PortfolioMetrics:
    """Portfolio performance and risk metrics"""
    weights: np.ndarray
    expected_return: float
    volatility: float
    sharpe_ratio: float
    sortino_ratio: float
    calmar_ratio: float
    max_drawdown: float
    var_95: float  # Value at Risk (95% confidence)
    cvar_95: float  # Conditional VaR (Expected shortfall)
    diversification_ratio: float
    concentration: float  # Herfindahl index
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for display"""
        return {
            'Expected Return': f"{self.expected_return:.2%}",
            'Volatility': f"{self.volatility:.2%}",
            'Sharpe Ratio': f"{self.sharpe_ratio:.2f}",
            'Sortino Ratio': f"{self.sortino_ratio:.2f}",
            'Calmar Ratio': f"{self.calmar_ratio:.2f}",
            'Max Drawdown': f"{self.max_drawdown:.2%}",
            'VaR (95%)': f"{self.var_95:.2%}",
            'CVaR (95%)': f"{self.cvar_95:.2%}",
            'Diversification Ratio': f"{self.diversification_ratio:.2f}",
            'Concentration (HHI)': f"{self.concentration:.4f}",
        }


class PortfolioOptimizer:
    """
    Portfolio optimization engine using Modern Portfolio Theory
    
    Supports multiple optimization methods and comprehensive risk metrics
    """
    
    def __init__(self, 
                 returns_data: pd.DataFrame,
                 risk_free_rate: float = 0.06,
                 confidence_level: float = 0.95):
        """
        Initialize optimizer with historical returns
        
        Args:
            returns_data: DataFrame with asset returns (daily/weekly/monthly)
            risk_free_rate: Annual risk-free rate (default 6%)
            confidence_level: For VaR/CVaR calculation (default 95%)
        """
        self.returns = returns_data
        self.risk_free_rate = risk_free_rate
        self.confidence_level = confidence_level
        self.n_assets = len(returns_data.columns)
        
        # Calculate statistics
        self._calculate_statistics()
        
    def _calculate_statistics(self):
        """Calculate return statistics and correlation matrix"""
        # Annualization factor (252 for daily, 52 for weekly, 12 for monthly)
        freq = self._detect_frequency()
        
        self.mean_returns = self.returns.mean() * freq
        self.cov_matrix = self.returns.cov() * freq
        self.corr_matrix = self.returns.corr()
        self.volatilities = self.returns.std() * np.sqrt(freq)
        self.frequency = freq
        
    def _detect_frequency(self) -> int:
        """Auto-detect return frequency (daily/weekly/monthly)"""
        if len(self.returns) > 1000:
            return 252  # Daily
        elif len(self.returns) > 250:
            return 52   # Weekly
        else:
            return 12   # Monthly
    
    def optimize(self, 
                method: OptimizationMethod = OptimizationMethod.MAX_SHARPE,
                constraints: Optional[List] = None,
                bounds: Tuple = (0, 1)) -> PortfolioMetrics:
        """
        Optimize portfolio weights
        
        Args:
            method: Optimization method to use
            constraints: Custom constraints (scipy format)
            bounds: Weight bounds per asset (default: 0-100%)
            
        Returns:
            PortfolioMetrics with optimized weights and performance
        """
        if method == OptimizationMethod.EQUAL_WEIGHT:
            weights = np.array([1.0 / self.n_assets] * self.n_assets)
        elif method == OptimizationMethod.INVERSE_VOLATILITY:
            weights = 1.0 / self.volatilities
            weights /= weights.sum()
        elif method == OptimizationMethod.RISK_PARITY:
            weights = self._risk_parity()
        elif method == OptimizationMethod.MAX_DIVERSIFICATION:
            weights = self._maximize_diversification(bounds, constraints)
        elif method == OptimizationMethod.MIN_VARIANCE:
            weights = self._minimize_variance(bounds, constraints)
        else:  # MAX_SHARPE (default)
            weights = self._maximize_sharpe(bounds, constraints)
        
        # Calculate metrics for optimized portfolio
        return self._calculate_metrics(weights)
    
    def _maximize_sharpe(self, bounds: Tuple, constraints: Optional[List]) -> np.ndarray:
        """Maximize Sharpe ratio"""
        def neg_sharpe(w):
            ret = np.sum(w * self.mean_returns)
            vol = np.sqrt(w @ self.cov_matrix @ w)
            return -(ret - self.risk_free_rate) / vol if vol > 0 else 0
        
        x0 = np.array([1.0 / self.n_assets] * self.n_assets)
        constraints_full = [{'type': 'eq', 'fun': lambda w: np.sum(w) - 1}]
        if constraints:
            constraints_full.extend(constraints)
        
        result = minimize(neg_sharpe, x0, method='SLSQP', 
                         bounds=[bounds] * self.n_assets,
                         constraints=constraints_full)
        
        return np.array(result.x)
    
    def _minimize_variance(self, bounds: Tuple, constraints: Optional[List]) -> np.ndarray:
        """Minimize portfolio variance"""
        def portfolio_variance(w):
            return w @ self.cov_matrix @ w
        
        x0 = np.array([1.0 / self.n_assets] * self.n_assets)
        constraints_full = [{'type': 'eq', 'fun': lambda w: np.sum(w) - 1}]
        if constraints:
            constraints_full.extend(constraints)
        
        result = minimize(portfolio_variance, x0, method='SLSQP',
                         bounds=[bounds] * self.n_assets,
                         constraints=constraints_full)
        
        return np.array(result.x)
    
    def _maximize_diversification(self, bounds: Tuple, constraints: Optional[List]) -> np.ndarray:
        """Maximize diversification ratio"""
        def neg_diversification(w):
            weighted_vol = np.sum(w * self.volatilities)
            portfolio_vol = np.sqrt(w @ self.cov_matrix @ w)
            return -weighted_vol / portfolio_vol if portfolio_vol > 0 else 0
        
        x0 = np.array([1.0 / self.n_assets] * self.n_assets)
        constraints_full = [{'type': 'eq', 'fun': lambda w: np.sum(w) - 1}]
        if constraints:
            constraints_full.extend(constraints)
        
        result = minimize(neg_diversification, x0, method='SLSQP',
                         bounds=[bounds] * self.n_assets,
                         constraints=constraints_full)
        
        return np.array(result.x)
    
    def _risk_parity(self) -> np.ndarray:
        """Calculate risk parity weights (inverse volatility normalized)"""
        inv_vol = 1.0 / self.volatilities
        return inv_vol / inv_vol.sum()
    
    def _calculate_metrics(self, weights: np.ndarray) -> PortfolioMetrics:
        """Calculate comprehensive portfolio metrics"""
        # Basic returns and risk
        portfolio_return = np.sum(weights * self.mean_returns)
        portfolio_vol = np.sqrt(weights @ self.cov_matrix @ weights)
        
        # Sharpe ratio
        sharpe = (portfolio_return - self.risk_free_rate) / portfolio_vol if portfolio_vol > 0 else 0
        
        # Sortino ratio (downside deviation)
        downside_returns = self.returns[self.returns < 0]
        downside_std = downside_returns.std() * np.sqrt(self.frequency)
        downside_std_val = downside_std if isinstance(downside_std, (int, float)) else downside_std.item() if hasattr(downside_std, 'item') else 0
        sortino = (portfolio_return - self.risk_free_rate) / downside_std_val if downside_std_val > 0 else 0
        
        # Simulate portfolio returns for drawdown and VaR
        portfolio_returns = self.returns @ weights
        
        # Maximum drawdown
        cum_returns = (1 + portfolio_returns).cumprod()
        running_max = cum_returns.expanding().max()
        drawdown = (cum_returns - running_max) / running_max
        max_dd = drawdown.min()
        
        # Calmar ratio
        calmar = portfolio_return / abs(max_dd) if max_dd != 0 else 0
        
        # VaR and CVaR (95% confidence)
        var_95 = np.percentile(portfolio_returns, (1 - self.confidence_level) * 100)
        cvar_95 = portfolio_returns[portfolio_returns <= var_95].mean()
        
        # Diversification ratio
        weighted_vol = np.sum(weights * self.volatilities)
        diversification_ratio = weighted_vol / portfolio_vol if portfolio_vol > 0 else 1.0
        
        # Concentration (Herfindahl-Hirschman Index)
        hhi = np.sum(weights ** 2)
        
        return PortfolioMetrics(
            weights=weights,
            expected_return=portfolio_return,
            volatility=portfolio_vol,
            sharpe_ratio=sharpe,
            sortino_ratio=sortino,
            calmar_ratio=calmar,
            max_drawdown=max_dd,
            var_95=var_95,
            cvar_95=cvar_95,
            diversification_ratio=diversification_ratio,
            concentration=hhi
        )
    
    def efficient_frontier(self, 
                          n_points: int = 50,
                          bounds: Tuple = (0, 1)) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate efficient frontier
        
        Args:
            n_points: Number of points on frontier
            bounds: Weight bounds
            
        Returns:
            (volatilities, returns) for frontier
        """
        # Target returns from minimum to maximum
        min_return = self.mean_returns.min()
        max_return = self.mean_returns.max()
        target_returns = np.linspace(min_return, max_return, n_points)
        
        frontier_vols = []
        frontier_rets = []
        
        for target_ret in target_returns:
            # Minimize variance subject to target return constraint
            def portfolio_variance(w):
                return w @ self.cov_matrix @ w
            
            constraints = [
                {'type': 'eq', 'fun': lambda w: np.sum(w) - 1},
                {'type': 'eq', 'fun': lambda w: np.sum(w * self.mean_returns) - target_ret}
            ]
            
            x0 = np.array([1.0 / self.n_assets] * self.n_assets)
            
            result = minimize(portfolio_variance, x0, method='SLSQP',
                            bounds=[bounds] * self.n_assets,
                            constraints=constraints)
            
            if result.success:
                vol = np.sqrt(result.fun)
                frontier_vols.append(vol)
                frontier_rets.append(target_ret)
        
        return np.array(frontier_vols), np.array(frontier_rets)
    
    def monte_carlo_simulation(self,
                              weights: np.ndarray,
                              n_simulations: int = 10000,
                              n_days: int = 252) -> Dict:
        """
        Monte Carlo simulation of portfolio performance
        
        Args:
            weights: Portfolio weights
            n_simulations: Number of simulations
            n_days: Time horizon in days
            
        Returns:
            Simulation results (percentiles, mean, std)
        """
        # Cholesky decomposition of covariance matrix
        L = np.linalg.cholesky(self.cov_matrix)
        
        # Simulated returns
        simulated_returns = np.zeros((n_simulations, n_days))
        
        for i in range(n_simulations):
            # Generate correlated random returns
            random_shocks = np.random.normal(0, 1, (n_days, self.n_assets))
            correlated_returns = random_shocks @ L.T + self.mean_returns / self.frequency
            
            # Portfolio value
            portfolio_val = 100 * np.cumprod(1 + correlated_returns @ weights)
            simulated_returns[i] = portfolio_val
        
        # Calculate percentiles
        return {
            'mean': simulated_returns.mean(axis=0),
            'std': simulated_returns.std(axis=0),
            'p5': np.percentile(simulated_returns, 5, axis=0),
            'p25': np.percentile(simulated_returns, 25, axis=0),
            'p50': np.percentile(simulated_returns, 50, axis=0),
            'p75': np.percentile(simulated_returns, 75, axis=0),
            'p95': np.percentile(simulated_returns, 95, axis=0),
        }
    
    def get_allocation_summary(self, weights: np.ndarray) -> pd.DataFrame:
        """Get formatted allocation summary"""
        allocation = pd.DataFrame({
            'Asset': self.returns.columns,
            'Weight': weights,
            'Weight %': weights * 100,
            'Expected Return': weights * self.mean_returns,
            'Volatility': self.volatilities,
            'Contribution to Risk': weights * np.sqrt(np.diag(self.cov_matrix))
        })
        return allocation.sort_values('Weight %', ascending=False)


class PairPortfolioOptimizer:
    """Specialized optimizer for pair trading portfolios"""
    
    def __init__(self, 
                 pair_returns: Dict[str, pd.Series],
                 correlation_matrix: pd.DataFrame,
                 risk_free_rate: float = 0.06):
        """
        Initialize pair portfolio optimizer
        
        Args:
            pair_returns: Dict of pair returns {pair_name: returns_series}
            correlation_matrix: Correlation between pairs
            risk_free_rate: Annual risk-free rate
        """
        self.pair_returns = pair_returns
        self.correlation_matrix = correlation_matrix
        self.risk_free_rate = risk_free_rate
        self.n_pairs = len(pair_returns)
    
    def optimize_pair_allocation(self, 
                                method: str = "max_sharpe",
                                target_correlation_threshold: float = 0.3) -> Dict:
        """
        Optimize allocation across multiple pairs
        
        Filter pairs by correlation to ensure diversification
        """
        # Create returns dataframe
        returns_df = pd.concat(self.pair_returns.values(), axis=1)
        returns_df.columns = list(self.pair_returns.keys())
        
        # Filter low-correlation pairs
        avg_correlation = self.correlation_matrix.values[
            np.triu_indices_from(self.correlation_matrix.values, k=1)
        ].mean()
        
        if avg_correlation > target_correlation_threshold:
            print(f"Warning: Pairs have high correlation ({avg_correlation:.2f})")
        
        # Optimize using main optimizer
        optimizer = PortfolioOptimizer(returns_df, self.risk_free_rate)
        
        method_map = {
            "max_sharpe": OptimizationMethod.MAX_SHARPE,
            "min_variance": OptimizationMethod.MIN_VARIANCE,
            "risk_parity": OptimizationMethod.RISK_PARITY,
        }
        
        metrics = optimizer.optimize(method=method_map.get(method, OptimizationMethod.MAX_SHARPE))
        
        return {
            'weights': dict(zip(self.pair_returns.keys(), metrics.weights)),
            'metrics': metrics,
            'allocation_summary': optimizer.get_allocation_summary(metrics.weights),
            'avg_pair_correlation': avg_correlation
        }
