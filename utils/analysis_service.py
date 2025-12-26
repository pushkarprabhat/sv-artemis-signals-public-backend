"""
analysis_service.py - Data Access Layer for Analysis
Handles all data retrieval and business logic separation for the Analyser page
Provides modular, reusable analysis functions with proper caching
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import logging

# Configure logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


class AnalysisDataLayer:
    """
    Data Access Layer for analysis operations
    Handles data retrieval from CSV files and caching
    """

    def __init__(self, data_dir: str = "data"):
        """
        Initialize the data access layer
        
        Args:
            data_dir: Base directory for data files
        """
        self.data_dir = Path(data_dir)
        self._cache: Dict = {}
        logger.info(f"AnalysisDataLayer initialized with data_dir: {data_dir}")

    def get_trades_data(
        self,
        strategy: str,
        timeframe: str,
        lookback_days: int
    ) -> pd.DataFrame:
        """
        Retrieve trades data for a given strategy and timeframe
        
        Args:
            strategy: Trading strategy name (e.g., 'Pair Trading', 'Momentum')
            timeframe: Timeframe (e.g., '15min', '30min', '60min', 'day')
            lookback_days: Number of days to look back
        
        Returns:
            DataFrame with trade data or empty DataFrame if no data found
        """
        # Create cache key
        cache_key = f"trades_{strategy}_{timeframe}_{lookback_days}"
        
        # Return from cache if available
        if cache_key in self._cache:
            logger.debug(f"Returning cached trades data for key: {cache_key}")
            return self._cache[cache_key]
        
        try:
            # Try to load from backtest results
            backtest_file = self.data_dir / "backtest_results.csv"
            
            if backtest_file.exists():
                df = pd.read_csv(backtest_file)
                
                # Filter by strategy if specified
                if strategy != "All" and "Strategy" in df.columns:
                    df = df[df["Strategy"] == strategy]
                
                # Filter by timeframe if present
                if "Timeframe" in df.columns and timeframe != "all":
                    df = df[df["Timeframe"] == timeframe]
                
                # Convert date column
                if "Date" in df.columns:
                    df["Date"] = pd.to_datetime(df["Date"])
                
                # Filter by lookback days
                cutoff_date = datetime.now() - timedelta(days=lookback_days)
                if "Date" in df.columns:
                    df = df[df["Date"] >= cutoff_date]
                
                logger.info(f"Loaded {len(df)} trades for {strategy}/{timeframe}")
                
                # Cache the result
                self._cache[cache_key] = df
                
                return df
            else:
                logger.warning(f"Backtest file not found: {backtest_file}")
                return pd.DataFrame()
                
        except Exception as e:
            logger.error(f"Error loading trades data: {str(e)}")
            return pd.DataFrame()

    def generate_sample_trades(
        self,
        num_trades: int = 15,
        strategy: str = "Pair Trading"
    ) -> pd.DataFrame:
        """
        Generate sample trades data for demonstration
        
        Args:
            num_trades: Number of sample trades to generate
            strategy: Strategy name for the trades
        
        Returns:
            DataFrame with sample trade data
        """
        logger.debug(f"Generating {num_trades} sample trades for {strategy}")
        
        dates = pd.date_range(
            start=datetime.now() - timedelta(days=20),
            periods=num_trades
        )
        
        # Generate realistic sample data
        entry_prices = np.random.uniform(100, 50000, num_trades)
        pnl_pcts = np.random.normal(0.5, 2.0, num_trades)  # Mean 0.5%, Std 2%
        exit_prices = entry_prices * (1 + pnl_pcts / 100)
        
        trades_data = pd.DataFrame({
            "Date": dates,
            "Pair": ["RELIANCE-SBIN"] * num_trades,
            "Entry": entry_prices,
            "Exit": exit_prices,
            "P&L": (exit_prices - entry_prices) * np.random.randint(1, 20, num_trades),
            "P&L %": pnl_pcts,
            "Duration": np.random.choice(["30m", "1h", "2h", "3h", "4h"], num_trades),
        })
        
        return trades_data

    def clear_cache(self) -> None:
        """Clear the data cache"""
        self._cache.clear()
        logger.info("Data cache cleared")


class AnalyticsCalculator:
    """
    Business logic layer for calculating trading analytics
    Decoupled from data access and UI
    """

    @staticmethod
    def calculate_performance_metrics(trades_df: pd.DataFrame) -> Dict:
        """
        Calculate comprehensive performance metrics from trades data
        
        Args:
            trades_df: DataFrame containing trades data
        
        Returns:
            Dictionary with calculated metrics
        """
        if trades_df.empty:
            logger.warning("Empty trades DataFrame provided to calculate_performance_metrics")
            return {
                "total_trades": 0,
                "winning_trades": 0,
                "losing_trades": 0,
                "win_rate": 0.0,
                "total_pnl": 0.0,
                "avg_trade_return": 0.0,
                "profit_factor": 0.0,
                "largest_win": 0.0,
                "largest_loss": 0.0,
            }
        
        try:
            # Extract P&L values
            pnl_values = trades_df["P&L"].astype(float)
            pnl_pcts = trades_df["P&L %"].astype(float)
            
            # Calculate basic metrics
            total_trades = len(trades_df)
            winning_trades = len(trades_df[pnl_values > 0])
            losing_trades = len(trades_df[pnl_values < 0])
            win_rate = (winning_trades / total_trades * 100) if total_trades > 0 else 0
            
            # Calculate PnL metrics
            total_pnl = pnl_values.sum()
            avg_trade_return = pnl_pcts.mean()
            
            # Profit factor
            gross_profit = pnl_values[pnl_values > 0].sum()
            gross_loss = abs(pnl_values[pnl_values < 0].sum())
            profit_factor = (gross_profit / gross_loss) if gross_loss > 0 else 0
            
            # Risk metrics
            largest_win = pnl_values.max()
            largest_loss = pnl_values.min()
            
            metrics = {
                "total_trades": int(total_trades),
                "winning_trades": int(winning_trades),
                "losing_trades": int(losing_trades),
                "win_rate": float(round(win_rate, 2)),
                "total_pnl": float(round(total_pnl, 2)),
                "avg_trade_return": float(round(avg_trade_return, 2)),
                "profit_factor": float(round(profit_factor, 2)),
                "largest_win": float(round(largest_win, 2)),
                "largest_loss": float(round(largest_loss, 2)),
                "gross_profit": float(round(gross_profit, 2)),
                "gross_loss": float(round(gross_loss, 2)),
            }
            
            logger.debug(f"Calculated metrics: {metrics}")
            return metrics
            
        except Exception as e:
            logger.error(f"Error calculating performance metrics: {str(e)}")
            return {}

    @staticmethod
    def calculate_risk_metrics(trades_df: pd.DataFrame) -> Dict:
        """
        Calculate risk-based metrics (Sharpe, Sortino, Drawdown)
        
        Args:
            trades_df: DataFrame containing trades data
        
        Returns:
            Dictionary with risk metrics
        """
        if trades_df.empty:
            return {
                "max_drawdown": 0.0,
                "sharpe_ratio": 0.0,
                "sortino_ratio": 0.0,
            }
        
        try:
            pnl_pcts = trades_df["P&L %"].astype(float)
            
            # Calculate Sharpe Ratio (assuming ~252 trading days per year)
            returns_mean = pnl_pcts.mean()
            returns_std = pnl_pcts.std()
            risk_free_rate = 0.10 / 252  # Daily risk-free rate
            
            sharpe_ratio = (
                (returns_mean - risk_free_rate) / returns_std * np.sqrt(252)
                if returns_std > 0
                else 0
            )
            
            # Calculate Sortino Ratio (only downside volatility)
            downside_returns = pnl_pcts[pnl_pcts < 0]
            downside_std = downside_returns.std() if len(downside_returns) > 0 else 0
            
            sortino_ratio = (
                (returns_mean - risk_free_rate) / downside_std * np.sqrt(252)
                if downside_std > 0
                else 0
            )
            
            # Calculate Maximum Drawdown
            cumulative_returns = (1 + pnl_pcts / 100).cumprod()
            running_max = cumulative_returns.expanding().max()
            drawdown = (cumulative_returns - running_max) / running_max * 100
            max_drawdown = drawdown.min()
            
            metrics = {
                "max_drawdown": float(round(max_drawdown, 2)),
                "sharpe_ratio": float(round(sharpe_ratio, 2)),
                "sortino_ratio": float(round(sortino_ratio, 2)),
            }
            
            logger.debug(f"Calculated risk metrics: {metrics}")
            return metrics
            
        except Exception as e:
            logger.error(f"Error calculating risk metrics: {str(e)}")
            return {}

    @staticmethod
    def calculate_cumulative_pnl(trades_df: pd.DataFrame) -> pd.Series:
        """
        Calculate cumulative P&L over time
        
        Args:
            trades_df: DataFrame containing trades data
        
        Returns:
            Series with cumulative P&L indexed by date
        """
        if trades_df.empty:
            logger.warning("Empty DataFrame provided to calculate_cumulative_pnl")
            return pd.Series()
        
        try:
            # Ensure Date column is datetime
            if "Date" in trades_df.columns:
                df_sorted = trades_df.sort_values("Date").copy()
                df_sorted["Date"] = pd.to_datetime(df_sorted["Date"])
                
                # Calculate cumulative P&L
                cumulative_pnl = df_sorted.set_index("Date")["P&L"].cumsum()
                
                logger.debug(f"Calculated cumulative P&L with {len(cumulative_pnl)} data points")
                return cumulative_pnl
            else:
                logger.error("Date column not found in trades_df")
                return pd.Series()
                
        except Exception as e:
            logger.error(f"Error calculating cumulative P&L: {str(e)}")
            return pd.Series()

    @staticmethod
    def calculate_monthly_metrics(trades_df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate metrics by month
        
        Args:
            trades_df: DataFrame containing trades data
        
        Returns:
            DataFrame with monthly performance metrics
        """
        if trades_df.empty:
            return pd.DataFrame()
        
        try:
            df_copy = trades_df.copy()
            df_copy["Date"] = pd.to_datetime(df_copy["Date"])
            df_copy["YearMonth"] = df_copy["Date"].dt.to_period("M")
            
            monthly_metrics = df_copy.groupby("YearMonth").agg({
                "P&L": ["sum", "count", "mean"],
                "P&L %": ["mean", "std"],
            }).round(2)
            
            logger.debug(f"Calculated monthly metrics for {len(monthly_metrics)} months")
            return monthly_metrics
            
        except Exception as e:
            logger.error(f"Error calculating monthly metrics: {str(e)}")
            return pd.DataFrame()


class AnalysisService:
    """
    High-level service combining data access and analytics
    Main interface for the Analyser page
    """

    def __init__(self, data_dir: str = "data"):
        """
        Initialize the analysis service
        
        Args:
            data_dir: Base directory for data files
        """
        self.data_layer = AnalysisDataLayer(data_dir)
        self.calculator = AnalyticsCalculator()
        logger.info("AnalysisService initialized")

    def run_analysis(
        self,
        strategy: str,
        timeframe: str,
        lookback_days: int
    ) -> Dict:
        """
        Run complete analysis and return all metrics
        
        Args:
            strategy: Trading strategy name
            timeframe: Timeframe for analysis
            lookback_days: Number of days to analyze
        
        Returns:
            Dictionary with all analysis results
        """
        logger.info(f"Starting analysis: strategy={strategy}, timeframe={timeframe}, lookback={lookback_days}")
        
        # Fetch trades data
        trades_df = self.data_layer.get_trades_data(strategy, timeframe, lookback_days)
        
        # If no real data, generate sample data for demonstration
        if trades_df.empty:
            logger.info("No real trades data found, generating sample data")
            trades_df = self.data_layer.generate_sample_trades(num_trades=15, strategy=strategy)
        
        # Calculate all metrics
        performance_metrics = self.calculator.calculate_performance_metrics(trades_df)
        risk_metrics = self.calculator.calculate_risk_metrics(trades_df)
        cumulative_pnl = self.calculator.calculate_cumulative_pnl(trades_df)
        monthly_metrics = self.calculator.calculate_monthly_metrics(trades_df)
        
        # Combine results
        analysis_results = {
            "trades_data": trades_df,
            "performance_metrics": performance_metrics,
            "risk_metrics": risk_metrics,
            "cumulative_pnl": cumulative_pnl,
            "monthly_metrics": monthly_metrics,
            "analysis_timestamp": datetime.now(),
            "analysis_params": {
                "strategy": strategy,
                "timeframe": timeframe,
                "lookback_days": lookback_days,
            }
        }
        
        logger.info(f"Analysis complete with {len(trades_df)} trades")
        return analysis_results

    def get_available_strategies(self) -> List[str]:
        """Get list of available strategies for analysis"""
        return ["Pair Trading", "Momentum", "Mean Reversion", "Volatility", "All"]

    def get_available_timeframes(self) -> List[str]:
        """Get list of available timeframes for analysis"""
        return ["15min", "30min", "60min", "day", "week", "month"]
