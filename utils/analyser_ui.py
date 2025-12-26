"""
analyser_ui.py - UI Presentation Layer for Analyser Page
Handles all Streamlit UI rendering and presentation logic
Decoupled from business logic and data access
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime
from typing import Dict, Any
import logging

logger = logging.getLogger(__name__)


class AnalyserUIRenderer:
    """
    Handles all UI rendering for the Analyser page
    Separates UI presentation from business logic
    """

    @staticmethod
    def render_settings_section(
        available_strategies: list,
        available_timeframes: list,
        available_lookback_options: list = None
    ) -> Dict[str, Any]:
        """
        Render settings/configuration section with user inputs
        
        Args:
            available_strategies: List of available trading strategies
            available_timeframes: List of available timeframes
            available_lookback_options: List of lookback day options
        
        Returns:
            Dictionary with user-selected parameters
        """
        if available_lookback_options is None:
            available_lookback_options = [10, 30, 60, 90, 180, 365]
        
        st.markdown("### ⚙️ [SETTINGS]")
        
        # Universe selection
        col_universe = st.columns(1)[0]
        with col_universe:
            selected_universe = st.selectbox(
                "Universe",
                ["NIFTY50", "NIFTY100", "NIFTY500", "All Stocks"],
                key="analyser_universe",
                help="Select the stock universe to analyze"
            )
        
        col1, col2, col3, col4, col5, col6 = st.columns(6)
        
        # Strategy selection
        with col1:
            selected_strategy = st.selectbox(
                "Strategy",
                available_strategies,
                key="analyser_strategy",
                help="Select a trading strategy to analyze"
            )
        
        # Timeframe selection
        with col2:
            selected_timeframe = st.selectbox(
                "Timeframe",
                available_timeframes,
                key="analyser_tf",
                help="Select the timeframe for analysis"
            )
        
        # Lookback period
        with col3:
            lookback_days = st.selectbox(
                "Lookback Period",
                available_lookback_options,
                key="analyser_lookback",
                help="Historical period to analyze"
            )
        
        # Min trades filter
        with col4:
            min_trades = st.number_input(
                "Min Trades",
                value=10,
                min_value=1,
                max_value=100,
                step=5,
                help="Minimum number of trades required for analysis"
            )
        
        # Confidence level filter
        with col5:
            confidence_level = st.slider(
                "Min Confidence",
                min_value=0.0,
                max_value=100.0,
                value=50.0,
                step=5.0,
                help="Minimum confidence level (%)"
            )
        
        # Run analysis button
        with col6:
            run_button = st.button(
                "🔍 RUN ANALYSIS",
                width="stretch",
                key="run_analysis_btn",
                help="Start strategy analysis with current parameters"
            )
        
        st.divider()
        
        return {
            "universe": selected_universe,
            "strategy": selected_strategy,
            "timeframe": selected_timeframe,
            "lookback_days": lookback_days,
            "min_trades": min_trades,
            "confidence_level": confidence_level,
            "run_analysis": run_button,
        }

    @staticmethod
    def render_performance_metrics(metrics: Dict[str, Any]) -> None:
        """
        Render key performance metrics as cards
        
        Args:
            metrics: Dictionary with performance metrics
        """
        st.markdown("### 📊 [PERFORMANCE METRICS]")
        
        col1, col2, col3, col4, col5, col6 = st.columns(6)
        
        metrics_list = [
            (col1, "Total Trades", metrics.get("total_trades", 0), ""),
            (col2, "Win Rate", f"{metrics.get('win_rate', 0):.1f}%", "trades_winning"),
            (col3, "Profit Factor", f"{metrics.get('profit_factor', 0):.2f}", "metric_growth"),
            (col4, "Avg Return", f"{metrics.get('avg_trade_return', 0):.2f}%", ""),
            (col5, "Total P&L", f"₹{metrics.get('total_pnl', 0):,.0f}", "metric_income"),
            (col6, "Largest Win", f"₹{metrics.get('largest_win', 0):,.0f}", "metric_growth"),
        ]
        
        for col, label, value, delta_type in metrics_list:
            with col:
                # Parse delta value if it exists
                delta = None
                if isinstance(value, str) and "%" in label:
                    try:
                        delta = float(value.split()[0])
                    except:
                        delta = None
                
                st.metric(label, value, delta=delta)
        
        st.divider()

    @staticmethod
    def render_risk_metrics(risk_metrics: Dict[str, Any]) -> None:
        """
        Render risk-related metrics
        
        Args:
            risk_metrics: Dictionary with risk metrics
        """
        st.markdown("### ⚠️ [RISK METRICS]")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            max_drawdown = risk_metrics.get("max_drawdown", 0)
            st.metric(
                "Max Drawdown",
                f"{max_drawdown:.2f}%",
                help="Maximum peak-to-trough decline"
            )
        
        with col2:
            sharpe_ratio = risk_metrics.get("sharpe_ratio", 0)
            st.metric(
                "Sharpe Ratio",
                f"{sharpe_ratio:.2f}",
                help="Risk-adjusted returns (higher is better)"
            )
        
        with col3:
            sortino_ratio = risk_metrics.get("sortino_ratio", 0)
            st.metric(
                "Sortino Ratio",
                f"{sortino_ratio:.2f}",
                help="Downside risk-adjusted returns"
            )
        
        with col4:
            win_rate = st.session_state.get("win_rate", 0)
            st.metric(
                "Win Rate",
                f"{win_rate:.1f}%",
                help="Percentage of winning trades"
            )
        
        st.divider()

    @staticmethod
    def render_cumulative_pnl_chart(cumulative_pnl: pd.Series, strategy: str, timeframe: str) -> None:
        """
        Render cumulative P&L line chart
        
        Args:
            cumulative_pnl: Series with cumulative P&L values
            strategy: Strategy name for chart title
            timeframe: Timeframe for chart title
        """
        if cumulative_pnl.empty:
            st.warning("No data available for cumulative P&L chart")
            return
        
        st.markdown("#### 📈 Cumulative P&L")
        
        # Create figure
        fig = go.Figure()
        
        # Add main line
        fig.add_trace(go.Scatter(
            x=cumulative_pnl.index,
            y=cumulative_pnl.values,
            mode='lines+markers',
            name='Cumulative P&L',
            line=dict(color='#00FF00', width=2),
            fill='tozeroy',
            fillcolor='rgba(0, 255, 0, 0.1)',
            hovertemplate='<b>%{x}</b><br>P&L: ₹%{y:,.0f}<extra></extra>'
        ))
        
        # Update layout
        fig.update_layout(
            title=f"{strategy} - Cumulative P&L ({timeframe})",
            xaxis_title="Date",
            yaxis_title="Cumulative P&L (₹)",
            template="plotly_dark",
            hovermode='x unified',
            height=400,
        )
        
        st.plotly_chart(fig, width='stretch')

    @staticmethod
    def render_trades_table(trades_df: pd.DataFrame) -> None:
        """
        Render formatted trades data table
        
        Args:
            trades_df: DataFrame with trades data
        """
        st.markdown("#### 📋 Trade Details")
        
        if trades_df.empty:
            st.warning("No trades data available")
            return
        
        # Prepare display dataframe
        display_df = trades_df.copy()
        
        # Format numeric columns
        if "P&L" in display_df.columns:
            display_df["P&L"] = display_df["P&L"].apply(lambda x: f"₹{x:,.0f}")
        
        if "P&L %" in display_df.columns:
            display_df["P&L %"] = display_df["P&L %"].apply(lambda x: f"{x:+.2f}%")
        
        if "Entry" in display_df.columns:
            display_df["Entry"] = display_df["Entry"].apply(lambda x: f"{x:,.2f}")
        
        if "Exit" in display_df.columns:
            display_df["Exit"] = display_df["Exit"].apply(lambda x: f"{x:,.2f}")
        
        # Display table
        st.dataframe(display_df, width='stretch', hide_index=True)

    @staticmethod
    def render_statistics_section(metrics: Dict[str, Any], risk_metrics: Dict[str, Any]) -> None:
        """
        Render detailed statistics breakdown
        
        Args:
            metrics: Performance metrics dictionary
            risk_metrics: Risk metrics dictionary
        """
        st.markdown("#### 📊 Strategy Statistics")
        
        col1, col2, col3 = st.columns(3)
        
        # Risk metrics column
        with col1:
            st.markdown("**Risk Metrics**")
            st.write(f"• Max Drawdown: {risk_metrics.get('max_drawdown', 0):.2f}%")
            st.write(f"• Sharpe Ratio: {risk_metrics.get('sharpe_ratio', 0):.2f}")
            st.write(f"• Sortino Ratio: {risk_metrics.get('sortino_ratio', 0):.2f}")
        
        # Trade metrics column
        with col2:
            st.markdown("**Trade Metrics**")
            avg_win = metrics.get("gross_profit", 0) / max(metrics.get("winning_trades", 1), 1)
            avg_loss = metrics.get("gross_loss", 0) / max(metrics.get("losing_trades", 1), 1)
            st.write(f"• Avg Win: ₹{avg_win:,.0f}")
            st.write(f"• Avg Loss: ₹{avg_loss:,.0f}")
            st.write(f"• Profit Factor: {metrics.get('profit_factor', 0):.2f}")
            risk_reward = abs(avg_win / avg_loss) if avg_loss != 0 else 0
            st.write(f"• Risk/Reward: 1:{risk_reward:.2f}")
        
        # Performance column
        with col3:
            st.markdown("**Performance**")
            total_return = metrics.get("total_pnl", 0)
            st.write(f"• Total Return: ₹{total_return:,.0f}")
            monthly_avg = total_return / max(metrics.get("total_trades", 1) / 20, 1)  # Rough estimate
            st.write(f"• Avg Per Trade: ₹{monthly_avg:,.0f}")
            st.write(f"• Win Rate: {metrics.get('win_rate', 0):.1f}%")
            st.write(f"• Total Trades: {metrics.get('total_trades', 0)}")
        
        st.divider()

    @staticmethod
    def render_export_section(trades_df: pd.DataFrame) -> None:
        """
        Render data export options
        
        Args:
            trades_df: DataFrame to export
        """
        st.markdown("#### 💾 Export Options")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("📥 Export Trades (CSV)", width="stretch"):
                if not trades_df.empty:
                    csv = trades_df.to_csv(index=False)
                    st.download_button(
                        label="Download CSV",
                        data=csv,
                        file_name=f"trades_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        mime="text/csv"
                    )
                else:
                    st.warning("No data to export")
        
        with col2:
            if st.button("📊 Export Report (Excel)", width="stretch"):
                st.info("Excel export coming soon...")
        
        with col3:
            if st.button("🗑️ Clear Results", width="stretch"):
                st.session_state.analysis_running = False
                st.session_state.analysis_results = None
                st.rerun()
        
        st.divider()

    @staticmethod
    def render_information_panel() -> None:
        """Render information and help panel"""
        st.markdown("### ℹ️ [ANALYSER INFORMATION]")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            **Available Strategies:**
            - **Pair Trading:** Cointegration-based statistical arbitrage
            - **Momentum:** Trend-following with technical indicators
            - **Mean Reversion:** Range-bound trading with volume confirmation
            - **Volatility:** GARCH-based volatility prediction
            
            **Timeframes:**
            - 15min: Intraday scalping
            - 30min: Short-term trading
            - 60min: Day trading
            - day: Swing trading
            - week: Position trading
            - month: Trend following
            """)
        
        with col2:
            st.markdown("""
            **Statistics Explained:**
            - **Sharpe Ratio:** Risk-adjusted returns (higher is better)
            - **Profit Factor:** Gross profit / Gross loss
            - **Max Drawdown:** Maximum peak-to-trough decline
            - **Win Rate:** % of profitable trades
            - **Sortino Ratio:** Downside risk-adjusted returns
            - **P&L:** Profit and Loss in absolute terms
            """)


class AnalyserSessionManager:
    """
    Manages Streamlit session state for the Analyser page
    Handles state initialization and updates
    """

    @staticmethod
    def initialize_session_state() -> None:
        """Initialize all required session state variables"""
        if 'analysis_running' not in st.session_state:
            st.session_state.analysis_running = False
            logger.debug("Initialized analysis_running to False")
        
        if 'analysis_results' not in st.session_state:
            st.session_state.analysis_results = None
            logger.debug("Initialized analysis_results to None")
        
        if 'analysis_start_time' not in st.session_state:
            st.session_state.analysis_start_time = None
            logger.debug("Initialized analysis_start_time to None")
        
        if 'win_rate' not in st.session_state:
            st.session_state.win_rate = 0.0
            logger.debug("Initialized win_rate to 0.0")

    @staticmethod
    def update_analysis_results(analysis_results: Dict[str, Any]) -> None:
        """
        Update session state with analysis results
        
        Args:
            analysis_results: Dictionary with analysis results
        """
        st.session_state.analysis_results = analysis_results
        st.session_state.analysis_running = True
        st.session_state.win_rate = analysis_results.get("performance_metrics", {}).get("win_rate", 0.0)
        logger.info(f"Updated session state with analysis results")

    @staticmethod
    def clear_analysis_results() -> None:
        """Clear analysis results from session state"""
        st.session_state.analysis_running = False
        st.session_state.analysis_results = None
        st.session_state.analysis_start_time = None
        logger.info("Cleared analysis results from session state")
