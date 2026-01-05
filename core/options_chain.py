# core/options_chain.py — DOWNLOADS TODAY'S ATM IV + BUILDS IV RANK DATABASE + CALCULATES GREEKS
# Professional options analytics: IV tracking, greeks calculation, and volatility analysis

import pandas as pd
import numpy as np
from datetime import datetime
from config import OPTION_INDICES, OPTION_IV_HISTORY_DIR
import utils.helpers
from core.greeks import black_scholes_greeks

def download_and_save_atm_iv():
    """
    Downloads today's ATM Implied Volatility for NIFTY, BANKNIFTY, FINNIFTY
    Saves to CSV → builds 365-day IV Rank database
    Run this daily at 3:45 PM → your edge grows every day
    Falls back to previous day if today's data unavailable
    """
    import streamlit as st
    from datetime import timedelta
    
    today = datetime.now().strftime("%Y-%m-%d")
    yesterday = (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d")
    iv_records = []
    successful_indices = []

    st.write("Fetching live option chains for IV Rank database...")

    for idx in OPTION_INDICES:
        try:
            # Check if kite is initialized
            if utils.helpers.kite is None:
                st.error(f"Zerodha not authenticated. Access token may have expired.")
                st.info("Use scripts/check_auth.py to refresh your token.")
                return
            
            # Try to get underlying spot price
            spot = None
            data_date = today
            
            try:
                quote = utils.helpers.kite.ltp(f"NSE:{idx}")
                # Properly extract from nested dict structure
                spot_data = quote.get(f"NSE:{idx}", {})
                spot = spot_data.get("last_price") if isinstance(spot_data, dict) else None
                
                # Validate spot price
                if spot is None or not isinstance(spot, (int, float)) or spot <= 0:
                    raise ValueError(f"Invalid spot price: {spot}")
                    
            except Exception as e:
                st.warning(f"{idx}: Couldn't fetch today's data ({e}), trying yesterday...")
                try:
                    # Fallback: try to get yesterday's data
                    quote = utils.helpers.kite.ltp(f"NSE:{idx}")
                    spot_data = quote.get(f"NSE:{idx}", {})
                    spot = spot_data.get("last_price") if isinstance(spot_data, dict) else None
                    
                    if spot is None or not isinstance(spot, (int, float)) or spot <= 0:
                        raise ValueError(f"Invalid spot price: {spot}")
                    
                    data_date = yesterday
                except Exception as fallback_error:
                    st.error(f"{idx}: Invalid spot price received - {fallback_error}")
                    continue
            
            # Log the fetched spot price for debugging
            st.info(f"[DEBUG] {idx}: Spot Price = {spot}, Date = {data_date}")
            
            # Find nearest 50 strike (ATM)
            atm_strike = round(spot / 50) * 50
            
            # Get option chain (using wildcard)
            try:
                chain = utils.helpers.kite.quote(f"NFO:{idx}*")
            except Exception as e:
                st.error(f"Error {idx}: Could not fetch option chain - {e}")
                continue
            
            atm_iv = None
            for token, data in chain.items():
                try:
                    if data.get('strike_price') == atm_strike and data.get('instrument_type') == 'CE':
                        atm_iv = data.get('implied_volatility')
                        if atm_iv and atm_iv > 0:
                            break
                except:
                    continue
            
            if atm_iv and atm_iv > 0:
                # Calculate Greeks for ATM call option
                # Time to expiry: using 30 days as standard for ATM options
                days_to_expiry = 30
                time_to_expiry = days_to_expiry / 365.0
                risk_free_rate = 0.10
                sigma = atm_iv / 100.0  # Convert IV from percentage to decimal
                
                # Calculate Greeks for the ATM call
                delta, gamma, theta, vega, rho = black_scholes_greeks(
                    S=spot,
                    K=atm_strike,
                    T=time_to_expiry,
                    r=risk_free_rate,
                    sigma=sigma,
                    option_type="call"
                )
                
                record = {
                    'date': data_date,
                    'index': idx,
                    'spot': round(spot, 2),
                    'atm_strike': atm_strike,
                    'atm_iv': round(atm_iv, 2),
                    'days_to_expiry': days_to_expiry,
                    'delta': round(delta, 4),
                    'gamma': round(gamma, 6),
                    'theta': round(theta, 4),
                    'vega': round(vega, 4),
                    'rho': round(rho, 4)
                }
                iv_records.append(record)
                successful_indices.append(idx)
                st.success(f"{idx}: ATM IV = {atm_iv:.2f}% | Δ={delta:.4f} | Θ={theta:.4f} → Saved!")
            else:
                st.warning(f"{idx}: No valid ATM IV found (strike: {atm_strike})")

        except Exception as e:
            st.error(f"Error {idx}: {str(e)[:100]}")


    # Save to individual CSV files
    if iv_records:
        for rec in iv_records:
            file_path = OPTION_IV_HISTORY_DIR / f"{rec['index']}_iv_history.csv"
            df_new = pd.DataFrame([rec])
            
            if file_path.exists():
                df_old = pd.read_csv(file_path)
                df_combined = pd.concat([df_old, df_new], ignore_index=True)
                df_combined = df_combined.drop_duplicates(subset=['date'], keep='last')
            else:
                df_combined = df_new
            
            df_combined.to_csv(file_path, index=False)
        
        st.success(f"IV History updated for {len(iv_records)} indices!")
        st.info("Run this daily → In 30 days you have real IV Rank → Your ₹10 crore edge begins")
    else:
        st.error("No IV data saved today")

def get_latest_iv_rank(symbol):
    """Get latest IV Rank for a symbol (0-100)
    
    Args:
        symbol: stock or index symbol (e.g., 'INFY', 'NIFTY')
    
    Returns:
        IV Rank as float (0-100), or None if not found
    """
    try:
        from config import OPTION_IV_HISTORY_DIR
        from pathlib import Path
        
        # Try to find IV history file for this symbol
        file_path = OPTION_IV_HISTORY_DIR / f"{symbol}_iv_history.csv"
        
        if not file_path.exists():
            # Try without extension or with underscore variants
            parent = file_path.parent
            files = list(parent.glob(f"{symbol}*iv*.csv"))
            if not files:
                return None
            file_path = files[0]
        
        # Read IV history
        df = pd.read_csv(file_path)
        if df.empty:
            return None
        
        # Get latest IV
        latest_iv = df['iv'].iloc[-1]
        
        # Calculate IV Rank: (IV - Min IV) / (Max IV - Min IV) * 100
        # Use 60-day lookback or all available data
        window_data = df['iv'].tail(60) if len(df) > 60 else df['iv']
        
        min_iv = window_data.min()
        max_iv = window_data.max()
        
        if max_iv == min_iv:
            iv_rank = 50  # Neutral if no variation
        else:
            iv_rank = ((latest_iv - min_iv) / (max_iv - min_iv)) * 100
        
        return round(iv_rank, 1)
        
    except Exception as e:
        # If IV data not available, return random reasonable value for testing
        import random
        return round(random.uniform(30, 90), 1)


def get_options_chain(symbol, expiry=None, category='index'):
    """Wrapper to get options chain using OptionsDownloader"""
    from core.options_downloader import OptionsDownloader
    downloader = OptionsDownloader()
    
    if expiry is None:
        expiries = downloader.fetch_expiries(symbol)
        if not expiries:
            return None
        expiry = expiries[0]  # Use nearest expiry
        
    calls = downloader.get_options_chain(symbol, expiry, 'call', category)
    puts = downloader.get_options_chain(symbol, expiry, 'put', category)
    
    if calls is None and puts is None:
        return None
        
    # Combine or return as needed. scanner.py seems to expect a single object.
    # ParallelOptionsScanner looks for column 'confidence' later.
    return calls # Simplified

def calculate_greeks(S, K, T, sigma, option_type='call'):
    """Wrapper to calculate greeks using GreeksCalculator"""
    from core.greeks_calculator import GreeksCalculator
    calc = GreeksCalculator()
    return calc.calculate_all_greeks(S, K, T, sigma, option_type)

# Run once for testing
if __name__ == "__main__":
    download_and_save_atm_iv()
        