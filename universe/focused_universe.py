# universe/focused_universe.py — Optimized subset of symbols for fast pair scanning
# Focuses on NIFTY50, NIFTY BANK, and NIFTY FINANCIAL SERVICES

NIFTY50_STOCKS = [
    'ADANIENT', 'ADANIPORTS', 'APOLLOHOSP', 'ASIANPAINT', 'AXISBANK', 'BAJAJ-AUTO', 
    'BAJAJFINSV', 'BAJFINANCE', 'BHARTIARTL', 'BPCL', 'BRITANNIA', 'CIPLA', 
    'COALINDIA', 'DIVISLAB', 'DRREDDY', 'EICHERMOT', 'GRASIM', 'HCLTECH', 
    'HDFCBANK', 'HDFCLIFE', 'HEROMOTOCO', 'HINDALCO', 'HINDUNILVR', 'ICICIBANK', 
    'INDUSINDBK', 'INFY', 'ITC', 'JSWSTEEL', 'KOTAKBANK', 'LT', 'LTIM', 'M&M', 
    'MARUTI', 'NESTLEIND', 'NTPC', 'ONGC', 'POWERGRID', 'RELIANCE', 'SBILIFE', 
    'SBIN', 'SUNPHARMA', 'TATACONSUM', 'TATAMOTORS', 'TATASTEEL', 'TCS', 
    'TECHM', 'TITAN', 'ULTRACEMCO', 'UPL', 'WIPRO'
]

BANK_STOCKS = [
    'AUBANK', 'AXISBANK', 'BANDHANBNK', 'FEDERALBNK', 'HDFCBANK', 'ICICIBANK', 
    'IDFCFIRSTB', 'INDUSINDBK', 'KOTAKBANK', 'PNB', 'SBIN', 'BANKBARODA'
]

FIN_STOCKS = [
    'CHOLAFIN', 'HDFCAMC', 'HDFCLIFE', 'ICICIGI', 'ICICIPRULI', 'MUTHOOTFIN', 
    'RECLTD', 'PFC', 'SHRIRAMFIN', 'SBICARD', 'SBILIFE'
]

# Combined focused universe (deduplicated)
FOCUSED_SYMBOLS = list(set(NIFTY50_STOCKS + BANK_STOCKS + FIN_STOCKS))

# Sector mapping for focused stocks
STOCK_SECTORS = {
    'HDFCBANK': 'Banking', 'ICICIBANK': 'Banking', 'SBIN': 'Banking', 'AXISBANK': 'Banking', 'KOTAKBANK': 'Banking',
    'INDUSINDBK': 'Banking', 'PNB': 'Banking', 'BANKBARODA': 'Banking', 'AUBANK': 'Banking', 'FEDERALBNK': 'Banking',
    'IDFCFIRSTB': 'Banking', 'BANDHANBNK': 'Banking',
    'RELIANCE': 'Energy', 'ONGC': 'Energy', 'BPCL': 'Energy', 'COALINDIA': 'Energy', 'NTPC': 'Energy', 'POWERGRID': 'Energy',
    'TCS': 'IT', 'INFY': 'IT', 'HCLTECH': 'IT', 'WIPRO': 'IT', 'TECHM': 'IT', 'LTIM': 'IT',
    'MARUTI': 'Automobile', 'TATAMOTORS': 'Automobile', 'M&M': 'Automobile', 'BAJAJ-AUTO': 'Automobile', 'EICHERMOT': 'Automobile', 'HEROMOTOCO': 'Automobile',
    'HINDUNILVR': 'FMCG', 'ITC': 'FMCG', 'NESTLEIND': 'FMCG', 'BRITANNIA': 'FMCG', 'TATACONSUM': 'FMCG',
    'SUNPHARMA': 'Healthcare', 'CIPLA': 'Healthcare', 'DRREDDY': 'Healthcare', 'DIVISLAB': 'Healthcare', 'APOLLOHOSP': 'Healthcare',
    'TATASTEEL': 'Metals', 'JSWSTEEL': 'Metals', 'HINDALCO': 'Metals',
    'LT': 'Construction', 'GRASIM': 'Cement', 'ULTRACEMCO': 'Cement',
    'BAJFINANCE': 'Financial Services', 'BAJAJFINSV': 'Financial Services', 'CHOLAFIN': 'Financial Services', 
    'SHRIRAMFIN': 'Financial Services', 'MUTHOOTFIN': 'Financial Services', 'RECLTD': 'Financial Services', 'PFC': 'Financial Services',
    'HDFCLIFE': 'Insurance', 'SBILIFE': 'Insurance', 'ICICIPRULI': 'Insurance', 'ICICIGI': 'Insurance',
    'BHARTIARTL': 'Telecom', 'TITAN': 'Consumer Durables', 'ADANIENT': 'Diversified', 'ADANIPORTS': 'Infrastructure',
    'ASIANPAINT': 'Consumer Durables', 'UPL': 'Chemicals', 'SBICARD': 'Financial Services', 'HDFCAMC': 'Financial Services',
    'AUBANK': 'Banking', 'PNB': 'Banking', 'FEDERALBNK': 'Banking', 'IDFCFIRSTB': 'Banking', 'BANDHANBNK': 'Banking', 'BANKBARODA': 'Banking'
}

def get_focused_universe():
    """Returns list of symbols in the focused universe"""
    return sorted(FOCUSED_SYMBOLS)

def get_stock_sector(symbol):
    """Returns the sector for a given stock symbol"""
    return STOCK_SECTORS.get(symbol, 'Other')
