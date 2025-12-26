# core/greeks.py — Black-Scholes Greeks (used in strangle)
import numpy as np
from scipy.stats import norm

def black_scholes_greeks(S, K, T, r=0.10, sigma=0.20, option_type="call"):
    """Return Delta, Gamma, Theta, Vega, Rho"""
    if T <= 0 or sigma <= 0:
        return 0,0,0,0,0
    
    d1 = (np.log(S/K) + (r + 0.5*sigma**2)*T) / (sigma*np.sqrt(T))
    d2 = d1 - sigma*np.sqrt(T)
    
    if option_type == "call":
        delta = norm.cdf(d1)
        theta = -(S * norm.pdf(d1) * sigma)/(2*np.sqrt(T)) - r*K*np.exp(-r*T)*norm.cdf(d2)
    else:
        delta = norm.cdf(d1) - 1
        theta = -(S * norm.pdf(d1) * sigma)/(2*np.sqrt(T)) + r*K*np.exp(-r*T)*norm.cdf(-d2)
    
    gamma = norm.pdf(d1) / (S * sigma * np.sqrt(T))
    vega = S * norm.pdf(d1) * np.sqrt(T)
    rho = K * T * np.exp(-r*T) * norm.cdf(d2) if option_type=="call" else -K*T*np.exp(-r*T)*norm.cdf(-d2)
    
    return delta, gamma, theta/365, vega/100, rho/100


def strangle_greeks(spot, lower_strike, upper_strike, time_to_expiry, r=0.10, sigma=0.20):
    """Calculate Greeks for strangle position"""
    delta_put, gamma_put, theta_put, vega_put, rho_put = black_scholes_greeks(spot, lower_strike, time_to_expiry, r, sigma, "put")
    delta_call, gamma_call, theta_call, vega_call, rho_call = black_scholes_greeks(spot, upper_strike, time_to_expiry, r, sigma, "call")
    
    total_delta = delta_put + delta_call
    total_gamma = gamma_put + gamma_call
    total_theta = theta_put + theta_call
    total_vega = vega_put + vega_call
    total_rho = rho_put + rho_call
    
    return {
        "delta": total_delta,
        "gamma": total_gamma,
        "theta": total_theta,
        "vega": total_vega,
        "rho": total_rho
    }
