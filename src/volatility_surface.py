
import os
import sys
import numpy as np
from matplotlib import cm
import matplotlib.pyplot as plt
from matplotlib import ticker
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.mlab import griddata
import pandas as pd
import math
from scipy.optimize import brentq
from scipy.stats import norm
from datetime import datetime
import yfinance as yf
from typing import Optional


class BuildVolatilitySurface():
    def __init__(self, symbol:str, expiration_date:Optional[str] = None):
        self.symbol = symbol
        self.extract_option_prices(self.symbol)


    def 

    def select_OTM_vols(data:pd.DataFrame):
        
        options_dfs = []
        
        for expiry in data['expiry'].unique():
            # Select available strikes for OTM Calls and OTM Puts
            otm_calls = data[(data['expiry'] == expiry)&(data['option_type'] == "Call")][data['strike'] > data['underlying_price']][['strike', 'impliedVolatility', 'expiry']]
            otm_puts = data[(data['expiry'] == expiry)&(data['option_type'] == "Put")][data['strike'] < data['underlying_price']][['strike', 'impliedVolatility', 'expiry']]

            # Adding an additional sorting as insurance
            otm_calls.sort_values(by="strike",ascending=True, inplace=True )
            otm_puts.sort_values(by="strike",ascending=True, inplace=True )

            options_dfs.append(otm_puts)
            options_dfs.append(otm_calls)

        options_data = pd.concat(options_dfs, ignore_index=True) if options_dfs else pd.DataFrame()

        return options_data

    def extract_option_prices(symbol:str="SPY", expiration_date:Optional[str] = None)->pd.DataFrame:
        """
        Extract option prices for a given stock ticker and optional expiration date.
        
        Parameters:
            symbol (str): Stock ticker symbol (e.g., 'AAPL')
            expiration_date (str, optional): The expiration date in 'YYYY-MM-DD' format. 
                                        If None, retrieves data for all available expiration dates.
                                        If specified but invalid, defaults to the first available expiration.
            
        Returns:
            pandas.DataFrame: DataFrame containing option chain data with columns:
                            contractSymbol, optionType, strike, lastPrice, bid, ask,
                            impliedVolatility, lastTradeDate, volume, openInterest
        Comments:
            openInterest: nr of active contracts for a particular option. Each contract typically represents 100 shares of the underlying asset.
                        High openInterest suggests greater liquidity and tighter Bid\Ask spread.
            volume: number of option contracts bought or sold on the most recent trading day.                 
        """
        try:
            # Create Ticker object
            stock = yf.Ticker(symbol)
            
            underlying_price = stock.info.get("regularMarketPrice", None)
            if underlying_price is None:
                # Fallback to last closing price from history if regularMarketPrice is unavailable
                recent_data = stock.history(period="1d")
                if not recent_data.empty:
                    underlying_price = recent_data["Close"].iloc[-1]
                else:
                    return {f"Unable to retrieve underlying price for {symbol}"}
            # Get the dividend yield
            dividendYield = stock.info.get("dividendYield", None)
            if underlying_price is None:
                dividendYield = 0.0

            # Get available expiration dates
            expiries = stock.options
            if not expiries:
                return {f"There is no options data available for {symbol}"}
            
            # Determine which expiries to process
            if expiration_date is None:
                selected_expiries = expiries
            else:
                if expiration_date in expiries:
                    selected_expiries = [expiration_date]
                else:
                    selected_expiries = [expiries[0]]
                    print(f"Expiration date {expiration_date} not found. Using first available: {expiries[0]}")
            

            # Initialize list to store DataFrames
            options_dfs = []
            
            # Fetch options chain for each selected expiration
            for expiry in selected_expiries:
                options = stock.option_chain(expiry)
                
                # Extract calls and puts
                calls = options.calls
                puts = options.puts

                # Add ticker, dividendYield, underlying price, expiration date, and option type
                calls["ticker"] = symbol
                puts["ticker"] = symbol

                calls["dividendYield"] = dividendYield
                puts["dividendYield"] = dividendYield

                calls["underlying_price"] = underlying_price
                puts["underlying_price"] = underlying_price
                calls["expiry"] = expiry
                puts["expiry"] = expiry
                calls["option_type"] = "Call"
                puts["option_type"] = "Put"
                
                # Select relevant columns to include in the output
                columns = ["contractSymbol", "strike", "lastPrice", "bid", "ask", "volume", "openInterest", 'lastTradeDate',
                        "impliedVolatility", "inTheMoney", "ticker", "underlying_price", "expiry", "option_type", 'dividendYield']
                calls = calls[[col for col in columns if col in calls.columns]]
                puts = puts[[col for col in columns if col in puts.columns]]

                # Append to options_dfs
                options_dfs.append(calls)
                options_dfs.append(puts)
            
            # Combine all DataFrames into a single DataFrame
            options_data = pd.concat(options_dfs, ignore_index=True) if options_dfs else pd.DataFrame()
            
            options_data['volume'] = options_data['volume'].fillna('-')
            options_data['openInterest'] = options_data['openInterest'].fillna('-')
            options_data['lastTradeDate'] = options_data['lastTradeDate'].apply(
                lambda x: x.strftime("%Y-%m-%d %H:%M:%S") if pd.notnull(x) else "-"
            )

            return options_data
        except Exception as e:
            print(f"Error fetching option prices for {symbol}: {str(e)}")
            return pd.DataFrame()
if __name__ == "__main__":
    
    df = extract_option_prices(symbol="SPY")

    df[(df['expiry']=="2025-08-29") & (df['strike'] == 550)]

    plt.plot(df[(df['expiry']=="2025-08-29")&(df['option_type']=="Call")]['strike'], df[(df['expiry']=="2025-08-29")&(df['option_type']=="Call")]['impliedVolatility'], label="Call")
    plt.plot(df[(df['expiry']=="2025-08-29")&(df['option_type']=="Put")]['strike'], df[(df['expiry']=="2025-08-29")&(df['option_type']=="Put")]['impliedVolatility'], label="Put")
    plt.legend()
    plt.show()

    from scipy.interpolate import CubicSpline

def interpolate_volatility(strikes, implied_vols):
    """
    Interpolate implied volatilities across strike prices using cubic spline.
    
    Parameters:
    - strikes (list of float): List of strike prices.
    - implied_vols (list of float): List of corresponding implied volatilities (as decimals, e.g., 0.2 for 20%).
    
    Returns:
    - callable: A function that takes a strike price (float) and returns the interpolated implied volatility.
    - tuple: (min_strike, max_strike) defining the valid range for interpolation.
    - None: If data is insufficient or an error occurs.
    """
    try:
        # Convert inputs to numpy arrays
        strikes = np.array(strikes, dtype=float)
        implied_vols = np.array(implied_vols, dtype=float)
        
        # Validate inputs
        if len(strikes) != len(implied_vols):
            print("Error: Strikes and implied volatilities must have the same length")
            return None, None
        if len(strikes) < 4:
            print(f"Error: At least 4 data points required for cubic spline interpolation, got {len(strikes)}")
            return None, None
        
        # Filter out invalid data (NaN or non-positive volatilities)
        valid_mask = (implied_vols > 0) & (~np.isnan(strikes)) & (~np.isnan(implied_vols))
        strikes = strikes[valid_mask]
        implied_vols = implied_vols[valid_mask]
        
        if len(strikes) < 4:
            print(f"Error: Insufficient valid data points ({len(strikes)}) after filtering")
            return None, None
        
        # Sort by strike price
        sorted_indices = np.argsort(strikes)
        strikes = strikes[sorted_indices]
        implied_vols = implied_vols[sorted_indices]
        
        # Check for duplicate strikes
        if len(np.unique(strikes)) < len(strikes):
            print("Error: Duplicate strike prices detected")
            return None, None
        
        # Create cubic spline interpolation
        cs = CubicSpline(strikes, implied_vols, bc_type='natural')
        
        # Define interpolation function
        def volatility_interpolator(strike):
            """
            Interpolate implied volatility for a given strike price.
            
            Parameters:
            - strike (float): Strike price to evaluate.
            
            Returns:
            - float: Interpolated implied volatility, or None if strike is out of range.
            """
            if not isinstance(strike, (int, float)) or strike < strikes[0] or strike > strikes[-1]:
                print(f"Strike {strike} is invalid or outside the valid range [{strikes[0]}, {strikes[-1]}]")
                return None
            return cs(strike)
        
        return volatility_interpolator, (strikes[0], strikes[-1])
    
    except Exception as e:
        print(f"Error interpolating volatility: {e}")
        return None, None

# Example usage
if __name__ == "__main__":
    # Sample data
    strikes = [100, 110, 120, 130, 140]
    implied_vols = [0.25, 0.20, 0.18, 0.20, 0.24]
    
    vol_func, strike_range = interpolate_volatility(strikes, implied_vols)
    
    if vol_func is not None:
        print(f"Valid strike range: {strike_range}")
        # Test interpolation
        test_strikes = [105, 115, 135]
        for strike in test_strikes:
            vol = vol_func(strike)
            if vol is not None:
                print(f"Interpolated implied volatility at strike {strike:.2f}: {vol * 100:.2f}%")
    else:
        print("Failed to interpolate volatility")