
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
import scipy.optimize

from datetime import datetime
import yfinance as yf
from typing import Optional


class BuildVolatilitySurface():
    def __init__(self, quote):
        self.quote = quote
        self.load_data()



    def extract_option_prices(self, symbol:str="SPY", expiration_date:Optional[str] = None)->pd.DataFrame:
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
                
                # Add ticker, underlying price, expiration date, and option type
                calls["ticker"] = symbol
                puts["ticker"] = symbol
                calls["underlying_price"] = underlying_price
                puts["underlying_price"] = underlying_price
                calls["expiry"] = expiry
                puts["expiry"] = expiry
                calls["option_type"] = "Call"
                puts["option_type"] = "Put"
                
                # Select relevant columns to include in the output
                columns = ["contractSymbol", "strike", "lastPrice", "bid", "ask", "volume", "openInterest", 'lastTradeDate',
                        "impliedVolatility", "inTheMoney", "ticker", "underlying_price", "expiry", "option_type"]
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
