import numpy as np
import pandas as pd
from datetime import datetime
import yfinance as yf
from matplotlib import cm
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Optional
from scipy.interpolate import interp1d,RectBivariateSpline
from functools import reduce


class BuildVolatilitySurface:
    def __init__(self, symbol: str, expiration_date: Optional[str] = None, fetch_data: bool = True):
        """
        Initialize with a stock symbol and optional expiration date.
        
        Parameters:
            symbol (str): Stock ticker symbol (e.g., 'SPY')
            fetch_data (bool): Whether to fetch option data during initialization (default: True)
        
        Raises:
            ValueError: If symbol is invalid or expiration_date is in the wrong format
        """
        if not isinstance(symbol, str):
            raise ValueError("Symbol must be a non-empty string")
        
        if expiration_date is not None:
            try:
                datetime.strptime(expiration_date, '%Y-%m-%d')
            except ValueError:
                raise ValueError("Expiration date must be in 'YYYY-MM-DD' format")
        
        self.symbol = symbol
        self.expiration_date = expiration_date
        self.data = pd.DataFrame()  # Initialize empty DataFrame
        
        if fetch_data:
            self.data = self.extract_option_prices(self.symbol, expiration_date=expiration_date)
    
    def plot_volatility_smile(self, data:pd.DataFrame, inputed_expiry:str=None):
        """
        Plot volatility smile for a specific expiry and the ATM level.
        
        Parameters:
            data (pd.DataFrame): DataFrame with 'strike', 'impliedVolatility', 'expiry', 'T', 'underlying_price', 'ticker'.
            inputed_expiry (Optional[str]): Expiry date in 'YYYY-MM-DD' format. If None, uses earliest expiry.
        """
        try:
            if (inputed_expiry is None):
                selected_expiry = data['expiry'].loc[0]

            else:
                selected_expiry = data[pd.to_datetime(data['expiry']) > datetime.strptime(inputed_expiry, '%Y-%m-%d')]['expiry'].unique()[0]

            vol_smile_data = data.loc[data['expiry'] == selected_expiry, ['strike', 'impliedVolatility']].reset_index(drop=True)
            if vol_smile_data.empty:
                raise ValueError(f"No data for expiry {selected_expiry}")

            underlying_price = data['underlying_price'].loc[0]
            expiry_set_as_a_ratio = data['T'].loc[0]
            ticker = data['ticker'].loc[0]

            sns.set_style("whitegrid")
            
            plt.figure(figsize=(10, 5))
            sns.lineplot(x=vol_smile_data['strike'], y=vol_smile_data['impliedVolatility'], marker='o', linewidth=2, markersize=5)
            # Interpolate implied volatility at underlying price
            interpolated_vol = np.interp(underlying_price, sorted(vol_smile_data['strike']), [vol_smile_data['impliedVolatility'][i] for i in np.argsort(vol_smile_data['strike'])])
            # Plot vertical line from (underlying_price, interpolated_vol) to (underlying_price, 0)
            plt.plot([underlying_price, underlying_price], [0, interpolated_vol], color='red', linestyle='--', label=f"Underlying Price: {underlying_price:.2f}")
            # Add a marker at the interpolated point
            plt.scatter([underlying_price], [interpolated_vol], s=10, zorder=5)
            
            plt.title(f"Volatility Smile for {ticker} with maturity on {selected_expiry} (T = {expiry_set_as_a_ratio:.4f})",
                    fontweight='bold', fontsize=12)
            plt.xlabel("Strike Price", fontsize=12)
            plt.ylabel("Implied Volatility", fontsize=12)
            plt.legend()
            plt.tight_layout()

            plt.show()

        except Exception as e:
            print(f"Error plotting Vol smile: {e}")

    def plot_volatility_surface(self, data:pd.DataFrame):
        try:
            if data.empty:
                raise ValueError("Input DataFrame is empty")
            # Create 3D plot
            fig = plt.figure(figsize=(12, 8))
            ax = fig.add_subplot(111, projection='3d')
            # Plot surface

            ax.plot_trisurf(data['strike'], data['T'], data['impliedVolatility'], cmap=cm.jet, linewidth=0.2)

            # Labels and title
            ax.set_xlabel('Strike Price')
            ax.set_ylabel('Time to Maturity (Years)')
            ax.set_zlabel('Implied Volatility')
            ax.set_title('Volatility Surface')
            plt.show()

        except Exception as e:
            print(f"Error plotting volatility surface: {e}")


    def get_vols_from_calls_and_puts(self,data:pd.DataFrame, vol_smile_construction_method:str="OTM_Calls_and_OTM_Puts")->pd.DataFrame:
        """
        Filter option data based on volatility smile construction method.
        
        Parameters:
            data (pd.DataFrame): DataFrame with option data.
            vol_smile_construction_method (str): Method to select options ('OTM_Calls_and_OTM_Puts', 'Calls', 'Puts').
        
        Returns:
            pd.DataFrame: Filtered DataFrame with selected options.
        """
        try:
            if data.empty:
                raise ValueError("Input DataFrame is empty")
            options_dfs = []
            
            for expiry in data['expiry'].unique():
                # Select available strikes for OTM Calls and OTM Puts
                if (vol_smile_construction_method == "OTM_Calls_and_OTM_Puts"):
                    otm_calls = data.loc[
                                            (data['expiry'] == expiry) & 
                                            (data['option_type'] == "Call")  & 
                                            (data['strike'] > data['underlying_price']),
                                            ['strike', 'impliedVolatility', 'T', 'expiry','ticker', 'underlying_price']
                                        ]
                    
                    otm_puts = data.loc[
                                            (data['expiry'] == expiry) & 
                                            (data['option_type'] == "Put") & 
                                            (data['strike'] < data['underlying_price']),
                                            ['strike', 'impliedVolatility', 'T', 'expiry','ticker', 'underlying_price']
                                        ]
                    
                    # Adding an additional sorting as insurance
                    otm_calls.sort_values(by="strike",ascending=True, inplace=True )
                    otm_puts.sort_values(by="strike",ascending=True, inplace=True )

                    options_dfs.append(otm_puts)
                    options_dfs.append(otm_calls)

                elif(vol_smile_construction_method == "Calls"):
                    # Select available strikes for OTM Calls and OTM Puts
                    calls = data.loc[
                                        (data['expiry'] == expiry) & 
                                        (data['option_type'] == "Call"),
                                        ['strike', 'impliedVolatility', 'T', 'expiry','ticker', 'underlying_price']
                                    ]
                    # Adding an additional sorting as insurance
                    calls.sort_values(by="strike",ascending=True, inplace=True )
                    options_dfs.append(calls)

                elif(vol_smile_construction_method =="Puts"):
                    # Select available strikes for OTM Calls and OTM Puts
                    puts = data.loc[
                                        (data['expiry'] == expiry) & 
                                        (data['option_type'] == "Put"),
                                        ['strike', 'impliedVolatility', 'T', 'expiry','ticker', 'underlying_price']
                                    ]
                    # Adding an additional sorting as insurance
                    puts.sort_values(by="strike",ascending=True, inplace=True )


                    options_dfs.append(puts)

                else:
                    raise ValueError("Allowed volatility construction methods include: ['OTM_Calls_and_OTM_Puts', 'Calls', 'Puts'] ")

            options_data = pd.concat(options_dfs, ignore_index=True) if options_dfs else pd.DataFrame()

            return options_data
        
        except Exception as e:
            print(f"Error in get_vols_from_calls_and_puts: {e}")
            return pd.DataFrame()

    
    def extract_option_prices(self,symbol:str="SPY", as_of_date:Optional[str]=None, expiration_date:Optional[str] = None)->pd.DataFrame:
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
            dividendYield = stock.info.get("dividendYield", 0.0) or 0.0


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
            
            as_of_date = pd.to_datetime(as_of_date) if as_of_date else datetime.now()
                
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
                # T - expiry is computed in seconds and then converted in days 
                calls['T'] =(pd.to_datetime(expiry) - as_of_date ).total_seconds()/(365*24*60)
                puts['T'] = (pd.to_datetime(expiry) - as_of_date ).total_seconds()/(365*24*60)
                
                # Select relevant columns to include in the output
                columns = ["contractSymbol", "strike", "lastPrice", "bid", "ask", "volume", "openInterest", 'lastTradeDate',
                        "impliedVolatility", "inTheMoney", "ticker", "underlying_price", "expiry","T", "option_type", 'dividendYield']
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
        
    def compute_implied_interest_rates( self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Computes implied interest rates per expiry using put-call parity, interpolating prices if strikes don't match.
        
        Parameters:
            data (pd.DataFrame): DataFrame with option's data
            
        Returns:
        data (pd.DataFrame):  Adds the implied interest rates as a column to the inputted dataframe
        """

        try:
            # Validate input DataFrame
            required_cols = ['strike', 'ticker', 'underlying_price', 'option_type', 'expiry', 'T', 'dividendYield', 'lastPrice']
            if not all(col in data.columns for col in required_cols):
                raise ValueError("DataFrame must contain 'strike', 'lastPrice','underlying_price', 'option_type', 'expiry' and 'T' columns")
            if data.empty:
                raise ValueError("Input DataFrame is empty")

            # Unique expiries
            expiries_unique = np.sort(data['T'].unique())
            #Get the divident yield - it is inputed as percentage
            q = data['dividendYield'][0]/100
            #Get the underlying price
            S = data['underlying_price'][0]
            
            implied_interest_rates = {}

            for T in expiries_unique:
                data_T = data[data['T'] == T]
                
                # Separate call and put data
                call_data = data_T.loc[data_T['option_type'] == "Call", ['strike', 'option_type', 'lastPrice']]
                put_data = data_T.loc[data_T['option_type'] == "Put", ['strike', 'option_type', 'lastPrice']]
                
                # Get unique strikes for calls and puts
                call_strikes = np.sort(call_data['strike'].unique())
                put_strikes = np.sort(put_data['strike'].unique())
                
                # Find common strikes
                common_strikes = np.intersect1d(call_strikes, put_strikes)
                
                # If no common strikes, interpolate to align strikes
                if len(common_strikes) == 0:
                    if len(call_strikes) < 2 or len(put_strikes) < 2:
                        implied_interest_rates[T] = np.nan  # Insufficient data for interpolation
                        continue
                    
                    # Interpolate call prices at put strikes
                    call_interp = interp1d(call_strikes, call_data.set_index('strike')['lastPrice'], 
                                        bounds_error=False, fill_value='extrapolate')
                    put_interp = interp1d(put_strikes, put_data.set_index('strike')['lastPrice'], 
                                        bounds_error=False, fill_value='extrapolate')
                    
                    # Use all unique strikes
                    all_strikes = np.sort(np.union1d(call_strikes, put_strikes))
                    rates = []
                    for K in all_strikes:
                        try:
                            C = call_interp(K)
                            P = put_interp(K)
                            if C >= 0 and P >= 0 and K > 0:
                                # Compute interest rate using Put-call parity: C - P = S e^(-qT) - K e^(-rT)
                                
                                r = -np.log((S * np.exp(-q * T) - (C - P)) / K) / T
                                if -1 < r < 1:  # Filter unrealistic rates
                                    rates.append(r)
                        except:
                            continue
                else:
                    # Use common strikes directly
                    rates = []
                    for K in common_strikes:
                        C = call_data[call_data['strike'] == K]['lastPrice'].iloc[0]
                        P = put_data[put_data['strike'] == K]['lastPrice'].iloc[0]
                        try:
                            if C >= 0 and P >= 0 and K > 0:
                                r = - np.log((S * np.exp(-q * T) - (C - P)) / K) / T

                                if -1 < r < 1:  # Filter unrealistic rates
                                    rates.append(r)
                        except:
                            continue
                
                # Average rates for this expiry
                implied_interest_rates[T] = np.nanmean(rates) if rates else np.nan

            # Adding average interest rates for each maturity T in the original dataset
            data['implied_interest_rate'] = data['T'].map(implied_interest_rates).astype(float)

            
            
            return data
        
        except Exception as e:
            print(f"Error computing implied interest rates: {e}")
            return {}
        
    def compute_local_volatility(self, data: pd.DataFrame, num_strikes: int = 50, num_expiries: int = 50) -> pd.DataFrame:
        """
        Compute local volatility surface from a DataFrame with option prices, strikes, maturities, and interest rates.
        Uses call prices directly or estimates them from put prices via put-call parity if call prices are missing.
        
        Parameters:
        data (pd.DataFrame): DataFrame with columns 'strike','T','option_type','impliedVolatility' 'lastPrice','underlying_price',
                            'dividendYield' and 'implied_interest_rate' to estimate missing call prices.


        num_strikes (int): Number of strike points in output grid (default: 50).
        num_expiries (int): Number of expiry points in output grid (default: 50).
        
        Returns:
            pd.DataFrame: DataFrame with columns 'strike', 'T', and 'LocalVolatility'.
        """
        try:

            #Update the inputted dataframe to include the implied interest rates
            data = self.compute_implied_interest_rates(data)

            # Validate input DataFrame
            required_cols = ['strike','T','option_type','impliedVolatility' ,'lastPrice','underlying_price','dividendYield', 'implied_interest_rate']

            if not all(col in data.columns for col in required_cols):
                raise ValueError("DataFrame must contain 'strike','T','option_type','impliedVolatility' 'lastPrice','underlying_price','dividendYield' and 'implied_interest_rate' columns")
            

            if data.empty:
                raise ValueError("Input DataFrame is empty")

            # Create a copy to avoid modifying the original
            df = data.copy()
            
            
            # Unique strikes and expiries
            # strikes_unique = np.sort(df['strike'].unique())
            # expiries_unique = np.sort(df['T'].unique())

            strikes_unique = np.sort(df[df['option_type']=="Call"]['strike'].unique())
            expiries_unique = np.sort(df[df['option_type']=="Call"]['T'].unique())


            S = df['underlying_price'][0]
            q = df['dividendYield'][0]

            
            # Interpolate call price surface, using put prices if call prices are missing
            # For Dupire equation only the prices of the Calls are required
            price_grid = np.zeros((len(expiries_unique), len(strikes_unique)))

            for i, T in enumerate(expiries_unique):
                r = df[df['T'] == T]['implied_interest_rate'].iloc[0]
                if np.isnan(r):
                    print(f"Warning: No valid interest rate for T={T}, using default r=0.012")
                    r = 0.012
                for j, K in enumerate(strikes_unique):

                    #Warning: for different maturites the Strikes are not the same 
                    subset = df[(df['T'] == T) & (df['strike'] == K)]

                    if not subset.empty:
                        call_subset = subset[subset['option_type'] == 'Call']
                        put_subset = subset[subset['option_type'] == 'Put']

                        if not call_subset.empty and not call_subset['lastPrice'].isna().any():
                            price_grid[i, j] = call_subset['lastPrice'].iloc[0]


                        elif not put_subset.empty and not put_subset['lastPrice'].isna().any():

                            # Use put-call parity: C = P + S e^(-qT) - K e^(-rT)
                            P = put_subset['lastPrice'].iloc[0]
                            price_grid[i, j] = P + S * np.exp(-q * T) - K * np.exp(-r * T)

                        else:
                            price_grid[i, j] = np.nan
                    else:
                        price_grid[i, j] = np.nan

            
            # Handle NaN in price grid
            price_grid = np.nan_to_num(price_grid, nan=0.0)


            interp_price = RectBivariateSpline(expiries_unique, strikes_unique, price_grid, kx=2, ky=2)
            
            # Create output grid
            strikes = np.linspace(min(strikes_unique), max(strikes_unique), num_strikes)
            expiries = np.linspace(min(expiries_unique), max(expiries_unique), num_expiries)
            local_vol_grid = np.zeros((num_expiries, num_strikes))


            # Interpolate interest rates
            # Get unique (T, implied_interest_rate) pairs only for Call options
            implied_rates_data = df[df['option_type'] == "Call"][['T', 'implied_interest_rate']].drop_duplicates()
        
            if len(implied_rates_data) < 2:
                print("Warning: Insufficient valid interest rate data, using default r=0.012 for all expiries.")
                interpolated_implied_rates = lambda x: 0.012
            else:
                # Sort by T to ensure monotonicity for interpolation
                implied_rates_data = implied_rates_data.sort_values('T')
                interpolated_implied_rates = interp1d(
                    implied_rates_data['T'],
                    implied_rates_data['implied_interest_rate'],
                    kind='linear',
                    bounds_error=False,
                    fill_value=0.012  # Default rate outside bounds
                )
                        
            # Small increments for numerical derivatives
            dT = 1e-2
            dK = 1e-2
            
            for i, T in enumerate(expiries):
                # Interpolate interest rate for this T
                r = interpolated_implied_rates(T)
                if np.isnan(r):
                    print(f"Warning: No valid interest rate for T={T}, using default r=0.012")
                    r = 0.012
                for j, K in enumerate(strikes):
                    # Get call price
                    C = interp_price(T, K)[0][0]
                    # a sanity check for the Interpolation results
                    if (T <= 0 or C < 0):
                        C = max(S - K, 0)
                    
                    # Numerical derivatives using interpolation
                    C_dT = interp_price(T + dT, K)[0][0] if T + dT > 0 else max(S - K, 0)
                    dC_dT = (C_dT - C) / dT
                    
                    C_dK = interp_price(T, K + dK)[0][0] if T > 0 else max(S - (K + dK), 0)
                    dC_dK = (C_dK - C) / dK
                    
                    C_dK2 = interp_price(T, K + 2*dK)[0][0] if T > 0 else max(S - (K + 2*dK), 0)
                    d2C_dK2 = (C_dK2 - 2*C_dK + C) / (dK**2)
                    
                    # Dupire formula
                    numerator = dC_dT + (r - q) * K * dC_dK + q * C
                    denominator = 0.5 * K**2 * d2C_dK2
                    try:
                        if denominator > 0 and numerator > 0:
                            local_vol_grid[i, j] = np.sqrt(numerator / denominator)
                        else:
                            local_vol_grid[i, j] = np.nan
                    except (ZeroDivisionError, ValueError):
                        local_vol_grid[i, j] = np.nan
            
            # Create output DataFrame
            strike_grid, expiry_grid = np.meshgrid(strikes, expiries)
            result_df = pd.DataFrame({
                'strike': strike_grid.ravel(),
                'T': expiry_grid.ravel(),
                'LocalVolatility': local_vol_grid.ravel()
            })
            
            return result_df
        
        except Exception as e:
            print(f"Error computing local volatility: {e}")
            return pd.DataFrame(columns=['strike', 'T', 'LocalVolatility'])


    def plot_Local_volatility_surface(self, data:pd.DataFrame):
        try:
            if data.empty:
                raise ValueError("Input DataFrame is empty")
            # Create 3D plot
            fig = plt.figure(figsize=(12, 8))
            ax = fig.add_subplot(111, projection='3d')
            # Plot surface

            ax.plot_trisurf(data['strike'], data['T'], data['LocalVolatility'], cmap=cm.jet, linewidth=0.2)

            # Labels and title
            ax.set_xlabel('Strike Price')
            ax.set_ylabel('Time to Maturity (Years)')
            ax.set_zlabel('Implied Volatility')
            ax.set_title('Volatility Surface')
            plt.show()

        except Exception as e:
            print(f"Error plotting volatility surface: {e}")



if __name__ == "__main__":
    
    vol_surface = BuildVolatilitySurface(symbol="SPY")
    # vol_surface = BuildVolatilitySurface(symbol="SPY", expiration_date='2025-12-30')
    # data = vol_surface.get_vols_from_calls_and_puts(vol_surface.data, "OTM_Calls_and_OTM_Puts")
    # vol_surface.plot_volatility_smile(data, inputed_expiry=None)
    # vol_surface.plot_volatility_surface(data)
    local_vol = vol_surface.compute_local_volatility(vol_surface.data)

    vol_surface.plot_Local_volatility_surface(local_vol)



