import numpy as np
from scipy.stats import norm
import typing



class ExtractImpliedVol:
    def __init__(self,method:str,  S:float, K:float, T:float, r:float,q:float, option_market_price:float, 
                 option_type:str='Call',  max_iterations:int=1e3, tol=1e-6):
        """
        Initialize the Implied Volatility calculator.
        
        Parameters:
            S (float): Spot price of the underlying asset
            K (float): Strike price
            T (float): Time to maturity (in years)
            r (float): Risk-free interest rate
            option_price (float): Observed market price of the option
            option_type (str): 'call' or 'put'
            q (float): Dividend yield (default = 0)
            """
        self.method = method
        self.S = S
        self.K = K
        self.T = T
        self.r = r
        self.q = q 
        self.option_market_price = option_market_price
        self.option_type = option_type
        self.max_iterations = max_iterations
        self.tol = tol

        if (option_type not in ("Call", "Put")):
            raise ValueError("option_type must be either 'Call' or 'Put'")

        if ( self.method== "Newton-Raphson"):
            self.implied_vol = self.Newton_Raphson_method_impied_vol(option_market_price=self.option_market_price, S=self.S, K=self.K, T=self.T, r=self.r, 
                                                  option_type=self.option_type, q=self.q, tol=self.tol, max_iterations=self.max_iterations, 
                                                  initial_guess_vol=4.0)
        elif ( method== "Bisect"):
            self.implied_vol = self.Bisection_method_implied_vol(option_market_price=option_market_price, S = S, K = K, T = T, r = r, option_type=option_type, 
                                              q = q, tol = tol, max_iterations=max_iterations, initial_min_vol=1e-6, initial_max_vol=5.0)
        elif ( method== "Secant"):
            self.implied_vol = self.Secant_method_implied_vol(option_market_price=option_market_price, S = S, K = K, T = T, r = r, option_type=option_type, q = q, 
                                           tol=tol, max_iterations=max_iterations, initial_min_vol=1e-6, initial_max_vol=5.0)
        elif (method =="Brent"):
            self.implied_vol = self.Brent_method_implied_vol(option_market_price=option_market_price, S = S, K = K, T = T, r = r, option_type=option_type, q = q, 
                                          tol = tol, max_iterations=max_iterations, initial_min_vol=1e-6, initial_max_vol=5.0)

        else:
            raise ValueError("Allowed methods include: ['Newton-Raphson', 'Bisect', 'Secant', 'Brent'] ")


    def __Black_Scholes_option_price(self, S:float, K:float, T:float, r:float, sigma:float, option_type:str ='Call', q:float=0.0)->float:
        """_
        Computes option price using Black-Scholes model
        Parameters:
            S (float): Current Underlying Price
            K (float): Strike Price
            T (float): Time to maturity set in years ( T/256 )
            r (float): risk-free interest rate
            sigma (float): Realized volatility
            option_type (str, optional): Allowed only 'Call' or 'Put'. Defaults to 'Call'.

        Raises:
            ValueError: option_type must be either 'Call' or 'Put'

        Returns:
            float: Option Price

        Comment: For testing purposed, the divident yield has been set to zero (q = 0).
        """
        d1 = (np.log(S / K) + (r - q + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        
        if option_type == 'Call':
            price = S * np.exp(-q * T) * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)

        else:
            price = K * np.exp(-r * T) * norm.cdf(-d2) - S*np.exp(-q * T) * norm.cdf(-d1)
    
        return price

    def __Black_Scholes_Vega(self, S:float, K:float, T:float, r:float, sigma:float, q:float=0.0)->float:
        """Computed Vega based on the closed-form solution estimated using Black-Scholes

        Args:
            S (float): Current Underlying Price
            K (float): Strike Price
            T (float): Time to maturity estimated as T/256
            r (float): risk-free interest rate
            sigma (float): Realized volatility

        Returns:
            float: Vega of an option
        """
        d1 = (np.log(S / K) + (r -q + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
        vega = S * np.exp(-q*T)*norm.pdf(d1) * np.sqrt(T)
        return vega

    def Newton_Raphson_method_impied_vol(self, option_market_price:float, S:float, K:float, T:float, r:float, 
                                        option_type:str='Call', q:float=0.0, tol:float=1e-9, max_iterations:int=1e3, 
                                        initial_guess_vol=5.0)->float:
        """
        Extract implied volatility using Newton-Raphson method
        Key Idea: The root of this function is the place where the function intercepts the x-axis

        x1 = x0 - f(x0)/f'(x0)
        
        Parameters:
            option_price(float): observed market price of the option
            S(float): current price of the underlying asset
            K(float): strike price of the option
            T(float): time to expiration (in years)
            r(float): risk-free interest rate
            option_type(str): 'Call' or 'Put'
            q(float): Dividend yield
            tol(float): tolerance for convergence
            max_iterations(int): maximum number of iterations
            
        Returns:
            implied volatility (sigma)
        """


        # Initial guess for implied volatility
        sigma = initial_guess_vol
        
        for i in range(int(max_iterations)):
            # Calculate price and vega with current sigma
            black_scholes_price = self.__Black_Scholes_option_price(S, K, T, r, sigma, option_type, q)
            vega = self.__Black_Scholes_Vega(S, K, T, r, sigma, q)
            
            # Check if vega is too small to avoid division by zero
            if vega < tol:
                break
                
            # Newton-Raphson update
            price_difference = option_market_price - black_scholes_price
            # In this case the sign is plus, due to the shape of curve (option_markt_price - BS price)/Vega
            sigma += price_difference / vega
        
            
            # Check for convergence
            if abs(price_difference) < tol:
                break
        
        return float(sigma)

    def Bisection_method_implied_vol(self, option_market_price:float, S:float, K:float, T:float, r:float, 
                                            option_type:str="Call", q:float=0.0,tol:float=1e-9, max_iterations:int=1e3,
                                            initial_min_vol:float=1e-6, initial_max_vol:float=5.0)->float:
        
        """Calculate implied volatility using the bisection method.

        Key Idea: The method sues a binary search approach. 
        Continuous function f in interval [a, b]. Values f(a) and f(b) are of opposite sign. Each iteration performs these steps:
        
        Parameters:
            market_price (float): Market price of the option
            S (float): Current stock price
            K (float): Strike price
            T (float): Time to maturity (in years)
            r (float): Risk-free interest rate
            option_type (str):'Call' or 'Put'
            q(float): Dividend yield
            tol (float): Tolerance for convergence
            max_iterations (int): Maximum number of iterations, it will be floored to a value of type int
            initial_min_vol (float): Initial guess of the lower bound of volatility
            initial_max_vol (float): Initial guess of the upper bound of volatility
        
        Returns:
            float: Implied volatility, or None if no convergence
        """
        # Initialize volatility bounds
        sigma_low = initial_min_vol  # Lower bound for volatility
        sigma_high = initial_max_vol   # Upper bound for volatility


        
        for _ in range(int(max_iterations)):

            # Calculate midpoint volatility
            sigma_mid = (sigma_low + sigma_high) / 2

            # Calculate option price at midpoint
            price_mid = self.__Black_Scholes_option_price(S, K, T, r, sigma_mid, option_type, q)
            
            # Check if price matches market price within tolerance
            if abs(price_mid - option_market_price) < tol:
                return sigma_mid
            
            # Adjust bounds based on whether price is too high or too low
            if price_mid < option_market_price:
                sigma_low = sigma_mid
            else:
                sigma_high = sigma_mid
                
            # Check if bounds are close enough
            if sigma_high - sigma_low < tol:
                return sigma_mid
        
        # Return None if no convergence within max iterations
        return None



    def Secant_method_implied_vol(self, option_market_price:float, S:float, K:float, T:float, r:float,
                                option_type:str="Call", q:float=0.0, tol:float=1e-9, max_iterations:int=1e3, 
                                initial_min_vol:float=1e-6, initial_max_vol:float=5.0)->float:
        """Extract implied volatility using the secant method.

        Key Idea: Similar to Newton-Raphson, except that the derivative is replaced by an approximation

        Parameters:
            option_market_price (float): Market price of the option
            S (float): Current stock price
            K (float): Strike price
            T (float): Time to maturity (in years)
            r (float): Risk-free interest rate
            option_type (str):'Call' or 'Put'
            q(float): Dividend yield
            tol (float): Tolerance for convergence
            max_iterations (int): Maximum number of iterations, it will be floored to a values of type int.
            initial_min_vol (float): Initial guess of the lower bound of volatility
            initial_max_vol (float): Initial guess of the upper bound of volatility
        
        Returns:
        float: Implied volatility, or None if no convergence
        """

        # Initial guesses of sigma_min and sigma_max 
        sigma_min = initial_min_vol
        sigma_max = initial_max_vol

        objective_function = lambda sigma: self.__Black_Scholes_option_price(S, K, T, r, sigma, option_type,q) - option_market_price 

        # Calculate initial option prices
        f0 = objective_function(sigma=sigma_min)
        f1 = objective_function(sigma=sigma_max)
        
        for _ in range(int(max_iterations)):

            # Check for division by zero or very small denominator
            if abs(f1 - f0) < tol:
                return sigma_max
            
            # Secant method update
            sigma_new = sigma_max - f1 * (sigma_max - sigma_min) / (f1 - f0)
            
            # Ensure volatility remains positive
            sigma_new = max(1e-8, sigma_new)
            
            # Calculate new option price
            f_new = objective_function(sigma=sigma_new)
            
            # Check for convergence
            if abs(f_new) < tol or abs(sigma_new - sigma_max) < tol:
                return sigma_new
            
            # Update values for next iteration
            sigma_min, sigma_max = sigma_max, sigma_new
            f0, f1 = f1, f_new
        
        # Return None if no convergence within max iterations
        return None


        
    def Brent_method_implied_vol(self, option_market_price:float,S:float, K:float, T:float, r:float, 
                                option_type:str='Call', q:float=0.0, tol:float=1e-6, max_iterations:int=1e3, 
                                initial_min_vol:float=1e-6, initial_max_vol:float=4.0)->float:
        """
        Extract implied volatility using Brent's method.
        Key Idea: Brent's Method combines inverse quadratic interpolation, Secant method and Bisect method

        Parameters:
            market_price (float): option market price
            S (float):Current stock price
            K (float): Strike price
            T (float): Time to expiration (in years, t/T)
            r (float) risk-free interest rate
            q (float):Dividend yield
            option_type (str): 'Call' or 'Put'
            tol (float): Tolerance for convergence
            max_iterations(int): Maximum number of iterations
            initial_min_vol (float): Initial guess of the lower bound of volatility
            initial_max_vol (float): Initial guess of the upper bound of volatility
        
        Returns:
        float - implied volatility
        """

        # Initial guesses for volatility
        a = initial_min_vol # Lower bound
        b = initial_max_vol    # Upper bound 
        
        objective_function = lambda sigma: self.__Black_Scholes_option_price(S, K, T, r, sigma, option_type, q) - option_market_price

        #Objective functions: difference between BS price and observed market prices
        fa = objective_function(sigma=a)
        fb = objective_function(sigma=b)
        
        # Check if root is bracketed - the function bracket must contain f(x) = 0
        # Required condition: f(simga_min) * f(sigma_max) < 0
        if (fa * fb >= 0):
            raise ValueError("Root not bracketed. Try different initial guesses.")
        
        # Check condition: |f(sigma_min)| >= |f(sigma_max)|
        # improves efficiency as sigma_min is closer to the root.
        if abs(fa) < abs(fb):
            a, b = b, a
            fa, fb = fb, fa
        
        # Initialize additional variables used Brent's method
        c = a  
        fc = fa
        d = b - a #step size from previous iteration
        e = d #step size from two iterations ago
        
        iter_count = 0
        while (iter_count < int(max_iterations)):
            if (abs(fb) < tol):
                return b  # Converged
            
            if (fa != fc and fb != fc):
                # Inverse quadratic interpolation
                s = a * fb * fc / ((fa - fb) * (fa - fc)) + \
                    b * fa * fc / ((fb - fa) * (fb - fc)) + \
                    c * fa * fb / ((fc - fa) * (fc - fb))
            else:
                # Secant method
                s = b - fb * (b - a) / (fb - fa)
            
            # Check if s is within bounds and satisfies convergence criteria
            if (s < (3 * a + b) / 4 or s > b) or \
            (abs(s - b) >= abs(b - c) / 2 and c != a) or \
            (abs(s - b) >= abs(c - d) / 2 and d != c):
                #If the above condition meets - the Bisection method activates
                s = (a + b) / 2
                d = e = b - a
            else:
                d = e
                e = b - s
            
            fs = objective_function(sigma=s)
            
            if abs(fs) < tol:
                return s  # Converged
            
            # Update points
            if fa * fs < 0:
                b = s
                fb = fs
            else:
                a = s
                fa = fs
            
            # Keep c as the previous b (before switch)
            if abs(fa) < abs(fb):
                a, b = b, a
                fa, fb = fb, fa
            c = a
            fc = fa
            
            iter_count += 1
        
        raise ValueError("Brent's method did not converge within maximum iterations.")


if __name__ == "__main__":
    
    ##############################################
    # Testing for Call Option
    ###############################################

    ExtractImpliedVol(method = "Newton-Raphson", S = 100, K = 105, T = 0.5, r = 0.03, q = 0.0, option_market_price= 5, 
                      option_type="Call", max_iterations=1e3, tol=1e-6).implied_vol

    ExtractImpliedVol(method = "Bisect", S = 100, K = 105, T = 0.5, r = 0.03, q = 0.0, option_market_price= 5, 
                      option_type="Call", max_iterations=1e3, tol=1e-6).implied_vol

    ExtractImpliedVol(method = "Secant", S = 100, K = 105, T = 0.5, r = 0.03, q = 0.0, option_market_price= 5, 
                    option_type="Call", max_iterations=1e3, tol=1e-6).implied_vol

    ExtractImpliedVol(method = "Brent", S = 100, K = 105, T = 0.5, r = 0.03, q = 0.0, option_market_price= 5, 
                option_type="Call", max_iterations=1e3, tol=1e-6).implied_vol

    ##############################################
    # Testing for Put Option
    ###############################################

    ExtractImpliedVol(method = "Newton-Raphson", S = 100, K = 95, T = 0.5, r = 0.03, q = 0.0, option_market_price= 5, 
                      option_type="Put", max_iterations=1e3, tol=1e-6).implied_vol

    ExtractImpliedVol(method = "Bisect", S = 100, K = 95, T = 0.5, r = 0.03, q = 0.0, option_market_price= 5, 
                      option_type="Put", max_iterations=1e3, tol=1e-6).implied_vol

    ExtractImpliedVol(method = "Secant", S = 100, K = 95, T = 0.5, r = 0.03, q = 0.0, option_market_price= 5, 
                    option_type="Put", max_iterations=1e3, tol=1e-6).implied_vol

    ExtractImpliedVol(method = "Brent", S = 100, K = 95, T = 0.5, r = 0.03, q = 0.0, option_market_price= 5, 
                option_type="Put", max_iterations=1e3, tol=1e-6).implied_vol














    # S = 100       # Current stock price
    # K = 105       # Strike price
    # T = 0.5       # Time to expiration (6 months)
    # r = 0.05      # Risk-free rate (5%)
    # option_type = 'Call' #option type

    # option_market_price = 5

    # newton_raphson_implied_vol = Newton_Raphson_method_impied_vol(option_market_price, S, K, T, r, option_type)
    # bisect_implied_vol = Bisection_method_implied_vol(option_market_price, S, K, T, r, option_type)
    # secant_implied_vol = Secant_method_implied_vol(option_market_price, S, K, T, r, option_type)
    # brent_implied_vol = Brent_method_implied_vol(option_market_price, S, K, T, r, option_type)
    # print(f"Newton-Raphson method vol: {newton_raphson_implied_vol:.5f}")
    # print(f"Bisection method vol: {bisect_implied_vol:.5f}")
    # print(f"Secant method vol: {secant_implied_vol:.5f}")
    # print(f"Brent method vol: {brent_implied_vol:.5f}")

    # print(f"Newton-Raphon:{Black_Scholes_option_price(S, K, T, r, newton_raphson_implied_vol):.5f}")
    # print(f"Bisection method:{Black_Scholes_option_price(S, K, T, r, bisect_implied_vol):.5f}")
    # print(f"Secant method:{Black_Scholes_option_price(S, K, T, r, secant_implied_vol):.5f}")
    # print(f"Brent method:{Black_Scholes_option_price(S, K, T, r, brent_implied_vol):.5f}")

    # option_type = 'Put' #option type

    # newton_raphson_implied_vol = Newton_Raphson_method_impied_vol(option_market_price, S, K, T, r, option_type)
    # bisect_implied_vol = Bisection_method_implied_vol(option_market_price, S, K, T, r, option_type)
    # secant_implied_vol = Secant_method_implied_vol(option_market_price, S, K, T, r, option_type)
    # brent_implied_vol = Brent_method_implied_vol(option_market_price, S, K, T, r, option_type)
    # print(f"Newton-Raphson method vol: {newton_raphson_implied_vol:.5f}")
    # print(f"Bisection method vol: {bisect_implied_vol:.5f}")
    # print(f"Secant method vol: {secant_implied_vol:.5f}")
    # print(f"Brent method vol: {brent_implied_vol:.5f}")

    # print(f"Newton-Raphon:{Black_Scholes_option_price(S, K, T, r, newton_raphson_implied_vol):.5f}")
    # print(f"Bisection method:{Black_Scholes_option_price(S, K, T, r, bisect_implied_vol):.5f}")
    # print(f"Secant method:{Black_Scholes_option_price(S, K, T, r, secant_implied_vol):.5f}")
    # print(f"Brent method:{Black_Scholes_option_price(S, K, T, r, brent_implied_vol):.5f}")