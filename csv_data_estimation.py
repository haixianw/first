import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from scipy.stats import norm
import os
from datetime import datetime
from matplotlib.backends.backend_pdf import PdfPages
import io
from sde_parameter_estimation import SDEParameterEstimator

# Set matplotlib to use a locale that supports UTF-8
mpl.rcParams['font.family'] = 'DejaVu Sans'

def load_futures_data(csv_file_path, price_column='收盘价', date_column='交易时间'):
    """
    Load futures price data from CSV file
    
    Parameters:
    - csv_file_path: Path to CSV file
    - price_column: Price column name (default: '收盘价')
    - date_column: Date column name (default: '交易时间')
    
    Returns:
    - prices: Price array (numpy array)
    - dates: Date array
    - dt: Estimated time step (annualized)
    """
    # Read CSV file
    try:
        data = pd.read_csv(csv_file_path)
        print(f"Successfully read CSV file with {len(data)} rows")
    except Exception as e:
        print(f"Error reading CSV file: {e}")
        return None, None, None
    
    # Print all column names for debugging
    print(f"CSV file columns: {data.columns.tolist()}")
    
    # Check if required columns exist
    required_columns = [price_column]
    if date_column:
        required_columns.append(date_column)
    
    missing_columns = [col for col in required_columns if col not in data.columns]
    if missing_columns:
        print(f"Missing required columns in CSV file: {missing_columns}")
        print(f"Available columns: {data.columns.tolist()}")
        return None, None, None
    
    # Handle thousands separator in price data
    if data[price_column].dtype == object:
        try:
            # Remove thousands separator and convert to float
            data[price_column] = data[price_column].str.replace(',', '').astype(float)
            print("Processed thousands separators in price data")
        except Exception as e:
            print(f"Error processing price data: {e}")
    
    # If date column exists, convert to datetime format and sort by date
    if date_column in data.columns:
        try:
            data[date_column] = pd.to_datetime(data[date_column])
            data = data.sort_values(by=date_column)
            
            # Calculate time step (daily average)
            time_diffs = data[date_column].diff()[1:].dt.days
            avg_days = time_diffs.mean()
            dt = 1/252 if avg_days < 2 else avg_days/365  # Assume daily data if average diff < 2
            
            dates = data[date_column].values
        except Exception as e:
            print(f"Error processing date column: {e}")
            dt = 1/252  # Default to daily data
            dates = np.arange(len(data))
    else:
        dt = 1/252  # Default to daily data
        dates = np.arange(len(data))
    
    # Extract price data
    prices = data[price_column].values
    
    # Check and handle missing values
    if np.isnan(prices).any():
        print(f"Warning: {np.isnan(prices).sum()} missing values in price data, will use forward fill")
        prices = pd.Series(prices).fillna(method='ffill').values
    
    print(f"Data loading complete: {len(prices)} price points, time step dt={dt:.6f} years")
    
    # Display data range
    print(f"Price range: Min={np.min(prices):.2f}, Max={np.max(prices):.2f}, Mean={np.mean(prices):.2f}")
    print(f"Date range: {pd.to_datetime(dates[0]).date()} to {pd.to_datetime(dates[-1]).date()}")
    
    return prices, dates, dt

def visualize_raw_data(prices, dates=None, title="Raw Price Data", return_fig=False):
    """
    Visualize raw price data
    """
    fig, ax = plt.subplots(figsize=(12, 6))
    
    if dates is not None and len(dates) == len(prices):
        ax.plot(dates, prices, 'b-')
        # Set appropriate date format
        fig.autofmt_xdate()
    else:
        ax.plot(prices, 'b-')
    
    ax.set_title(title)
    ax.set_ylabel("Price")
    ax.grid(True)
    
    if return_fig:
        return fig
    else:
        plt.tight_layout()
        plt.show()

def analyze_returns(prices, log_returns=True, return_fig=False):
    """
    Analyze price returns distribution
    """
    if log_returns:
        # Calculate log returns
        returns = np.diff(np.log(prices))
        return_type = "Log Returns"
    else:
        # Calculate simple returns
        returns = np.diff(prices) / prices[:-1]
        return_type = "Simple Returns"
    
    # Plot returns distribution
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Time series
    ax1.plot(returns, 'b-')
    ax1.set_title(f"{return_type} Time Series")
    ax1.set_xlabel("Time")
    ax1.set_ylabel(return_type)
    ax1.grid(True)
    
    # Histogram
    ax2.hist(returns, bins=50, density=True, alpha=0.7)
    ax2.set_title(f"{return_type} Distribution")
    ax2.set_xlabel(return_type)
    ax2.set_ylabel("Frequency Density")
    ax2.grid(True)
    
    # Calculate statistics
    mean_return = np.mean(returns)
    std_return = np.std(returns)
    skew = np.mean(((returns - mean_return) / std_return) ** 3)
    kurtosis = np.mean(((returns - mean_return) / std_return) ** 4) - 3
    
    # Display statistics
    stats_text = f"Mean: {mean_return:.6f}\nStd Dev: {std_return:.6f}\nSkewness: {skew:.4f}\nKurtosis: {kurtosis:.4f}"
    ax2.text(0.95, 0.95, stats_text, transform=ax2.transAxes, verticalalignment='top', horizontalalignment='right', bbox=dict(facecolor='white', alpha=0.7))
    
    plt.tight_layout()
    
    if return_fig:
        return fig, returns
    else:
        plt.show()
        return returns

def calculate_parameter_significance(estimated_params, prices, L_estimated, dt):
    """
    Calculate parameter significance using Hessian-based standard errors
    
    Parameters:
    - estimated_params: Estimated parameters
    - prices: Observed prices
    - L_estimated: Estimated liquidity
    - dt: Time step
    
    Returns:
    - param_stats: DataFrame with parameter statistics
    """
    from scipy.optimize import approx_fprime
    
    # Create a temporary estimator object
    temp_estimator = SDEParameterEstimator(prices, dt)
    
    # Function to compute negative log-likelihood
    def nll_func(params):
        return temp_estimator.negative_log_likelihood(params)
    
    # Compute Hessian numerically (approximation)
    eps = 1e-5  # Step size for finite difference
    n_params = len(estimated_params)
    hessian = np.zeros((n_params, n_params))
    
    for i in range(n_params):
        for j in range(n_params):
            # Compute second partial derivative
            def f_i(x_i):
                params_i = np.copy(estimated_params)
                params_i[i] = x_i
                return approx_fprime([params_i[j]], lambda x: nll_func(
                    np.array([p if k != j else x[0] for k, p in enumerate(params_i)])
                ), eps)[0]
            
            hessian[i, j] = approx_fprime([estimated_params[i]], f_i, eps)[0]
    
    # Ensure Hessian is symmetric
    hessian = (hessian + hessian.T) / 2
    
    try:
        # Compute covariance matrix as inverse of Hessian
        # We use pseudo-inverse for numerical stability
        covariance = np.linalg.pinv(hessian)
        
        # Extract standard errors (square root of diagonal elements)
        std_errors = np.sqrt(np.diag(covariance))
        
        # Z-scores and p-values
        z_scores = estimated_params / std_errors
        p_values = 2 * (1 - norm.cdf(np.abs(z_scores)))
        
        # Create results DataFrame
        param_names = ["r", "alpha", "beta", "theta_bar", "sigma_S", "sigma_L", "rho1", "rho2", "rho3"]
        
        param_stats = pd.DataFrame({
            'Parameter': param_names,
            'Estimate': estimated_params,
            'Std Error': std_errors,
            'z-value': z_scores,
            'Pr(>|z|)': p_values,
            'Significance': ['***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else '.' if p < 0.1 else '' for p in p_values]
        })
        
        return param_stats
    except np.linalg.LinAlgError:
        print("Warning: Could not compute parameter significance due to numerical issues")
        
        # Create simplified results DataFrame without significance
        param_names = ["r", "alpha", "beta", "theta_bar", "sigma_S", "sigma_L", "rho1", "rho2", "rho3"]
        param_stats = pd.DataFrame({
            'Parameter': param_names,
            'Estimate': estimated_params
        })
        
        return param_stats

def plot_estimated_vs_true(L_estimated, dates=None, title="Estimated Liquidity Process", return_fig=False):
    """
    Plot the estimated latent process (liquidity)
    """
    fig, ax = plt.subplots(figsize=(12, 6))
    
    if dates is not None and len(dates) == len(L_estimated):
        ax.plot(dates, L_estimated, 'r-', label='Estimated Liquidity (L)')
        # Set appropriate date format
        fig.autofmt_xdate()
    else:
        ax.plot(L_estimated, 'r-', label='Estimated Liquidity (L)')
    
    ax.set_title(title)
    ax.set_ylabel("Liquidity")
    ax.set_xlabel("Time")
    ax.legend()
    ax.grid(True)
    
    if return_fig:
        return fig
    else:
        plt.tight_layout()
        plt.show()

def save_results_to_file(prices, dates, estimated_params, L_estimated, param_stats, output_file, dt):
    """
    Save estimation results to a PDF file
    
    Parameters:
    - prices: Observed prices
    - dates: Date array
    - estimated_params: Estimated parameters
    - L_estimated: Estimated liquidity
    - param_stats: Parameter statistics DataFrame
    - output_file: Output file path
    - dt: Time step
    """
    # Create PDF file
    with PdfPages(output_file) as pdf:
        # First page: Title and summary
        plt.figure(figsize=(8.5, 11))
        plt.axis('off')
        plt.text(0.5, 0.95, "SDE Parameter Estimation Results", ha='center', fontsize=16, weight='bold')
        plt.text(0.5, 0.9, f"Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", ha='center', fontsize=10)
        
        # Summary information
        summary_text = (
            f"Data Summary:\n"
            f"- Number of observations: {len(prices)}\n"
            f"- Time period: {pd.to_datetime(dates[0]).date()} to {pd.to_datetime(dates[-1]).date()}\n"
            f"- Time step (dt): {dt:.6f} years\n"
            f"- Mean price: {np.mean(prices):.2f}\n"
            f"- Price volatility: {np.std(np.diff(np.log(prices))) * np.sqrt(252):.4f} (annualized)\n\n"
            
            f"Model Specification:\n"
            f"δS_t = r S_t δt + β L_t S_t δW_t^γ + σ_S S_t δW_t^S\n"
            f"δL_t = α(θ̄ - L_t)δt + σ_L δW_t^L\n\n"
            
            f"Estimation Method:\n"
            f"Extended Kalman Filter with Maximum Likelihood Estimation"
        )
        plt.text(0.1, 0.8, summary_text, fontsize=10, verticalalignment='top', family='monospace')
        
        # Parameter estimates table as text
        plt.text(0.5, 0.45, "Parameter Estimates", ha='center', fontsize=12, weight='bold')
        
        # Convert parameter stats to string for display
        param_table = param_stats.to_string(index=False)
        plt.text(0.1, 0.4, param_table, fontsize=9, verticalalignment='top', family='monospace')
        
        # Note on significance
        sig_note = (
            "Significance codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1"
        )
        plt.text(0.1, 0.1, sig_note, fontsize=8, color='gray')
        
        pdf.savefig()
        plt.close()
        
        # Raw price data
        fig = visualize_raw_data(prices, dates, title="Soybean Meal Futures Prices", return_fig=True)
        pdf.savefig(fig)
        plt.close(fig)
        
        # Returns analysis
        fig, _ = analyze_returns(prices, log_returns=True, return_fig=True)
        pdf.savefig(fig)
        plt.close(fig)
        
        # Estimated liquidity
        fig = plot_estimated_vs_true(L_estimated, dates, title="Estimated Liquidity Process", return_fig=True)
        pdf.savefig(fig)
        plt.close(fig)
        
        # Create a combined plot of prices and liquidity
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
        
        if dates is not None:
            ax1.plot(dates, prices, 'b-')
            ax2.plot(dates, L_estimated, 'r-')
            fig.autofmt_xdate()
        else:
            ax1.plot(prices, 'b-')
            ax2.plot(L_estimated, 'r-')
            
        ax1.set_title("Soybean Meal Futures Prices")
        ax1.set_ylabel("Price")
        ax1.grid(True)
        
        ax2.set_title("Estimated Liquidity Process")
        ax2.set_ylabel("Liquidity")
        ax2.set_xlabel("Time")
        ax2.grid(True)
        
        plt.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)
    
    print(f"Results saved to {output_file}")

def run_parameter_estimation(csv_file_path, price_column='收盘价', date_column='交易时间', 
                            initial_params=None, plot_result=True, analyze_data=True,
                            save_results=True, output_file=None):
    """
    Load data from CSV file and run SDE parameter estimation
    
    Parameters:
    - csv_file_path: Path to CSV file
    - price_column: Price column name
    - date_column: Date column name
    - initial_params: Initial parameter estimates, if None use default values
    - plot_result: Whether to plot results
    - analyze_data: Whether to analyze data distribution
    - save_results: Whether to save results to file
    - output_file: Output file path, if None generates default name
    
    Returns:
    - estimated_params: Estimated parameters
    - L_estimated: Estimated liquidity sequence
    """
    # Load data
    prices, dates, dt = load_futures_data(csv_file_path, price_column, date_column)
    if prices is None:
        print("Data loading failed, cannot continue")
        return None, None
    
    # Visualize raw data
    if plot_result:
        visualize_raw_data(prices, dates, title="Soybean Meal Futures Prices")
    
    # Analyze returns (optional)
    if analyze_data:
        analyze_returns(prices)
    
    # If no initial parameters provided, use default values
    if initial_params is None:
        # Set initial parameters based on data characteristics
        # These are rough estimates, can be adjusted based on specific data
        r = 0.03  # Annualized risk-free rate
        alpha = 0.5  # Mean-reversion speed
        beta = 0.2  # Liquidity impact coefficient
        theta_bar = 1.0  # Long-term average liquidity
        sigma_S = np.std(np.diff(np.log(prices))) * np.sqrt(252 if dt <= 1/200 else 1/dt)  # Annualized price volatility
        sigma_L = sigma_S * 0.5  # Liquidity volatility (rough estimate as half of price volatility)
        rho1 = 0.3  # Correlation coefficients
        rho2 = 0.2
        rho3 = 0.1
        
        initial_params = (r, alpha, beta, theta_bar, sigma_S, sigma_L, rho1, rho2, rho3)
        
        print("\nAutomatically set initial parameters based on data:")
        print(f"r (risk-free rate): {r:.4f}")
        print(f"alpha (mean-reversion speed): {alpha:.4f}")
        print(f"beta (liquidity impact): {beta:.4f}")
        print(f"theta_bar (long-term liquidity): {theta_bar:.4f}")
        print(f"sigma_S (price volatility): {sigma_S:.4f}")
        print(f"sigma_L (liquidity volatility): {sigma_L:.4f}")
        print(f"rho1, rho2, rho3 (correlations): {rho1:.2f}, {rho2:.2f}, {rho3:.2f}")
    
    # Create estimator
    estimator = SDEParameterEstimator(prices, dt)
    
    # Estimate parameters
    print("\nStarting parameter estimation...")
    estimated_params, L_estimated = estimator.estimate_parameters(initial_params)
    
    # Calculate parameter significance
    print("Calculating parameter significance...")
    param_stats = calculate_parameter_significance(estimated_params, prices, L_estimated, dt)
    
    # Print results
    print("\nParameter Estimation Results:")
    print(param_stats.to_string(index=False))
    
    # Visualize results
    if plot_result:
        estimator.plot_results(prices, L_estimated, estimated_params, 
                             title="Soybean Meal Futures SDE Parameter Estimation")
    
    # Save results to file
    if save_results:
        if output_file is None:
            # Generate default output filename
            base_name = os.path.splitext(csv_file_path)[0]
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = f"{base_name}_results_{timestamp}.pdf"
            
        save_results_to_file(prices, dates, estimated_params, L_estimated, param_stats, output_file, dt)
    
    return estimated_params, L_estimated, param_stats

if __name__ == "__main__":
    # Soybean Meal Futures CSV file path
    csv_file_path = "soybean_meal_futures.csv"
    
    # Column names based on the provided file
    price_column = '收盘价'  # Price column name
    date_column = '交易时间'  # Date column name
    
    # Run parameter estimation with output file saved to current directory
    estimated_params, L_estimated, param_stats = run_parameter_estimation(
        csv_file_path, 
        price_column=price_column, 
        date_column=date_column,
        save_results=True
    ) 