import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy import signal, stats
from scipy.stats import entropy

#########################
# Log Returns
def extract_log_returns(df: pd.DataFrame,
                        price_col: str = "close",
                        ticker_col: str = "tr_ric",
                        time_col: str = "timestamp"):
    """
    Returns a dict:
        { ticker : np.ndarray of log-returns }
    """
    returns_by_ticker = {}

    for ticker, df_t in df[ticker_col]:
        df_t = df_t.sort_values(time_col)

        prices = df_t[price_col].values.astype(float)

        # log-returns
        log_prices = np.log(prices)
        rets = np.diff(log_prices)

        # drop pathological series
        if len(rets) < 30 or np.any(~np.isfinite(rets)):
            continue

        returns_by_ticker[ticker] = rets

    return returns_by_ticker

##############################################################
#DFA

def detrended_fluctuation_analysis(time_series, min_window=4, max_window=None, num_windows=10):
    """
    Perform Detrended Fluctuation Analysis on a time series.

    Parameters:
    -----------
    time_series : array-like
        The time series data (non-stationary)
    min_window : int
        Minimum window size for scaling
    max_window : int
        Maximum window size for scaling (default: len(time_series)//4)
    num_windows : int
        Number of window sizes to use (linear spacing)

    Returns:
    --------
    alpha : float
        The DFA exponent (Hurst exponent)
    scales : array
        The window sizes used
    fluctuations : array
        The fluctuation values for each scale
    """

    # Convert to numpy array and handle any NaN values
    ts = np.asarray(time_series).flatten()
    ts = ts[~np.isnan(ts)]

    N = len(ts)

    # default max_window
    if max_window is None:
        max_window = N // 4

    # max_window bounds
    max_window = min(max_window, N // 4)

    # linear spacing for window sizes
    scales = np.linspace(min_window, max_window, num_windows, dtype=int)
    scales = np.unique(scales)  # Remove duplicates

    # Calculate cumulative sum (profile)
    # Subtracting mean first (standard DFA procedure)
    mean_ts = np.mean(ts)
    profile = np.cumsum(ts - mean_ts)

    fluctuations = []

    # Non-overlapping segments
    for scale in scales:
        # Number of segments
        n_segments = N // scale

        if n_segments < 1:
            continue

        # Calculate fluctuation for each segment
        segment_fluctuations = []

        for seg in range(n_segments):
            # Extract segment
            start_idx = seg * scale
            end_idx = start_idx + scale
            segment = profile[start_idx:end_idx]

            # Fit polynomial trend (linear detrending)
            x = np.arange(scale)
            coeffs = np.polyfit(x, segment, 1)
            trend = np.polyval(coeffs, x)

            # Calculate fluctuation (variance from trend)
            fluctuation = np.sqrt(np.mean((segment - trend) ** 2))
            segment_fluctuations.append(fluctuation)

        # Average fluctuation across all segments for this scale
        F_scale = np.mean(segment_fluctuations)
        fluctuations.append(F_scale)

    fluctuations = np.array(fluctuations)
    scales = scales[:len(fluctuations)]


    # F(n) ~ n^alpha
    log_scales = np.log(scales)
    log_fluctuations = np.log(fluctuations)

    # Linear regression in log-log space
    coeffs = np.polyfit(log_scales, log_fluctuations, 1)
    alpha = coeffs[0]

    return alpha, scales, fluctuations

################################################

def calculate_dfa_for_all_tickers(df, min_length=50, min_window=4, max_window=None, num_windows=10):
    """
    Calculate DFA alpha for all tickers in the dataframe.

    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame with columns 'tr_ric' and 'close'
    min_length : int
        Minimum required length for a time series to be processed
    min_window : int
        Minimum window size for DFA
    max_window : int
        Maximum window size for DFA
    num_windows : int
        Number of window sizes to use

    Returns:
    --------
    results_df : pandas.DataFrame
        DataFrame with columns ['tr_ric', 'alpha', 'n_samples', 'status']
    """

    # Get unique tickers
    tickers = df['tr_ric'].unique()

    results = []

    print(f"Processing {len(tickers)} time series...")

    for i, ticker in enumerate(tickers):
        # Progress indicator
        if (i + 1) % 100 == 0:
            print(f"Processed {i + 1}/{len(tickers)} time series...")

        try:
            # Time series for this ticker
            ticker_data = df[df['tr_ric'] == ticker]['close'].values

            # Remove NaN values
            ticker_data = ticker_data[~np.isnan(ticker_data)]

            n_samples = len(ticker_data)

            if n_samples < min_length:
                results.append({
                    'tr_ric': ticker,
                    'alpha': np.nan,
                    'n_samples': n_samples,
                    'status': 'too_short'
                })
                continue

            # DFA
            alpha, _, _ = detrended_fluctuation_analysis(
                ticker_data,
                min_window=min_window,
                max_window=max_window,
                num_windows=num_windows
            )

            # alpha range [0, 2]
            if not (0 <= alpha <= 2):
                status = 'out_of_range'
            else:
                status = 'success'

            results.append({
                'tr_ric': ticker,
                'alpha': alpha,
                'n_samples': n_samples,
                'status': status
            })

        except Exception as e:
            results.append({
                'tr_ric': ticker,
                'alpha': np.nan,
                'n_samples': len(df[df['tr_ric'] == ticker]),
                'status': f'error: {str(e)}'
            })

    print(f"Processing complete!")

    # Results dataframe
    results_df = pd.DataFrame(results)


    # Valid alphas
    valid_alphas = results_df[results_df['status'] == 'success']['alpha']
    if len(valid_alphas) > 0:
        print("Alpha Statistics (valid series only):")
        print(f"  Mean: {valid_alphas.mean():.4f}")
        print(f"  Median: {valid_alphas.median():.4f}")
        print(f"  Std Dev: {valid_alphas.std():.4f}")
        print(f"  Min: {valid_alphas.min():.4f}")
        print(f"  Max: {valid_alphas.max():.4f}")

    return results_df

#########################################################

def visualize_alpha_distribution(results_df, bins=30, figsize=(12, 6)):
    """
    Create a histogram visualization of DFA alpha values.

    Parameters:
    -----------
    results_df : pandas.DataFrame
        Results from calculate_dfa_for_all_tickers()
    bins : int or array
        Number of bins or bin edges for histogram
    figsize : tuple
        Figure size (width, height)
    """

    # Calculated alphas
    valid_results = results_df[results_df['status'] == 'success'].copy()
    alphas = valid_results['alpha'].values

    if len(alphas) == 0:
        print("No valid alpha values to plot!")
        return

    fig, ax = plt.subplots(figsize=figsize)

    counts, bin_edges, patches = ax.hist(
        alphas,
        bins=bins,
        color='steelblue',
        alpha=0.7,
        edgecolor='black',
        linewidth=0.5
    )

    # Reference lines for theoretical values
    ax.axvline(0.5, color='red', linestyle='--', linewidth=2,
               label='α = 0.5 (Random Walk)', alpha=0.7)
    ax.axvline(1.0, color='green', linestyle='--', linewidth=2,
               label='α = 1.0 (1/f Noise)', alpha=0.7)

    # Labels and title
    ax.set_xlabel('DFA Alpha (Hurst Exponent)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Frequency (Number of Time Series)', fontsize=12, fontweight='bold')
    ax.set_title(f'Distribution of DFA Alpha Values Across {len(alphas)} Time Series',
                 fontsize=14, fontweight='bold', pad=20)

    # Set x-axis limits to [0, 2] for theoretical range
    ax.set_xlim(0, 2)

    ax.grid(True, alpha=0.3, linestyle=':', linewidth=0.5)
    ax.set_axisbelow(True)

    # Legend
    ax.legend(loc='upper right', fontsize=10)

    # Stats
    stats_text = f'n = {len(alphas)}\n'
    stats_text += f'Mean = {alphas.mean():.3f}\n'
    stats_text += f'Median = {np.median(alphas):.3f}\n'
    stats_text += f'Std = {alphas.std():.3f}\n'
    stats_text += f'Range = [{alphas.min():.3f}, {alphas.max():.3f}]'

    ax.text(0.02, 0.98, stats_text,
            transform=ax.transAxes,
            fontsize=10,
            verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    # Interpretation regions
    ax.axvspan(0, 0.5, alpha=0.1, color='orange', label='Anti-persistent')
    ax.axvspan(0.5, 1.0, alpha=0.1, color='gray', label='Persistent')
    ax.axvspan(1.0, 2.0, alpha=0.1, color='purple', label='Strong Trend')

    plt.tight_layout()
    plt.show()

    print("DISTRIBUTION ANALYSIS: ")

    print("\nInterpretation:")
    anti_persistent = (alphas < 0.5).sum()
    persistent = ((alphas >= 0.5) & (alphas < 1.0)).sum()
    strong_trend = (alphas >= 1.0).sum()

    print(f"  Anti-persistent (α < 0.5): {anti_persistent} ({100 * anti_persistent / len(alphas):.1f}%)")
    print(f"  Persistent (0.5 ≤ α < 1.0): {persistent} ({100 * persistent / len(alphas):.1f}%)")
    print(f"  Strong trend (α ≥ 1.0): {strong_trend} ({100 * strong_trend / len(alphas):.1f}%)")


###################################################
# COMPLEXITY
###################################################

def spectral_entropy(x, sf=1.0, method='welch', nperseg=None, normalize=True):
    """
    Compute spectral entropy of a time series using Welch's method.

    Spectral entropy measures the complexity of the frequency spectrum.
    It quantifies how evenly power is distributed across frequencies.

    Parameters:
    -----------
    x : np.ndarray
        1D time series (log returns)
    sf : float
        Sampling frequency (default: 1.0 for daily data)
    method : str
        Method to compute PSD ('welch' or 'fft')
    nperseg : int
        Length of each segment for Welch's method (default: None = auto)
    normalize : bool
        If True, normalize to range [0, 1]

    Returns:
    --------
    Spectral entropy (float)
    - Low SE: Power concentrated in few frequencies (periodic)
    - High SE: Power spread across many frequencies (complex/random)
    """
    x = np.asarray(x)

    # Remove any NaN or infinite values
    x = x[np.isfinite(x)]

    if len(x) < 2:
        return np.nan

    # Compute power spectral density using Welch's method
    if method == 'welch':
        # Set nperseg if not provided
        if nperseg is None:
            nperseg = min(256, len(x))

        # Compute PSD
        freqs, psd = signal.welch(x, fs=sf, nperseg=nperseg)

    elif method == 'fft':
        # Compute PSD using FFT
        fft_vals = np.fft.rfft(x)
        psd = np.abs(fft_vals) ** 2
        freqs = np.fft.rfftfreq(len(x), 1/sf)

    else:
        raise ValueError("method must be 'welch' or 'fft'")

    # Remove zero and negative values
    psd = psd[psd > 0]

    if len(psd) == 0:
        return np.nan

    # Normalize PSD to get probability distribution
    psd_norm = psd / psd.sum()

    # Compute Shannon entropy
    se = entropy(psd_norm, base=np.e)  # Using natural log (ln)

    if normalize:
        # Normalize by maximum possible entropy
        se_max = np.log(len(psd_norm))
        if se_max > 0:
            se = se / se_max

    return se

################################################################

def compute_spectral_entropy_across_assets(df_real,
                                          extract_log_returns_func,
                                          sf=1.0,
                                          method='welch',
                                          nperseg=None,
                                          normalize=True,
                                          min_length=50):
    """
    Computes Spectral Entropy for each ticker.

    Parameters:
    -----------
    df_real : pandas.DataFrame
        DataFrame with columns 'tr_ric' and price data
    extract_log_returns_func : function
        Function that extracts log returns from df_real
    sf : float
        Sampling frequency (1.0 for daily data)
    method : str
        'welch' or 'fft'
    nperseg : int
        Segment length for Welch method
    normalize : bool
        If True, normalize SE to [0, 1]
    min_length : int
        Minimum time series length

    Returns:
    --------
    DataFrame with columns ['tr_ric', 'spectral_entropy', 'n_obs', 'volatility', 'status']
    """

    print(f"Extracting log returns from {len(df_real['tr_ric'].unique())} time series...")

    # Extract returns
    try:
        returns_by_ticker = extract_log_returns_func(df_real)
    except Exception as e:
        print(f"Error extracting returns: {e}")
        return pd.DataFrame()

    results = []
    tickers = list(returns_by_ticker.keys())

    print(f"Computing Spectral Entropy for {len(tickers)} time series...")
    print(f"Parameters: method={method}, normalize={normalize}, min_length={min_length}")

    for i, ticker in enumerate(tickers):
        # Progress indicator
        if (i + 1) % 100 == 0:
            print(f"Processed {i + 1}/{len(tickers)} time series...")

        try:
            rets = returns_by_ticker[ticker]

            # Remove NaN values
            rets = rets[~np.isnan(rets)]
            n_obs = len(rets)

            # Quality control: check if series is long enough
            if n_obs < min_length:
                results.append({
                    "tr_ric": ticker,
                    "spectral_entropy": np.nan,
                    "n_obs": n_obs,
                    "volatility": np.nan,
                    "status": "too_short"
                })
                continue

            # Calculate volatility
            volatility = np.std(rets, ddof=1)

            if volatility <= 0 or not np.isfinite(volatility):
                results.append({
                    "tr_ric": ticker,
                    "spectral_entropy": np.nan,
                    "n_obs": n_obs,
                    "volatility": volatility,
                    "status": "invalid_volatility"
                })
                continue

            # Compute spectral entropy
            se = spectral_entropy(rets, sf=sf, method=method,
                                 nperseg=nperseg, normalize=normalize)

            # Check if valid
            if not np.isfinite(se):
                status = "invalid_se"
            else:
                status = "success"

            results.append({
                "tr_ric": ticker,
                "spectral_entropy": se,
                "n_obs": n_obs,
                "volatility": volatility,
                "status": status
            })

        except Exception as e:
            results.append({
                "tr_ric": ticker,
                "spectral_entropy": np.nan,
                "n_obs": len(returns_by_ticker.get(ticker, [])),
                "volatility": np.nan,
                "status": f"error: {str(e)}"
            })

    print(f"Processing complete!")

    # Create results dataframe
    results_df = pd.DataFrame(results)

    # Print summary statistics
    print("\n" + "="*60)
    print("SUMMARY STATISTICS - SPECTRAL ENTROPY")
    print("="*60)
    print(f"Total time series: {len(results_df)}")
    print(f"Successfully processed: {(results_df['status'] == 'success').sum()}")
    print(f"Too short: {(results_df['status'] == 'too_short').sum()}")
    print(f"Invalid volatility: {(results_df['status'] == 'invalid_volatility').sum()}")
    print(f"Invalid SE: {(results_df['status'] == 'invalid_se').sum()}")
    print(f"Errors: {(results_df['status'].str.contains('error', na=False)).sum()}")
    print("\n")

    # Statistics for valid SE values
    valid_se = results_df[results_df['status'] == 'success']['spectral_entropy']
    if len(valid_se) > 0:
        print("Spectral Entropy Statistics:")
        print(f"  Mean: {valid_se.mean():.4f}")
        print(f"  Median: {valid_se.median():.4f}")
        print(f"  Std Dev: {valid_se.std():.4f}")
        print(f"  Min: {valid_se.min():.4f}")
        print(f"  Max: {valid_se.max():.4f}")

        # Categorization based on paper thresholds
        if normalize:
            print("\n  Note: Values are normalized to [0, 1]")
            print("  For paper categories (X < 1, 1 ≤ X < 9, 9 ≤ X),")
            print("  you may need to use unnormalized SE (normalize=False)")
        else:
            cat_A = (valid_se < 1.0).sum()
            cat_B = ((valid_se >= 1.0) & (valid_se < 9.0)).sum()
            cat_C = (valid_se >= 9.0).sum()

            print("\nCategorization (Paper thresholds):")
            print(f"  Category A (SE < 1): {cat_A} ({100*cat_A/len(valid_se):.1f}%)")
            print(f"  Category B (1 ≤ SE < 9): {cat_B} ({100*cat_B/len(valid_se):.1f}%)")
            print(f"  Category C (9 ≤ SE): {cat_C} ({100*cat_C/len(valid_se):.1f}%)")

    print("="*60)

    return results_df

##########################################################

def visualize_spectral_entropy_distribution(results_df, bins=30, figsize=(12, 6)):
    """
    Create a histogram visualization of Spectral Entropy values.

    Parameters:
    -----------
    results_df : pandas.DataFrame
        Results from compute_spectral_entropy_across_assets()
    bins : int or array
        Number of bins or bin edges for histogram
    figsize : tuple
        Figure size (width, height)
    """

    # Filter for successfully calculated spectral entropy
    valid_results = results_df[results_df['status'] == 'success'].copy()
    se_values = valid_results['spectral_entropy'].values

    if len(se_values) == 0:
        print("No valid Spectral Entropy values to plot!")
        return

    # Create figure
    fig, ax = plt.subplots(figsize=figsize)

    # Create histogram
    counts, bin_edges, patches = ax.hist(
        se_values,
        bins=bins,
        color='dodgerblue',
        alpha=0.7,
        edgecolor='black',
        linewidth=0.5
    )

    # Add reference lines for category thresholds (paper categories)
    ax.axvline(1.0, color='orange', linestyle='--', linewidth=2,
               label='Category A/B Threshold (SE=1)', alpha=0.7)
    ax.axvline(9.0, color='red', linestyle='--', linewidth=2,
               label='Category B/C Threshold (SE=9)', alpha=0.7)

    # Labels and title
    ax.set_xlabel('Spectral Entropy', fontsize=12, fontweight='bold')
    ax.set_ylabel('Frequency (Number of Time Series)', fontsize=12, fontweight='bold')
    ax.set_title(f'Spectral Entropy Distribution Across {len(se_values)} Time Series',
                 fontsize=14, fontweight='bold', pad=20)

    # Add grid for better readability
    ax.grid(True, alpha=0.3, linestyle=':', linewidth=0.5)
    ax.set_axisbelow(True)

    # Add legend
    ax.legend(loc='upper right', fontsize=10)

    # Add text box with statistics
    stats_text = f'n = {len(se_values)}\n'
    stats_text += f'Mean = {se_values.mean():.3f}\n'
    stats_text += f'Median = {np.median(se_values):.3f}\n'
    stats_text += f'Std = {se_values.std():.3f}\n'
    stats_text += f'Range = [{se_values.min():.3f}, {se_values.max():.3f}]'

    ax.text(0.02, 0.98, stats_text,
            transform=ax.transAxes,
            fontsize=10,
            verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='lightcyan', alpha=0.5))

    # Add interpretation regions (background shading)
    x_min, x_max = ax.get_xlim()
    ax.axvspan(x_min, 1.0, alpha=0.1, color='yellow', label='Low Complexity')
    ax.axvspan(1.0, 9.0, alpha=0.1, color='green', label='Moderate Complexity')
    ax.axvspan(9.0, x_max, alpha=0.1, color='red', label='High Complexity')

    plt.tight_layout()
    plt.show()

    # Print additional insights about gaps
    print("\n" + "="*60)
    print("DISTRIBUTION ANALYSIS - SPECTRAL ENTROPY DIVERSITY")
    print("="*60)

    # Analyze gaps in the distribution
    bin_width = bin_edges[1] - bin_edges[0]
    empty_bins = np.where(counts == 0)[0]

    if len(empty_bins) > 0:
        print(f"Empty intervals detected: {len(empty_bins)} out of {len(counts)} bins")
        print(f"Bin width: {bin_width:.4f}")
        print("\nEmpty interval ranges:")
        for idx in empty_bins[:10]:  # Show first 10
            print(f"  [{bin_edges[idx]:.4f}, {bin_edges[idx+1]:.4f}]")
        if len(empty_bins) > 10:
            print(f"  ... and {len(empty_bins) - 10} more")
    else:
        print("No empty intervals detected - excellent spectral entropy diversity!")

    print("\nComplexity Interpretation (Paper Categories):")
    cat_A = (se_values < 1.0).sum()
    cat_B = ((se_values >= 1.0) & (se_values < 9.0)).sum()
    cat_C = (se_values >= 9.0).sum()

    print(f"  Category A (SE < 1): {cat_A} ({100*cat_A/len(se_values):.1f}%)")
    print(f"    → Low complexity, power concentrated in few frequencies")
    print(f"  Category B (1 ≤ SE < 9): {cat_B} ({100*cat_B/len(se_values):.1f}%)")
    print(f"    → Moderate complexity, typical financial behavior")
    print(f"  Category C (9 ≤ SE): {cat_C} ({100*cat_C/len(se_values):.1f}%)")
    print(f"    → High complexity, power spread across many frequencies")
    print("="*60)


#######################################################
# NORMALITY
#######################################################
def compute_kurtosis(returns):
    return stats.kurtosis(returns, fisher=True)  # Fisher=True gives excess kurtosis

def compute_skewness(returns):
    return stats.skew(returns)

def compute_god_jb(returns):
    differences = np.diff(returns)

    if len(differences) < 2:
        return np.nan

    # Jarque-Bera test statistic
    # JB = (n/6) * (S^2 + (K^2)/4)
    # where S = skewness, K = excess kurtosis
    n = len(differences)
    s = stats.skew(differences)
    k = stats.kurtosis(differences, fisher=True)

    jb_stat = (n / 6) * (s**2 + (k**2) / 4)

    return jb_stat

def compute_normality_features_for_all_tickers(df_real,
                                               extract_log_returns_func,
                                               min_length=50):
    """
    Compute normality features (Kurtosis, Skewness, GoD) for all tickers.

    Parameters:
    -----------
    df_real : pandas.DataFrame
        DataFrame with columns 'tr_ric' and price data
    extract_log_returns_func : function
        Your function that extracts log returns from df_real
    min_length : int
        Minimum required length for a time series to be processed

    Returns:
    --------
    results_df : pandas.DataFrame
        DataFrame with columns ['tr_ric', 'kurtosis', 'skewness', 'god_jb',
                                'n_obs', 'volatility', 'status']
    """

    print(f"Extracting log returns from {len(df_real['tr_ric'].unique())} time series...")

    # Extract returns using your function
    try:
        returns_by_ticker = extract_log_returns_func(df_real)
    except Exception as e:
        print(f"Error extracting returns: {e}")
        return pd.DataFrame()

    tickers = list(returns_by_ticker.keys())
    results = []

    print(f"Computing normality features for {len(tickers)} time series...")
    print(f"Parameters: min_length={min_length}")

    for i, ticker in enumerate(tickers):
        # Progress indicator
        if (i + 1) % 100 == 0:
            print(f"Processed {i + 1}/{len(tickers)} time series...")

        try:
            rets = returns_by_ticker[ticker]

            # Remove NaN values
            rets = rets[~np.isnan(rets)]
            n_obs = len(rets)

            # Quality control: check if series is long enough
            if n_obs < min_length:
                results.append({
                    'tr_ric': ticker,
                    'kurtosis': np.nan,
                    'skewness': np.nan,
                    'god_jb': np.nan,
                    'n_obs': n_obs,
                    'volatility': np.nan,
                    'status': 'too_short'
                })
                continue

            # Calculate volatility
            volatility = np.std(rets, ddof=1)

            # Check for valid volatility
            if volatility <= 0 or not np.isfinite(volatility):
                results.append({
                    'tr_ric': ticker,
                    'kurtosis': np.nan,
                    'skewness': np.nan,
                    'god_jb': np.nan,
                    'n_obs': n_obs,
                    'volatility': volatility,
                    'status': 'invalid_volatility'
                })
                continue

            # Compute normality features
            kurt = compute_kurtosis(rets)
            skew = compute_skewness(rets)
            god = compute_god_jb(rets)

            # Check if all features are valid
            if not all(np.isfinite([kurt, skew, god])):
                status = 'invalid_features'
            else:
                status = 'success'

            results.append({
                'tr_ric': ticker,
                'kurtosis': kurt,
                'skewness': skew,
                'god_jb': god,
                'n_obs': n_obs,
                'volatility': volatility,
                'status': status
            })

        except Exception as e:
            # Handle any errors gracefully
            results.append({
                'tr_ric': ticker,
                'kurtosis': np.nan,
                'skewness': np.nan,
                'god_jb': np.nan,
                'n_obs': len(returns_by_ticker.get(ticker, [])),
                'volatility': np.nan,
                'status': f'error: {str(e)}'
            })

    print(f"Processing complete!")

    # Create results dataframe
    results_df = pd.DataFrame(results)

    # Print summary statistics
    print("\n" + "="*60)
    print("SUMMARY STATISTICS")
    print("="*60)
    print(f"Total time series: {len(results_df)}")
    print(f"Successfully processed: {(results_df['status'] == 'success').sum()}")
    print(f"Too short: {(results_df['status'] == 'too_short').sum()}")
    print(f"Invalid volatility: {(results_df['status'] == 'invalid_volatility').sum()}")
    print(f"Invalid features: {(results_df['status'] == 'invalid_features').sum()}")
    print(f"Errors: {(results_df['status'].str.contains('error', na=False)).sum()}")
    print("\n")

    # Statistics for valid features
    valid_results = results_df[results_df['status'] == 'success']
    if len(valid_results) > 0:
        print("KURTOSIS Statistics:")
        print(f"  Mean: {valid_results['kurtosis'].mean():.4f}")
        print(f"  Median: {valid_results['kurtosis'].median():.4f}")
        print(f"  Std Dev: {valid_results['kurtosis'].std():.4f}")
        print(f"  Range: [{valid_results['kurtosis'].min():.4f}, {valid_results['kurtosis'].max():.4f}]")

        light_tail = (valid_results['kurtosis'] < 0).sum()
        normal_tail = ((valid_results['kurtosis'] >= 0) & (valid_results['kurtosis'] <= 1)).sum()
        heavy_tail = (valid_results['kurtosis'] > 1).sum()
        print(f"  Light tail (< 0): {light_tail} ({100*light_tail/len(valid_results):.1f}%)")
        print(f"  Normal-like (0-1): {normal_tail} ({100*normal_tail/len(valid_results):.1f}%)")
        print(f"  Heavy tail (> 1): {heavy_tail} ({100*heavy_tail/len(valid_results):.1f}%)")

        print("\nSKEWNESS Statistics:")
        print(f"  Mean: {valid_results['skewness'].mean():.4f}")
        print(f"  Median: {valid_results['skewness'].median():.4f}")
        print(f"  Std Dev: {valid_results['skewness'].std():.4f}")
        print(f"  Range: [{valid_results['skewness'].min():.4f}, {valid_results['skewness'].max():.4f}]")

        left_skew = (valid_results['skewness'] < -0.5).sum()
        symmetric = ((valid_results['skewness'] >= -0.5) & (valid_results['skewness'] <= 0.5)).sum()
        right_skew = (valid_results['skewness'] > 0.5).sum()
        print(f"  Left skew (< -0.5): {left_skew} ({100*left_skew/len(valid_results):.1f}%)")
        print(f"  Symmetric (-0.5 to 0.5): {symmetric} ({100*symmetric/len(valid_results):.1f}%)")
        print(f"  Right skew (> 0.5): {right_skew} ({100*right_skew/len(valid_results):.1f}%)")

        print("\nGOD (Jarque-Bera) Statistics:")
        print(f"  Mean: {valid_results['god_jb'].mean():.4f}")
        print(f"  Median: {valid_results['god_jb'].median():.4f}")
        print(f"  Std Dev: {valid_results['god_jb'].std():.4f}")
        print(f"  Range: [{valid_results['god_jb'].min():.4f}, {valid_results['god_jb'].max():.4f}]")

        # JB critical value at 5% significance level is approximately 5.99 (chi-square with 2 df)
        normal_like = (valid_results['god_jb'] < 5.99).sum()
        non_normal = (valid_results['god_jb'] >= 5.99).sum()
        print(f"  Normal-like (JB < 5.99): {normal_like} ({100*normal_like/len(valid_results):.1f}%)")
        print(f"  Non-normal (JB ≥ 5.99): {non_normal} ({100*non_normal/len(valid_results):.1f}%)")
    print("="*60)

    return results_df


def visualize_kurtosis_distribution(results_df, bins=30, figsize=(12, 6)):
    """Visualize Kurtosis distribution"""

    valid_results = results_df[results_df['status'] == 'success'].copy()
    values = valid_results['kurtosis'].values

    if len(values) == 0:
        print("No valid kurtosis values to plot!")
        return

    fig, ax = plt.subplots(figsize=figsize)

    counts, bin_edges, patches = ax.hist(
        values, bins=bins, color='coral', alpha=0.7,
        edgecolor='black', linewidth=0.5
    )

    # Reference lines
    ax.axvline(0, color='red', linestyle='--', linewidth=2,
               label='Normal Distribution (Kurt=0)', alpha=0.7)

    # Labels
    ax.set_xlabel('Excess Kurtosis', fontsize=12, fontweight='bold')
    ax.set_ylabel('Frequency', fontsize=12, fontweight='bold')
    ax.set_title(f'Kurtosis Distribution Across {len(values)} Time Series',
                 fontsize=14, fontweight='bold', pad=20)

    ax.grid(True, alpha=0.3, linestyle=':', linewidth=0.5)
    ax.set_axisbelow(True)
    ax.legend(loc='upper right', fontsize=10)

    # Stats box
    stats_text = f'n = {len(values)}\n'
    stats_text += f'Mean = {values.mean():.3f}\n'
    stats_text += f'Median = {np.median(values):.3f}\n'
    stats_text += f'Std = {values.std():.3f}\n'
    stats_text += f'Range = [{values.min():.3f}, {values.max():.3f}]'

    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
            fontsize=10, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    # Background shading
    x_min, x_max = ax.get_xlim()
    ax.axvspan(x_min, 0, alpha=0.1, color='blue')  # Light tail
    ax.axvspan(0, x_max, alpha=0.1, color='red')   # Heavy tail

    plt.tight_layout()
    plt.show()


def visualize_skewness_distribution(results_df, bins=30, figsize=(12, 6)):
    """Visualize Skewness distribution"""

    valid_results = results_df[results_df['status'] == 'success'].copy()
    values = valid_results['skewness'].values

    if len(values) == 0:
        print("No valid skewness values to plot!")
        return

    fig, ax = plt.subplots(figsize=figsize)

    counts, bin_edges, patches = ax.hist(
        values, bins=bins, color='mediumpurple', alpha=0.7,
        edgecolor='black', linewidth=0.5
    )

    # Reference lines
    ax.axvline(0, color='green', linestyle='--', linewidth=2,
               label='Symmetric (Skew=0)', alpha=0.7)

    # Labels
    ax.set_xlabel('Skewness', fontsize=12, fontweight='bold')
    ax.set_ylabel('Frequency', fontsize=12, fontweight='bold')
    ax.set_title(f'Skewness Distribution Across {len(values)} Time Series',
                 fontsize=14, fontweight='bold', pad=20)

    ax.grid(True, alpha=0.3, linestyle=':', linewidth=0.5)
    ax.set_axisbelow(True)
    ax.legend(loc='upper right', fontsize=10)

    # Stats box
    stats_text = f'n = {len(values)}\n'
    stats_text += f'Mean = {values.mean():.3f}\n'
    stats_text += f'Median = {np.median(values):.3f}\n'
    stats_text += f'Std = {values.std():.3f}\n'
    stats_text += f'Range = [{values.min():.3f}, {values.max():.3f}]'

    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
            fontsize=10, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    # Background shading
    x_min, x_max = ax.get_xlim()
    ax.axvspan(x_min, 0, alpha=0.1, color='orange')  # Left skew
    ax.axvspan(0, x_max, alpha=0.1, color='purple')  # Right skew

    plt.tight_layout()
    plt.show()


def visualize_god_distribution(results_df, bins=30, figsize=(12, 6)):
    """Visualize Gaussianity of Differences (JB statistic) distribution"""

    valid_results = results_df[results_df['status'] == 'success'].copy()
    values = valid_results['god_jb'].values

    if len(values) == 0:
        print("No valid GoD values to plot!")
        return

    fig, ax = plt.subplots(figsize=figsize)

    counts, bin_edges, patches = ax.hist(
        values, bins=bins, color='seagreen', alpha=0.7,
        edgecolor='black', linewidth=0.5
    )

    # Reference lines
    ax.axvline(5.99, color='red', linestyle='--', linewidth=2,
               label='Critical Value (α=0.05)', alpha=0.7)

    # Labels
    ax.set_xlabel('Jarque-Bera Statistic (GoD)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Frequency', fontsize=12, fontweight='bold')
    ax.set_title(f'Gaussianity of Differences (JB) Distribution Across {len(values)} Time Series',
                 fontsize=14, fontweight='bold', pad=20)

    ax.grid(True, alpha=0.3, linestyle=':', linewidth=0.5)
    ax.set_axisbelow(True)
    ax.legend(loc='upper right', fontsize=10)

    # Stats box
    stats_text = f'n = {len(values)}\n'
    stats_text += f'Mean = {values.mean():.3f}\n'
    stats_text += f'Median = {np.median(values):.3f}\n'
    stats_text += f'Std = {values.std():.3f}\n'
    stats_text += f'Range = [{values.min():.3f}, {values.max():.3f}]'

    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
            fontsize=10, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    # Background shading
    x_min, x_max = ax.get_xlim()
    ax.axvspan(x_min, 5.99, alpha=0.1, color='green')  # Normal-like
    ax.axvspan(5.99, x_max, alpha=0.1, color='yellow') # Non-normal

    plt.tight_layout()
    plt.show()



# ============================================================================
# Feature Categorization Functions
# ============================================================================

def categorize_spectral_entropy(values):
    """
    Categorize Spectral Entropy into 3 categories:
    A: X < 1, B: 1 ≤ X < 9, C: 9 ≤ X
    """
    categories = []
    for val in values:
        if np.isnan(val):
            categories.append(np.nan)
        elif val < 1:
            categories.append('A')
        elif val < 9:
            categories.append('B')
        else:
            categories.append('C')
    return np.array(categories)


def categorize_kurtosis(values):
    """
    Categorize Kurtosis into 3 categories:
    A: X < -0.3, B: -0.3 ≤ X < 0.3, C: 0.3 ≤ X
    """
    categories = []
    for val in values:
        if np.isnan(val):
            categories.append(np.nan)
        elif val < -0.3:
            categories.append('A')
        elif val < 0.3:
            categories.append('B')
        else:
            categories.append('C')
    return np.array(categories)


def categorize_skewness(values):
    """
    Categorize Skewness into 3 categories:
    A: X < -0.3, B: -0.3 ≤ X < 0.3, C: 0.3 ≤ X
    """
    categories = []
    for val in values:
        if np.isnan(val):
            categories.append(np.nan)
        elif val < -0.3:
            categories.append('A')
        elif val < 0.3:
            categories.append('B')
        else:
            categories.append('C')
    return np.array(categories)


def categorize_god(values):
    """
    Categorize GoD (JB statistic) into 2 categories:
    A: X < 0.02, B: 0.02 ≤ X
    """
    categories = []
    for val in values:
        if np.isnan(val):
            categories.append(np.nan)
        elif val < 0.02:
            categories.append('A')
        else:
            categories.append('B')
    return np.array(categories)


def categorize_dfa(values):
    """
    Categorize DFA into 7 categories:
    A: X < 0.45, B: 0.45 ≤ X < 0.55, C: 0.55 ≤ X < 0.95,
    D: 0.95 ≤ X < 1.05, E: 1.05 ≤ X < 1.45, F: 1.45 ≤ X < 1.55, G: 1.55 ≤ X
    """
    categories = []
    for val in values:
        if np.isnan(val):
            categories.append(np.nan)
        elif val < 0.45:
            categories.append('A')
        elif val < 0.55:
            categories.append('B')
        elif val < 0.95:
            categories.append('C')
        elif val < 1.05:
            categories.append('D')
        elif val < 1.45:
            categories.append('E')
        elif val < 1.55:
            categories.append('F')
        else:
            categories.append('G')
    return np.array(categories)

# ============================================================================
# Shannon Entropy Calculation
# ============================================================================

def calculate_shannon_entropy(categories):
    """
    Calculate Shannon entropy H(X) = -Σ p(xi) × ln(p(xi))

    Parameters:
    -----------
    categories : array-like
        Array of category labels (e.g., ['A', 'B', 'C', ...])

    Returns:
    --------
    H(X) : float
        Shannon entropy
    """
    # Remove NaN values
    categories = categories[~pd.isna(categories)]

    if len(categories) == 0:
        return np.nan

    # Count occurrences of each category
    unique, counts = np.unique(categories, return_counts=True)

    # Calculate proportions (probabilities)
    proportions = counts / len(categories)

    # Calculate Shannon entropy using natural log
    # H(X) = -Σ p(xi) × ln(p(xi))
    entropy = -np.sum(proportions * np.log(proportions))

    return entropy


def calculate_max_entropy(num_categories):
    """
    Calculate maximum entropy Hmax(X) = ln(S)
    where S is the number of categories

    Parameters:
    -----------
    num_categories : int
        Number of categories

    Returns:
    --------
    Hmax : float
        Maximum entropy
    """
    return np.log(num_categories)


def calculate_normalized_entropy(H, Hmax):
    """
    Calculate normalized entropy HE(X) = H(X) / Hmax(X)

    Parameters:
    -----------
    H : float
        Shannon entropy
    Hmax : float
        Maximum entropy

    Returns:
    --------
    HE : float
        Normalized entropy (evenness) between 0 and 1
    """
    if Hmax == 0:
        return np.nan
    return H / Hmax


# ============================================================================
# Main Diversity Score Calculation
# ============================================================================

def calculate_diversity_score(dfa_results,
                              spectral_entropy_results,
                              normality_results):
    """
    Calculate multivariate entropy diversity score from all features.

    Parameters:
    -----------
    dfa_results : pandas.DataFrame
        Results from DFA analysis (must have 'tr_ric' and 'alpha' columns)
    spectral_entropy_results : pandas.DataFrame
        Results from spectral entropy (must have 'tr_ric' and 'spectral_entropy' columns)
    normality_results : pandas.DataFrame
        Results from normality features (must have 'tr_ric', 'kurtosis', 'skewness', 'god_jb' columns)

    Returns:
    --------
    diversity_score : float
        Overall diversity score (0 to 1)
    table1 : pandas.DataFrame
        Proportion table
    table2 : pandas.DataFrame
        Entropy scores table
    """

    # Merge all results on tr_ric
    merged = dfa_results[['tr_ric', 'alpha']].merge(
        spectral_entropy_results[['tr_ric', 'spectral_entropy']],
        on='tr_ric', how='inner'
    ).merge(
        normality_results[['tr_ric', 'kurtosis', 'skewness', 'god_jb']],
        on='tr_ric', how='inner'
    )

    print(f"Total time series with all features: {len(merged)}")

    # Categorize each feature
    categories = {
        'DFA': categorize_dfa(merged['alpha'].values),
        'Spectral Entropy': categorize_spectral_entropy(merged['spectral_entropy'].values),
        'Kurtosis': categorize_kurtosis(merged['kurtosis'].values),
        'Skewness': categorize_skewness(merged['skewness'].values),
        'GoD': categorize_god(merged['god_jb'].values)
    }

    # Define number of categories for each feature
    num_categories = {
        'DFA': 7,
        'Spectral Entropy': 3,
        'Kurtosis': 3,
        'Skewness': 3,
        'GoD': 2
    }

    # ========================================================================
    # TABLE 1: Proportion of dataset relative to time series characteristic
    # ========================================================================

    all_possible_cats = ['A', 'B', 'C', 'D', 'E', 'F', 'G']
    proportion_data = []

    for feature_name, cats in categories.items():
        # Remove NaN
        valid_cats = cats[~pd.isna(cats)]
        total = len(valid_cats)

        if total == 0:
            continue

        # Calculate proportions for each category
        proportions = {'Feature': feature_name}
        unique, counts = np.unique(valid_cats, return_counts=True)

        for cat in all_possible_cats:
            if cat in unique:
                idx = np.where(unique == cat)[0][0]
                proportions[cat] = counts[idx] / total
            else:
                proportions[cat] = np.nan

        proportion_data.append(proportions)

    table1 = pd.DataFrame(proportion_data)
    table1 = table1[['Feature', 'A', 'B', 'C', 'D', 'E', 'F', 'G']]

    # ========================================================================
    # TABLE 2: Entropy scores for each metric
    # ========================================================================

    entropy_data = []
    normalized_entropies = []

    for feature_name, cats in categories.items():
        # Calculate H(X)
        H = calculate_shannon_entropy(cats)

        # Calculate Hmax
        Hmax = calculate_max_entropy(num_categories[feature_name])

        # Calculate HE (normalized entropy)
        HE = calculate_normalized_entropy(H, Hmax)

        entropy_data.append({
            'Feature': feature_name,
            'H(X)': H,
            'Hmax': Hmax,
            'HE': HE
        })

        if not np.isnan(HE):
            normalized_entropies.append(HE)

    table2 = pd.DataFrame(entropy_data)

    # ========================================================================
    # FINAL DIVERSITY SCORE: H = (1/k) × Σ HE(X^k)
    # ========================================================================

    if len(normalized_entropies) > 0:
        diversity_score = np.mean(normalized_entropies)
    else:
        diversity_score = np.nan

    return diversity_score, table1, table2

