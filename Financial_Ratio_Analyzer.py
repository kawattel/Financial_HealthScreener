import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import cm
from scipy.stats import linregress
import numpy as np
import yfinance as yf

# Load the updated data (Update with YOUR file path)
csv_path = '/Users/Cleaned_Tech_Financial_Ratios.csv'
df = pd.read_csv(csv_path)

# Combine FB and META data under the ticker META
df['TICKER'] = df['TICKER'].replace('FB', 'META')

# Group by TICKER and qdate and calculate the mean for numeric columns only
numeric_cols = df.select_dtypes(include=['number']).columns  # Select numeric columns
df = df.groupby(['TICKER', 'qdate'], as_index=False)[numeric_cols].mean()  # Only aggregate numeric columns

# Ensure `qdate` is in datetime format
df['qdate'] = pd.to_datetime(df['qdate'], errors='coerce')  # Handle invalid dates gracefully

# Snap all `qdate` values to the nearest quarter-end
df['qdate'] = df['qdate'] + pd.offsets.QuarterEnd(0)  # Adjust all dates to the nearest quarter-end

# Calculate Debt-to-Equity Ratio
if 'totdebt_invcap' in df.columns and 'equity_invcap' in df.columns:
    # Avoid division by zero
    df['de_ratio'] = df['totdebt_invcap'] / df['equity_invcap'].replace(0, np.nan)
    # Handle missing values
    df['de_ratio'] = df['de_ratio'].interpolate(method='linear').fillna(method='bfill').fillna(method='ffill')
else:
    st.error("Required columns ('totdebt_invcap', 'equity_invcap') are missing for calculating Debt-to-Equity Ratio.")

# Step 1: Handle missing data for key financial ratios
for ratio in ['npm', 'roa', 'roe', 'curr_ratio', 'quick_ratio']:  # Adjust based on your columns
    if ratio in df.columns:
        df[ratio] = df[ratio].interpolate(method='linear').fillna(method='ffill')

# Cleaning Efficiency Ratios
def clean_efficiency_ratios(df, cols):
    """
    Cleans missing values in the specified columns by:
    - Interpolating missing values
    - Backward and forward filling edge cases
    """
    for col in cols:
        # Interpolate missing values for smooth time-series filling
        df[col] = df[col].interpolate(method='linear')
        
        # Backward and forward fill remaining missing values
        df[col] = df[col].fillna(method='bfill').fillna(method='ffill')
    return df

# Clean the efficiency ratios
efficiency_ratios = ['inv_turn', 'pay_turn']  # Add the column names here
df = clean_efficiency_ratios(df, efficiency_ratios)

# Extend Data Summary Section with Efficiency Ratios
with st.expander("📊 Data Summary", expanded=False):
    # Number of unique stocks
    num_stocks = df['TICKER'].nunique()

    # Date range
    date_min = pd.to_datetime(df['qdate']).min()
    date_max = pd.to_datetime(df['qdate']).max()

    # Missing values summary (including efficiency ratios)
    efficiency_ratios = ['inv_turn', 'at_turn', 'rect_turn', 'pay_turn']  # Efficiency ratios
    key_columns = ['npm', 'roa', 'roe', 'curr_ratio', 'quick_ratio', 'de_ratio'] + efficiency_ratios
    missing_values = df[key_columns].isnull().sum().sum()  # Total missing in key columns
    missing_ratios = (df[key_columns].isnull().sum() / len(df) * 100).to_dict()  # Percent missing for key columns

    # Display summary details
    st.write(f"**Number of Stocks:** {num_stocks}")
    st.write(f"**Date Range:** {date_min.date()} to {date_max.date()}")
    st.write(f"**Total Missing Values (Key Columns):** {missing_values}")

    # Display missing values for key columns
    if missing_values > 0:
        st.markdown("### Missing Values for Key Columns")
        for col, pct in missing_ratios.items():
            if pct > 0:
                st.write(f"- `{col}`: {pct:.2f}% missing")

    # Display dataset preview (includes efficiency ratios)
    st.write("### Sample of Dataset")
    st.dataframe(df[key_columns + ['TICKER', 'qdate']].head(5))  # Show only relevant columns

# Dashboard Title
st.title("Financial Statement Analyzer - Tech Stocks")

# About Section in a dropdown
with st.sidebar.expander("About This Dashboard", expanded=False):
    st.write(
        """
        This dashboard provides a comprehensive analysis of financial metrics for 
        leading tech companies. Users can:
        
        - Compare financial ratios like Net Profit Margin (NPM), Return on Assets (ROA), and others over time.
        - Analyze historical price trends for selected stocks.
        - View industry benchmarks and apply confidence intervals to evaluate performance.
        - Use moving averages or trend lines to observe long-term patterns.

        **Stock Price Note:**  
        Real-time and Historical stock price data rely on the `yfinance` library. During weekends, holidays, 
        or other times when markets are closed, live stock prices may not update correctly.
        

        This tool is designed for investors, analysts, and anyone interested in 
        understanding financial performance within the tech sector. Select options 
        from the sidebar to customize your analysis!
        """
    )


# Sidebar for user input
tickers = st.sidebar.multiselect("Select Stock Tickers for Comparison:", df['TICKER'].unique())
ratio_options = {
    'npm': 'Net Profit Margin',
    'roa': 'Return on Assets',
    'roe': 'Return on Equity',
    'curr_ratio': 'Current Ratio',
    'quick_ratio': 'Quick Ratio',
}
selected_ratio_key = st.sidebar.selectbox(
    "Select Financial Ratio:",
    list(ratio_options.keys()),
    format_func=lambda x: ratio_options[x]  # Display full names in the dropdown
)
selected_ratio_label = ratio_options[selected_ratio_key]  # Get the full label from the dictionary

# Efficiency Ratios Dropdown (New Section)
efficiency_ratio_options = {
    'inv_turn': 'Inventory Turnover',
    'at_turn': 'Asset Turnover',
    'rect_turn': 'Receivables Turnover',
    'pay_turn': 'Payables Turnover',
}
selected_efficiency_ratio_key = st.sidebar.selectbox(
    "Select Efficiency Ratio:",
    list(efficiency_ratio_options.keys()),
    format_func=lambda x: efficiency_ratio_options[x],  # Display full names in dropdown
)

# Efficiency Ratio Selection Label (Full Name)
selected_efficiency_ratio_label = efficiency_ratio_options[selected_efficiency_ratio_key]  # Get full label


# Solvency Ratios Dropdown
solvency_ratio_options = {
    'de_ratio': 'Debt-to-Equity Ratio'
}
selected_solvency_ratio_key = st.sidebar.selectbox(
    "Select Solvency Ratio:",
    list(solvency_ratio_options.keys()),
    format_func=lambda x: solvency_ratio_options[x]
)

# Add option to select confidence level
confidence_level = st.sidebar.selectbox("Confidence Level for Confidence Interval (%)", [80, 90, 95], index=1)
z_values = {80: 1.28, 90: 1.645, 95: 1.96}  # Z-scores for corresponding confidence levels
z = z_values[confidence_level]  # Get the Z-value based on the user's selection

# Add toggle for confidence interval, trend line, and moving averages
show_confidence_interval = st.sidebar.checkbox("Show Confidence Interval (Financial Ratios Only)", value=False)
show_trend_line = st.sidebar.checkbox("Show Trend Line", value=False)
show_moving_avg = st.sidebar.checkbox("Show Moving Average (4Q)", value=False)

# Add option to use full date range
use_full_date_range = st.sidebar.checkbox("Use Full Date Range", value=True)

# Get the min and max dates from the dataset
min_date = df['qdate'].min()
max_date = df['qdate'].max()

if use_full_date_range:
    # Use the full date range
    filtered_df = df
    full_date_range = pd.date_range(min_date, max_date, freq='Q')
else:
    # Use the custom date range, restricted to available dates
    start_date = st.sidebar.date_input("Start Date", value=min_date, min_value=min_date, max_value=max_date)
    end_date = st.sidebar.date_input("End Date", value=max_date, min_value=min_date, max_value=max_date)

    # Ensure the end date is not earlier than the start date
    if start_date > end_date:
        st.error("Error: End date must fall after the start date.")
    else:
        filtered_df = df[(df['qdate'] >= pd.Timestamp(start_date)) & (df['qdate'] <= pd.Timestamp(end_date))]
        full_date_range = pd.date_range(filtered_df['qdate'].min(), filtered_df['qdate'].max(), freq='Q')

# Sidebar Toggle for Ratio Ranking
show_ranking = st.sidebar.checkbox("Show Financial Ratio Ranking", value=False)

if show_ranking:
    # Financial Ratio Ranking Section
    st.subheader("🏆 Financial Ratio Ranking")

    # Select a ratio for ranking
    ranking_ratio_key = st.selectbox("Select Financial Ratio for Ranking:", list(ratio_options.keys()))
    ranking_ratio_label = ratio_options[ranking_ratio_key]  # Get the full name of the ratio

    # Get the latest available values of the selected ratio for all stocks
    latest_data = filtered_df.groupby("TICKER").last()  # Get the last entry for each stock
    ranking_data = latest_data[ranking_ratio_key].sort_values(ascending=False)  # Sort stocks by the ratio

    # Define constant colors for each stock ticker (if not already defined globally)
    unique_tickers = filtered_df['TICKER'].unique()  # Get all unique tickers
    color_map = {
        ticker: plt.cm.tab10(i % 10) for i, ticker in enumerate(sorted(unique_tickers))  # Assign consistent colors
    }

    # Use the color mapping for bar chart colors
    bar_colors = [color_map[ticker] for ticker in ranking_data.index]  # Map colors to tickers

    # Plot the ranking as a bar chart
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(
        ranking_data.index,
        ranking_data.values,
        color=bar_colors,  # Use constant colors
        edgecolor="black"
    )
    ax.set_title(f"{ranking_ratio_label} Ranking", fontsize=16, fontweight="bold")
    ax.set_ylabel(ranking_ratio_label, fontsize=12)
    ax.set_xlabel("Stock Ticker", fontsize=12)
    plt.xticks(rotation=45, fontsize=10)

    # Display the bar chart in the dashboard
    st.pyplot(fig)


# Sidebar Toggle for Efficiency Ratio Ranking
# Sidebar Toggle for Efficiency Ratio Ranking
show_efficiency_ranking = st.sidebar.checkbox("Show Efficiency Ratio Ranking", value=False)

if show_efficiency_ranking:
    st.subheader("🏆 Efficiency Ratio Ranking")

    # Select an efficiency ratio for ranking
    efficiency_ranking_key = st.selectbox(
        "Select Efficiency Ratio for Ranking:",
        list(efficiency_ratio_options.keys()),
        format_func=lambda x: efficiency_ratio_options[x]  # Display full names
    )
    efficiency_ranking_label = efficiency_ratio_options[efficiency_ranking_key]

    # Use a separate filtered DataFrame for ranking
    ranking_df = filtered_df.copy()  # Create a copy to avoid modifying the original
    latest_efficiency_data = ranking_df.groupby("TICKER").last()  # Get the last entry for each stock
    efficiency_ranking_data = latest_efficiency_data[efficiency_ranking_key].sort_values(ascending=False)  # Sort stocks

    # Define constant colors for each stock ticker
    unique_tickers = filtered_df['TICKER'].unique()  # Use the global unique tickers
    color_map = {
        ticker: plt.cm.tab10(i % 10) for i, ticker in enumerate(sorted(unique_tickers))  # Assign consistent colors
    }

    # Map colors to tickers
    bar_colors = [color_map[ticker] for ticker in efficiency_ranking_data.index]

    # Plot the ranking as a bar chart
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(
        efficiency_ranking_data.index,
        efficiency_ranking_data.values,
        color=bar_colors,
        edgecolor="black"
    )
    ax.set_title(f"{efficiency_ranking_label} Ranking", fontsize=16, fontweight="bold")
    ax.set_ylabel(efficiency_ranking_label, fontsize=12)
    ax.set_xlabel("Stock Ticker", fontsize=12)
    plt.xticks(rotation=45, fontsize=10)

    # Display the bar chart in the dashboard
    st.pyplot(fig)


# Add checkbox for Solvency Ratio Ranking
show_solvency_ranking = st.sidebar.checkbox("Show Solvency Ratio Ranking", value=False)

if show_solvency_ranking:
    st.subheader("🏆 Solvency Ratio Ranking")

    # Select a solvency ratio for ranking
    solvency_ranking_key = st.selectbox(
        "Select Solvency Ratio for Ranking:",
        list(solvency_ratio_options.keys()),
        format_func=lambda x: solvency_ratio_options[x]
    )
    solvency_ranking_label = solvency_ratio_options[solvency_ranking_key]

    # Get the latest available values of the selected ratio for all stocks
    latest_data = filtered_df.groupby("TICKER").last()  # Get the last entry for each stock
    ranking_data = latest_data[solvency_ranking_key].sort_values(ascending=False)  # Sort stocks by the ratio

    # Define constant colors for each stock ticker (if not already defined globally)
    unique_tickers = filtered_df['TICKER'].unique()  # Get all unique tickers
    color_map = {
        ticker: plt.cm.tab10(i % 10) for i, ticker in enumerate(sorted(unique_tickers))  # Assign consistent colors
    }

    # Use the color mapping for bar chart colors
    bar_colors = [color_map[ticker] for ticker in ranking_data.index]  # Map colors to tickers

    # Plot the ranking as a bar chart
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(
        ranking_data.index,
        ranking_data.values,
        color=bar_colors,  # Use consistent colors
        edgecolor="black"
    )

    # Add chart title and labels
    ax.set_title(f"{solvency_ranking_label} Ranking", fontsize=16, fontweight="bold")
    ax.set_ylabel(solvency_ranking_label, fontsize=12)
    ax.set_xlabel("Stock Ticker", fontsize=12)

    # Manually set the Y-axis range here (optional, modify as needed)
    y_min = -2  # Default minimum value (e.g., 0 for solvency ratios)
    y_max = ranking_data.max() + (ranking_data.max() * 0.1)  # Add 10% padding above max value
    ax.set_ylim(y_min, y_max)

    # Rotate X-axis labels for better readability
    plt.xticks(rotation=45, fontsize=10)

    # Display the bar chart in the dashboard
    st.pyplot(fig)



# Function to fetch live stock price with up/down indication
def get_live_price_with_arrow(ticker):
    try:
        stock = yf.Ticker(ticker)
        # Fetch live data for the current day
        live_data = stock.history(period="1d")
        if live_data.empty:
            return "Market data unavailable"
        
        current_price = live_data['Close'].iloc[-1]  # Last closing price
        prev_close = live_data['Close'].iloc[0]  # Previous close price
        
        # Determine if the stock is up or down
        if current_price > prev_close:
            return f"${current_price:.2f} ↑"  # Up, add green arrow
        elif current_price < prev_close:
            return f"${current_price:.2f} ↓"  # Down, add red arrow
        else:
            return f"${current_price:.2f}"  # No change
    except Exception as e:
        return f"Error fetching price: {e}"

# Sidebar integration for real-time stock prices
show_live_prices = st.sidebar.checkbox("Show Live Stock Prices", value=False)

if show_live_prices and tickers:
    st.subheader("Live Stock Prices")
    for ticker in tickers:
        price_output = get_live_price_with_arrow(ticker)
        # Use colored arrows
        if "↑" in price_output:
            st.markdown(f"{ticker}: <span style='color:green;'>{price_output}</span>", unsafe_allow_html=True)
        elif "↓" in price_output:
            st.markdown(f"{ticker}: <span style='color:red;'>{price_output}</span>", unsafe_allow_html=True)
        else:
            st.write(f"{ticker}: {price_output}")

# Function to fetch historical stock prices
def get_historical_prices(ticker, start_date, end_date):
    stock = yf.Ticker(ticker)
    return stock.history(start=start_date, end=end_date)

# Sidebar integration for historical price display (separate from the chart)
show_price_history = st.sidebar.checkbox("Show Historical Price Trends", value=False)

if show_price_history and tickers:
    st.subheader("Historical Stock Price Trends")
    for ticker in tickers:
        try:
            # Fetch historical prices for the selected date range
            price_data = get_historical_prices(
                ticker,
                filtered_df['qdate'].min(),
                filtered_df['qdate'].max()
            )

            # Plot historical prices
            fig_price, ax_price = plt.subplots(figsize=(12, 5))  # Adjust figure size
            ax_price.plot(
                price_data.index,
                price_data['Close'],
                linestyle='-',
                marker='o',
                markersize=4,  # Set marker size for clarity
                linewidth=2,  # Set thicker line width for better visibility
                label=f"{ticker} Prices"
            )
            ax_price.set_title(f"Price Trend for {ticker}", fontsize=16, fontweight='bold')
            ax_price.set_xlabel("Date", fontsize=12)
            ax_price.set_ylabel("Close Price (USD)", fontsize=12)
            ax_price.grid(visible=True, linestyle='--', alpha=0.6)  # Make gridlines subtle but visible
            ax_price.legend(loc="best", fontsize=12)  # Improve legend readability

            # Format x-axis for better readability
            ax_price.xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%Y-%m'))  # Year-Month format
            plt.xticks(rotation=45, fontsize=10)  # Rotate and adjust size of x-axis labels
            plt.yticks(fontsize=10)

            # Display the chart
            st.pyplot(fig_price)

        except Exception as e:
            st.write(f"Could not fetch historical prices for {ticker}. Error: {e}")

# Step 4: Plot the financial ratio over time for each selected ticker
# Change the white title text for the Financial Ratio Chart
st.subheader("Financial Ratio Comparison Over Time")
fig, ax = plt.subplots(figsize=(10, 6))

# Create a colormap
colors = cm.get_cmap("tab10", len(tickers))

for idx, ticker in enumerate(tickers):
    # Filter and align stock data to the global date range
    stock_data = (
        filtered_df[filtered_df['TICKER'] == ticker]
        .set_index('qdate')
        .reindex(full_date_range, fill_value=None)  # Align to global quarterly date range
        .interpolate(method='linear')  # Fill missing values
        .fillna(method='bfill')  # Fill gaps at the start
        .fillna(method='ffill')  # Fill gaps at the end
        .reset_index()
    )

    # Plot stock data
    ax.plot(
        stock_data['index'],  # Aligned dates
        stock_data[selected_ratio_key],
        color=colors(idx),  # Assign color based on colormap
        marker='o',
        linestyle='-',
        label=ticker,
    )

# Step 5: Calculate industry benchmark with real confidence intervals
industry_stats = (
    filtered_df.groupby('qdate')[selected_ratio_key]
    .agg(['mean', 'std', 'count'])  # Compute mean, standard deviation, and count
    .reindex(full_date_range)  # Align to global quarterly date range
)

# Calculate Standard Error of the Mean (SEM)
industry_stats['sem'] = industry_stats['std'] / industry_stats['count']**0.5

# Calculate Margin of Error based on selected confidence level
industry_stats['margin_of_error'] = z * industry_stats['sem']

# Calculate Confidence Interval
industry_stats['lower_bound'] = industry_stats['mean'] - industry_stats['margin_of_error']
industry_stats['upper_bound'] = industry_stats['mean'] + industry_stats['margin_of_error']

# Fill missing values (if needed)
industry_stats = industry_stats.fillna(method='bfill').fillna(method='ffill')

# Consisten Gridline Styling for charts
gridline_style = {'linestyle': '--', 'linewidth': 0.7, 'alpha': 0.7}

# Plot industry benchmark
ax.plot(
    full_date_range,
    industry_stats['mean'],
    color='red',
    linestyle='--',
    linewidth=2,
    label='Industry Benchmark',
)

# Toggle confidence interval
if show_confidence_interval:
    ax.fill_between(
        full_date_range,
        industry_stats['lower_bound'],  # Lower bound
        industry_stats['upper_bound'],  # Upper bound
        color="red",
        alpha=0.2,
        label=f"{confidence_level}% Confidence Interval"
    )

# Calculate and plot trend line if enabled
if show_trend_line:
    x = np.arange(len(full_date_range))  # X-axis values as integers
    y = industry_stats['mean'].values  # Y-axis values
    slope, intercept, _, _, _ = linregress(x, y)  # Linear regression
    trendline = slope * x + intercept  # Calculate trendline
    ax.plot(
        full_date_range,
        trendline,
        color='blue',
        linestyle='--',
        label='Trend Line'
    )

# Calculate and plot moving average if enabled
if show_moving_avg:
    moving_avg = industry_stats['mean'].rolling(window=4).mean()  # 4-quarter moving average
    ax.plot(
        full_date_range,
        moving_avg,
        color='green',
        linestyle='--',
        linewidth=2,
        label='4Q Moving Average (Benchmark)'  # Updated label for clarity
    )

# Add gridlines
ax.grid(**gridline_style)

# Set plot labels and legend
ax.set_title(f"{selected_ratio_label} Comparison Over Time", fontsize=16, fontweight="bold")
ax.set_xlabel("Date", fontsize=12)
ax.set_ylabel(selected_ratio_label, fontsize=12)
plt.xticks(fontsize=10, rotation=45)
plt.yticks(fontsize=10)
ax.legend(title="Ticker", loc="best", fontsize=10, title_fontsize=12)


#  Apply consistent Y-axis range
global_min = filtered_df[selected_ratio_key].min()  # Minimum for the selected ratio
global_max = filtered_df[selected_ratio_key].max()  # Maximum for the selected ratio
ax.set_ylim(global_min, global_max + .1)

# Display the chart
st.pyplot(fig)

# Plot Efficiency Ratio Chart
# Change the white title text for the Financial Ratio Chart
st.subheader("Efficiency Ratio Comparison Over Time")
fig_efficiency, ax_efficiency = plt.subplots(figsize=(10, 6))

# Create a colormap for individual companies
colors = cm.get_cmap("tab10", len(tickers))

# Plot individual company data for efficiency ratios
for idx, ticker in enumerate(tickers):
    stock_data = (
        filtered_df[filtered_df['TICKER'] == ticker]
        .set_index('qdate')
        .reindex(full_date_range, fill_value=None)
        .interpolate(method='linear')
        .fillna(method='bfill')
        .fillna(method='ffill')
        .reset_index()
    )
    ax_efficiency.plot(
        stock_data['index'],
        stock_data[selected_efficiency_ratio_key],
        color=colors(idx),
        marker='o',
        linestyle='-',
        label=ticker,
    )

# Calculate industry statistics for efficiency ratios
industry_stats = (
    filtered_df.groupby('qdate')[selected_efficiency_ratio_key]
    .agg(['mean', 'std', 'count'])
    .reindex(full_date_range, fill_value=None)
    .interpolate(method='linear')
    .fillna(method='bfill')
    .fillna(method='ffill')
)
industry_stats['sem'] = industry_stats['std'] / industry_stats['count']**0.5
industry_stats['margin_of_error'] = z * industry_stats['sem']
industry_stats['lower_bound'] = industry_stats['mean'] - industry_stats['margin_of_error']
industry_stats['upper_bound'] = industry_stats['mean'] + industry_stats['margin_of_error']

# Plot industry benchmark (median)
industry_benchmark = (
    filtered_df.groupby('qdate')[selected_efficiency_ratio_key]
    .median()
    .reindex(full_date_range, fill_value=None)
    .interpolate(method='linear')
    .fillna(method='bfill')
    .fillna(method='ffill')
)
ax_efficiency.plot(
    full_date_range,
    industry_benchmark,
    color='red',
    linestyle='--',
    linewidth=2,
    label='Industry Median Benchmark',
)

# Add trend line if enabled
if show_trend_line:
    x = np.arange(len(full_date_range))  # Convert dates to numerical values
    y = industry_benchmark.values  # Use the benchmark median for the trend line
    slope, intercept, _, _, _ = linregress(x, y)
    trendline = slope * x + intercept
    ax_efficiency.plot(
        full_date_range,
        trendline,
        color='blue',
        linestyle='--',
        label='Trend Line'
    )

# Add moving average if enabled
if show_moving_avg:
    moving_avg = industry_stats['mean'].rolling(window=4).mean()  # 4-quarter moving average
    ax_efficiency.plot(
        full_date_range,
        moving_avg,
        color='green',
        linestyle='--',
        linewidth=2,
        label='4Q Moving Average (Benchmark)'
    )

# Add titles, labels, legend, and gridlines
ax_efficiency.set_title(f"{selected_efficiency_ratio_label} Comparison Over Time", fontsize=16, fontweight="bold")
ax_efficiency.set_xlabel("Date", fontsize=12)
ax_efficiency.set_ylabel(selected_efficiency_ratio_label, fontsize=12)
ax_efficiency.legend(title="Ticker", loc="best", fontsize=10)
plt.xticks(rotation=45)
plt.yticks(fontsize=10)
gridline_style = {'linestyle': '--', 'linewidth': 0.7, 'alpha': 0.7}
ax_efficiency.grid(**gridline_style)

# Display the chart
st.pyplot(fig_efficiency)

# Solvency Ratios Visualization
# Display the chart for the selected solvency ratio
if selected_solvency_ratio_key:  # Ensure this block handles all selected tickers
    # Change the white title text for the Financial Ratio Chart
    st.subheader("Solvency Ratio Comparison Over Time")


    # Initialize the plot
    fig, ax = plt.subplots(figsize=(10, 6))

    # Plot the solvency ratio for all selected tickers
    for ticker in tickers:
        stock_data = (
            filtered_df[filtered_df['TICKER'] == ticker]
            .set_index('qdate')
            .reindex(full_date_range, fill_value=None)
            .interpolate(method='linear')
            .fillna(method='bfill')
            .fillna(method='ffill')
            .reset_index()
        )
        ax.plot(
            stock_data['index'],
            stock_data[selected_solvency_ratio_key],
            label=ticker,
            marker='o',
            linestyle='-',
        )

    # Calculate and plot the industry benchmark
    industry_benchmark = (
        filtered_df.groupby('qdate')[selected_solvency_ratio_key]
        .median()
        .reindex(full_date_range, fill_value=None)
        .interpolate(method='linear')
        .fillna(method='bfill')
        .fillna(method='ffill')
    )

    ax.plot(
        full_date_range,
        industry_benchmark,
        color='red',
        linestyle='--',
        linewidth=2,
        label='Industry Median Benchmark'
    )

    # Add 4-quarter moving average if enabled
    if show_moving_avg:
        moving_avg = industry_benchmark.rolling(window=4).mean()  # 4-quarter moving average
        ax.plot(
            full_date_range,
            moving_avg,
            color='green',
            linestyle='--',
            linewidth=2,
            label='4Q Moving Average (Benchmark)'
        )

    # Add trend line if enabled
    if show_trend_line:
        x = np.arange(len(full_date_range))  # Convert dates to numerical values
        y = industry_benchmark.values  # Use the benchmark median for the trend line
        slope, intercept, _, _, _ = linregress(x, y)
        trendline = slope * x + intercept
        ax.plot(
            full_date_range,
            trendline,
            color='blue',
            linestyle='--',
            label='Trend Line'
        )

    # Add title, labels, and legend
    ax.set_title(f"{solvency_ratio_options[selected_solvency_ratio_key]} Comparison Over Time", fontsize=16, fontweight="bold")
    ax.set_xlabel("Date", fontsize=12)
    ax.set_ylabel(solvency_ratio_options[selected_solvency_ratio_key], fontsize=12)
    plt.xticks(rotation=45, fontsize=10)
    ax.legend(title="Ticker", loc="best", fontsize=10)
    ax.grid(**gridline_style)

    # Display the chart
    st.pyplot(fig)


# Special Notes for QCOM and ORCL (Rendered Independently)
if 'ORCL' in tickers:
    st.warning(
        """
        **Special Note About Oracle (ORCL) Debt-to-Equity 2021 Data:**  
        - Oracle experienced a significant stock buyback program, a 0.40 dollar per share dividend, 
          and a 1.25 billion dollar net loss in 2021 due to a legal judgment.  
        - These events caused dramatic fluctuations in Oracle's Debt-to-Equity Ratio, 
          including periods of negative equity in 2022.  
        - **When comparing Oracle with other tickers, the chart will appear distorted due to Oracle's extreme values during this time.**
        """,
        icon="⚠️"
    )

if 'QCOM' in tickers:
    st.warning(
        """
        **Special Note About Qualcomm (QCOM) Debt-to-Equity 2018-2019 Data:**  
        - In 2018, Qualcomm settled a significant legal dispute with Apple, resulting in a one-time payment of $4.5–4.7 billion.  
        - This settlement caused a spike in Qualcomm's Debt-to-Equity Ratio, as recorded in Q3 2018.  
        - In 2015, Qualcomm initiated a $5 billion stock buyback program and dividend payments, impacting its equity values significantly.  
        - **These events will distort comparisons with other companies, so exercise caution when analyzing QCOM's data for these periods.**
        """,
        icon="⚠️"
    )

# Step 7: Export data to CSV
csv = df.to_csv(index=False)  # Convert the DataFrame to CSV format
st.download_button(
    label="Download Financial Data as CSV",
    data=csv,
    file_name='financial_ratios.csv',
    mime='text/csv'
)

# Author Footnote
st.markdown(
    """
    ---
    **Author:** Kurt W 2024  
    *This dashboard is my personal project for financial analysis and visualization.*
    """
)

# Run with your unique streamlit file path
