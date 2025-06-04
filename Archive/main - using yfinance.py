import requests
import streamlit as st
from datetime import date
import yfinance as yf
from prophet import Prophet
from prophet.diagnostics import cross_validation, performance_metrics
from prophet.plot import plot_plotly
from plotly import graph_objs as go
import pandas as pd
import numpy as np
from plotly.subplots import make_subplots
import ta
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error
import itertools

# Main Title
st.title('Forecasting Tool for Swing Trading')

with st.expander('-- Welcome! Click here to expand --'):
    st.write("""
    -- Welcome to the Prophet Forecasting App: A Data Science Exploration --

    **by Shane Peterson**

    This tool demonstrates the application of the Prophet forecasting model for predicting short-term price movements in stocks. This project focuses on showcasing the following data science skills:

    * **Time Series Forecasting:** Utilizing the Prophet model to predict stock prices.
    * **Hyperparameter Tuning:** Optimizing model performance through hyperparameter tuning. (see About section below)
    * **Model Evaluation:** Assessing model performance using appropriate metrics (e.g., SMAPE, RMSE). (see Accuracy Metrics section below)

    The app includes visualizations such as candlestick charts, moving averages, and Bollinger Bands to provide context for the forecast.

    **Disclaimer: This app is for educational and demonstrative purposes only. It is not a financial recommendation and should not be used for actual trading decisions.**

    For more technical details please refer to the About section and the Appendix.
    """)

### Ticker Selection Searchbar
st.subheader('-- Choose a Stock --')
with st.expander('-- Click here to expand --'):
    selected_stock = st.text_input("Enter Symbol (Ticker List in Appendix)", value="goog").upper()

    # Get Ticker Metadata
    # ------------------------------------------------------------------
    FINNHUB_API_KEY = 'cuaq7shr01qof06j5bfgcuaq7shr01qof06j5bg0'
    EXCHANGE_CODE = 'US' 

    url = f'https://finnhub.io/api/v1/stock/symbol?exchange={EXCHANGE_CODE}&token={FINNHUB_API_KEY}'

    try:
        response = requests.get(url)
        response.raise_for_status()  # Raise an exception for bad status codes
        tickers_data = response.json()

        # Extract ticker symbols from the response
        tickers = [item['symbol'] for item in tickers_data] 

        # print(tickers)

    except requests.exceptions.RequestException as e:
        data_load_state = st.text(f"-- Error fetching data: {e} --")

    # ------------------------------------------------------------------
    # Extract ticker name using symbol
    # ------------------------------------------------------------------
    for item in tickers_data:
        if item['symbol'] == selected_stock:
            ticker_name = item['description']
        
    # Get ticker raw data
    @st.cache_data # caches data from different tickers
    def load_data(ticker):
        """
        Downloads historical market data for a given ticker symbol.

        Parameters:
        ticker (str): The ticker symbol of the stock to download data for.

        Returns:
        pd.DataFrame: A DataFrame containing the historical market data for the specified ticker.
        """
        data = yf.download(ticker, period='max') # returns relevant data in df
        data.reset_index(inplace=True) # reset multindex, output is index list of tuples
        cols = list(data.columns) # convert index to list
        cols[0] = ('Date', '') 
        cols = [i[0] for i in cols] # return first element of cols tuples
        data.columns = cols # set as column names
        data['Date'] = pd.to_datetime(data['Date']).dt.date
        return data

    # Check input against list of tickers
    data_load_state = st.text("-- Loading Data... --")
    if selected_stock in tickers:
        data = load_data(selected_stock)
        data_load_state.text(f"-- {ticker_name} Data Loaded. --")
    else:
        data_load_state.text(f"-- '{selected_stock}' is not a valid Symbol. Please enter a symbol from the Ticker List in the Appendix below. --")

    # Change Data to datetime64[ns] datatype
    data.Date = pd.to_datetime(data.Date)
    data.Date = data.Date.astype('datetime64[ns]')

    # ------------------------------------------------------------------
    # Feature Engineering
    # ------------------------------------------------------------------

    # Get Ticker Stats
    # ------------------------------------------------------------------
    stats = {}
    for item in tickers_data:
        if item['symbol'] == selected_stock:
            stats['Symbol'] = item['symbol']
            # stats['Name'] = item['Name']
            stats['Current Price'] = round(data.Close.iloc[-1], 2)
            stats['Current Volume'] = data.Volume.iloc[-1]
            data['daily_returns'] = data.Close.pct_change()
            volatility = data.daily_returns.std() * np.sqrt(252)
            if volatility < 0.2:
                category = "Low"
                percentiles=(0.15, 0.85)
            elif volatility < 0.4:
                category = "Medium-Low"
                percentiles=(0.1, 0.9)
            elif volatility < 0.6:
                category = "Medium"
                percentiles=(0.1, 0.9)
            elif volatility < 0.8:
                category = "Medium-High"
                percentiles=(0.05, 0.95)
            else:
                category = "High"
                percentiles=(0.05, 0.95)
            stats['Annualized Volatility'] = category
            stats['Percentage Change'] = str(round(data['daily_returns'].mean() * 100, 4)) + ' %'
            stats['IPO'] = min(data.Date)
            stats['Historical Low'] = round(min(data.Low), 2)
            stats['HL Date'] = data.Date[data.Low.idxmin()]
            stats['Historical High'] = round(max(data.High), 2)
            stats['HH Date'] = data.Date[data.High.idxmax()]
    stats_window_df = pd.DataFrame(stats, index=[0])

    # Get Stock Age & Set training & forecast periods
    # ------------------------------------------------------------------
    if len(data)/365 < 8:
        period_unit = int(len(data)/4)
        forecast_period = period_unit
        train_period = len(data)
        stock_age = 'young'
    else:
        period_unit = 365
        forecast_period = period_unit
        train_period = forecast_period * 4 if volatility < 0.6 else forecast_period * 8
        stock_age = 'seasoned'

    # Get stats window
    st.write(stats_window_df)

# ------------------------------------------------------------------
# Process Indicators
# ------------------------------------------------------------------
data['SMA50'] = data['Close'].rolling(window=50).mean()
indicator_bb = ta.volatility.BollingerBands(close=data['Close'], window=20, window_dev=2)
data['bb_upper'] = indicator_bb.bollinger_hband()
data['bb_lower'] = indicator_bb.bollinger_lband()

# ------------------------------------------------------------------
# FORECASTING
# ------------------------------------------------------------------

# Windsorize Function
# ------------------------------------------------------------------
def dynamic_winsorize(df, column, window_size=30, percentiles=percentiles):
    """
    Winsorizes data within a rolling window.

    Args:
        df: DataFrame containing the data.
        column: Name of the column to winsorize.
        window_size: Size of the rolling window.
        percentiles: Tuple containing the lower and upper percentiles.

    Returns:
        DataFrame with the winsorized column.
    """

    df['rolling_lower'] = df[column].rolling(window=window_size).quantile(percentiles[0])
    df['rolling_upper'] = df[column].rolling(window=window_size).quantile(percentiles[1])

    df['winsorized'] = df[column]
    df.loc[df[column] < df['rolling_lower'], 'winsorized'] = df['rolling_lower']
    df.loc[df[column] > df['rolling_upper'], 'winsorized'] = df['rolling_upper']

    return df

# Apply dynamic winsorization to raw data
data = dynamic_winsorize(data, 'Close')

# Get training data
# ------------------------------------------------------------------
df_train = data[['Date', 'Close', 'winsorized']]
df_train = df_train.rename(columns={'Date': 'ds'})

df_train = df_train[-train_period:] 

# Lambda function for cross validation metrics
# ------------------------------------------------------------------
cv_func = lambda model_name: cross_validation(model_name, 
                                              initial=f'{train_period} days', 
                                              period=f'{period_unit} days', 
                                              horizon=f'{forecast_period} days')

# Get metrics for baseline & winsorized models
# ------------------------------------------------------------------

scores_df = pd.DataFrame(columns=['mse', 'rmse', 'mae', 'smape'])

@st.cache_resource
def model_drafts(df_train, scores_df=scores_df):
    """
    Trains Prophet models on 'Close' and 'winsorized' columns, evaluates their performance, and updates the scores DataFrame.

    Parameters:
    df_train (pd.DataFrame): DataFrame containing the training data with 'ds' (date) and 'Close'/'winsorized' columns.
    scores_df (pd.DataFrame): DataFrame to store the performance metrics of the models.

    Returns:
    pd.DataFrame: Updated scores DataFrame with performance metrics for 'Close' and 'winsorized' models.
    """
    for i in ['Close', 'winsorized']:
        m = Prophet()
        df_train_renamed = df_train[['ds', i]].rename(columns={i: 'y'})
        m.fit(df_train_renamed)
        df_cv = cv_func(m)
        df_p = performance_metrics(df_cv, rolling_window=1)
        scores_df = pd.concat([scores_df, df_p[['mse', 'rmse', 'mae', 'smape']]], ignore_index=True)
    return scores_df

data_load_state = st.text("-- Please wait while the Baseline & Winsorized models train... --")
if len(df_train) > 0:
    scores_df = model_drafts(df_train)
    data_load_state.text("-- Baseline & Winsorized models Trained. --")
else:
    data_load_state.text("-- Error in training models. --")

# Train final model on best performing model draft
if scores_df.iloc[0]['rmse'] < scores_df.iloc[1]['rmse']:
    df_train = df_train.rename(columns={'Close': 'y'})
else:
    df_train = df_train.rename(columns={'winsorized': 'y'})

# Prepare for grid search of combos of all pararmeters
# ------------------------------------------------------------------
param_grid = {
    'changepoint_prior_scale': [0.001, 0.01, 0.1, 0.5],
    'seasonality_prior_scale': [0.01, 0.1, 1.0, 10.0]
}
# Generate combos of all pararmeters
# ------------------------------------------------------------------
all_params = [dict(zip(param_grid.keys(), v)) for v in itertools.product(*param_grid.values())]

@st.cache_resource
def tune_and_train_final_model(df_train, all_params, forecast_period, scores_df=scores_df):
    """
    Tunes hyperparameters, trains the final Prophet model, evaluates its performance, and generates a forecast.

    Parameters:
    df_train (pd.DataFrame): DataFrame containing the training data with 'ds' (date) and target columns.
    all_params (list): List of dictionaries containing hyperparameter combinations to be tested.
    forecast_period (int): Number of periods to forecast into the future.
    scores_df (pd.DataFrame): DataFrame to store the performance metrics of the models.

    Returns:
    tuple: A tuple containing the trained model, updated scores DataFrame, forecast DataFrame, and best hyperparameters dictionary.
    """
    rmses = []
    for params in all_params:
        m = Prophet(**params).fit(df_train)
        df_cv = cv_func(m)
        df_p = performance_metrics(df_cv, rolling_window=1)
        rmses.append(df_p['rmse'].values[0])

    # Find best parameters
    tuning_results = pd.DataFrame(all_params)
    tuning_results['rmse'] = rmses
    best_params_dict = dict(tuning_results.sort_values('rmse').reset_index(drop=True).drop('rmse', axis='columns').iloc[0])

    m = Prophet(**best_params_dict)
    m.fit(df_train)
    df_cv = cv_func(m)
    df_p = performance_metrics(df_cv, rolling_window=1)
    scores_df = pd.concat([scores_df, df_p[['mse', 'rmse', 'mae', 'smape']]], ignore_index=True)
    future = m.make_future_dataframe(periods=forecast_period)
    forecast = m.predict(future)
    
    return m, scores_df, forecast, best_params_dict

data_load_state = st.text("-- Please wait while the Final Model trains... --")
if len(df_train) > 0:
    m, scores_df, forecast, best_params_dict = tune_and_train_final_model(df_train, all_params, forecast_period)
    data_load_state.text("-- Final Model Trained. --")
else:
    data_load_state.text("-- Error in training final model. --")

# Merge entire forecast w actual data & indicators
# ------------------------------------------------------------------
forecast_candlestick_df = pd.merge(
    left=data,
    right=forecast,
    right_on='ds',
    left_on='Date',
    how='right')[['ds', 'Open', 'High', 'Low', 'Close', 'yhat', 'yhat_lower', 'yhat_upper', 'SMA50', 'bb_upper', 'bb_lower']]
forecast_candlestick_df.rename(columns={'ds': 'Date'}, inplace=True) # keep naming convention and ds data. Date does not contain forecast date values.

# Get metrics 
# ------------------------------------------------------------------
scores_df.index = ['Baseline Model', 'Winsorized Model', 'Final Model']
scores_df = scores_df.reindex(sorted(scores_df.columns), axis=1)

# Function & Indicators for Forecasted Candlestick Graph
# ------------------------------------------------------------------
@st.cache_resource
def plot_forecast(data):
    """
    Generates a Plotly figure for the forecasted candlestick graph.

    Parameters:
    data (pd.DataFrame): DataFrame containing the forecast data and indicators.

    Returns:
    go.Figure: Plotly figure object with the forecasted candlestick graph.
    """
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data['Date'], 
                         y=data['yhat_lower'], 
                         line=dict(color='lightblue', width=2), 
                         name='Forecast Lower Bound'))
    # Add upper forecast bound (yhat_upper)
    fig.add_trace(go.Scatter(x=data['Date'], 
                             y=data['yhat_upper'], 
                             line=dict(color='lightblue', width=2), 
                             name='Forecast Upper Bound',
                             fill='tonextx', # Fill between forecast and upper bound
                             fillcolor=None,
                             opacity=0.01))
    # Add forecast line (yhat)
    fig.add_trace(go.Scatter(x=data['Date'], 
                             y=data['yhat'], 
                             line=dict(color='blue', width=2), 
                             name='Forecast',
                             mode='lines'))
    # Add upper Bollinger Band
    fig.add_trace(go.Scatter(x=data['Date'], 
                             y=data['bb_upper'], 
                             line=dict(color='red', width=1), 
                             name='Upper BB',
                             visible='legendonly'))
    # Add lower Bollinger Band
    fig.add_trace(go.Scatter(x=data['Date'], 
                             y=data['bb_lower'], 
                             line=dict(color='green', width=1), 
                             name='Lower BB',
                             visible='legendonly'))
    # Add SMA trace
    fig.add_trace(go.Scatter(x=data['Date'], 
                            y=data['SMA50'], 
                            name='SMA50', 
                            line=dict(color='black', width=2, dash='dash'),
                            visible='legendonly'))
    # Add candlestick trace
    fig.add_trace(go.Candlestick(x=data['Date'],
                open=data['Open'],
                high=data['High'],
                low=data['Low'],
                close=data['Close'],
                name='Candlestick'))
    # Labels
    fig.layout.update(
        title_text=f"Forecast for Time Series Data: {ticker_name} '{selected_stock}'",
                xaxis_rangeslider_visible=True,
                yaxis_title='Price',
                xaxis_title='Date')
    # Calculate default date range
    end_date = data['Date'].max()
    start_date = end_date - pd.Timedelta(days=period_unit*((train_period/period_unit)-1))
    fig.update_xaxes(range=[start_date, end_date])
    st.plotly_chart(fig)

plot_forecast(forecast_candlestick_df)


# Chart Tips
# ------------------------------------------------------------------
st.subheader('-- Chart Tips --')
with st.expander('Click here to expand'):
    st.write('* Use the slider (above) to select a date range')
    st.write('* Click items in the legend to show/hide indicators')
    st.write('* Hover in the upper-right corner of graph to reveal controls. Go fullscreen and explore!')

# Accuracy Metrics
# ------------------------------------------------------------------
st.subheader('**-- Accuracy Metrics --**')
st.write('Model Accuracy Score:')
st.subheader(f'{100-(round(scores_df['smape'].iloc[2]*100, 2))}%')


# Metrics notes
# ------------------------------------------------------------------
st.write('-- More Metrics --')
with st.expander('Click here to expand'):
    st.subheader('-- Model Iterations --')
    st.write("The tables below display the performance metrics for each model iteration. The 'Baseline Model' uses the raw closing prices, while the 'Winsorized Model' applies dynamic winsorization to the closing prices. The 'Final Model' is the best-performing model after hyperparameter tuning.")
    st.write('-- Baseline Model --')
    st.dataframe(scores_df.loc[['Baseline Model']], width=500)
    st.write('-- Winsorized Model --')
    st.dataframe(scores_df.loc[['Winsorized Model']], width=500)
    st.write('-- Final Model --')
    st.dataframe(scores_df.loc[['Final Model']], width=500)
    st.write("In the context of time series forecasting, 'error' refers to the difference between the actual value of a variable at a specific point in time and the value predicted by a forecasting model. In this case, the metrics will specifically measure the error between the stock's closing price and the forecast trained on the closing price.")
    st.write(f"* Mean Absolute Error (MAE) - a MAE of {round(scores_df['mae'].iloc[2], 4)} implies that, on average, the model's predictions are off by approximately ${round(scores_df['mae'].iloc[2], 2)}.")
    st.write(f"* Symmetric Mean Absolute Percentage Error (smape) - a smape of {round(scores_df['smape'].iloc[2], 4)} means that, on average, the model's predictions are {round(scores_df['smape'].iloc[2] * 100, 2)}% off from the actual values.")
    st.write('* Mean Squared Error (MSE) - this squares the errors, giving more weight to larger errors. A lower MSE indicates better accuracy.')
    st.write(f"* Root Mean Squared Error (RMSE) -  The square root of MSE. It is in the same units as the original data, making it easier to interpret. The RMSE of {round(scores_df['rmse'].iloc[2], 4)} suggests that the model's predictions can deviate from the actual values by up to ${round(scores_df['rmse'].iloc[2], 2)} in some cases.")

# ------------------------------------------------------------------
### ABOUT
# ------------------------------------------------------------------

about_str = f"""
**-- The Tool --**

As a passionate swing trader, I developed this application to streamline my decision-making process. It leverages fundamental data science concepts, including data engineering and analytics, to provide actionable insights. 

The app features a user-friendly interface with a candlestick chart, Bollinger Bands, and a Simple Moving Average (SMA) for visual analysis of price trends. 

**-- The Model --**

To enhance my swing trading strategy, I've integrated a Prophet forecasting model, fine-tuned with techniques like winsorization and hyperparameter tuning to optimize its accuracy. 

Key Model Enhancements:
* Adaptive Winsorization: The winsorization thresholds are dynamically adjusted based on the stock's volatility. 
* Adaptive Training Data: The training data size is dynamically adjusted based on the stock's volatility and available data.

By combining these refinements with a cross-validated grid search to optimize changepoint_prior_scale and seasonality_prior_scale, this application provides a robust forecasting tool.

For '{selected_stock}', optimal values are: changepoint_prior_scale: {best_params_dict["changepoint_prior_scale"]:.3f}, seasonality_prior_scale: {best_params_dict["seasonality_prior_scale"]:.3f}.

Cross-validation ensures that the hyperparameters selected are not overfitted to a specific subset of the data. By evaluating the model's performance on multiple subsets of the data during the grid search, we can select hyperparameters that generalize better to unseen data and potentially improve the model's out-of-sample performance. Check out Model Iterations in the More Metrics section (above) to observe the model's improvement over its learning cycles.

**-- Swing Trading --**

Swing trading focuses on capturing short-term price movements. By combining the forecasting model with visual aids like candlestick charts, Bollinger Bands, and SMAs, I'm able to identify potential entry and exit points with greater confidence, ultimately refining my trading decisions.

By selecting a stock ticker, the app displays important background information like historical highs/lows, percentage change, volatility, and current price alongside the chart. This comprehensive tool empowers more informed trading decisions and refined trading strategies.
"""

st.subheader('-- About --')
with st.expander('Click here to expand'):
    st.write(about_str)

# ------------------------------------------------------------------
### APPENDIX
# ------------------------------------------------------------------

st.subheader('-- Appendix --') # button to hide / unhide
with st.expander('Click here to expand'):
    st.subheader('-- Ticker List --') # button to hide / unhide
    st.write(pd.DataFrame(tickers_data))
    st.subheader('-- Forecast Components --')
    fig2 = m.plot_components(forecast)
    st.write(fig2)
    st.subheader('-- Forecast Grid --')
    st.write(forecast)
    st.subheader('-- Raw Data --') # button to hide / unhide
    st.write(data)
