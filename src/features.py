import pandas as pd

def moving_average(series, window):
    return series.rolling(window).mean()

def rsi(series, period=14):
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(period).mean()
    avg_loss = loss.rolling(period).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))

def volatility(series, window=5):
    return series.pct_change().rolling(window).std()

def create_features(df, ma_short_window, ma_long_window, rsi_period, vol_window):

    #Adds 5 ML features: return, ma_short, ma_long, rsi, volatility
    Parameters are configurable.
    
    df = df.copy()
    df["return"] = df["close"].pct_change()
    df["ma_short"] = moving_average(df["close"], ma_short_window)
    df["ma_long"] = moving_average(df["close"], ma_long_window)
    df["rsi"] = rsi(df["close"], rsi_period)
    df["volatility"] = volatility(df["close"], vol_window)
    return df.dropna()


