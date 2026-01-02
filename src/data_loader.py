import yfinance as yf
import pandas as pd

def load_crypto_data(symbol="BTC-USD", start="2024-06-01"):
    """
    Download historical price data from Yahoo Finance.
    """
    df = yf.download(symbol, start=start, progress=False)[["Close"]].reset_index()
    df.columns = ["date", "close"]
    if df.empty:
        raise ValueError(f"No data returned for {symbol}")
    return df.dropna()

