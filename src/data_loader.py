import yfinance as yf
import pandas as pd

def load_btc_data(start="2018-01-01"):
    btc = yf.download("BTC-USD", start=start, progress=False)
    df = btc[["Close"]].reset_index()
    df.columns = ["date", "close"]
    return df.dropna()
