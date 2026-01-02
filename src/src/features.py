def moving_average(series, window):
    return series.rolling(window).mean()

def volatility(series, window):
    return series.pct_change().rolling(window).std()
