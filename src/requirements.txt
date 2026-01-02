import matplotlib.pyplot as plt
import numpy as np

def calculate_backtest_metrics(df):
    
    #Calculates standard backtesting metrics:
    Total return, annualized return, annualized volatility, Sharpe ratio, max drawdown
    
    total_return = df["cumulative_returns"].iloc[-1] - 1
    days = (df["date"].iloc[-1] - df["date"].iloc[0]).days
    annualized_return = (1 + total_return) ** (365 / days) - 1
    annualized_vol = df["strategy_returns"].std() * np.sqrt(252)
    
    sharpe_ratio = annualized_return / annualized_vol if annualized_vol != 0 else np.nan

    rolling_max = df["cumulative_returns"].cummax()
    drawdown = (df["cumulative_returns"] - rolling_max) / rolling_max
    max_drawdown = drawdown.min()

    metrics = {
        "Total Return": total_return,
        "Annualized Return": annualized_return,
        "Annualized Volatility": annualized_vol,
        "Sharpe Ratio": sharpe_ratio,
        "Max Drawdown": max_drawdown
    }
    return metrics

def backtest(df, y_pred, plot=True):
    
    #Backtesting strategy using predicted positions.
    
    df = df.copy()
    df = df.iloc[-len(y_pred):]
    df["position"] = y_pred
    df["strategy_returns"] = df["return"] * df["position"]
    df["cumulative_returns"] = (1 + df["strategy_returns"]).cumprod()
    df["buy_and_hold"] = (1 + df["return"]).cumprod()

    metrics = calculate_backtest_metrics(df)
    print("Backtest Metrics")
    for k, v in metrics.items():
        print(f"{k}: {v:.4f}")

    if plot:
        plt.figure(figsize=(12,6))
        plt.plot(df["date"], df["cumulative_returns"], label="Strategy")
        plt.plot(df["date"], df["buy_and_hold"], label="Buy & Hold")
        plt.xlabel("Date")
        plt.ylabel("Cumulative Returns")
        plt.title("Backtest: Strategy vs Buy & Hold")
        plt.legend()
        plt.show()

    return df, metrics



