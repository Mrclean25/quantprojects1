from src.data_loader import load_btc_data
from src.features import moving_average, volatility

df = load_btc_data()

df["ma"] = moving_average(df["close"], 5)
df["vol"] = volatility(df["close"], 5)

print(df.tail())
