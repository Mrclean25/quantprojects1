import logging
from data_loader import load_crypto_data
from features import create_features
from strategy import prepare_data, train_and_test
from backtest import backtest


# Logging setup

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


# PARAMETERS (5)

symbol = "BTC-USD"        # Crypto symbol
start_date = "2024-06-01" # Data start date
ma_short_window = 5       # Short MA
ma_long_window = 10       # Long MA
rsi_period = 5            # RSI period
vol_window = 5            # Volatility window
test_size = 0.3           # Train/test split fraction


# Workflow

logging.info(f"Loading data for {symbol} starting {start_date}...")
df = load_crypto_data(symbol, start=start_date)
logging.info(f"Data loaded: {len(df)} rows")

logging.info("Creating features...")
df = create_features(df, ma_short_window, ma_long_window, rsi_period, vol_window)
logging.info(f"Features created: {df.columns.tolist()}")

X, y = prepare_data(df)
logging.info(f"Prepared ML data: X shape {X.shape}, y shape {y.shape}")

logging.info("Training and testing model...")
model, metrics, X_test, y_test, y_test_pred = train_and_test(X, y, test_size=test_size)

logging.info(f"Train Accuracy: {metrics['train_acc']:.3f}")
logging.info(f"Test Accuracy: {metrics['test_acc']:.3f}")
logging.info(f"Precision: {metrics['precision']:.3f}")
logging.info(f"Recall: {metrics['recall']:.3f}")
logging.info(f"F1 Score: {metrics['f1']:.3f}")

logging.info("Backtesting strategy...")
df_backtest, backtest_metrics = backtest(df, y_test_pred)
