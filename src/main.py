import argparse
import logging
from data_loader import load_crypto_data
from features import create_features
from ml_model import prepare_data, train_and_test


# Setup logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


# Argument parser

parser = argparse.ArgumentParser(description="Beginner Quant Project")
parser.add_argument("--symbol", type=str, default="BTC-USD", help="Crypto symbol, e.g., BTC-USD")
parser.add_argument("--start", type=str, default="2018-01-01", help="Start date YYYY-MM-DD")
args = parser.parse_args()


# Main workflow

def main(symbol, start):
    logging.info(f"Loading data for {symbol} starting {start}...")
    df = load_crypto_data(symbol, start=start)
    logging.info(f"Data loaded: {len(df)} rows")

    logging.info("Creating features...")
    df = create_features(df)
    logging.info(f"Features created: {df.columns.tolist()}")

    X, y = prepare_data(df)
    logging.info(f"Prepared ML data: X shape {X.shape}, y shape {y.shape}")

    logging.info("Training and testing model...")
    model, train_acc, test_acc = train_and_test(X, y)

    logging.info(f"Train Accuracy: {round(train_acc,3)}")
    logging.info(f"Test Accuracy: {round(test_acc,3)}")

if __name__ == "__main__":
    main(args.symbol, args.start)
