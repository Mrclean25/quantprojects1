from src.data_loader import load_btc_data
from src.features import create_features
from src.ml_model import prepare_data, train_and_test

df = load_btc_data(start="2018-01-01")

df = create_features(df)

X, y = prepare_data(df)

model, train_acc, test_acc = train_and_test(X, y)

print("Train Accuracy:", round(train_acc, 3))
print("Test Accuracy:", round(test_acc, 3))
