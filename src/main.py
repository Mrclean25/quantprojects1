from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

def prepare_data(df):
    
    #Prepare features X and target y for ML model.
    Target: 1 if next day close > current close, else 0
   
    df = df.copy()
    df["target"] = (df["close"].shift(-1) > df["close"]).astype(int)
    df = df.dropna()
    features = ["return", "ma_short", "ma_long", "rsi", "volatility"]
    X = df[features]
    y = df["target"]
    return X, y

def train_and_test(X, y):
    
    #Train a Logistic Regression model and return train/test accuracy.
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, shuffle=False)
    model = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(C=0.5, max_iter=1000))
    ])
    model.fit(X_train, y_train)
    train_acc = accuracy_score(y_train, model.predict(X_train))
    test_acc = accuracy_score(y_test, model.predict(X_test))
    return model, train_acc, test_acc
