"""
prediction.py — Train & serve price predictions
Model: Random Forest (per asset) with rolling technical features.
Models are saved to /models/ so they are trained once and reused.
"""

import os
import pickle
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_percentage_error
from utils.data_loader import get_asset_df, ASSETS

MODELS_DIR = os.path.join(os.path.dirname(__file__), '..', 'models')
os.makedirs(MODELS_DIR, exist_ok=True)


# ─── Feature Engineering ────────────────────────────────────────────────────

def _build_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create rolling + lag features from Price column.
    Returns DataFrame with features + target (next-day price).
    """
    d = df[['Date', 'Price']].copy()

    # Lag features
    for lag in [1, 2, 3, 5, 10]:
        d[f'lag_{lag}'] = d['Price'].shift(lag)

    # Rolling stats
    for w in [5, 10, 20, 50]:
        d[f'ma_{w}']  = d['Price'].rolling(w).mean()
        d[f'std_{w}'] = d['Price'].rolling(w).std()

    # Daily return
    d['daily_ret']  = d['Price'].pct_change()
    d['ret_5d']     = d['Price'].pct_change(5)

    # RSI (14)
    delta    = d['Price'].diff()
    gain     = delta.clip(lower=0)
    loss     = -delta.clip(upper=0)
    avg_gain = gain.ewm(com=13, min_periods=14).mean()
    avg_loss = loss.ewm(com=13, min_periods=14).mean()
    rs       = avg_gain / avg_loss.replace(0, np.nan)
    d['rsi'] = 100 - 100 / (1 + rs)

    # Target: next-day price
    d['target'] = d['Price'].shift(-1)

    d.dropna(inplace=True)
    return d


FEATURE_COLS = [
    'lag_1', 'lag_2', 'lag_3', 'lag_5', 'lag_10',
    'ma_5', 'ma_10', 'ma_20', 'ma_50',
    'std_5', 'std_10', 'std_20', 'std_50',
    'daily_ret', 'ret_5d', 'rsi'
]


# ─── Train ───────────────────────────────────────────────────────────────────

def train(asset: str, force: bool = False) -> dict:
    """
    Train a RandomForest model for the given asset.
    Saves model + scaler to disk.
    Returns training metrics.
    """
    model_path  = os.path.join(MODELS_DIR, f'{asset}_rf.pkl')
    scaler_path = os.path.join(MODELS_DIR, f'{asset}_scaler.pkl')

    if os.path.exists(model_path) and not force:
        return {'status': 'already_trained', 'asset': asset}

    df = get_asset_df(asset)
    data = _build_features(df)

    X = data[FEATURE_COLS].values
    y = data['target'].values

    # 80/20 split (time-ordered — no shuffle)
    split = int(len(X) * 0.8)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    # Scale features
    scaler = MinMaxScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s  = scaler.transform(X_test)

    # Train
    model = RandomForestRegressor(
        n_estimators=200,
        max_depth=10,
        min_samples_split=5,
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train_s, y_train)

    # Evaluate
    preds = model.predict(X_test_s)
    mape  = mean_absolute_percentage_error(y_test, preds) * 100

    # Save
    with open(model_path,  'wb') as f: pickle.dump(model,  f)
    with open(scaler_path, 'wb') as f: pickle.dump(scaler, f)

    return {
        'status':       'trained',
        'asset':        asset,
        'train_rows':   split,
        'test_rows':    len(X_test),
        'mape_pct':     round(mape, 2),
        'accuracy_pct': round(100 - mape, 2)
    }


def train_all(force: bool = False) -> list:
    """Train models for every asset. Call once on first deploy."""
    return [train(asset, force=force) for asset in ASSETS]


# ─── Predict ─────────────────────────────────────────────────────────────────

def _load_model(asset: str):
    model_path  = os.path.join(MODELS_DIR, f'{asset}_rf.pkl')
    scaler_path = os.path.join(MODELS_DIR, f'{asset}_scaler.pkl')

    if not os.path.exists(model_path):
        train(asset)

    with open(model_path,  'rb') as f: model  = pickle.load(f)
    with open(scaler_path, 'rb') as f: scaler = pickle.load(f)
    return model, scaler


def predict_next(asset: str) -> dict:
    """
    Predict the next trading day's price for the given asset.
    Returns predicted price + confidence band (±1 std of tree predictions).
    """
    model, scaler = _load_model(asset)

    df   = get_asset_df(asset)
    data = _build_features(df)

    # Use the last available row as input
    last_row  = data[FEATURE_COLS].iloc[-1].values.reshape(1, -1)
    last_row_s = scaler.transform(last_row)

    # Individual tree predictions for uncertainty estimate
    tree_preds = np.array([t.predict(last_row_s)[0] for t in model.estimators_])
    pred_price = float(np.mean(tree_preds))
    pred_std   = float(np.std(tree_preds))

    current_price = float(df['Price'].iloc[-1])
    change_pct    = (pred_price - current_price) / current_price * 100

    return {
        'asset':          asset,
        'current_price':  round(current_price, 2),
        'predicted_price': round(pred_price, 2),
        'lower_bound':    round(pred_price - pred_std, 2),
        'upper_bound':    round(pred_price + pred_std, 2),
        'change_pct':     round(change_pct, 2),
        'direction':      'UP' if change_pct > 0 else 'DOWN',
        'last_date':      str(df['Date'].iloc[-1].date())
    }


def predict_sequence(asset: str, days: int = 30) -> list:
    """
    Iteratively predict `days` future prices.
    Note: error accumulates over longer horizons — treat as trend, not exact.
    """
    model, scaler = _load_model(asset)

    df   = get_asset_df(asset)
    data = _build_features(df)

    # Start from last known feature row
    last_features = data[FEATURE_COLS].iloc[-1].values.copy()
    last_price    = float(df['Price'].iloc[-1])
    last_date     = df['Date'].iloc[-1]

    results = []
    current_price = last_price
    features = last_features.copy()

    for i in range(1, days + 1):
        feat_scaled = scaler.transform(features.reshape(1, -1))
        next_price  = float(model.predict(feat_scaled)[0])

        future_date = last_date + pd.Timedelta(days=i)
        results.append({
            'Date':  future_date.strftime('%Y-%m-%d'),
            'Price': round(next_price, 2)
        })

        # Roll features forward: shift lags, update MA approximations
        features[0] = next_price          # lag_1
        features[1] = features[0]         # lag_2 ← old lag_1
        features[2] = features[1]         # lag_3 ← old lag_2
        features[3] = features[2]         # lag_5 approx
        features[4] = features[3]         # lag_10 approx
        current_price = next_price

    return results


if __name__ == '__main__':
    result = train('Apple', force=True)
    print(result)
    print(predict_next('Apple'))