"""
eda.py — Exploratory Data Analysis & Technical Indicators
All functions return plain dicts/lists (JSON-serialisable) so the
Flask backend can pass them directly to the frontend.
"""

import pandas as pd
import numpy as np
from utils.data_loader import get_asset_df, get_all_prices, ASSETS


# ─── Helpers ────────────────────────────────────────────────────────────────

def _to_records(df: pd.DataFrame) -> list:
    """Convert DataFrame to list of dicts with ISO date strings."""
    d = df.copy()
    if 'Date' in d.columns:
        d['Date'] = d['Date'].dt.strftime('%Y-%m-%d')
    return d.to_dict(orient='records')


# ─── 1. Price History ────────────────────────────────────────────────────────

def price_history(asset: str, period: str = 'all') -> list:
    """
    Return Date + Price for charting.
    period: 'all' | '1y' | '6m' | '3m' | '1m'
    """
    df = get_asset_df(asset)

    cutoffs = {'1y': 365, '6m': 182, '3m': 91, '1m': 30}
    if period in cutoffs:
        cutoff = df['Date'].max() - pd.Timedelta(days=cutoffs[period])
        df = df[df['Date'] >= cutoff]

    return _to_records(df[['Date', 'Price']])


# ─── 2. Moving Averages ──────────────────────────────────────────────────────

def moving_averages(asset: str, windows: list = [20, 50, 200]) -> list:
    """
    Return Date, Price, MA_20, MA_50, MA_200 (only those in windows).
    """
    df = get_asset_df(asset)[['Date', 'Price']].copy()

    for w in windows:
        df[f'MA_{w}'] = df['Price'].rolling(window=w).mean().round(2)

    return _to_records(df)


# ─── 3. RSI ─────────────────────────────────────────────────────────────────

def rsi(asset: str, period: int = 14) -> list:
    """Relative Strength Index."""
    df = get_asset_df(asset)[['Date', 'Price']].copy()

    delta = df['Price'].diff()
    gain  = delta.clip(lower=0)
    loss  = -delta.clip(upper=0)

    avg_gain = gain.ewm(com=period - 1, min_periods=period).mean()
    avg_loss = loss.ewm(com=period - 1, min_periods=period).mean()

    rs = avg_gain / avg_loss.replace(0, np.nan)
    df['RSI'] = (100 - 100 / (1 + rs)).round(2)

    return _to_records(df[['Date', 'RSI']].dropna())


# ─── 4. MACD ─────────────────────────────────────────────────────────────────

def macd(asset: str, fast: int = 12, slow: int = 26, signal: int = 9) -> list:
    """MACD line, Signal line, Histogram."""
    df = get_asset_df(asset)[['Date', 'Price']].copy()

    ema_fast   = df['Price'].ewm(span=fast, adjust=False).mean()
    ema_slow   = df['Price'].ewm(span=slow, adjust=False).mean()
    df['MACD'] = (ema_fast - ema_slow).round(4)
    df['Signal'] = df['MACD'].ewm(span=signal, adjust=False).mean().round(4)
    df['Histogram'] = (df['MACD'] - df['Signal']).round(4)

    return _to_records(df[['Date', 'MACD', 'Signal', 'Histogram']].dropna())


# ─── 5. Bollinger Bands ──────────────────────────────────────────────────────

def bollinger_bands(asset: str, window: int = 20, num_std: float = 2.0) -> list:
    """Upper band, Middle (SMA), Lower band."""
    df = get_asset_df(asset)[['Date', 'Price']].copy()

    sma = df['Price'].rolling(window).mean()
    std = df['Price'].rolling(window).std()

    df['Upper']  = (sma + num_std * std).round(2)
    df['Middle'] = sma.round(2)
    df['Lower']  = (sma - num_std * std).round(2)

    return _to_records(df[['Date', 'Price', 'Upper', 'Middle', 'Lower']].dropna())


# ─── 6. Volatility ───────────────────────────────────────────────────────────

def volatility(asset: str, window: int = 30) -> list:
    """Rolling annualised volatility (%)."""
    df = get_asset_df(asset)[['Date', 'Price']].copy()
    daily_ret = df['Price'].pct_change()
    df['Volatility'] = (daily_ret.rolling(window).std() * np.sqrt(252) * 100).round(2)
    return _to_records(df[['Date', 'Volatility']].dropna())


# ─── 7. Daily Returns ────────────────────────────────────────────────────────

def daily_returns(asset: str) -> list:
    df = get_asset_df(asset)[['Date', 'Price']].copy()
    df['Return'] = df['Price'].pct_change().mul(100).round(4)
    return _to_records(df[['Date', 'Return']].dropna())


# ─── 8. Correlation Matrix ───────────────────────────────────────────────────

def correlation_matrix() -> dict:
    """
    Return a correlation matrix of all asset prices.
    Result: { 'labels': [...], 'matrix': [[...], ...] }
    """
    df = get_all_prices().drop(columns=['Date'])
    # Rename columns: 'Apple_Price' → 'Apple'
    df.columns = [c.replace('_Price', '') for c in df.columns]
    corr = df.corr().round(3)
    return {
        'labels': corr.columns.tolist(),
        'matrix': corr.values.tolist()
    }


# ─── 9. Asset Statistics ─────────────────────────────────────────────────────

def asset_stats(asset: str) -> dict:
    """
    Summary stats for one asset.
    """
    df = get_asset_df(asset)
    price = df['Price']
    daily_ret = price.pct_change().dropna()

    return {
        'asset':          asset,
        'start_date':     str(df['Date'].min().date()),
        'end_date':       str(df['Date'].max().date()),
        'current_price':  round(float(price.iloc[-1]), 2),
        'all_time_high':  round(float(price.max()), 2),
        'all_time_low':   round(float(price.min()), 2),
        'mean_price':     round(float(price.mean()), 2),
        'std_price':      round(float(price.std()), 2),
        'total_return_pct': round(
            (price.iloc[-1] - price.iloc[0]) / price.iloc[0] * 100, 2
        ),
        'avg_daily_return_pct': round(float(daily_ret.mean() * 100), 4),
        'annualised_volatility_pct': round(
            float(daily_ret.std() * np.sqrt(252) * 100), 2
        ),
        'sharpe_ratio':   round(
            float((daily_ret.mean() / daily_ret.std()) * np.sqrt(252)), 3
        ) if daily_ret.std() > 0 else None,
    }


# ─── 10. Volume Analysis ─────────────────────────────────────────────────────

def volume_history(asset: str) -> list:
    df = get_asset_df(asset)[['Date', 'Volume']].dropna()
    return _to_records(df)


# ─── 11. Multi-asset Comparison ──────────────────────────────────────────────

def normalised_comparison(assets: list) -> list:
    """
    Normalise all assets to 100 at start date for fair comparison.
    Returns list of { Date, Asset1, Asset2, ... }
    """
    frames = {}
    for asset in assets:
        df = get_asset_df(asset)[['Date', 'Price']].set_index('Date')
        first = df['Price'].dropna().iloc[0]
        frames[asset] = (df['Price'] / first * 100).round(2)

    combined = pd.DataFrame(frames).reset_index()
    combined['Date'] = combined['Date'].dt.strftime('%Y-%m-%d')
    return combined.to_dict(orient='records')


if __name__ == '__main__':
    print(asset_stats('Apple'))
    print()
    r = rsi('Bitcoin')
    print('RSI (last 3):', r[-3:])