import pandas as pd
import numpy as np
import os

DATA_PATH = os.path.join(os.path.dirname(__file__), '..', 'data', 'Stock_Market_Dataset.csv')

ASSETS = [
    'Natural_Gas', 'Crude_oil', 'Copper', 'Bitcoin', 'Platinum',
    'Ethereum', 'S&P_500', 'Nasdaq_100', 'Apple', 'Tesla',
    'Microsoft', 'Silver', 'Google', 'Nvidia', 'Berkshire',
    'Netflix', 'Amazon', 'Meta', 'Gold'
]

STRING_PRICE_COLS = [
    'Bitcoin_Price', 'Ethereum_Price', 'S&P_500_Price',
    'Nasdaq_100_Price', 'Berkshire_Price', 'Gold_Price'
]


def _clean_price(series: pd.Series) -> pd.Series:
    return series.astype(str).str.replace(',', '', regex=False).astype(float)


def load_raw() -> pd.DataFrame:
    df = pd.read_csv(DATA_PATH)

    df.drop(columns=[c for c in df.columns if 'Unnamed' in c], inplace=True)

    df['Date'] = pd.to_datetime(df['Date'], dayfirst=True)
    df.sort_values('Date', inplace=True)
    df.reset_index(drop=True, inplace=True)

    for col in STRING_PRICE_COLS:
        if col in df.columns:
            df[col] = _clean_price(df[col])

    vol_cols = [c for c in df.columns if 'Vol' in c]
    df[vol_cols] = df[vol_cols].ffill().bfill()

    # ✅ CRITICAL FIX: Replace NaN with None (for valid JSON)
    df = df.replace({np.nan: None})

    return df


def get_asset_df(asset: str) -> pd.DataFrame:
    df = load_raw()

    price_col = f'{asset}_Price'
    vol_col   = f'{asset}_Vol.'

    if price_col not in df.columns:
        raise ValueError(f"Asset '{asset}' not found. Available: {ASSETS}")

    asset_df = df[['Date', price_col]].copy()
    asset_df.rename(columns={price_col: 'Price'}, inplace=True)

    asset_df['Price'] = asset_df['Price'].astype(str).str.replace(',', '', regex=False)
    asset_df['Price'] = pd.to_numeric(asset_df['Price'], errors='coerce')

    if vol_col in df.columns:
        asset_df['Volume'] = df[vol_col].values
    else:
        asset_df['Volume'] = None

    # ✅ Remove invalid prices
    asset_df.dropna(subset=['Price'], inplace=True)

    # ✅ Add Moving Average (this was likely missing / causing NaN issues)
    asset_df['MA_20'] = asset_df['Price'].rolling(window=20).mean()

    # ✅ Convert NaN → None again AFTER rolling
    asset_df = asset_df.replace({np.nan: None})

    asset_df.reset_index(drop=True, inplace=True)

    return asset_df


def get_all_prices() -> pd.DataFrame:
    df = load_raw()
    price_cols = ['Date'] + [c for c in df.columns if c.endswith('_Price')]

    result = df[price_cols].copy()

    # ✅ Ensure JSON safe
    result = result.replace({np.nan: None})

    return result


def get_summary() -> dict:
    df = load_raw()
    price_cols = [c for c in df.columns if c.endswith('_Price')]

    return {
        'start_date': str(df['Date'].min().date()),
        'end_date':   str(df['Date'].max().date()),
        'total_rows': len(df),
        'assets':     ASSETS,
        'asset_count': len(ASSETS),
        'null_prices': {
            col: int(pd.isna(df[col]).sum())
            for col in price_cols
            if pd.isna(df[col]).sum() > 0
        }
    }


if __name__ == '__main__':
    print(get_summary())
    print()
    print(get_asset_df('Apple').tail())