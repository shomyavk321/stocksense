import pandas as pd
import numpy as np
import os

DATA_PATH = os.path.join(os.path.dirname(__file__), '..', 'data', 'Stock_Market_Dataset.csv')

# All assets in the dataset
ASSETS = [
    'Natural_Gas', 'Crude_oil', 'Copper', 'Bitcoin', 'Platinum',
    'Ethereum', 'S&P_500', 'Nasdaq_100', 'Apple', 'Tesla',
    'Microsoft', 'Silver', 'Google', 'Nvidia', 'Berkshire',
    'Netflix', 'Amazon', 'Meta', 'Gold'
]

# Assets whose price columns are stored as strings with commas
STRING_PRICE_COLS = [
    'Bitcoin_Price', 'Ethereum_Price', 'S&P_500_Price',
    'Nasdaq_100_Price', 'Berkshire_Price', 'Gold_Price'
]


def _clean_price(series: pd.Series) -> pd.Series:
    """Strip commas and convert to float."""
    return series.astype(str).str.replace(',', '', regex=False).astype(float)


def load_raw() -> pd.DataFrame:
    """Load CSV and return a clean DataFrame."""
    df = pd.read_csv(DATA_PATH)

    # Drop unnamed index column
    df.drop(columns=[c for c in df.columns if 'Unnamed' in c], inplace=True)

    # Parse dates
    df['Date'] = pd.to_datetime(df['Date'], dayfirst=True)
    df.sort_values('Date', inplace=True)
    df.reset_index(drop=True, inplace=True)

    # Fix comma-formatted price columns
    for col in STRING_PRICE_COLS:
        if col in df.columns:
            df[col] = _clean_price(df[col])

    # Fill missing volume values with forward fill, then backward fill
    vol_cols = [c for c in df.columns if 'Vol' in c]
    df[vol_cols] = df[vol_cols].ffill().bfill()

    return df


def get_asset_df(asset: str) -> pd.DataFrame:
    """
    Return a clean DataFrame for a single asset with columns:
    Date, Price, Volume
    """
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
        asset_df['Volume'] = np.nan

    asset_df.dropna(subset=['Price'], inplace=True)
    asset_df.reset_index(drop=True, inplace=True)

    return asset_df


def get_all_prices() -> pd.DataFrame:
    """Return a wide DataFrame with Date + all Price columns only."""
    df = load_raw()
    price_cols = ['Date'] + [c for c in df.columns if c.endswith('_Price')]
    return df[price_cols].copy()


def get_summary() -> dict:
    """
    Quick summary of the dataset:
    - date range
    - number of assets
    - row count
    - null counts per price column
    """
    df = load_raw()
    price_cols = [c for c in df.columns if c.endswith('_Price')]

    return {
        'start_date': str(df['Date'].min().date()),
        'end_date':   str(df['Date'].max().date()),
        'total_rows': len(df),
        'assets':     ASSETS,
        'asset_count': len(ASSETS),
        'null_prices': {
            col: int(df[col].isna().sum())
            for col in price_cols
            if df[col].isna().sum() > 0
        }
    }


if __name__ == '__main__':
    print(get_summary())
    print()
    print(get_asset_df('Apple').tail())