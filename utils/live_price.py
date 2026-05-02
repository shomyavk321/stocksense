"""
live_price.py — Fetch live/latest prices using yfinance.
Falls back to last known dataset price if live fetch fails.
"""

import yfinance as yf
from utils.data_loader import get_asset_df, ASSETS

# Map dataset asset names → Yahoo Finance tickers
TICKER_MAP = {
    'Natural_Gas':  'NG=F',
    'Crude_oil':    'CL=F',
    'Copper':       'HG=F',
    'Bitcoin':      'BTC-USD',
    'Platinum':     'PL=F',
    'Ethereum':     'ETH-USD',
    'S&P_500':      '^GSPC',
    'Nasdaq_100':   '^NDX',
    'Apple':        'AAPL',
    'Tesla':        'TSLA',
    'Microsoft':    'MSFT',
    'Silver':       'SI=F',
    'Google':       'GOOGL',
    'Nvidia':       'NVDA',
    'Berkshire':    'BRK-A',
    'Netflix':      'NFLX',
    'Amazon':       'AMZN',
    'Meta':         'META',
    'Gold':         'GC=F',
}


def get_live_price(asset: str) -> dict:
    """
    Fetch latest price for one asset from Yahoo Finance.
    Returns dict with price, change, change_pct, and source.
    """
    ticker_sym = TICKER_MAP.get(asset)
    if not ticker_sym:
        raise ValueError(f"Unknown asset: {asset}")

    try:
        ticker = yf.Ticker(ticker_sym)
        info   = ticker.fast_info

        current  = round(float(info.last_price), 2)
        prev_close = round(float(info.previous_close), 2)
        change   = round(current - prev_close, 2)
        change_pct = round((change / prev_close) * 100, 2) if prev_close else 0.0

        return {
            'asset':       asset,
            'ticker':      ticker_sym,
            'price':       current,
            'prev_close':  prev_close,
            'change':      change,
            'change_pct':  change_pct,
            'direction':   'UP' if change >= 0 else 'DOWN',
            'source':      'live'
        }

    except Exception as e:
        # Fallback to last dataset price
        df = get_asset_df(asset)
        last_price = round(float(df['Price'].iloc[-1]), 2)
        return {
            'asset':      asset,
            'ticker':     ticker_sym,
            'price':      last_price,
            'prev_close': None,
            'change':     None,
            'change_pct': None,
            'direction':  None,
            'source':     'dataset_fallback',
            'error':      str(e)
        }


def get_all_live_prices() -> list:
    """Fetch live prices for all assets."""
    return [get_live_price(asset) for asset in ASSETS]


def get_live_ticker_bar() -> list:
    """
    Lightweight version for a scrolling ticker bar.
    Returns: [{ asset, price, change_pct, direction }, ...]
    """
    results = []
    for asset in ASSETS:
        data = get_live_price(asset)
        results.append({
            'asset':      asset,
            'price':      data['price'],
            'change_pct': data['change_pct'],
            'direction':  data['direction'],
        })
    return results


if __name__ == '__main__':
    print(get_live_price('Apple'))
    print(get_live_price('Bitcoin'))