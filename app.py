"""
app.py — StockSense Flask Backend
Run: python app.py
"""

from flask import Flask, jsonify, request
from flask_cors import CORS

from utils.data_loader import ASSETS, get_summary
from utils.eda import (
    price_history, moving_averages, rsi, macd,
    bollinger_bands, volatility, daily_returns,
    correlation_matrix, asset_stats, volume_history,
    normalised_comparison
)
from utils.prediction import predict_next, predict_sequence, train_all
from utils.live_price import get_live_price, get_live_ticker_bar

app = Flask(__name__)
CORS(app)  # Allow frontend (React/HTML) to call this API


# ─── Utility ─────────────────────────────────────────────────────────────────

def error(msg: str, code: int = 400):
    return jsonify({'error': msg}), code


def validate_asset(asset: str):
    if asset not in ASSETS:
        return error(f"Unknown asset '{asset}'. Valid: {ASSETS}", 404)
    return None


# ─── General ─────────────────────────────────────────────────────────────────

@app.route('/')
def index():
    return jsonify({'message': 'StockSense API running ✓', 'version': '1.0'})


@app.route('/api/assets')
def list_assets():
    """List all available assets."""
    return jsonify({'assets': ASSETS})


@app.route('/api/summary')
def dataset_summary():
    """Dataset overview: date range, row count, assets."""
    return jsonify(get_summary())


# ─── Live Prices ─────────────────────────────────────────────────────────────

@app.route('/api/live/ticker')
def live_ticker():
    """Lightweight data for the top scrolling ticker bar."""
    return jsonify(get_live_ticker_bar())


@app.route('/api/live/<asset>')
def live_price(asset):
    err = validate_asset(asset)
    if err: return err
    return jsonify(get_live_price(asset))


# ─── EDA ─────────────────────────────────────────────────────────────────────

@app.route('/api/eda/<asset>/stats')
def eda_stats(asset):
    err = validate_asset(asset)
    if err: return err
    return jsonify(asset_stats(asset))


@app.route('/api/eda/<asset>/price')
def eda_price(asset):
    err = validate_asset(asset)
    if err: return err
    period = request.args.get('period', 'all')
    return jsonify(price_history(asset, period))


@app.route('/api/eda/<asset>/ma')
def eda_ma(asset):
    err = validate_asset(asset)
    if err: return err
    windows = request.args.get('windows', '20,50,200')
    w = [int(x) for x in windows.split(',')]
    return jsonify(moving_averages(asset, w))


@app.route('/api/eda/<asset>/rsi')
def eda_rsi(asset):
    err = validate_asset(asset)
    if err: return err
    return jsonify(rsi(asset))


@app.route('/api/eda/<asset>/macd')
def eda_macd(asset):
    err = validate_asset(asset)
    if err: return err
    return jsonify(macd(asset))


@app.route('/api/eda/<asset>/bollinger')
def eda_bollinger(asset):
    err = validate_asset(asset)
    if err: return err
    return jsonify(bollinger_bands(asset))


@app.route('/api/eda/<asset>/volatility')
def eda_volatility(asset):
    err = validate_asset(asset)
    if err: return err
    return jsonify(volatility(asset))


@app.route('/api/eda/<asset>/returns')
def eda_returns(asset):
    err = validate_asset(asset)
    if err: return err
    return jsonify(daily_returns(asset))


@app.route('/api/eda/<asset>/volume')
def eda_volume(asset):
    err = validate_asset(asset)
    if err: return err
    return jsonify(volume_history(asset))


@app.route('/api/eda/correlation')
def eda_correlation():
    return jsonify(correlation_matrix())


# ─── Comparison ──────────────────────────────────────────────────────────────

@app.route('/api/compare')
def compare():
    """
    Compare multiple assets normalised to 100.
    Query: ?tickers=Apple,Tesla,Nvidia
    """
    tickers_param = request.args.get('tickers', '')
    assets = [t.strip() for t in tickers_param.split(',') if t.strip()]

    if not assets:
        return error("Provide ?tickers=Asset1,Asset2")

    invalid = [a for a in assets if a not in ASSETS]
    if invalid:
        return error(f"Unknown assets: {invalid}")

    return jsonify(normalised_comparison(assets))


# ─── Prediction ──────────────────────────────────────────────────────────────

@app.route('/api/predict/<asset>')
def predict(asset):
    """Predict next trading day's price."""
    err = validate_asset(asset)
    if err: return err
    return jsonify(predict_next(asset))


@app.route('/api/predict/<asset>/sequence')
def predict_seq(asset):
    """
    Predict next N days.
    Query: ?days=30
    """
    err = validate_asset(asset)
    if err: return err
    days = int(request.args.get('days', 30))
    days = min(days, 90)  # cap at 90
    return jsonify(predict_sequence(asset, days))


# ─── Admin ───────────────────────────────────────────────────────────────────

@app.route('/api/admin/train-all', methods=['POST'])
def admin_train_all():
    """Trigger training for all assets. Call once after deploy."""
    force = request.json.get('force', False) if request.json else False
    results = train_all(force=force)
    return jsonify(results)


# ─── Run ─────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    app.run(debug=True, port=5000)