#!/usr/bin/env python3
import json, os, pickle, numpy as np
from pathlib import Path

def load_pickle(path):
    if os.path.exists(path):
        try:
            with open(path, 'rb') as f:
                return pickle.load(f)
        except Exception as e:
            print(f"[WARN] Could not load pickle {path}: {e}")
    return None

def main():
    # Load features
    feat_path = os.path.join('data', 'osint_features.json')
    if not os.path.exists(feat_path):
        print("OSINT features not found. Run osint_researcher first.")
        return
    with open(feat_path) as f:
        feat = json.load(f)

    # Load market data to get intervals
    market_path = os.path.join('data', 'market_raw.json')
    if not os.path.exists(market_path):
        print("Market data not found. Run market_collector first.")
        return
    with open(market_path) as f:
        market_data = json.load(f)
    market_intervals = market_data.get('intervals', [])
    if not market_intervals:
        print("No intervals in market data.")
        return

    # Build feature vector (same as used in training)
    feat_vec = np.array([
        feat.get('temp_forecast_mean',28.0),
        feat.get('temp_forecast_std',1.5),
        feat.get('cloud_cover_pct',50.0),
        feat.get('precip_forecast_mm',0.0),
        feat.get('wind_speed_avg',2.0),
        feat.get('sea_breeze_flag',0),
        feat.get('weekday',0)
    ], dtype=float).reshape(1, -1)

    # Try to load logistic model
    model_dir = Path(os.path.dirname(__file__)) / 'models'
    model_path = model_dir / 'logistic_model.pkl'
    model = load_pickle(str(model_dir / 'logistic_model.pkl'))

    if model is not None:
        # Predict probabilities for each class
        probs = model.predict_proba(feat_vec)[0]  # shape (n_classes,)
        # Ensure length matches market_intervals
        if len(probs) != len(market_intervals):
            # If mismatch, fallback to uniform
            probs = np.full(len(market_intervals), 1.0/len(market_intervals))
    else:
        # Fallback: simple heuristic (use forecast mean as mu, forecast std as sigma, assume normal distribution)
        mu = feat.get('temp_forecast_mean',28.0)
        sigma = feat.get('temp_forecast_std',1.5)
        if sigma <= 0:
            sigma = 1.0
        import math
        def norm_cdf(x, mu=0.0, sigma=1.0):
            return 0.5 * (1 + math.erf((x - mu) / (sigma * math.sqrt(2))))
        probs = []
        for iv in market_intervals:
            p = norm_cdf(iv['hi'], mu, sigma) - norm_cdf(iv['lo'], mu, sigma)
            if p < 0:
                p = 0.0
            probs.append(p)
        total = sum(probs)
        if total > 0:
            probs = [p / total for p in probs]
        else:
            probs = [1.0/len(market_intervals)] * len(market_intervals)

    # Build output JSON
    out_intervals = []
    for iv, p in zip(market_intervals, probs):
        out_intervals.append({
            "lo": iv['lo'],
            "hi": iv['hi'],
            "p": float(p)
        })
    out_data = {
        "date": market_data.get('date'),
        "intervals": out_intervals
    }
    os.makedirs('data', exist_ok=True)
    out_path = os.path.join('data', 'model_probs.json')
    with open(out_path, 'w') as f:
        json.dump(out_data, f, indent=2)
    print(f"[INFO] Model probabilities saved to {out_path}")

if __name__ == '__main__':
    main()