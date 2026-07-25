#!/usr/bin/env python3
import json
import os
import sys
import pickle
import numpy as np
from pathlib import Path

def load_pickle(path):
    if os.path.exists(path):
        try:
            with open(path, 'rb') as f:
                return pickle.load(f)
        except Exception as e:
            print(f"[WARN] Could not load pickle {path}: {e}")
    return None

def apply_calibration(model, X, calibrators, n_classes):
    probs = model.predict_proba(X)
    calibrated_probs = np.zeros_like(probs)
    for c in range(n_classes):
        calibrated_probs[:, c] = calibrators[c].predict(probs[:, c])
    row_sums = calibrated_probs.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1.0
    calibrated_probs = calibrated_probs / row_sums
    return calibrated_probs

def main():
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

    # Load features
    feat_path = os.path.join('data', 'osint_features.json')
    if not os.path.exists(feat_path):
        print("OSINT features not found. Run osint_researcher first.")
        return
    with open(feat_path) as f:
        feat = json.load(f)

    # Build feature vector (13 features, same as training)
    feat_vec = np.array([
        feat.get('temp_forecast_mean', 28.0),
        feat.get('temp_forecast_std', 1.5),
        feat.get('cloud_cover_pct', 50.0),
        feat.get('precip_forecast_mm', 0.0),
        feat.get('wind_speed_avg', 2.0),
        feat.get('sea_breeze_flag', 0),
        feat.get('weekday', 0),
        0.0,  # hist_err - not available at prediction time
        feat.get('cloud_cover_pct', 50.0) ** 2 / 10000.0,  # cloud_sq
        feat.get('cloud_cover_pct', 50.0) * feat.get('sea_breeze_flag', 0) / 100.0,  # cloud_x_sea
        feat.get('temp_forecast_mean', 28.0) * feat.get('cloud_cover_pct', 50.0) / 100.0,  # fc_mean_x_cloud
        0.0,  # seasonal_sin - would need date
        0.0,  # seasonal_cos
    ], dtype=float).reshape(1, -1)

    # Try to load best calibrated model
    model_dir = Path(os.path.dirname(__file__)) / 'models'
    model_bundle = load_pickle(model_dir / 'best_model.pkl')

    if model_bundle is not None:
        model = model_bundle['model']
        calibrators = model_bundle['calibrators']
        unique_classes = model_bundle.get('unique_classes', [])
        idx_to_class = model_bundle.get('idx_to_class', {})
        n_mapped_classes = len(calibrators)

        print(f"[INFO] Loaded {model_bundle.get('model_name', 'unknown')} model with {len(calibrators)} calibrators")

        # Get calibrated probabilities
        probs_mapped = apply_calibration(model, feat_vec, calibrators, n_mapped_classes)

        # Expand to full interval space (12 intervals)
        probs_full = np.zeros((1, len(market_intervals)))
        for mapped_idx, orig_idx in idx_to_class.items():
            if orig_idx < len(market_intervals):
                probs_full[0, orig_idx] = probs_mapped[0, mapped_idx]

        # Normalize
        total = probs_full.sum()
        if total > 0:
            probs_full = probs_full / total

    else:
        # Fallback: simple heuristic (normal distribution)
        print("[WARN] No trained model found, using heuristic fallback")
        import math
        def norm_cdf(x, mu=0.0, sigma=1.0):
            return 0.5 * (1 + math.erf((x - mu) / (sigma * math.sqrt(2))))
        mu = feat.get('temp_forecast_mean', 28.0)
        sigma = feat.get('temp_forecast_std', 1.5)
        if sigma <= 0:
            sigma = 1.0
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
            probs = [1.0 / len(market_intervals)] * len(market_intervals)
        probs_full = np.array([probs])

    # Build output JSON
    out_intervals = []
    for iv, p in zip(market_intervals, probs_full[0]):
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
    print(f"Model probabilities saved to {out_path}")
    print(json.dumps(out_data, indent=2))

if __name__ == '__main__':
    main()