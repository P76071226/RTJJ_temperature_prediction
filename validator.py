#!/usr/bin/env python3
import json
import os
import sys
from datetime import datetime, timezone

def main():
    # Load market data (implied probabilities)
    market_path = os.path.join('data', 'market_raw.json')
    if not os.path.exists(market_path):
        print("Market data not found. Run market_collector first.")
        sys.exit(1)
    with open(market_path) as f:
        market_data = json.load(f)
    market_intervals = market_data.get('intervals', [])
    if not market_intervals:
        print("No intervals in market data.")
        sys.exit(1)

    # Load model probabilities
    model_path = os.path.join('data', 'model_probs.json')
    if not os.path.exists(model_path):
        print("Model probabilities not found. Run model_builder first.")
        sys.exit(1)
    with open(model_path) as f:
        model_data = json.load(f)
    model_intervals = model_data.get('intervals', [])
    if not model_intervals:
        print("No intervals in model data.")
        sys.exit(1)

    # Load OSINT features to get observed temperature (METAR temperature as proxy)
    features_path = os.path.join('data', 'osint_features.json')
    if not os.path.exists(features_path):
        print("OSINT features not found. Run osint_researcher first.")
        sys.exit(1)
    with open(features_path) as f:
        features = json.load(f)

    observed_temp = features.get('metar_temp_c')
    if observed_temp is None:
        # fallback: use forecast mean if available
        observed_temp = features.get('temp_forecast_mean', 0.0)
        print(f"Warning: METAR temperature not found, using forecast mean {observed_temp}°C as proxy.")

    # Ensure both have same intervals (lo, hi) in same order; we'll assume they are aligned
    # If lengths differ, we can match by lo,hi but for simplicity assume same.
    if len(market_intervals) != len(model_intervals):
        print("Warning: market and model intervals length mismatch")
        # We'll still proceed by matching via index up to min length
    n = min(len(market_intervals), len(model_intervals))
    diffs = []
    for i in range(n):
        mkt = market_intervals[i]['p']
        mdl = model_intervals[i]['p']
        diff = abs(mdl - mkt)
        diffs.append(diff)
    max_diff = max(diffs) if diffs else 0.0
    # Threshold for signal (as discussed, 8% to 10%)
    threshold = 0.08
    signal = max_diff > threshold
    # Build validation log
    max_idx = diffs.index(max(diffs)) if diffs else -1
    signal_interval = None
    if max_idx >= 0 and max_idx < len(market_intervals):
        signal_interval = {
            "lo": market_intervals[max_idx]['lo'],
            "hi": market_intervals[max_idx]['hi'],
            "market_p": market_intervals[max_idx]['p'],
            "model_p": model_intervals[max_idx]['p'],
            "diff": diffs[max_idx]
        }
    # Build validation log with observed temperature
    validation_log = {
        "date": market_data.get('date'),
        "observed_temp": observed_temp,  # stored ground truth (proxy)
        "max_abs_diff": max_diff,
        "threshold": threshold,
        "signal": signal,
        "signal_interval": signal_interval,
        "timestamp": datetime.now(timezone.utc).isoformat()
    }
    os.makedirs('data', exist_ok=True)
    output_path = os.path.join('data', 'validation_log.json')
    with open(output_path, 'w') as f:
        json.dump(validation_log, f, indent=2)
    print(f"Validation log saved to {output_path}")
    print(json.dumps(validation_log, indent=2))

if __name__ == '__main__':
    main()