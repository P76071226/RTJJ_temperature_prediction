#!/usr/bin/env python3
"""Backtest RJTT temperature prediction pipeline.

Reads each archived run (from archive/YYYYMMDD_HHMMSS/), loads the
stored files, computes the observed temperature (from validation_log.json
or from Open-Meteo historical API if not found), and calculates
probabilistic scores: Brier score, log score, Ranked Probability Score (RPS),
directional accuracy, top-interval hit rate, and a simulated P&L.

Results are printed to stdout and saved to backtest_results.csv
in the project directory.
"""

import json
import os
import csv
import subprocess
import sys
import math
import datetime

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
ARCHIVE_DIR = os.path.join(PROJECT_DIR, 'archive')
SIGNAL_THRESHOLD = 0.08  # same as validator.py
MAX_SPREAD = 0.04

def load_json(path):
    try:
        with open(path, 'r') as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return None

def compute_scores(market_intervals, model_intervals, actual_temp):
    mkt = sorted(market_intervals, key=lambda x: x['lo'])
    mdl = sorted(model_intervals, key=lambda x: x['lo'])
    n = min(len(mkt), len(mdl))
    lo = [mkt[i]['lo'] for i in range(n)]
    hi = [mkt[i]['hi'] for i in range(n)]
    p_mkt = [mkt[i]['p'] for i in range(n)]
    p_mdl = [mdl[i]['p'] for i in range(n)]

    actual_idx = None
    for i in range(n):
        if lo[i] <= actual_temp < hi[i]:
            actual_idx = i
            break
    if actual_idx is None:
        if n > 0 and abs(actual_temp - hi[-1]) < 1e-9:
            actual_idx = n - 1
        else:
            actual_idx = min(range(n), key=lambda i: abs(actual_temp - (lo[i]+hi[i])/2))

    eps = 1e-12
    p_actual = max(p_mdl[actual_idx], eps) if 0 <= actual_idx < n else eps
    log_score = -math.log(p_actual)

    brier = sum((p_mdl[i] - (1 if i == actual_idx else 0))**2 for i in range(n)) / n

    # RPS
    cum_mdl = [sum(p_mdl[:i+1]) for i in range(n)]
    cum_obs = [1 if i >= actual_idx else 0 for i in range(n)]
    rps = sum((cum_mdl[i] - cum_obs[i])**2 for i in range(n)) / n

    # Directional accuracy: compare midpoints
    def mid(i): return (lo[i] + hi[i]) / 2.0
    mkt_mid = sum(p_mkt[i] * mid(i) for i in range(n))
    mdl_mid = sum(p_mdl[i] * mid(i) for i in range(n))
    direction_correct = 1 if ((actual_temp - mkt_mid) * (mdl_mid - mkt_mid) > 0) else 0

    top_idx = p_mdl.index(max(p_mdl)) if p_mdl else -1
    top_hit = 1 if top_idx == actual_idx else 0

    # Simulated P&L
    best_edge = 0.0
    best_idx = -1
    best_side = 0
    for i in range(n):
        edge = p_mdl[i] - p_mkt[i]
        if abs(edge) > best_edge:
            best_edge = abs(edge)
            best_idx = i
            best_side = 1 if edge > 0 else -1
    took = (best_edge >= SIGNAL_THRESHOLD)
    pnl = best_edge if best_idx == actual_idx else -best_edge if took else 0.0

    return {
        'brier': brier,
        'log_score': log_score,
        'rps': rps,
        'direction_correct': direction_correct,
        'top_hit': top_hit,
        'actual_temp': actual_temp,
        'mkt_mid': mkt_mid,
        'mdl_mid': mdl_mid,
        'took_trade': took,
        'pnl': pnl,
        'edge_used': best_edge if took else 0.0,
        'trade_side': best_side,
        'signal_interval_idx': best_idx,
        'signal_interval_mid': mid(best_idx) if 0 <= best_idx < n else None
    }

def main():
    if not os.path.isdir(ARCHIVE_DIR):
        print(f"Archive directory not found: {ARCHIVE_DIR}")
        return

    results = []
    folders = sorted(os.listdir(ARCHIVE_DIR))
    for folder_name in folders:
        folder_path = os.path.join(ARCHIVE_DIR, folder_name)
        if not os.path.isdir(folder_path):
            continue
        market_data = load_json(os.path.join(folder_path, 'market_raw.json'))
        model_data = load_json(os.path.join(folder_path, 'model_probs.json'))
        validation_data = load_json(os.path.join(folder_path, 'validation_log.json'))
        if not market_data or not model_data:
            continue

        # Get observed temperature: prefer validation_log.json observed_temp
        actual_temp = None
        if validation_data and 'observed_temp' in validation_data:
            actual_temp = validation_data['observed_temp']
        else:
            # fallback: osint_features.json metar_temp_c
            features = load_json(os.path.join(folder_path, 'osint_features.json'))
            if features and 'metar_temp_c' in features:
                actual_temp = features['metar_temp_c']
        if actual_temp is None:
            print(f"Skipping {folder_name}: no observed temperature found")
            continue

        # also try to fetch via Open-Meteo historical (if requests available)
        # but we skip for simplicity; use local data only

        scores = compute_scores(
            market_data.get('intervals', []),
            model_data.get('intervals', []),
            actual_temp
        )
        scores['date'] = market_data.get('date', '?')
        results.append(scores)
        print(f"[{scores['date']}] Brier={scores['brier']:.4f}  Log={scores['log_score']:.3f}  "
              f"RPS={scores['rps']:.4f}  DirAcc={scores['direction_correct']}  "
              f"TopHit={scores['top_hit']}  Trade={scores['took_trade']}  "
              f"PnL={scores['pnl']:.3f}  actual={scores['actual_temp']}°C")

    csv_path = os.path.join(PROJECT_DIR, 'backtest_results.csv')
    if results:
        fieldnames = ['date', 'brier', 'log_score', 'rps', 'direction_correct',
                      'top_hit', 'actual_temp', 'mkt_mid', 'mdl_mid',
                      'took_trade', 'pnl', 'edge_used', 'trade_side']
        with open(csv_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for r in results:
                writer.writerow(r)
        print(f"\nResults saved to {csv_path}")
        N = len(results)
        avg_brier = sum(r['brier'] for r in results) / N
        avg_log = sum(r['log_score'] for r in results) / N
        avg_rps = sum(r['rps'] for r in results) / N
        dir_acc = sum(r['direction_correct'] for r in results) / N * 100
        top_hit = sum(r['top_hit'] for r in results) / N * 100
        trade_cnt = sum(1 for r in results if r['took_trade'])
        trade_pnl = sum(r['pnl'] for r in results if r['took_trade'])
        print(f"\n{'='*50}")
        print(f"Backtest over {N} archived runs")
        print(f"{'='*50}")
        print(f"Avg Brier score:      {avg_brier:.4f} (lower=better)")
        print(f"Avg Log score:        {avg_log:.3f} (lower=better)")
        print(f"Avg RPS:              {avg_rps:.4f} (lower=better)")
        print(f"Directional accuracy: {dir_acc:.1f}%")
        print(f"Top interval hit:     {top_hit:.1f}%")
        print(f"Trades taken:         {trade_cnt}/{N}")
        print(f"Total trade P&L:      {trade_pnl:.3f}")
        print(f"Avg trade PnL:        {trade_pnl/max(trade_cnt,1):.3f}")
    else:
        print("No valid archives found to backtest.")

if __name__ == '__main__':
    main()