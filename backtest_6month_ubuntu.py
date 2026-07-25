import json
import numpy as np
import pandas as pd
import joblib
from pathlib import Path

# Load historical actual temperatures
actual_temps = pd.read_csv('/root/hermes_research/rjtt_project/historical_temps_6months.csv', index_col=0, parse_dates=True)['actual_max_temp']

# Load trained model
model_data = joblib.load('/root/hermes_research/rjtt_project/models/xgboost_calibrated.pkl')
model = model_data['model']
calibrators = model_data['calibrators']
classes = model_data['unique_classes']
intervals = model_data['intervals']  # list of {'lo': x, 'hi': y}

# Load archive data
archive_dir = Path('/root/hermes_research/rjtt_project/archive')
results = []

for day_dir in sorted(archive_dir.iterdir()):
    if not day_dir.is_dir():
        continue
    
    market_file = day_dir / 'market_raw.json'
    osint_file = day_dir / 'osint_features.json'
    model_file = day_dir / 'model_probs.json'
    
    if not market_file.exists() or not osint_file.exists() or not model_file.exists():
        continue
    
    with open(market_file) as f:
        market_data = json.load(f)
    
    with open(osint_file) as f:
        osint = json.load(f)
    
    with open(model_file) as f:
        model_data_json = json.load(f)
    
    try:
        date_str = day_dir.name.split('_')[0]
        date = pd.Timestamp(date_str)
    except:
        continue
    
    if date not in actual_temps.index:
        continue
    actual_temp = actual_temps[date]
    
    # Market intervals and probs
    market_intervals = market_data['intervals']
    market_probs = [i['p'] for i in market_intervals]
    
    # Model intervals and probs
    model_intervals = model_data_json['intervals']
    model_probs = [i['p'] for i in model_intervals]
    
    # Find actual interval
    actual_interval_idx = None
    for i, interval in enumerate(market_intervals):
        lo, hi = interval['lo'], interval['hi']
        if lo <= actual_temp < hi:
            actual_interval_idx = i
            break
    
    if actual_interval_idx is None:
        continue
    
    # Build features
    fc_mean = osint.get('temp_forecast_mean', osint.get('temp_mean', 0))
    fc_std = osint.get('temp_forecast_std', osint.get('temp_std', 0))
    cloud_mean = osint.get('cloud_cover_pct', 0)
    precip = osint.get('precip_forecast_mm', 0)
    wind = osint.get('wind_speed_avg', 0)
    sea_breeze = osint.get('sea_breeze_flag', 0)
    dow = date.dayofweek
    doy = date.dayofyear
    
    hist_err = results[-1]['error'] if results else 0
    
    features = np.array([[
        fc_mean, fc_std, cloud_mean, precip, wind, sea_breeze, dow,
        np.sin(2*np.pi*doy/365), np.cos(2*np.pi*doy/365),
        cloud_mean**2, cloud_mean*sea_breeze, fc_mean*cloud_mean, hist_err
    ]])
    
    # Use stored model probabilities
    calibrated_probas = np.array(model_probs)
    calibrated_probas = calibrated_probas / calibrated_probas.sum()
    
    market_probs_arr = np.array(market_probs)
    market_probs_arr = market_probs_arr / market_probs_arr.sum() if market_probs_arr.sum() > 0 else np.ones_like(market_probs_arr) / len(market_probs_arr)
    
    # Metrics
    y_true = np.zeros(len(market_intervals))
    y_true[actual_interval_idx] = 1
    brier = np.mean((calibrated_probas - y_true) ** 2)
    log_score = -np.log(max(calibrated_probas[actual_interval_idx], 1e-10))
    cdf_true = np.cumsum(y_true)
    cdf_pred = np.cumsum(calibrated_probas)
    rps = np.mean((cdf_pred - cdf_true) ** 2)
    
    model_top = np.argmax(calibrated_probas)
    market_top = np.argmax(market_probs_arr)
    dir_correct = 1 if model_top == actual_interval_idx else 0
    market_dir_correct = 1 if market_top == actual_interval_idx else 0
    top_hit = 1 if actual_interval_idx == model_top else 0
    market_top_hit = 1 if actual_interval_idx == market_top else 0
    
    error = fc_mean - actual_temp
    
    threshold = 0.08
    edge = calibrated_probas - market_probs_arr
    signal_intervals = np.where(edge > threshold)[0]
    
    pnl = 0
    if len(signal_intervals) > 0:
        for idx in signal_intervals:
            if idx == actual_interval_idx:
                pnl += (1/market_probs_arr[idx]) - 1 if market_probs_arr[idx] > 0 else 0
            else:
                pnl -= 1
    
    results.append({
        'date': date, 'actual_temp': actual_temp, 'actual_interval': f"{market_intervals[actual_interval_idx]['lo']}-{market_intervals[actual_interval_idx]['hi']}",
        'fc_mean': fc_mean, 'cloud': cloud_mean, 'brier': brier, 'log_score': log_score, 'rps': rps,
        'dir_correct': dir_correct, 'market_dir_correct': market_dir_correct, 'top_hit': top_hit,
        'market_top_hit': market_top_hit, 'error': error, 'pnl': pnl,
        'model_top': f"{model_intervals[model_top]['lo']}-{model_intervals[model_top]['hi']}",
        'market_top': f"{market_intervals[market_top]['lo']}-{market_intervals[market_top]['hi']}",
    })
    
    print(f"{date.date()} | Actual: {actual_temp:.1f}C ({market_intervals[actual_interval_idx]['lo']}-{market_intervals[actual_interval_idx]['hi']}) | FC: {fc_mean:.1f} | Brier: {brier:.4f} | Log: {log_score:.4f} | Dir: {dir_correct} | PnL: {pnl:.2f}")

df_results = pd.DataFrame(results)
print("="*80)
print("BACKTEST SUMMARY")
print("="*80)
print(f"Total days: {len(df_results)}")
if len(df_results) > 0:
    print(f"Mean Brier Score: {df_results['brier'].mean():.4f}")
    print(f"Mean Log Score: {df_results['log_score'].mean():.4f}")
    print(f"Mean RPS: {df_results['rps'].mean():.4f}")
    print(f"Directional Accuracy: {df_results['dir_correct'].mean():.2%}")
    print(f"Market Directional Accuracy: {df_results['market_dir_correct'].mean():.2%}")
    print(f"Top Interval Hit Rate: {df_results['top_hit'].mean():.2%}")
    print(f"Market Top Interval Hit Rate: {df_results['market_top_hit'].mean():.2%}")
    print(f"Mean Forecast Error (FC - Actual): {df_results['error'].mean():.2f}C")
    print(f"MAE: {df_results['error'].abs().mean():.2f}C")
    print(f"Total PnL: {df_results['pnl'].sum():.2f}")
    print(f"Mean PnL/day: {df_results['pnl'].mean():.2f}")
    print(f"Win days: {(df_results['pnl'] > 0).sum()} / {len(df_results)}")
    
    df_results['temp_bin'] = pd.cut(df_results['actual_temp'], bins=[0,15,20,25,30,35,40])
    print("\nPerformance by temperature range:")
    for bin_name, group in df_results.groupby('temp_bin'):
        if len(group) > 0:
            print(f"  {bin_name}: n={len(group)}, Brier={group['brier'].mean():.4f}, Dir={group['dir_correct'].mean():.2%}, PnL={group['pnl'].mean():.2f}")

df_results.to_pickle('/root/hermes_research/rjtt_project/backtest_6month_results.pkl')
print("\nResults saved to backtest_6month_results.pkl")
