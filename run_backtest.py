import json
import numpy as np
import pandas as pd
import joblib
from pathlib import Path

actual_temps = pd.read_csv('/root/hermes_research/rjtt_project/historical_temps_6months.csv', index_col=0, parse_dates=True)['actual_max_temp']

model_data = joblib.load('/root/hermes_research/rjtt_project/models/xgboost_calibrated.pkl')
model = model_data['model']
calibrators = model_data['calibrators']
class_indices = model_data['class_indices']
classes = model_data['classes']

archive_dir = Path('/root/hermes_research/rjtt_project/archive')
results = []

for day_dir in sorted(archive_dir.iterdir()):
    if not day_dir.is_dir():
        continue
    osint_file = day_dir / 'osint_features.json'
    validation_file = day_dir / 'validation.json'
    if not osint_file.exists() or not validation_file.exists():
        continue
    with open(osint_file) as f:
        osint = json.load(f)
    with open(validation_file) as f:
        validation = json.load(f)
    try:
        date_str = day_dir.name.split('_')[0]
        date = pd.Timestamp(date_str)
    except:
        continue
    if date not in actual_temps.index:
        continue
    actual_temp = actual_temps[date]
    intervals = validation['intervals']
    actual_interval_idx = None
    for i, interval in enumerate(intervals):
        if interval == '24-25':
            lo, hi = 24, 25
        elif interval == '35+':
            lo, hi = 35, 100
        else:
            lo, hi = map(int, interval.split('-'))
        if lo <= actual_temp < hi:
            actual_interval_idx = i
            break
    if actual_interval_idx is None:
        continue
    fc_mean = osint.get('forecast_mean', osint.get('temp_mean', 0))
    fc_std = osint.get('forecast_std', osint.get('temp_std', 0))
    cloud_mean = osint.get('cloud_cover_mean', 0)
    precip = osint.get('precipitation_sum', 0)
    wind = osint.get('wind_speed_max', 0)
    sea_breeze = 1 if osint.get('sea_breeze_flag', False) else 0
    dow = date.dayofweek
    doy = date.dayofyear
    hist_err = 0
    if len(results) > 0:
        hist_err = results[-1]['error']
    features = np.array([[
        fc_mean, fc_std, cloud_mean, precip, wind, sea_breeze, dow,
        np.sin(2*np.pi*doy/365), np.cos(2*np.pi*doy/365),
        cloud_mean**2, cloud_mean*sea_breeze, fc_mean*cloud_mean, hist_err
    ]])
    probas = model.predict_proba(features)[0]
    calibrated_probas = np.zeros_like(probas)
    for i, cls in enumerate(classes):
        cal = calibrators.get(int(cls))
        if cal is not None:
            p_clipped = np.clip(probas[i], 1e-6, 1-1e-6)
            calibrated_probas[i] = cal.predict(p_clipped.reshape(-1, 1))
        else:
            calibrated_probas[i] = probas[i]
    calibrated_probas = calibrated_probas / calibrated_probas.sum()
    market_probs = np.array(validation['market_probs'])
    market_probs = market_probs / market_probs.sum() if market_probs.sum() > 0 else np.ones_like(market_probs) / len(market_probs)
    y_true = np.zeros(len(intervals))
    y_true[actual_interval_idx] = 1
    brier = np.mean((calibrated_probas - y_true) ** 2)
    log_score = -np.log(max(calibrated_probas[actual_interval_idx], 1e-10))
    cdf_true = np.cumsum(y_true)
    cdf_pred = np.cumsum(calibrated_probas)
    rps = np.mean((cdf_pred - cdf_true) ** 2)
    model_top = np.argmax(calibrated_probas)
    market_top = np.argmax(market_probs)
    dir_correct = 1 if model_top == actual_interval_idx else 0
    market_dir_correct = 1 if market_top == actual_interval_idx else 0
    top_hit = 1 if actual_interval_idx == model_top else 0
    market_top_hit = 1 if actual_interval_idx == market_top else 0
    error = fc_mean - actual_temp
    threshold = 0.08
    edge = calibrated_probas - market_probs
    signal_intervals = np.where(edge > threshold)[0]
    pnl = 0
    if len(signal_intervals) > 0:
        for idx in signal_intervals:
            if idx == actual_interval_idx:
                pnl += (1/market_probs[idx]) - 1 if market_probs[idx] > 0 else 0
            else:
                pnl -= 1
    results.append({
        'date': date, 'actual_temp': actual_temp, 'actual_interval': intervals[actual_interval_idx],
        'fc_mean': fc_mean, 'cloud': cloud_mean, 'brier': brier, 'log_score': log_score, 'rps': rps,
        'dir_correct': dir_correct, 'market_dir_correct': market_dir_correct, 'top_hit': top_hit,
        'market_top_hit': market_top_hit, 'error': error, 'pnl': pnl,
        'model_top': intervals[model_top], 'market_top': intervals[market_top],
    })
    print(f"{date.date()} | Actual: {actual_temp:.1f}C ({intervals[actual_interval_idx]}) | FC: {fc_mean:.1f} | Brier: {brier:.4f} | Log: {log_score:.4f} | Dir: {dir_correct} | PnL: {pnl:.2f}")

df_results = pd.DataFrame(results)
print("="*80)
print("BACKTEST SUMMARY (6 months)")
print("="*80)
print(f"Total days: {len(df_results)}")
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
print("
Performance by temperature range:")
for bin_name, group in df_results.groupby('temp_bin'):
    if len(group) > 0:
        print(f"  {bin_name}: n={len(group)}, Brier={group['brier'].mean():.4f}, Dir={group['dir_correct'].mean():.2%}, PnL={group['pnl'].mean():.2f}")
df_results.to_pickle('/root/hermes_research/rjtt_project/backtest_6month_results.pkl')
print("
Results saved to backtest_6month_results.pkl")
