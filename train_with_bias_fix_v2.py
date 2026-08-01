#!/usr/bin/env python3
import json, os, pickle, time, numpy as np, pandas as pd
from pathlib import Path
from datetime import datetime, timedelta
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.isotonic import IsotonicRegression
import xgboost as xgb
import lightgbm as lgb
from sklearn.neural_network import MLPClassifier

archive = Path('/root/hermes_research/rjtt_project/archive')
model_dir = Path('/root/hermes_research/rjtt_project/models')
model_dir.mkdir(exist_ok=True)

# Monthly climatology
MONTHLY_CLIMATOLOGY = {1: 8.8375, 2: 12.282142857142857, 3: 13.980645161290322, 4: 19.873333333333335, 5: 24.216129032258067, 6: 24.05666666666667, 7: 29.186956521739127, 8: 30.5, 9: 27.0, 10: 22.0, 11: 17.0, 12: 11.0}

# Load cached historical temps
hist_df = pd.read_csv('/root/hermes_research/rjtt_project/historical_temps_6months.csv')
hist_df['date'] = pd.to_datetime(hist_df['date'])
true_temps = dict(zip(hist_df['date'].dt.strftime('%Y-%m-%d'), hist_df['actual_max_temp']))

EXTENDED_INTERVALS = [{'lo': 24, 'hi': 25, 'p': 0.02}, {'lo': 25, 'hi': 26, 'p': 0.08}, {'lo': 26, 'hi': 27, 'p': 0.20}, {'lo': 27, 'hi': 28, 'p': 0.35}, {'lo': 28, 'hi': 29, 'p': 0.25}, {'lo': 29, 'hi': 30, 'p': 0.08}, {'lo': 30, 'hi': 31, 'p': 0.02}, {'lo': 31, 'hi': 32, 'p': 0.0}, {'lo': 32, 'hi': 33, 'p': 0.0}, {'lo': 33, 'hi': 34, 'p': 0.0}, {'lo': 34, 'hi': 35, 'p': 0.0}, {'lo': 35, 'hi': 36, 'p': 0.0}]
intervals = EXTENDED_INTERVALS
n_class = len(intervals)

def interval_index(obs, intervals):
    for i, iv in enumerate(intervals):
        if iv['lo'] <= obs < iv['hi']: return i
    if abs(obs - intervals[-1]['hi']) < 1e-9: return len(intervals) - 1
    if obs < intervals[0]['lo']: return 0
    return len(intervals) - 1

def parse_folder_time(folder_name):
    try: return datetime.strptime(folder_name, '%Y%m%d_%H%M%S')
    except ValueError: return None

def multiclass_brier(y_true, y_prob):
    n_samples = len(y_true); n_classes = y_prob.shape[1]
    y_true_oh = np.zeros((n_samples, n_classes))
    y_true_oh[np.arange(n_samples), y_true] = 1.0
    return np.mean(np.sum((y_prob - y_true_oh) ** 2, axis=1))

def calibrate_isotonic(model, X_train, y_train, X_cal, y_cal, n_classes):
    probs = model.predict_proba(X_cal)
    calibrators = []; calibrated_probs = np.zeros_like(probs)
    for c in range(n_classes):
        y_binary = (y_cal == c).astype(float); prob_c = probs[:, c]
        iso = IsotonicRegression(out_of_bounds='clip'); iso.fit(prob_c, y_binary)
        calibrators.append(iso); calibrated_probs[:, c] = iso.predict(prob_c)
    row_sums = calibrated_probs.sum(axis=1, keepdims=True); row_sums[row_sums == 0] = 1.0
    return calibrators, calibrated_probs / row_sums

X_list = []; y_list = []; dates_used = []
print(f'Processing archive folders with {n_class} intervals...')

for folder in sorted(archive.iterdir()):
    if not folder.is_dir(): continue
    folder_time = parse_folder_time(folder.name)
    if folder_time is None: continue
    date_str = folder_time.strftime('%Y-%m-%d')
    true_max_temp = true_temps.get(date_str)
    if true_max_temp is None:
        print(f'  Skipping {date_str}: No cached true max temp'); continue
    y_idx = interval_index(true_max_temp, intervals)
    feat_path = folder / 'osint_features.json'
    if not feat_path.exists(): continue
    feat = json.load(open(feat_path))
    hist_err = 0.0
    prev_time = folder_time - timedelta(days=1)
    prev_folder = archive / prev_time.strftime('%Y%m%d_%H%M%S')
    if prev_folder.is_dir():
        prev_feat_path = prev_folder / 'osint_features.json'
        prev_val_path = prev_folder / 'validation_log.json'
        if prev_feat_path.exists() and prev_val_path.exists():
            prev_feat = json.load(open(prev_feat_path))
            prev_val = json.load(open(prev_val_path))
            prev_obs = prev_val.get('observed_temp', 0.0)
            prev_fc = prev_feat.get('temp_forecast_mean', 0.0)
            hist_err = prev_obs - prev_fc
    fc_mean = feat.get('temp_forecast_mean', 28.0)
    month = folder_time.month
    monthly_climo = MONTHLY_CLIMATOLOGY.get(month, 20.0)
    fc_anomaly = fc_mean - monthly_climo
    feat_vec = np.array([
        fc_mean, feat.get('temp_forecast_std', 1.5), feat.get('cloud_cover_pct', 50.0),
        feat.get('precip_forecast_mm', 0.0), feat.get('wind_speed_avg', 2.0),
        feat.get('sea_breeze_flag', 0), feat.get('weekday', 0), hist_err,
        feat.get('cloud_cover_pct', 50.0) ** 2 / 10000,
        feat.get('cloud_cover_pct', 50.0) * feat.get('sea_breeze_flag', 0) / 100,
        fc_mean * feat.get('cloud_cover_pct', 50.0) / 100,
        np.sin(2 * np.pi * folder_time.timetuple().tm_yday / 365.25),
        np.cos(2 * np.pi * folder_time.timetuple().tm_yday / 365.25),
        fc_anomaly
    ], dtype=float)
    X_list.append(feat_vec); y_list.append(y_idx); dates_used.append(date_str)
    print(f'  {date_str}: true_max={true_max_temp:.1f}C -> bin {y_idx} ({intervals[y_idx]["lo"]}-{intervals[y_idx]["hi"]}C) | fc_anomaly={fc_anomaly:.2f}')

X = np.vstack(X_list); y = np.array(y_list, dtype=int)
print(f'Training samples: {len(X)}'); print(f'Features: {X.shape[1]}')
print(f'Classes present: {sorted(np.unique(y))} (out of {n_class})')
print(f'Date range: {dates_used[0]} to {dates_used[-1]}')

unique_classes = sorted(np.unique(y))
class_to_idx = {c: i for i, c in enumerate(unique_classes)}
idx_to_class = {i: c for i, c in enumerate(unique_classes)}
y_mapped = np.array([class_to_idx[c] for c in y])
n_mapped_classes = len(unique_classes)
print(f'Mapped classes: {unique_classes} -> {list(range(n_mapped_classes))}')

X_train, X_cal, y_train, y_cal = train_test_split(X, y_mapped, test_size=0.2, random_state=42, stratify=y_mapped if n_mapped_classes > 1 else None)
print(f'Train: {len(X_train)}, Calibration: {len(X_cal)}')

print('=== Training XGBoost ===')
xgb_model = xgb.XGBClassifier(objective='multi:softprob', num_class=n_mapped_classes, n_estimators=200, max_depth=5, learning_rate=0.1, subsample=0.8, colsample_bytree=0.8, random_state=42, n_jobs=-1, eval_metric='mlogloss')
xgb_model.fit(X_train, y_train)
xgb_calibrators, xgb_cal_probs = calibrate_isotonic(xgb_model, X_train, y_train, X_cal, y_cal, n_mapped_classes)
xgb_brier = multiclass_brier(y_cal, xgb_cal_probs)
print(f'XGBoost calibrated Brier: {xgb_brier:.4f}')

print('=== Training LightGBM ===')
lgb_model = lgb.LGBMClassifier(objective='multiclass', num_class=n_mapped_classes, n_estimators=200, max_depth=5, learning_rate=0.1, subsample=0.8, colsample_bytree=0.8, random_state=42, n_jobs=-1, verbosity=-1)
lgb_model.fit(X_train, y_train)
lgb_calibrators, lgb_cal_probs = calibrate_isotonic(lgb_model, X_train, y_train, X_cal, y_cal, n_mapped_classes)
lgb_brier = multiclass_brier(y_cal, lgb_cal_probs)
print(f'LightGBM calibrated Brier: {lgb_brier:.4f}')

print('=== Training MLP ===')
mlp_model = MLPClassifier(hidden_layer_sizes=(64, 32), activation='relu', solver='adam', alpha=0.001, batch_size=32, learning_rate='adaptive', max_iter=500, random_state=42, early_stopping=True, validation_fraction=0.1)
mlp_model.fit(X_train, y_train)
mlp_calibrators, mlp_cal_probs = calibrate_isotonic(mlp_model, X_train, y_train, X_cal, y_cal, n_mapped_classes)
mlp_brier = multiclass_brier(y_cal, mlp_cal_probs)
print(f'MLP calibrated Brier: {mlp_brier:.4f}')

print('=== Training Logistic Regression ===')
lr_model = LogisticRegression(solver='lbfgs', max_iter=1000, random_state=42)
lr_model.fit(X_train, y_train)
lr_calibrators, lr_cal_probs = calibrate_isotonic(lr_model, X_train, y_train, X_cal, y_cal, n_mapped_classes)
lr_brier = multiclass_brier(y_cal, lr_cal_probs)
print(f'Logistic calibrated Brier: {lr_brier:.4f}')

models = {'xgboost': (xgb_model, xgb_calibrators, xgb_brier), 'lightgbm': (lgb_model, lgb_calibrators, lgb_brier), 'mlp': (mlp_model, mlp_calibrators, mlp_brier), 'logistic': (lr_model, lr_calibrators, lr_brier)}
best_name, (best_model, best_calibrators, best_brier) = min(models.items(), key=lambda x: x[1][2])
print(f'Best model: {best_name} (Brier: {best_brier:.4f})')

feature_names = ['temp_forecast_mean', 'temp_forecast_std', 'cloud_cover_pct', 'precip_forecast_mm', 'wind_speed_avg', 'sea_breeze_flag', 'weekday', 'hist_err', 'cloud_sq', 'cloud_x_sea', 'fc_mean_x_cloud', 'seasonal_sin', 'seasonal_cos', 'fc_anomaly']

model_bundle = {'model': best_model, 'calibrators': best_calibrators, 'model_name': best_name, 'intervals': intervals, 'feature_names': feature_names, 'n_class': n_class, 'n_mapped_classes': n_mapped_classes, 'unique_classes': unique_classes, 'class_to_idx': class_to_idx, 'idx_to_class': idx_to_class, 'brier_score': best_brier, 'trained_on_dates': dates_used, 'calibration_method': 'manual_isotonic', 'monthly_climatology': MONTHLY_CLIMATOLOGY}
pickle.dump(model_bundle, open(model_dir / 'best_model.pkl', 'wb'))
print(f'Best model ({best_name}) saved to {model_dir}/best_model.pkl')

all_models = {'xgboost': (xgb_model, xgb_calibrators), 'lightgbm': (lgb_model, lgb_calibrators), 'mlp': (mlp_model, mlp_calibrators), 'logistic': (lr_model, lr_calibrators)}
for name, (model, calibrators) in all_models.items():
    pickle.dump({'model': model, 'calibrators': calibrators, 'model_name': name, 'intervals': intervals, 'feature_names': feature_names, 'n_class': n_class, 'n_mapped_classes': n_mapped_classes, 'unique_classes': unique_classes, 'class_to_idx': class_to_idx, 'idx_to_class': idx_to_class}, open(model_dir / f'{name}_calibrated.pkl', 'wb'))
print('All calibrated models saved')

if hasattr(xgb_model, 'feature_importances_'):
    print('XGBoost Feature Importances:')
    for fname, imp in zip(feature_names, xgb_model.feature_importances_): print(f'  {fname}: {imp:.4f}')
if hasattr(lgb_model, 'feature_importances_'):
    print('LightGBM Feature Importances:')
    for fname, imp in zip(feature_names, lgb_model.feature_importances_): print(f'  {fname}: {imp:.4f}')
