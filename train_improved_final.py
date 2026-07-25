#!/usr/bin/env python3
"""
Improved training script with:
1. Real daily maximum temperature as target (from Open-Meteo historical API)
2. Extended intervals to cover actual temperature range
3. XGBoost / LightGBM / MLP models
4. Manual Isotonic Regression calibration
5. Feature engineering: hist_err, cloud_sq, cloud_x_sea, fc_mean_x_cloud, seasonal sin/cos
"""
import json
import os
import pickle
import time
import numpy as np
import requests
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

# Use extended intervals that cover the actual temperature range (up to 35°C)
EXTENDED_INTERVALS = [
    {"lo": 24, "hi": 25, "p": 0.02},
    {"lo": 25, "hi": 26, "p": 0.08},
    {"lo": 26, "hi": 27, "p": 0.20},
    {"lo": 27, "hi": 28, "p": 0.35},
    {"lo": 28, "hi": 29, "p": 0.25},
    {"lo": 29, "hi": 30, "p": 0.08},
    {"lo": 30, "hi": 31, "p": 0.02},
    {"lo": 31, "hi": 32, "p": 0.0},
    {"lo": 32, "hi": 33, "p": 0.0},
    {"lo": 33, "hi": 34, "p": 0.0},
    {"lo": 34, "hi": 35, "p": 0.0},
    {"lo": 35, "hi": 36, "p": 0.0},
]
intervals = EXTENDED_INTERVALS
n_class = len(intervals)

def interval_index(obs, intervals):
    for i, iv in enumerate(intervals):
        if iv['lo'] <= obs < iv['hi']:
            return i
    if abs(obs - intervals[-1]['hi']) < 1e-9:
        return len(intervals) - 1
    if obs < intervals[0]['lo']:
        return 0
    return len(intervals) - 1

def get_true_max_temp(date_str):
    url = (
        f"https://archive-api.open-meteo.com/v1/archive?"
        f"latitude=35.55&longitude=139.78&"
        f"start_date={date_str}&end_date={date_str}&"
        f"daily=temperature_2m_max&timezone=Asia/Tokyo"
    )
    try:
        resp = requests.get(url, timeout=10)
        if resp.status_code == 200:
            data = resp.json()
            if 'daily' in data and 'temperature_2m_max' in data['daily']:
                temps = data['daily']['temperature_2m_max']
                if temps and temps[0] is not None:
                    return float(temps[0])
    except Exception as e:
        print(f"  Error fetching true max temp for {date_str}: {e}")
    return None

def parse_folder_time(folder_name):
    try:
        return datetime.strptime(folder_name, '%Y%m%d_%H%M%S')
    except ValueError:
        return None

def multiclass_brier(y_true, y_prob):
    """Multiclass Brier score: mean squared error between one-hot true and prob"""
    n_samples = len(y_true)
    n_classes = y_prob.shape[1]
    # One-hot encode true labels
    y_true_oh = np.zeros((n_samples, n_classes))
    y_true_oh[np.arange(n_samples), y_true] = 1.0
    return np.mean(np.sum((y_prob - y_true_oh) ** 2, axis=1))

def calibrate_isotonic(model, X_train, y_train, X_cal, y_cal, n_classes):
    probs = model.predict_proba(X_cal)
    calibrators = []
    calibrated_probs = np.zeros_like(probs)
    for c in range(n_classes):
        y_binary = (y_cal == c).astype(float)
        prob_c = probs[:, c]
        iso = IsotonicRegression(out_of_bounds='clip')
        iso.fit(prob_c, y_binary)
        calibrators.append(iso)
        calibrated_probs[:, c] = iso.predict(prob_c)
    row_sums = calibrated_probs.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1.0
    calibrated_probs = calibrated_probs / row_sums
    return calibrators, calibrated_probs

def apply_calibration(model, X, calibrators, n_classes):
    probs = model.predict_proba(X)
    calibrated_probs = np.zeros_like(probs)
    for c in range(n_classes):
        calibrated_probs[:, c] = calibrators[c].predict(probs[:, c])
    row_sums = calibrated_probs.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1.0
    calibrated_probs = calibrated_probs / row_sums
    return calibrated_probs

X_list = []
y_list = []
dates_used = []

print(f"Processing archive folders with {n_class} intervals...")

for folder in sorted(archive.iterdir()):
    if not folder.is_dir():
        continue
    folder_time = parse_folder_time(folder.name)
    if folder_time is None:
        continue
    
    date_str = folder_time.strftime('%Y-%m-%d')
    true_max_temp = get_true_max_temp(date_str)
    if true_max_temp is None:
        print(f"  Skipping {date_str}: Could not fetch true max temp")
        continue
    
    y_idx = interval_index(true_max_temp, intervals)
    
    feat_path = folder / 'osint_features.json'
    if not feat_path.exists():
        continue
    feat = json.load(open(feat_path))
    
    hist_err = 0.0
    prev_time = folder_time - timedelta(days=1)
    prev_folder_name = prev_time.strftime('%Y%m%d_%H%M%S')
    prev_folder = archive / prev_folder_name
    if prev_folder.is_dir():
        prev_feat_path = prev_folder / 'osint_features.json'
        prev_val_path = prev_folder / 'validation_log.json'
        if prev_feat_path.exists() and prev_val_path.exists():
            prev_feat = json.load(open(prev_feat_path))
            prev_val = json.load(open(prev_val_path))
            prev_obs = prev_val.get('observed_temp', 0.0)
            prev_fc = prev_feat.get('temp_forecast_mean', 0.0)
            hist_err = prev_obs - prev_fc
    
    feat_vec = np.array([
        feat.get('temp_forecast_mean', 28.0),
        feat.get('temp_forecast_std', 1.5),
        feat.get('cloud_cover_pct', 50.0),
        feat.get('precip_forecast_mm', 0.0),
        feat.get('wind_speed_avg', 2.0),
        feat.get('sea_breeze_flag', 0),
        feat.get('weekday', 0),
        hist_err,
        feat.get('cloud_cover_pct', 50.0) ** 2 / 10000,
        feat.get('cloud_cover_pct', 50.0) * feat.get('sea_breeze_flag', 0) / 100,
        feat.get('temp_forecast_mean', 28.0) * feat.get('cloud_cover_pct', 50.0) / 100,
        np.sin(2 * np.pi * folder_time.timetuple().tm_yday / 365.25),
        np.cos(2 * np.pi * folder_time.timetuple().tm_yday / 365.25),
    ], dtype=float)
    
    X_list.append(feat_vec)
    y_list.append(y_idx)
    dates_used.append(date_str)
    
    print(f"  {date_str}: true_max={true_max_temp:.1f}°C -> bin {y_idx} ({intervals[y_idx]['lo']}-{intervals[y_idx]['hi']}°C)")
    time.sleep(0.1)

X = np.vstack(X_list)
y = np.array(y_list, dtype=int)

print(f"\nTraining samples: {len(X)}")
print(f"Features: {X.shape[1]}")
print(f"Classes present: {sorted(np.unique(y))} (out of {n_class})")
print(f"Date range: {dates_used[0]} to {dates_used[-1]}")

# Remap class labels to 0-based continuous
unique_classes = sorted(np.unique(y))
class_to_idx = {c: i for i, c in enumerate(unique_classes)}
idx_to_class = {i: c for i, c in enumerate(unique_classes)}
y_mapped = np.array([class_to_idx[c] for c in y])
n_mapped_classes = len(unique_classes)

print(f"\nMapped classes: {unique_classes} -> {list(range(n_mapped_classes))}")
print(f"Number of mapped classes: {n_mapped_classes}")

X_train, X_cal, y_train, y_cal = train_test_split(X, y_mapped, test_size=0.2, random_state=42, stratify=y_mapped)
print(f"\nTrain: {len(X_train)}, Calibration: {len(X_cal)}")

# MODEL 1: XGBoost
print("\n=== Training XGBoost ===")
xgb_model = xgb.XGBClassifier(
    objective='multi:softprob',
    num_class=n_mapped_classes,
    n_estimators=200,
    max_depth=5,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    n_jobs=-1,
    eval_metric='mlogloss'
)
xgb_model.fit(X_train, y_train)

print("Calibrating XGBoost with manual Isotonic Regression...")
xgb_calibrators, xgb_cal_probs = calibrate_isotonic(xgb_model, X_train, y_train, X_cal, y_cal, n_mapped_classes)
xgb_brier = multiclass_brier(y_cal, xgb_cal_probs)
print(f"XGBoost calibrated Brier: {xgb_brier:.4f}")

# MODEL 2: LightGBM
print("\n=== Training LightGBM ===")
lgb_model = lgb.LGBMClassifier(
    objective='multiclass',
    num_class=n_mapped_classes,
    n_estimators=200,
    max_depth=5,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    n_jobs=-1,
    verbosity=-1
)
lgb_model.fit(X_train, y_train)

print("Calibrating LightGBM...")
lgb_calibrators, lgb_cal_probs = calibrate_isotonic(lgb_model, X_train, y_train, X_cal, y_cal, n_mapped_classes)
lgb_brier = multiclass_brier(y_cal, lgb_cal_probs)
print(f"LightGBM calibrated Brier: {lgb_brier:.4f}")

# MODEL 3: MLP
print("\n=== Training MLP ===")
mlp_model = MLPClassifier(
    hidden_layer_sizes=(64, 32),
    activation='relu',
    solver='adam',
    alpha=0.001,
    batch_size=32,
    learning_rate='adaptive',
    max_iter=500,
    random_state=42,
    early_stopping=True,
    validation_fraction=0.1
)
mlp_model.fit(X_train, y_train)

print("Calibrating MLP...")
mlp_calibrators, mlp_cal_probs = calibrate_isotonic(mlp_model, X_train, y_train, X_cal, y_cal, n_mapped_classes)
mlp_brier = multiclass_brier(y_cal, mlp_cal_probs)
print(f"MLP calibrated Brier: {mlp_brier:.4f}")

# MODEL 4: Logistic Regression
print("\n=== Training Logistic Regression ===")
lr_model = LogisticRegression(solver='lbfgs', max_iter=1000, random_state=42)
lr_model.fit(X_train, y_train)

print("Calibrating Logistic Regression...")
lr_calibrators, lr_cal_probs = calibrate_isotonic(lr_model, X_train, y_train, X_cal, y_cal, n_mapped_classes)
lr_brier = multiclass_brier(y_cal, lr_cal_probs)
print(f"Logistic calibrated Brier: {lr_brier:.4f}")

# Select best model
models = {
    'xgboost': (xgb_model, xgb_calibrators, xgb_brier),
    'lightgbm': (lgb_model, lgb_calibrators, lgb_brier),
    'mlp': (mlp_model, mlp_calibrators, mlp_brier),
    'logistic': (lr_model, lr_calibrators, lr_brier),
}

best_name, (best_model, best_calibrators, best_brier) = min(models.items(), key=lambda x: x[1][2])
print(f"\n=== Best model: {best_name} (Brier: {best_brier:.4f}) ===")

model_bundle = {
    'model': best_model,
    'calibrators': best_calibrators,
    'model_name': best_name,
    'intervals': intervals,
    'feature_names': [
        'temp_forecast_mean', 'temp_forecast_std', 'cloud_cover_pct',
        'precip_forecast_mm', 'wind_speed_avg', 'sea_breeze_flag',
        'weekday', 'hist_err', 'cloud_sq', 'cloud_x_sea',
        'fc_mean_x_cloud', 'seasonal_sin', 'seasonal_cos'
    ],
    'n_class': n_class,
    'n_mapped_classes': n_mapped_classes,
    'unique_classes': unique_classes,
    'class_to_idx': class_to_idx,
    'idx_to_class': idx_to_class,
    'brier_score': best_brier,
    'trained_on_dates': dates_used,
    'calibration_method': 'manual_isotonic'
}

pickle.dump(model_bundle, open(model_dir / 'best_model.pkl', 'wb'))
print(f"[INFO] Best model ({best_name}) saved to {model_dir/'best_model.pkl'}")

# Save all models
all_models = {
    'xgboost': (xgb_model, xgb_calibrators),
    'lightgbm': (lgb_model, lgb_calibrators),
    'mlp': (mlp_model, mlp_calibrators),
    'logistic': (lr_model, lr_calibrators),
}
for name, (model, calibrators) in all_models.items():
    pickle.dump({
        'model': model,
        'calibrators': calibrators,
        'model_name': name,
        'intervals': intervals,
        'feature_names': model_bundle['feature_names'],
        'n_class': n_class,
        'n_mapped_classes': n_mapped_classes,
        'unique_classes': unique_classes,
        'class_to_idx': class_to_idx,
        'idx_to_class': idx_to_class,
    }, open(model_dir / f'{name}_calibrated.pkl', 'wb'))

print("[INFO] All calibrated models saved")

# Feature importance
if hasattr(xgb_model, 'feature_importances_'):
    print("\nXGBoost Feature Importances:")
    for fname, imp in zip(model_bundle['feature_names'], xgb_model.feature_importances_):
        print(f"  {fname}: {imp:.4f}")

if hasattr(lgb_model, 'feature_importances_'):
    print("\nLightGBM Feature Importances:")
    for fname, imp in zip(model_bundle['feature_names'], lgb_model.feature_importances_):
        print(f"  {fname}: {imp:.4f}")