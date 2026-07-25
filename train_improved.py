#!/usr/bin/env python3
"""
Improved training script with:
1. Real daily max temperature as target (Open-Meteo historical API)
2. XGBoost / LightGBM / MLP models
3. Calibration (Isotonic regression / Platt scaling)
"""
import json
import os
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta
import requests
import time

# ML models
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import train_test_split, cross_val_predict
from sklearn.metrics import brier_score_loss, log_loss
import xgboost as xgb
import lightgbm as lgb

archive = Path('/root/hermes_research/rjtt_project/archive')

# 1. Get reference market intervals
sample_market = json.load(open(next(archive.iterdir()) / 'market_raw.json'))
intervals = sample_market.get('intervals', [])
n_class = len(intervals)

print(f"Found {n_class} temperature intervals")

def interval_index(obs, intervals):
    """Map observed max temp to interval index"""
    for i, iv in enumerate(intervals):
        if iv['lo'] <= obs < iv['hi']:
            return i
    if abs(obs - intervals[-1]['hi']) < 1e-9:
        return len(intervals) - 1
    return -1

def parse_folder_time(folder_name):
    try:
        return datetime.strptime(folder_name, '%Y%m%d_%H%M%S')
    except ValueError:
        return None

def get_true_max_temp(date_str):
    """
    Fetch true daily max temperature from Open-Meteo historical API
    date_str format: '2026-07-19'
    """
    url = "https://archive-api.open-meteo.com/v1/archive"
    params = {
        'latitude': 35.55,
        'longitude': 139.78,
        'start_date': date_str,
        'end_date': date_str,
        'daily': 'temperature_2m_max',
        'timezone': 'Asia/Tokyo'
    }
    try:
        resp = requests.get(url, params=params, timeout=10)
        if resp.status_code == 200:
            data = resp.json()
            if 'daily' in data and 'temperature_2m_max' in data['daily']:
                return data['daily']['temperature_2m_max'][0]
    except Exception as e:
        print(f"  Warning: Failed to fetch true max temp for {date_str}: {e}")
    return None

# Collect training data
X_list = []
y_list = []
dates_used = []

print("Processing archive folders...")
for folder in sorted(archive.iterdir()):
    if not folder.is_dir():
        continue
    
    folder_time = parse_folder_time(folder.name)
    if folder_time is None:
        continue
    
    # Load features
    feat_path = folder / 'osint_features.json'
    if not feat_path.exists():
        continue
    feat = json.load(open(feat_path))
    
    # Get date string from folder name (YYYYMMDD)
    date_str = folder_time.strftime('%Y-%m-%d')
    
    # Get TRUE daily max temperature from Open-Meteo historical API
    true_max_temp = get_true_max_temp(date_str)
    if true_max_temp is None:
        print(f"  Skipping {date_str}: Could not fetch true max temp")
        continue
    
    y_idx = interval_index(true_max_temp, intervals)
    if y_idx < 0:
        print(f"  Skipping {date_str}: true max temp {true_max_temp} outside intervals")
        continue
    
    # Historical error: yesterday observed - yesterday forecast mean
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
    
    # Feature vector (enhanced)
    feat_vec = np.array([
        feat.get('temp_forecast_mean', 28.0),
        feat.get('temp_forecast_std', 1.5),
        feat.get('cloud_cover_pct', 50.0),
        feat.get('precip_forecast_mm', 0.0),
        feat.get('wind_speed_avg', 2.0),
        feat.get('sea_breeze_flag', 0),
        feat.get('weekday', 0),
        hist_err,
        # New features
        feat.get('cloud_cover_pct', 50.0) ** 2 / 10000,  # cloud_sq
        feat.get('cloud_cover_pct', 50.0) * feat.get('sea_breeze_flag', 0) / 100,  # cloud_x_sea
        feat.get('temp_forecast_mean', 28.0) * feat.get('cloud_cover_pct', 50.0) / 100,  # fc_mean_x_cloud
        np.sin(2 * np.pi * folder_time.timetuple().tm_yday / 365.25),  # seasonal sin
        np.cos(2 * np.pi * folder_time.timetuple().tm_yday / 365.25),  # seasonal cos
    ], dtype=float)
    
    X_list.append(feat_vec)
    y_list.append(y_idx)
    dates_used.append(date_str)
    
    # Rate limit API calls
    time.sleep(0.1)

X = np.vstack(X_list)
y = np.array(y_list, dtype=int)

print(f"\nTraining samples: {len(X)}")
print(f"Features: {X.shape[1]}")
print(f"Classes: {n_class}")
print(f"Date range: {dates_used[0]} to {dates_used[-1]}")

# Train-test split for calibration
X_train, X_cal, y_train, y_cal = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

print(f"\nTrain: {len(X_train)}, Calibration: {len(X_cal)}")

# ============================================
# MODEL 1: XGBoost
# ============================================
print("\n=== Training XGBoost ===")
xgb_model = xgb.XGBClassifier(
    objective='multi:softprob',
    num_class=n_class,
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

# Calibrate with Isotonic Regression
print("Calibrating XGBoost with Isotonic Regression...")
xgb_calibrated = CalibratedClassifierCV(xgb_model, method='isotonic', cv='prefit')
xgb_calibrated.fit(X_cal, y_cal)

# Evaluate calibration
xgb_probs_train = xgb_model.predict_proba(X_train)
xgb_probs_cal = xgb_calibrated.predict_proba(X_cal)

# Brier score
xgb_brier = brier_score_loss(y_cal, np.max(xgb_probs_cal, axis=1))
print(f"XGBoost calibrated Brier (max prob): {xgb_brier:.4f}")

# ============================================
# MODEL 2: LightGBM
# ============================================
print("\n=== Training LightGBM ===")
lgb_model = lgb.LGBMClassifier(
    objective='multiclass',
    num_class=n_class,
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

print("Calibrating LightGBM with Isotonic Regression...")
lgb_calibrated = CalibratedClassifierCV(lgb_model, method='isotonic', cv='prefit')
lgb_calibrated.fit(X_cal, y_cal)

lgb_probs_cal = lgb_calibrated.predict_proba(X_cal)
lgb_brier = brier_score_loss(y_cal, np.max(lgb_probs_cal, axis=1))
print(f"LightGBM calibrated Brier (max prob): {lgb_brier:.4f}")

# ============================================
# MODEL 3: MLPClassifier
# ============================================
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

print("Calibrating MLP with Isotonic Regression...")
mlp_calibrated = CalibratedClassifierCV(mlp_model, method='isotonic', cv='prefit')
mlp_calibrated.fit(X_cal, y_cal)

mlp_probs_cal = mlp_calibrated.predict_proba(X_cal)
mlp_brier = brier_score_loss(y_cal, np.max(mlp_probs_cal, axis=1))
print(f"MLP calibrated Brier (max prob): {mlp_brier:.4f}")

# ============================================
# MODEL 4: Logistic Regression (baseline with calibration)
# ============================================
print("\n=== Training Logistic Regression (with calibration) ===")
lr_model = LogisticRegression(solver='lbfgs', max_iter=1000, random_state=42)
lr_model.fit(X_train, y_train)

print("Calibrating Logistic Regression...")
lr_calibrated = CalibratedClassifierCV(lr_model, method='isotonic', cv='prefit')
lr_calibrated.fit(X_cal, y_cal)

lr_probs_cal = lr_calibrated.predict_proba(X_cal)
lr_brier = brier_score_loss(y_cal, np.max(lr_probs_cal, axis=1))
print(f"Logistic calibrated Brier (max prob): {lr_brier:.4f}")

# ============================================
# Select best model
# ============================================
models = {
    'xgboost': (xgb_calibrated, xgb_brier),
    'lightgbm': (lgb_calibrated, lgb_brier),
    'mlp': (mlp_calibrated, mlp_brier),
    'logistic': (lr_calibrated, lr_brier),
}

best_name, (best_model, best_brier) = min(models.items(), key=lambda x: x[1][1])
print(f"\n=== Best model: {best_name} (Brier: {best_brier:.4f}) ===")

# Save best model
model_dir = Path('/root/hermes_research/rjtt_project/models')
model_dir.mkdir(exist_ok=True)

model_bundle = {
    'model': best_model,
    'model_name': best_name,
    'intervals': intervals,
    'feature_names': [
        'temp_forecast_mean', 'temp_forecast_std', 'cloud_cover_pct',
        'precip_forecast_mm', 'wind_speed_avg', 'sea_breeze_flag',
        'weekday', 'hist_err', 'cloud_sq', 'cloud_x_sea',
        'fc_mean_x_cloud', 'seasonal_sin', 'seasonal_cos'
    ],
    'n_class': n_class,
    'brier_score': best_brier,
    'trained_on_dates': dates_used,
    'calibration_method': 'isotonic'
}

pickle.dump(model_bundle, open(model_dir / 'best_model.pkl', 'wb'))
print(f"[INFO] Best model ({best_name}) saved to {model_dir/'best_model.pkl'}")

# Also save all models for comparison
all_models = {
    'xgboost': xgb_calibrated,
    'lightgbm': lgb_calibrated,
    'mlp': mlp_calibrated,
    'logistic': lr_calibrated,
}
for name, model in all_models.items():
    pickle.dump({
        'model': model,
        'model_name': name,
        'intervals': intervals,
        'feature_names': model_bundle['feature_names'],
        'n_class': n_class,
    }, open(model_dir / f'{name}_calibrated.pkl', 'wb'))

print("[INFO] All calibrated models saved")

# Print feature importance for tree models
if hasattr(xgb_model, 'feature_importances_'):
    print("\nXGBoost Feature Importances:")
    for fname, imp in zip(model_bundle['feature_names'], xgb_model.feature_importances_):
        print(f"  {fname}: {imp:.4f}")

if hasattr(lgb_model, 'feature_importances_'):
    print("\nLightGBM Feature Importances:")
    for fname, imp in zip(model_bundle['feature_names'], lgb_model.feature_importances_):
        print(f"  {fname}: {imp:.4f}")