#!/usr/bin/env python3
"""
Improved training script with:
1. Real daily maximum temperature as target (from Open-Meteo historical API)
2. Extended intervals to cover actual temperature range
3. XGBoost / LightGBM / MLP models
4. Isotonic regression calibration
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
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import brier_score_loss
import xgboost as xgb
import lightgbm as lgb
from sklearn.neural_network import MLPClassifier

archive = Path('/root/hermes_research/rjtt_project/archive')
model_dir = Path('/root/hermes_research/rjtt_project/models')
model_dir.mkdir(exist_ok=True)

# Use mock intervals that cover the actual temperature range (up to 35°C)
EXTENDED_INTERVALS = [
    {"lo": 24, "hi": 25, "p": 0.02},
    {"lo": 25, "hi": 26, "p": 0.08},
    {"lo": 26, "hi": 27, "p": 0.20},
    {"lo": 27, "hi": 28, "p": 0.35},
    {"lo": 28, "hi": 29, "p": 0.25},
    {"lo": 29, "hi": 30, "p": 0.08},
    {"lo": 30, "hi": 31, "p": 0.02},
    {"lo": 31, "hi": 32, "p": 0.0},  # extended
    {"lo": 32, "hi": 33, "p": 0.0},  # extended
    {"lo": 33, "hi": 34, "p": 0.0},  # extended
    {"lo": 34, "hi": 35, "p": 0.0},  # extended
    {"lo": 35, "hi": 36, "p": 0.0},  # extended
]
intervals = EXTENDED_INTERVALS
n_class = len(intervals)

def interval_index(obs, intervals):
    """Map observed temperature to interval index (0..n_class-1)"""
    for i, iv in enumerate(intervals):
        if iv['lo'] <= obs < iv['hi']:
            return i
    if abs(obs - intervals[-1]['hi']) < 1e-9:
        return len(intervals) - 1
    if obs < intervals[0]['lo']:
        return 0
    return len(intervals) - 1

def get_true_max_temp(date_str):
    """Fetch true daily maximum temperature from Open-Meteo historical API"""
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

X_list = []
y_list = []
dates_used = []

print(f"Processing archive folders with {n_class} intervals...")
print(f"Intervals: {[(iv['lo'], iv['hi']) for iv in intervals]}")

for folder in sorted(archive.iterdir()):
    if not folder.is_dir():
        continue
    folder_time = parse_folder_time(folder.name)
    if folder_time is None:
        continue
    
    date_str = folder_time.strftime('%Y-%m-%d')
    
    # Get true daily max temperature
    true_max_temp = get_true_max_temp(date_str)
    if true_max_temp is None:
        print(f"  Skipping {date_str}: Could not fetch true max temp")
        continue
    
    y_idx = interval_index(true_max_temp, intervals)
    
    # Load features
    feat_path = folder / 'osint_features.json'
    if not feat_path.exists():
        continue
    feat = json.load(open(feat_path))
    
    # Historical error
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
    
    # Feature vector (13 features)
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
        np.sin(2 * np.pi * folder_time.timetuple().tm_yday / 365.25),  # seasonal_sin
        np.cos(2 * np.pi * folder_time.timetuple().tm_yday / 365.25),  # seasonal_cos
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

# ============================================
# KEY FIX: Remap class labels to 0-based continuous
# ============================================
unique_classes = sorted(np.unique(y))
class_to_idx = {c: i for i, c in enumerate(unique_classes)}
idx_to_class = {i: c for i, c in enumerate(unique_classes)}
y_mapped = np.array([class_to_idx[c] for c in y])
n_mapped_classes = len(unique_classes)

print(f"\nMapped classes: {unique_classes} -> {list(range(n_mapped_classes))}")
print(f"Number of mapped classes: {n_mapped_classes}")

# Train-test split for calibration
X_train, X_cal, y_train, y_cal = train_test_split(X, y_mapped, test_size=0.2, random_state=42, stratify=y_mapped)
print(f"\nTrain: {len(X_train)}, Calibration: {len(X_cal)}")

# ============================================
# MODEL 1: XGBoost
# ============================================
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

print("Calibrating XGBoost with Isotonic Regression...")
xgb_calibrated = CalibratedClassifierCV(xgb_model, method='isotonic', cv='prefit')
xgb_calibrated.fit(X_cal, y_cal)

xgb_probs_cal = xgb_calibrated.predict_proba(X_cal)
xgb_brier = brier_score_loss(y_cal, np.max(xgb_probs_cal, axis=1))
print(f"XGBoost calibrated Brier (max prob): {xgb_brier:.4f}")

# ============================================
# MODEL 2: LightGBM
# ============================================
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
# MODEL 4: Logistic Regression (baseline)
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
    'xgboost': (xgb_calibrated, xgb_brier, unique_classes, class_to_idx, idx_to_class),
    'lightgbm': (lgb_calibrated, lgb_brier, unique_classes, class_to_idx, idx_to_class),
    'mlp': (mlp_calibrated, mlp_brier, unique_classes, class_to_idx, idx_to_class),
    'logistic': (lr_calibrated, lr_brier, unique_classes, class_to_idx, idx_to_class),
}

best_name, (best_model, best_brier, _, _, _) = min(models.items(), key=lambda x: x[1][1])
print(f"\n=== Best model: {best_name} (Brier: {best_brier:.4f}) ===")

# Save best model with class mapping
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
    'n_mapped_classes': n_mapped_classes,
    'unique_classes': unique_classes,
    'class_to_idx': class_to_idx,
    'idx_to_class': idx_to_class,
    'brier_score': best_brier,
    'trained_on_dates': dates_used,
    'calibration_method': 'isotonic'
}

pickle.dump(model_bundle, open(model_dir / 'best_model.pkl', 'wb'))
print(f"[INFO] Best model ({best_name}) saved to {model_dir/'best_model.pkl'}")

# Also save all models for comparison
all_models = {
    'xgboost': (xgb_calibrated, unique_classes, class_to_idx, idx_to_class),
    'lightgbm': (lgb_calibrated, unique_classes, class_to_idx, idx_to_class),
    'mlp': (mlp_calibrated, unique_classes, class_to_idx, idx_to_class),
    'logistic': (lr_calibrated, unique_classes, class_to_idx, idx_to_class),
}
for name, (model, uq, c2i, i2c) in all_models.items():
    pickle.dump({
        'model': model,
        'model_name': name,
        'intervals': intervals,
        'feature_names': model_bundle['feature_names'],
        'n_class': n_class,
        'n_mapped_classes': n_mapped_classes,
        'unique_classes': uq,
        'class_to_idx': c2i,
        'idx_to_class': i2c,
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