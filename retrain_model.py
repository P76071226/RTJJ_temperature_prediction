#!/usr/bin/env python3
"""
Retrain XGBoost model on all available archive data (40 days) with actual temps.
"""
import json
import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.isotonic import IsotonicRegression
import xgboost as xgb
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# 1. Load historical actual temperatures
# ============================================================
actual_temps = pd.read_csv('/root/hermes_research/rjtt_project/historical_temps_6months.csv', 
                           index_col=0, parse_dates=True)['actual_max_temp']
print(f"Historical temps: {len(actual_temps)} days ({actual_temps.index.min()} to {actual_temps.index.max()})")

# ============================================================
# 2. Extract features from archive
# ============================================================
archive_dir = Path('/root/hermes_research/rjtt_project/archive')
dirs = list(sorted(archive_dir.iterdir()))

X_list = []
y_list = []
dates_list = []

for d in dirs:
    if not d.is_dir():
        continue
    market = d / 'market_raw.json'
    osint = d / 'osint_features.json'
    model_probs = d / 'model_probs.json'
    if not (market.exists() and osint.exists() and model_probs.exists()):
        continue
    
    with open(market) as f:
        market_data = json.load(f)
    with open(osint) as f:
        osint_data = json.load(f)
    with open(model_probs) as f:
        model_data = json.load(f)
    
    try:
        date_str = d.name.split('_')[0]
        date = pd.Timestamp(date_str)
    except:
        continue
    
    if date not in actual_temps.index:
        continue
    
    actual_temp = actual_temps[date]
    
    # Find actual interval (using model intervals which go to 36C)
    intervals = model_data['intervals']
    actual_idx = None
    for i, interval in enumerate(intervals):
        lo, hi = interval['lo'], interval['hi']
        if lo <= actual_temp < hi:
            actual_idx = i
            break
    if actual_idx is None and actual_temp >= 35:
        actual_idx = len(intervals) - 1
    
    if actual_idx is None:
        continue
    
    # Build features
    fc_mean = osint_data.get('temp_forecast_mean', 0)
    fc_std = osint_data.get('temp_forecast_std', 0)
    cloud = osint_data.get('cloud_cover_pct', 0)
    precip = osint_data.get('precip_forecast_mm', 0)
    wind = osint_data.get('wind_speed_avg', 0)
    sea_breeze = osint_data.get('sea_breeze_flag', 0)
    dow = date.dayofweek
    doy = date.dayofyear
    
    # Historical error (previous day forecast error)
    hist_err = 0
    if len(X_list) > 0:
        hist_err = X_list[-1][0] - y_list[-1]  # fc_mean - actual_idx
    
    features = [
        fc_mean, fc_std, cloud, precip, wind, sea_breeze, dow,
        np.sin(2*np.pi*doy/365), np.cos(2*np.pi*doy/365),
        cloud**2, cloud*sea_breeze, fc_mean*cloud, hist_err
    ]
    
    X_list.append(features)
    y_list.append(actual_idx)
    dates_list.append(date)

X = np.array(X_list)
y = np.array(y_list)
print(f"\nTraining samples: {len(X)}")
print(f"Class distribution: {np.bincount(y)}")
print(f"Date range: {dates_list[0]} to {dates_list[-1]}")

# ============================================================
# 3. Train XGBoost
# ============================================================
print("\nTraining XGBoost...")
intervals = [
    {'lo': 24, 'hi': 25}, {'lo': 25, 'hi': 26}, {'lo': 26, 'hi': 27},
    {'lo': 27, 'hi': 28}, {'lo': 28, 'hi': 29}, {'lo': 29, 'hi': 30},
    {'lo': 30, 'hi': 31}, {'lo': 31, 'hi': 32}, {'lo': 32, 'hi': 33},
    {'lo': 33, 'hi': 34}, {'lo': 34, 'hi': 35}, {'lo': 35, 'hi': 36},
    {'lo': 36, 'hi': 37}
]

model = xgb.XGBClassifier(
    n_estimators=200,
    max_depth=4,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    objective='multi:softprob',
    num_class=len(intervals),
    eval_metric='mlogloss',
    random_state=42,
    verbosity=0
)

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
model.fit(X_train, y_train)

train_acc = (model.predict(X_train) == y_train).mean()
val_acc = (model.predict(X_val) == y_val).mean()
print(f"Train accuracy: {train_acc:.2%}")
print(f"Val accuracy: {val_acc:.2%}")

# ============================================================
# 4. Isotonic Calibration per class
# ============================================================
print("\nCalibrating with Isotonic Regression...")
probas_val = model.predict_proba(X_val)
calibrators = {}
classes = np.arange(len(intervals))

for cls in classes:
    mask = (y_val == cls).astype(int)
    probs = probas_val[:, cls]
    
    if mask.sum() >= 3 and mask.sum() < len(mask):
        try:
            cal = IsotonicRegression(out_of_bounds='clip')
            cal.fit(probs, mask)
            calibrators[int(cls)] = cal
        except:
            calibrators[int(cls)] = None
    else:
        calibrators[int(cls)] = None

# Test calibration
calibrated_val = np.zeros_like(probas_val)
for i, cls in enumerate(classes):
    cal = calibrators.get(int(cls))
    if cal is not None:
        calibrated_val[:, i] = cal.predict(np.clip(probas_val[:, i], 1e-6, 1-1e-6))
    else:
        calibrated_val[:, i] = probas_val[:, i]
calibrated_val = calibrated_val / calibrated_val.sum(axis=1, keepdims=True)

y_true_onehot = np.eye(len(intervals))[y_val]
brier_before = np.mean((probas_val - y_true_onehot) ** 2)
brier_after = np.mean((calibrated_val - y_true_onehot) ** 2)
print(f"Brier before calibration: {brier_before:.4f}")
print(f"Brier after calibration:  {brier_after:.4f}")

# ============================================================
# 5. Save model
# ============================================================
interval_labels = []
for i in intervals:
    interval_labels.append(str(i['lo']) + '-' + str(i['hi']))

model_data = {
    'model': model,
    'calibrators': calibrators,
    'class_indices': {int(c): i for i, c in enumerate(classes)},
    'unique_classes': classes,
    'intervals': intervals,
    'interval_labels': interval_labels,
    'feature_names': ['fc_mean', 'fc_std', 'cloud', 'precip', 'wind', 'sea_breeze', 'dow',
                      'sin_doy', 'cos_doy', 'cloud_sq', 'cloud_x_sea', 'fc_x_cloud', 'hist_err'],
    'n_class': len(intervals),
    'n_mapped_classes': len(calibrators),
    'train_dates': [str(d) for d in dates_list],
    'val_brier': brier_after
}

joblib.dump(model_data, '/root/hermes_research/rjtt_project/models/xgboost_calibrated_v2.pkl')
print("\nSaved model to models/xgboost_calibrated_v2.pkl")
print("Intervals: " + str(interval_labels))
print("Features: " + str(model_data['feature_names']))