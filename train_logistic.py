#!/usr/bin/env python3
import json, os, pickle, numpy as np
from pathlib import Path
from datetime import datetime
import requests
from sklearn.linear_model import LogisticRegression

archive = Path('/root/hermes_research/rjtt_project/archive')
X_list = []
y_list = []

# 取第一天的市場區間作為參考（假設所有天的區間相同）
sample_market = json.load(open(next(archive.iterdir())/'market_raw.json'))
intervals = sample_market.get('intervals', [])
n_class = len(intervals)

def interval_index(obs, intervals):
    for i, iv in enumerate(intervals):
        if iv['lo'] <= obs < iv['hi']:
            return i
    if abs(obs - intervals[-1]['hi']) < 1e-9:
        return len(intervals)-1
    return -1

def get_observed_temp(date_str):
    lat, lon = 35.55, 139.78
    url = f"https://archive-api.open-meteo.com/v1/archive?latitude={lat}&longitude={lon}&start_date={date_str}&end_date={date_str}&daily=temperature_2m_max&timezone=UTC"
    try:
        resp = requests.get(url, timeout=10)
        resp.raise_for_status()
        data = resp.json()
        return float(data['daily']['temperature_2m_max'][0])
    except Exception as e:
        print(f"[WARN] Could not fetch observed temp for {date_str}: {e}")
        return None

for folder in sorted(archive.iterdir()):
    if not folder.is_dir():
        continue
    market_path = folder / 'market_raw.json'
    if not market_path.is_file():
        continue
    market_data = json.load(open(market_path))
    date_str = market_data.get('date')
    if not date_str:
        continue
    obs = get_observed_temp(date_str)
    if obs is None:
        continue
    feat = json.load(open(folder/'osint_features.json'))
    feat_vec = np.array([
        feat.get('temp_forecast_mean',28.0),
        feat.get('temp_forecast_std',1.5),
        feat.get('cloud_cover_pct',50.0),
        feat.get('precip_forecast_mm',0.0),
        feat.get('wind_speed_avg',2.0),
        feat.get('sea_breeze_flag',0),
        feat.get('weekday',0)
    ], dtype=float)
    X_list.append(feat_vec)
    y_list.append(interval_index(obs, intervals))

X = np.vstack(X_list)
y = np.array(y_list, dtype=int)
print(f"[INFO] Loaded {len(X)} samples")
print(f"[INFO] Unique classes observed: {np.unique(y)}")

model = LogisticRegression(solver='lbfgs', max_iter=1000)
model.fit(X, y)

model_dir = Path('/root/hermes_research/rjtt_project/models')
model_dir.mkdir(exist_ok=True)
pickle.dump(model, open(model_dir/'logistic_model.pkl', 'wb'))
print(f"[INFO] Logistic model saved to {model_dir/'logistic_model.pkl'}")
print(f"[INFO] 訓練樣本數: {len(X)}")
print(f"[INFO] Number of classes: {len(model.classes_)}")