#!/usr/bin/env python3
import json, os, pickle, numpy as np
from pathlib import Path
from datetime import datetime, timedelta
import statsmodels.api as sm
from statsmodels.miscmodels.ordinal_model import OrderedModel

archive = Path('/root/hermes_research/rjtt_project/archive')
X_list = []
y_list = []   # 0,1,2,... 對應區間索引

# 取第一天的市場區間作為參考（假設所有天的區間相同）
sample_market = json.load(open(next(archive.iterdir())/'market_raw.json'))
intervals = sample_market.get('intervals', [])
n_class = len(intervals)

def interval_index(obs, intervals):
    """把觀測最高溫映射到區間索引（0..n_class-1）"""
    for i, iv in enumerate(intervals):
        if iv['lo'] <= obs < iv['hi']:
            return i
    # 若剛好等於上界，則放進最後一個區間
    if abs(obs - intervals[-1]['hi']) < 1e-9:
        return len(intervals)-1
    return -1  # 在正常資料中不應該發生

def parse_folder_time(folder_name):
    # folder_name format: YYYYMMDD_HHMMSS
    try:
        dt = datetime.strptime(folder_name, '%Y%m%d_%H%M%S')
        return dt
    except ValueError:
        return None

for folder in sorted(archive.iterdir()):
    if not folder.is_dir():
        continue
    folder_time = parse_folder_time(folder.name)
    if folder_time is None:
        continue
    # 特徵
    feat = json.load(open(folder/'osint_features.json'))
    val  = json.load(open(folder/'validation_log.json'))
    obs  = val.get('observed_temp')
    if obs is None:
        continue

    # 歷史誤差：昨日的觀測 - 昨日的預報平均
    hist_err = 0.0
    prev_time = folder_time - timedelta(days=1)
    prev_folder_name = prev_time.strftime('%Y%m%d_%H%M%S')
    prev_folder = archive / prev_folder_name
    if prev_folder.is_dir():
        prev_feat = json.load(open(prev_folder/'osint_features.json'))
        prev_val  = json.load(open(prev_folder/'validation_log.json'))
        hist_err = prev_val.get('observed_temp',0.0) - prev_feat.get('temp_forecast_mean',0.0)

    # 特徵向量：基本特徵
    feat_vec = np.array([
        feat.get('temp_forecast_mean',28.0),
        feat.get('temp_forecast_std',1.5),
        feat.get('cloud_cover_pct',50.0),
        feat.get('precip_forecast_mm',0.0),
        feat.get('wind_speed_avg',2.0),
        feat.get('sea_breeze_flag',0),
        feat.get('weekday',0),
        hist_err
    ], dtype=float)
    X_list.append(feat_vec)
    y_list.append(interval_index(obs, intervals))

X = np.vstack(X_list)
y = np.array(y_list, dtype=int)

# 有序 logit 模型（logit link）不需要截距
ord_model = OrderedModel(y, X, distr='logit')
result = ord_model.fit(method='bfgs', disp=False)

# 保存模型
model_dir = Path('/root/hermes_research/rjtt_project/models')
model_dir.mkdir(exist_ok=True)
pickle.dump(result, open(model_dir/'ordinal_model.pkl', 'wb'))
print(f"[INFO] Ordinal model saved to {model_dir/'ordinal_model.pkl'}")
print(f"[INFO] 訓練樣本數: {len(X)}")
