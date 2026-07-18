#!/usr/bin/env python3
import json
import os
import sys
import subprocess
from datetime import datetime, timedelta
import math

def try_fetch_json(url):
    """Try to fetch JSON from a URL using curl."""
    try:
        result = subprocess.run(['curl', '-s', '-S', url], capture_output=True, text=True, timeout=10)
        if result.returncode == 0 and result.stdout.strip():
            try:
                return json.loads(result.stdout)
            except json.JSONDecodeError:
                # Not JSON, return text
                return result.stdout
        else:
            print(f"Failed to fetch {url}: {result.stderr}")
            return None
    except Exception as e:
        print(f"Exception fetching {url}: {e}")
        return None

def get_features():
    target = datetime.utcnow() + timedelta(days=1)
    date_str = target.strftime('%Y-%m-%d')
    features = {"date": date_str}
    # 1. JMA / Open-Meteo forecast for Tokyo (approx coordinates for Haneda)
    # Use Open-Meteo as proxy
    url = "https://api.open-meteo.com/v1/forecast?latitude=35.55&longitude=139.78&hourly=temperature_2m,cloudcover,precipitation,windspeed,winddirection&timezone=Asia%2FTokyo&forecast_days=1"
    data = try_fetch_json(url)
    if data and 'hourly' in data:
        hourly = data['hourly']
        temps = hourly.get('temperature_2m', [])
        clouds = hourly.get('cloudcover', [])
        precip = hourly.get('precipitation', [])
        windsp = hourly.get('windspeed', [])
        winddir = hourly.get('winddirection', [])
        if temps:
            features['temp_forecast_mean'] = sum(temps) / len(temps)
            features['temp_forecast_std'] = math.sqrt(sum((t - features['temp_forecast_mean'])**2 for t in temps) / len(temps)) if len(temps) > 1 else 0.0
        if clouds:
            features['cloud_cover_pct'] = sum(clouds) / len(clouds)
        if precip:
            # total precipitation forecast for the day
            features['precip_forecast_mm'] = sum(precip)
        if windsp:
            features['wind_speed_avg'] = sum(windsp) / len(windsp)
        if winddir:
            # average wind direction (vector average)
            sin_sum = sum(math.sin(math.radians(d)) for d in winddir)
            cos_sum = sum(math.cos(math.radians(d)) for d in winddir)
            features['wind_dir_avg'] = math.degrees(math.atan2(sin_sum, cos_sum)) % 360
    # 2. METAR observations for RJTT (past 6-12 hours)
    metar_url = "https://tgftp.nws.noaa.gov/data/observations/metar/stations/RJTT.TXT"
    metar_text = try_fetch_json(metar_url)
    if metar_text and isinstance(metar_text, str):
        lines = [l.strip() for l in metar_text.split('\\n') if l.strip()]
        # Take the most recent METAR (first line)
        if lines:
            metar = lines[0]
            features['metar_raw'] = metar
            # Parse temperature and dewpoint (format like ... 18/10 ...)
            import re
            # Find pattern like XX/YY
            m = re.search(r'(\d{2})/(\d{2})', metar)
            if m:
                temp = int(m.group(1))
                dew = int(m.group(2))
                features['metar_temp_c'] = temp
                features['metar_dewpoint_c'] = dew
            # Wind: look for pattern like XXXKT or XXXMPS
            # Simplified: just note if wind speed present
            if 'KT' in metar or 'MPS' in metar:
                features['metar_wind_present'] = True
            # Cloud coverage: look for FEW,SCT,BKN,OVC
            if any(c in metar for c in ['FEW','SCT','BKN','OVC']):
                features['metar_cloud_present'] = True
            # Sea breeze detection: wind direction from south (180) or sea? For simplicity, if wind direction between 130 and 230, flag sea breeze
            wdir = features.get('wind_dir_avg', features.get('metar_wind_dir', 0))
            if 130 <= wdir <= 230:
                features['sea_breeze_flag'] = 1
            else:
                features['sea_breeze_flag'] = 0
    # 3. Time features
    features['month'] = target.month
    features['weekday'] = target.weekday()  # Monday=0
    features['day'] = target.day
    # 4. If we couldn't get real data, fill with sensible defaults/mock
    # Ensure essential keys exist
    if 'temp_forecast_mean' not in features:
        features['temp_forecast_mean'] = 28.0
    if 'temp_forecast_std' not in features:
        features['temp_forecast_std'] = 1.5
    if 'cloud_cover_pct' not in features:
        features['cloud_cover_pct'] = 50.0
    if 'sea_breeze_flag' not in features:
        features['sea_breeze_flag'] = 0
    if 'precip_forecast_mm' not in features:
        features['precip_forecast_mm'] = 0.0
    if 'wind_speed_avg' not in features:
        features['wind_speed_avg'] = 3.0
    # Remove raw text fields we don't need to keep in JSON
    features.pop('metar_raw', None)
    return features

def main():
    features = get_features()
    os.makedirs('data', exist_ok=True)
    output_path = os.path.join('data', 'osint_features.json')
    with open(output_path, 'w') as f:
        json.dump(features, f, indent=2)
    print(f"OSINT features saved to {output_path}")
    print(json.dumps(features, indent=2))

if __name__ == '__main__':
    main()