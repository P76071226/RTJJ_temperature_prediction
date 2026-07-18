#!/usr/bin/env python3
import json
import os
import re
import subprocess
import sys
from datetime import datetime, timedelta

def get_target_date():
    return (datetime.utcnow() + timedelta(days=1)).strftime('%Y-%m-%d')

def date_to_slug_parts(date_str):
    dt = datetime.strptime(date_str, '%Y-%m-%d')
    return dt.strftime('%B').lower(), dt.strftime('%d').lstrip('0'), dt.strftime('%Y')

def build_market_slugs(date_str):
    month_name, day, year = date_to_slug_parts(date_str)
    slugs = []
    for temp in range(27, 36):
        slugs.append(f"highest-temperature-in-tokyo-on-{month_name}-{day}-{year}-{temp}c")
    slugs.append(f"highest-temperature-in-tokyo-on-{month_name}-{day}-{year}-36corhigher")
    return slugs

def fetch_market_data(slug):
    cmd = ['polymarket-cli', 'market', '--slug', slug, '--json']
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        if result.returncode != 0:
            print(f"CLI error for {slug}: {result.stderr.strip()}")
            return None
        return json.loads(result.stdout)
    except Exception as e:
        print(f"Exception fetching {slug}: {e}")
        return None

def extract_yes_price(market_data):
    for outcome in market_data.get('odds', []):
        if outcome.get('outcome') == 'Yes':
            return float(outcome.get('price', 0))
    return 0.0

def build_intervals_from_markets(date_str, market_probs):
    intervals = []
    lower_temps = [(24, 25), (25, 26), (26, 27)]
    for lo, hi in lower_temps:
        intervals.append({"lo": lo, "hi": hi, "p": 0.0})
    for temp in range(27, 36):
        p = market_probs.get(temp, 0.0)
        intervals.append({"lo": temp, "hi": temp + 1, "p": p})
    p_36plus = market_probs.get('36+', 0.0)
    intervals.append({"lo": 36, "hi": 40, "p": p_36plus})
    total = sum(i["p"] for i in intervals)
    if total > 0:
        for i in intervals:
            i["p"] = i["p"] / total
    else:
        n = len(intervals)
        for i in intervals:
            i["p"] = 1.0 / n
    return intervals

def get_market_data():
    target_date = get_target_date()
    slugs = build_market_slugs(target_date)
    print(f"Fetching {len(slugs)} markets for {target_date}...")
    market_probs = {}
    success_count = 0
    for slug in slugs:
        data = fetch_market_data(slug)
        if data is None:
            print(f"  x {slug}: failed")
            continue
        if slug.endswith('36corhigher'):
            temp_key = '36+'
        else:
            match = re.search(r'-(\d+)c$', slug)
            if match:
                temp_key = int(match.group(1))
            else:
                print(f"  x {slug}: could not parse temperature")
                continue
        yes_price = extract_yes_price(data)
        market_probs[temp_key] = yes_price
        success_count += 1
        print(f"  ok {slug}: Yes={yes_price:.4f}")
    if success_count == 0:
        print("All market fetches failed, using mock data")
        return get_mock_data(target_date)
    if success_count < len(slugs):
        print(f"Warning: only {success_count}/{len(slugs)} markets fetched successfully")
    intervals = build_intervals_from_markets(target_date, market_probs)
    market_data = {
        "date": target_date,
        "intervals": intervals,
        "source": "polymarket",
        "markets_fetched": success_count,
        "markets_total": len(slugs)
    }
    return market_data

def get_mock_data(target_date):
    intervals = [
        {"lo": 24, "hi": 25, "p": 0.02},
        {"lo": 25, "hi": 26, "p": 0.08},
        {"lo": 26, "hi": 27, "p": 0.20},
        {"lo": 27, "hi": 28, "p": 0.35},
        {"lo": 28, "hi": 29, "p": 0.25},
        {"lo": 29, "hi": 30, "p": 0.08},
        {"lo": 30, "hi": 31, "p": 0.02},
    ]
    return {
        "date": target_date,
        "intervals": intervals,
        "source": "mock",
        "markets_fetched": 0,
        "markets_total": 0
    }

def main():
    market_data = get_market_data()
    os.makedirs('data', exist_ok=True)
    output_path = os.path.join('data', 'market_raw.json')
    with open(output_path, 'w') as f:
        json.dump(market_data, f, indent=2)
    print(f"\nMarket data saved to {output_path}")
    print(json.dumps(market_data, indent=2))

if __name__ == '__main__':
    main()
