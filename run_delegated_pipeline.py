#!/usr/bin/env python3
import json
TASKS = [
  {
    "name": "market_collector",
    "goal": "Fetch Polymarket binary markets for target date and normalize to 13 temperature intervals (24-36\u00b0C). Steps: 1. Read target date from data/market_raw.json or use tomorrow. 2. Run polymarket-cli for 10 temperature thresholds (27-36\u00b0C). 3. Parse yes/no probabilities. 4. Normalize to 13 intervals (24-36\u00b0C in 1\u00b0C bins). 5. Save to data/market_raw.json with structure: {\"date\": \"...\", \"intervals\": [...], \"source\": \"polymarket\", \"markets_fetched\": N, \"markets_total\": 10}",
    "context": "Working dir: /root/hermes_research/rjtt_project. Polymarket CLI at /home/.hermes/skills/research/polymarket/scripts/polymarket.py. Target date: tomorrow (Aug 3, 2026). Save to /root/hermes_research/rjtt_project/data/market_raw.json",
    "toolsets": [
      "terminal",
      "file"
    ]
  },
  {
    "name": "osint_researcher",
    "goal": "Gather weather forecast and observational data for Tokyo Haneda (RJTT). Steps: 1. Read target date from data/market_raw.json. 2. Call Open-Meteo forecast API (35.55, 139.78) for hourly temps, cloud, precip, wind. 3. Calculate daily max temp forecast mean/std. 4. Call JMA METAR API for current observations at RJTT. 5. Compute sea breeze flag. 6. Add time features (month, weekday, day). 7. Save to data/osint_features.json with all 12+ features",
    "context": "Working dir: /root/hermes_research/rjtt_project. Read target date from /root/hermes_research/rjtt_project/data/market_raw.json. Open-Meteo: https://api.open-meteo.com/v1/forecast. JMA METAR: https://www.jma.go.jp/bosai/amedas/data/meta/. Save to /root/hermes_research/rjtt_project/data/osint_features.json",
    "toolsets": [
      "terminal",
      "file",
      "web"
    ]
  },
  {
    "name": "model_builder",
    "goal": "Load trained XGBoost + Isotonic model and predict temperature interval probabilities. Steps: 1. Read market intervals from data/market_raw.json. 2. Read features from data/osint_features.json. 3. Build 14-feature vector including fc_anomaly bias correction. 4. Load best_model.pkl (XGBoost + 6 Isotonic calibrators). 5. Apply calibration to get probabilities for mapped classes. 6. Expand to full 13-interval space. 7. Normalize and save to data/model_probs.json",
    "context": "Working dir: /root/hermes_research/rjtt_project. Model at /root/hermes_research/rjtt_project/models/best_model.pkl. Monthly climatology: {\"1\": 8.8, \"2\": 12.3, \"3\": 14.0, \"4\": 19.9, \"5\": 24.2, \"6\": 24.1, \"7\": 29.2, \"8\": 30.5, \"9\": 27.0, \"10\": 22.0, \"11\": 17.0, \"12\": 11.0}. Save to /root/hermes_research/rjtt_project/data/model_probs.json",
    "toolsets": [
      "terminal",
      "file"
    ]
  },
  {
    "name": "validator",
    "goal": "Compare model probabilities vs market probabilities, detect arbitrage edges > 8pp. Steps: 1. Read market and model probabilities. 2. Align by interval (lo/hi). 3. Compute absolute difference for each interval. 4. Find max absolute difference and corresponding interval. 5. If max_diff > 0.08 (8pp threshold), signal = true. 6. Get observed_temp from METAR. 7. Save validation_log.json",
    "context": "Working dir: /root/hermes_research/rjtt_project. Threshold: 0.08. Read from /root/hermes_research/rjtt_project/data/market_raw.json and /root/hermes_research/rjtt_project/data/model_probs.json. Save to /root/hermes_research/rjtt_project/data/validation_log.json",
    "toolsets": [
      "terminal",
      "file"
    ]
  },
  {
    "name": "report_generator",
    "goal": "Create HTML dashboard with Chart.js visualizations. Steps: 1. Read all data files (market_raw, osint_features, model_probs, validation_log). 2. Generate HTML with: Signal card (BUY/SELL/HOLD with edge), Market vs Model probability comparison chart, Feature importance chart, OSINT features summary, Historical signal performance table. 3. Save to ~/hermes_reports/rjtt/report_YYYYMMDD_HHMMSS.html",
    "context": "Working dir: /root/hermes_research/rjtt_project. Report dir: ~/hermes_reports/rjtt. Read all data from /root/hermes_research/rjtt_project/data/. Save HTML with timestamped filename.",
    "toolsets": [
      "terminal",
      "file"
    ]
  }
]

def get_task_configs():
    return TASKS

if __name__ == "__main__":
    print(json.dumps(TASKS, indent=2))
