
#!/usr/bin/env python3
"""
Hermes Sub-Agent Orchestrator for RJTT Pipeline

This script runs INSIDE the Hermes agent environment and uses delegate_task
to spawn 5 specialized sub-agents for the pipeline.
"""
import json
import sys
from datetime import datetime, timezone

# Import delegate_task from hermes_tools (available in Hermes agent context)
try:
    from hermes_tools import delegate_task
except ImportError:
    # Fallback for testing outside Hermes
    delegate_task = None

BASE_DIR = '/root/hermes_research/rjtt_project'
DATA_DIR = BASE_DIR + '/data'

TASKS = [
    {
        'name': 'market_collector',
        'goal': 'Fetch Polymarket binary markets for target date and normalize to 13 temperature intervals (24-36°C). Steps: 1. Read target date from data/market_raw.json or use tomorrow. 2. Run polymarket-cli for 10 temperature thresholds (27-36°C). 3. Parse yes/no probabilities. 4. Normalize to 13 intervals (24-36°C in 1°C bins). 5. Save to data/market_raw.json with structure: {"date": "...", "intervals": [...], "source": "polymarket", "markets_fetched": N, "markets_total": 10}',
        'context': f'Working dir: {BASE_DIR}. Polymarket CLI at /home/.hermes/skills/research/polymarket/scripts/polymarket.py. Target date: tomorrow. Save to {DATA_DIR}/market_raw.json',
        'toolsets': ['terminal', 'file']
    },
    {
        'name': 'osint_researcher',
        'goal': 'Gather weather forecast and observational data for Tokyo Haneda (RJTT). Steps: 1. Read target date from data/market_raw.json. 2. Call Open-Meteo forecast API (35.55, 139.78) for hourly temps, cloud, precip, wind. 3. Calculate daily max temp forecast mean/std. 4. Call JMA METAR API for current observations at RJTT. 5. Compute sea breeze flag. 6. Add time features (month, weekday, day). 7. Save to data/osint_features.json with all 12+ features',
        'context': f'Working dir: {BASE_DIR}. Read target date from {DATA_DIR}/market_raw.json. Open-Meteo: https://api.open-meteo.com/v1/forecast. JMA METAR: https://www.jma.go.jp/bosai/amedas/data/meta/. Save to {DATA_DIR}/osint_features.json',
        'toolsets': ['terminal', 'file', 'web']
    },
    {
        'name': 'model_builder',
        'goal': 'Load trained XGBoost + Isotonic model and predict temperature interval probabilities. Steps: 1. Read market intervals from data/market_raw.json. 2. Read features from data/osint_features.json. 3. Build 14-feature vector including fc_anomaly bias correction. 4. Load best_model.pkl (XGBoost + 6 Isotonic calibrators). 5. Apply calibration to get probabilities for mapped classes. 6. Expand to full 13-interval space. 7. Normalize and save to data/model_probs.json',
        'context': f'Working dir: {BASE_DIR}. Model at {BASE_DIR}/models/best_model.pkl. Monthly climatology: {{1:8.8, 2:12.3, 3:14.0, 4:19.9, 5:24.2, 6:24.1, 7:29.2, 8:30.5, 9:27.0, 10:22.0, 11:17.0, 12:11.0}}. Save to {DATA_DIR}/model_probs.json',
        'toolsets': ['terminal', 'file']
    },
    {
        'name': 'validator',
        'goal': 'Compare model probabilities vs market probabilities, detect arbitrage edges > 8pp. Steps: 1. Read market and model probabilities. 2. Align by interval (lo/hi). 3. Compute absolute difference for each interval. 4. Find max absolute difference and corresponding interval. 5. If max_diff > 0.08 (8pp threshold), signal = true. 6. Get observed_temp from METAR. 7. Save validation_log.json',
        'context': f'Working dir: {BASE_DIR}. Threshold: 0.08. Read from {DATA_DIR}/market_raw.json and {DATA_DIR}/model_probs.json. Save to {DATA_DIR}/validation_log.json',
        'toolsets': ['terminal', 'file']
    },
    {
        'name': 'report_generator',
        'goal': 'Create HTML dashboard with Chart.js visualizations. Steps: 1. Read all data files (market_raw, osint_features, model_probs, validation_log). 2. Generate HTML with: Signal card (BUY/SELL/HOLD with edge), Market vs Model probability comparison chart, Feature importance chart, OSINT features summary, Historical signal performance table. 3. Save to ~/hermes_reports/rjtt/report_YYYYMMDD_HHMMSS.html',
        'context': f'Working dir: {BASE_DIR}. Report dir: ~/hermes_reports/rjtt. Read all data from {DATA_DIR}/. Save HTML with timestamped filename.',
        'toolsets': ['terminal', 'file']
    }
]

def run_delegated_pipeline():
    """Run the full pipeline using Hermes delegate_task for each stage."""
    if delegate_task is None:
        print('ERROR: delegate_task not available. Run this inside Hermes agent.')
        return False
    
    print(f'Starting RJTT delegated pipeline at {datetime.now(timezone.utc).isoformat()}')
    
    for task in TASKS:
        print(f'
=== Delegating: {task["name"]} ===')
        try:
            result = delegate_task(
                goal=task['goal'],
                context=task['context'],
                toolsets=task['toolsets'],
                role='leaf'
            )
            print(f'Result: {result}')
            # Check if sub-agent succeeded
            if isinstance(result, list) and len(result) > 0:
                status = result[0].get('status', 'unknown')
                if status != 'completed':
                    print(f'FAILED: {task["name"]} returned status {status}')
                    return False
            print(f'SUCCESS: {task["name"]}')
        except Exception as e:
            print(f'FAILED: {task["name"]} - {e}')
            return False
    
    # Archive and summary (run in main agent)
    print('
=== Archiving run ===')
    import subprocess
    subprocess.run([sys.executable, BASE_DIR + '/orchestrator_delegated.py'], cwd=BASE_DIR, check=False)
    
    return True

if __name__ == '__main__':
    success = run_delegated_pipeline()
    sys.exit(0 if success else 1)
