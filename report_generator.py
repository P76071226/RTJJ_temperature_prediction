#!/usr/bin/env python3
import json
import os
import sys
import subprocess
from datetime import datetime

def try_install(package):
    try:
        __import__(package)
    except ImportError:
        print(f"Installing {package}...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])

def main():
    # Ensure we can import send_message from hermes_tools
    try:
        from hermes_tools import send_message
        HAVE_SEND_MESSAGE = True
    except Exception as e:
        print(f"Could not import send_message from hermes_tools: {e}")
        HAVE_SEND_MESSAGE = False

    # Load data
    market_path = os.path.join('data', 'market_raw.json')
    model_path = os.path.join('data', 'model_probs.json')
    features_path = os.path.join('data', 'osint_features.json')
    validation_path = os.path.join('data', 'validation_log.json')
    for p, name in [(market_path, 'market'), (model_path, 'model'), (features_path, 'features'), (validation_path, 'validation')]:
        if not os.path.exists(p):
            print(f"{name} data not found at {p}. Run preceding steps.")
            sys.exit(1)

    with open(market_path) as f:
        market_data = json.load(f)
    with open(model_path) as f:
        model_data = json.load(f)
    with open(features_path) as f:
        features = json.load(f)
    with open(validation_path) as f:
        validation = json.load(f)

    date_str = market_data.get('date', 'unknown')
    # Build HTML report
    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>RJTT Temperature Prediction Report – {date_str}</title>
<style>
body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; margin: 40px; background: #f8f9fa; color: #212529; }}
h1, h2 {{ color: #0d6efd; }}
.container {{ max-width: 900px; margin: auto; background: white; padding: 20px; box-shadow: 0 0 10px rgba(0,0,0,0.1); border-radius: 8px; }}
table {{ width: 100%; border-collapse: collapse; margin: 15px 0; }}
th, td {{ padding: 8px 12px; border: 1px solid #dee2e6; text-align: left; }}
th {{ background-color: #e9ecef; }}
tr:nth-child(even) {{ background-color: #f8f9fa; }}
.signal {{ display: inline-block; padding: 6px 12px; background: {'#d4edda' if validation.get('signal') else '#f8d7da'}; color: {'#155724' if validation.get('signal') else '#721c24'}; border-radius: 4px; font-weight: bold; margin: 10px 0; }}
.note {{ font-size: 0.9em; color: #6c757d; }}
</style>
</head>
<body>
<div class="container">
<h1>RJTT Temperature Prediction Report</h1>
<p><strong>Date:</strong> {date_str}</p>
<p><strong>Generated:</strong> {datetime.utcnow().isoformat()}Z</p>

<div class="signal">
{"🚨 SIGNAL DETECTED: Model vs Market edge exceeds threshold" if validation.get('signal') else "✅ No significant signal"}
</div>

<h2>Market Implied Probabilities</h2>
<table>
<thead><tr><th>Interval (℃)</th><th>Implied Probability</th></tr></thead>
<tbody>
"""
    for interval in market_data.get('intervals', []):
        lo = interval['lo']
        hi = interval['hi']
        p = interval.get('p', 0)
        html += f"<tr><td>{lo}–{hi}</td><td>{p:.3f}</td></tr>\n"
    html += "</tbody></table>\n"

    html += "<h2>Model Predicted Probabilities</h2>\n<table>\n<thead><tr><th>Interval (℃)</th><th>Model Probability</th></tr></thead>\n<tbody>\n"
    for interval in model_data.get('intervals', []):
        lo = interval['lo']
        hi = interval['hi']
        p = interval.get('p', 0)
        html += f"<tr><td>{lo}–{hi}</td><td>{p:.3f}</td></tr>\n"
    html += "</tbody></table>\n"

    html += "<h2>Key OSINT Features</h2>\n<table>\n<thead><tr><th>Feature</th><th>Value</th></tr></thead>\n<tbody>\n"
    # Select a subset of features to display
    display_features = [
        ('temp_forecast_mean', 'Forecast Mean Temperature (℃)'),
        ('temp_forecast_std', 'Forecast Std Dev (℃)'),
        ('cloud_cover_pct', 'Cloud Cover (%)'),
        ('sea_breeze_flag', 'Sea Breeze Flag (1=yes)'),
        ('precip_forecast_mm', 'Forecast Precipitation (mm)'),
        ('wind_speed_avg', 'Average Wind Speed (m/s)'),
        ('metar_temp_c', 'METAR Temperature (℃)'),
        ('metar_dewpoint_c', 'METAR Dewpoint (℃)'),
        ('month', 'Month'),
        ('weekday', 'Weekday (0=Mon)'),
    ]
    for key, label in display_features:
        val = features.get(key, 'N/A')
        if isinstance(val, float):
            val_disp = f"{val:.2f}"
        else:
            val_disp = str(val)
        html += f"<tr><td>{label}</td><td>{val_disp}</td></tr>\n"
    html += "</tbody></table>\n"

    html += "<h2>Validation Results</h2>\n<table>\n<thead><tr><th>Metric</th><th>Value</th></tr></thead>\n<tbody>\n"
    html += f"<tr><td>Max Absolute Difference (Model vs Market)</td><td>{validation.get('max_abs_diff', 0):.4f}</td></tr>\n"
    html += f"<tr><td>Threshold for Signal</td><td>{validation.get('threshold', 0):.4f}</td></tr>\n"
    html += f"<tr><td>Signal Present</td><td>{'YES' if validation.get('signal') else 'NO'}</td></tr>\n"
    sig_int = validation.get('signal_interval')
    if sig_int:
        html += f"<tr><td>Signal Interval</td><td>{sig_int['lo']}–{sig_int['hi']}℃ (Market {sig_int['market_p']:.3f}, Model {sig_int['model_p']:.3f}, Diff {sig_int['diff']:.4f})</td></tr>\n"
    html += "</tbody></table>\n"

    html += "<div class=\"note\">"
    html += "This report is generated automatically by the RJTT temperature prediction pipeline. "
    html += "Data sources: Polymarket (market implied probabilities), Open-Meteo (forecast), NOAA METAR (observations). "
    html += "Model: Advanced regression model (if pretrained models available) else heuristic. "
    html += "Signal threshold: 0.08 (8 percentage points). "
    html += "</div>"
    html += "</div>"
    html += "</body></html>"

    # Ensure report directory exists
    report_dir = os.path.expanduser('~/hermes_reports/rjtt')
    os.makedirs(report_dir, exist_ok=True)
    timestamp = datetime.utcnow().strftime('%Y%m%d_%H%M%S')
    report_path = os.path.join(report_dir, f'report_{timestamp}.html')
    with open(report_path, 'w') as f:
        f.write(html)
    print(f"Report saved to {report_path}")

    # If signal is True, we DO NOT send via Telegram here; the orchestrator will handle it.
    if validation.get('signal'):
        print("Signal detected; Telegram notification will be sent by orchestrator.")
    else:
        print("No signal; report saved only.")

if __name__ == '__main__':
    main()