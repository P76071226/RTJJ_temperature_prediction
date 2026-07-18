# RJTT Temperature Prediction Pipeline

**Tokyo Haneda (RJTT) highest temperature prediction for Polymarket arbitrage.**

Automated pipeline: market data -> OSINT weather features -> ML model -> validation -> HTML report -- running every 6 hours via Hermes cron on Ubuntu (proot-distro).

---

## System Architecture

ORCHESTRATOR (orchestrator.py) -- runs every 6h via cron
   |
   +-- MARKET COLLECTOR (market_collector.py)
   |     * polymarket-cli fetches 10 binary markets for target date
   |     * Builds 13-bin probability distribution
   |     * Output: data/market_raw.json
   |
   +-- OSINT RESEARCHER (osint_researcher.py)
   |     * Open-Meteo forecast API (temp, cloud, precip, wind)
   |     * JMA METAR observations (current temp)
   |     * Engineered features: sea-breeze flag, cloud^2, lagged error, interactions
   |     * Output: data/osint_features.json
   |
   +-- MODEL BUILDER (model_builder.py)
   |     * Loads models/logistic_model.pkl (multinomial LogisticRegression)
   |     * Predicts P(temp in bin) for 13 intervals
   |     * Fallback: heuristic normal-dist if model missing
   |     * Output: data/model_probs.json
   |
   +-- VALIDATOR (validator.py)
   |     * Computes max|model_prob - market_prob| per bin
   |     * Signal threshold: 0.08 (8 pp)
   |     * Records observed_temp from METAR
   |     * Output: data/validation_log.json
   |
   +-- REPORT GENERATOR (report_generator.py)
         * Jinja2 HTML report: market vs model probs, features, signal
         * Saves to /home/hermes_reports/rjtt/report_YYYYMMDD_HHMMSS.html
         * Orchestrator archives everything under archive/YYYYMMDD_HHMMSS/

## Repository Structure

RJTT_temperature_prediction/
|-- .gitignore
|-- README.md
|-- orchestrator.py
|-- market_collector.py
|-- osint_researcher.py
|-- model_builder.py
|-- validator.py
|-- report_generator.py
|-- train_logistic_model.py
|-- backtest_final.py
|-- backtest_results.csv
|-- models/
|   |-- logistic_model.pkl
|-- data/              # git-ignored runtime intermediates
|-- archive/           # git-ignored immutable run snapshots
|-- hermes_tools.py    # minimal stub for cron execution

---

## Installation & Setup

### Prerequisites
- Ubuntu (or proot-distro Ubuntu on Termux)
- Python 3.10+
- polymarket-cli (pip install polymarket-cli)
- scikit-learn, statsmodels, pandas, numpy, requests, jinja2

### Quick Start

1. Enter Ubuntu environment (Termux proot-distro):
   proot-distro login ubuntu

2. Activate venv:
   source ~/ml_env/bin/activate

3. Install dependencies:
   pip install scikit-learn statsmodels pandas numpy requests jinja2 polymarket-cli

4. Verify Polymarket CLI:
   polymarket-cli --help

5. Run once manually:
   cd /root/hermes_research/rjtt_project
   python orchestrator.py

6. View latest report:
   python -m http.server 8080 --directory /home/hermes_reports/rjtt
   # Open http://localhost:8080

### Requirements.txt
scikit-learn>=1.3
statsmodels>=0.14
pandas>=2.0
numpy>=1.24
requests>=2.31
jinja2>=3.1
polymarket-cli>=0.2

---

## Automated Execution (Cron)

The pipeline runs every 6 hours via Hermes cron job:

| Cron Job | Schedule | Command |
|----------|----------|---------|
| rjtt_temp_prediction | every 360m | proot-distro login ubuntu -- bash -c source
