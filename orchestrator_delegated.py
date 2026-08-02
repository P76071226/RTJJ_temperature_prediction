#!/usr/bin/env python3
import json, os, sys, shutil, subprocess
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, os.path.dirname(__file__))
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / 'data'
ARCHIVE_DIR = BASE_DIR / 'archive'
REPORT_DIR = Path.home() / 'hermes_reports' / 'rjtt'

def run_delegated_pipeline():
    print(f'Starting RJTT delegated pipeline at {datetime.now(timezone.utc).isoformat()}')
    scripts = ['market_collector.py', 'osint_researcher.py', 'model_builder.py', 'validator.py', 'report_generator.py']
    all_ok = True
    for script in scripts:
        script_path = BASE_DIR / script
        print(f'\n--- Running {script} ---')
        result = subprocess.run([sys.executable, str(script_path)], capture_output=True, text=True, cwd=BASE_DIR)
        if result.returncode != 0:
            print(f'FAILED: {script}')
            print('STDOUT:', result.stdout)
            print('STDERR:', result.stderr)
            all_ok = False
            break
        else:
            print(f'SUCCESS: {script}')
            print(result.stdout)
    if all_ok:
        archive_run(datetime.now(timezone.utc))
        print_summary()
        # Run action optimizer after successful pipeline
        run_action_optimizer()
    return all_ok

def run_action_optimizer():
    print('\n=== RUNNING ACTION OPTIMIZER ===')
    try:
        result = subprocess.run([sys.executable, str(BASE_DIR / 'action_optimizer.py')], 
                              capture_output=True, text=True, cwd=BASE_DIR, timeout=120)
        print(result.stdout)
        if result.stderr:
            print('STDERR:', result.stderr)
        if result.returncode == 0:
            print('Action optimizer completed successfully')
        else:
            print(f'Action optimizer exited with code {result.returncode}')
    except Exception as e:
        print(f'Action optimizer error: {e}')

def archive_run(run_time):
    folder_name = run_time.strftime('%Y%m%d_%H%M%S')
    archive_path = ARCHIVE_DIR / folder_name
    archive_path.mkdir(parents=True, exist_ok=True)
    for src, dst in [(DATA_DIR / 'market_raw.json', 'market_raw.json'), (DATA_DIR / 'osint_features.json', 'osint_features.json'), (DATA_DIR / 'model_probs.json', 'model_probs.json'), (DATA_DIR / 'validation_log.json', 'validation_log.json')]:
        if src.exists(): shutil.copy2(src, archive_path / dst)
        else: print(f'Warning: {src} not found')
    if REPORT_DIR.exists():
        html_files = sorted(REPORT_DIR.glob('*.html'), key=lambda f: f.stat().st_mtime, reverse=True)
        if html_files: shutil.copy2(html_files[0], archive_path / html_files[0].name)
    print(f'Archived run to {archive_path}')

def print_summary():
    try:
        with open(DATA_DIR / 'validation_log.json') as f: val = json.load(f)
        signal = val.get('signal', False)
        observed = val.get('observed_temp', 'N/A')
        max_diff = val.get('max_abs_diff', 0.0)
        threshold = val.get('threshold', 0.08)
        signal_int = val.get('signal_interval')
        interval_str = ''
        if signal_int: interval_str = f', Signal interval {signal_int["lo"]}-{signal_int["hi"]}℃ (M:{signal_int["market_p"]:.3f}, Mod:{signal_int["model_p"]:.3f})'
        latest_report = None
        if REPORT_DIR.exists():
            html_files = sorted(REPORT_DIR.glob('*.html'), key=lambda f: f.stat().st_mtime, reverse=True)
            if html_files: latest_report = html_files[0]
        summary = f'RJTT pipeline run {datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")} UTC - Signal: {"YES" if signal else "NO"} - Observed temp: {observed}°C - Max diff: {max_diff:.4f} (thr {threshold}){interval_str}'
        if latest_report: summary += f' - Report: {latest_report}'
        print(f'\n{summary}')
    except Exception as e: print(f'Failed to create summary: {e}')

if __name__ == '__main__':
    success = run_delegated_pipeline()
    sys.exit(0 if success else 1)
