#!/usr/bin/env python3
import subprocess
import sys
import os
import shutil
from datetime import datetime, timezone
import json

def run_script(script_name):
    """Run a Python script and return (success, stdout, stderr)."""
    script_path = os.path.join(os.path.dirname(__file__), script_name)
    if not os.path.exists(script_path):
        print(f"ERROR: Script not found: {script_path}")
        return False, "", f"Not found: {script_path}"
    print(f"--- Running {script_name} ---")
    result = subprocess.run(
        [sys.executable, script_path],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        print(f"FAILED: {script_name}")
        print("STDOUT:", result.stdout)
        print("STDERR:", result.stderr)
        return False, result.stdout, result.stderr
    else:
        print(f"SUCCESS: {script_name}")
        print(result.stdout)
        return True, result.stdout, result.stderr

def archive_run(base_dir, run_time):
    """Copy all artefacts of this run into a dated folder."""
    # run_time is a datetime with timezone info (we use UTC)
    folder_name = run_time.strftime("%Y%m%d_%H%M%S")
    archive_dir = os.path.join(base_dir, "archive", folder_name)
    os.makedirs(archive_dir, exist_ok=True)

    # Files we want to preserve
    to_copy = [
        ("data/market_raw.json", "market_raw.json"),
        ("data/osint_features.json", "osint_features.json"),
        ("data/model_probs.json", "model_probs.json"),
        ("data/validation_log.json", "validation_log.json"),
    ]
    for src, dst in to_copy:
        src_path = os.path.join(base_dir, src)
        dst_path = os.path.join(archive_dir, dst)
        if os.path.exists(src_path):
            shutil.copy2(src_path, dst_path)
        else:
            print(f"Warning: expected file not found – {src}")

    # HTML report (may be in the reports directory)
    report_dir = os.path.expanduser('~/hermes_reports/rjtt')
    if os.path.isdir(report_dir):
        for fname in os.listdir(report_dir):
            if fname.lower().endswith('.html'):
                src_path = os.path.join(report_dir, fname)
                dst_path = os.path.join(archive_dir, fname)
                shutil.copy2(src_path, dst_path)

    print(f"Archived run to {archive_dir}")

def main():
    start_time = datetime.now(timezone.utc)
    print(f"Starting RJTT temperature prediction pipeline at {start_time.isoformat()}")

    steps = [
        "market_collector.py",
        "osint_researcher.py",
        "model_builder.py",
        "validator.py",
        "report_generator.py",
    ]

    all_ok = True
    for script in steps:
        ok, out, err = run_script(script)
        if not ok:
            all_ok = False
            break  # stop on first failure

    end_time = datetime.now(timezone.utc)
    if all_ok:
        print(f"Pipeline completed successfully at {end_time.isoformat()}")
        # Archive the successful run
        archive_run(os.path.dirname(__file__), end_time)
        # After archiving, print a short summary for telegram delivery
        try:
            validation_path = os.path.join(os.path.dirname(__file__), 'data', 'validation_log.json')
            if os.path.exists(validation_path):
                with open(validation_path) as f:
                    val = json.load(f)
                signal = val.get('signal', False)
                observed = val.get('observed_temp', 'N/A')
                max_diff = val.get('max_abs_diff', 0.0)
                threshold = val.get('threshold', 0.08)
                signal_int = val.get('signal_interval')
                interval_str = ""
                if signal_int:
                    interval_str = f", Signal interval {signal_int['lo']}-{signal_int['hi']}℃ (M:{signal_int['market_p']:.3f}, Mod:{signal_int['model_p']:.3f})"
                # Determine latest report path
                report_dir = os.path.expanduser('~/hermes_reports/rjtt')
                latest_report = None
                if os.path.isdir(report_dir):
                    files = [f for f in os.listdir(report_dir) if f.lower().endswith('.html')]
                    if files:
                        files.sort(key=lambda f: os.path.getmtime(os.path.join(report_dir, f)), reverse=True)
                        latest_report = os.path.join(report_dir, files[0])
                summary = f"RJTT pipeline run {end_time.strftime('%Y-%m-%d %H:%M:%S')} UTC - Signal: {'YES' if signal else 'NO'} - Observed temp: {observed}°C - Max diff: {max_diff:.4f} (thr {threshold}){interval_str}"
                if latest_report:
                    summary += f" - Report: {latest_report}"
                print(summary)
            else:
                print("Validation log not found.")
        except Exception as e:
            print(f"Failed to create summary: {e}")
    else:
        print(f"Pipeline failed at {end_time.isoformat()}")
        sys.exit(1)

if __name__ == '__main__':
    main()