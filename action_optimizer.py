#!/usr/bin/env python3
"""
Action Optimizer for RJTT Pipeline

After each validation run, analyze results and propose/execute improvement actions.
Optimization target: Directional Accuracy & Top Interval Hit Rate
"""
import json, os, subprocess, sys
from datetime import datetime, timedelta
from pathlib import Path
import numpy as np

BASE_DIR = Path('/root/hermes_research/rjtt_project')
DATA_DIR = BASE_DIR / 'data'
ARCHIVE_DIR = BASE_DIR / 'archive'
MODEL_DIR = BASE_DIR / 'models'

ACTION_LOG = BASE_DIR / 'action_log.jsonl'

def load_recent_performance(window_days=7):
    logs = []
    cutoff = datetime.now() - timedelta(days=window_days)
    for folder in sorted(ARCHIVE_DIR.iterdir()):
        if not folder.is_dir(): continue
        try:
            folder_time = datetime.strptime(folder.name, '%Y%m%d_%H%M%S')
        except: continue
        if folder_time < cutoff: continue
        val_path = folder / 'validation_log.json'
        if not val_path.exists(): continue
        try:
            val = json.load(open(val_path))
            val['folder_time'] = folder_time.isoformat()
            logs.append(val)
        except: pass
    return logs

def analyze_performance(logs):
    if not logs:
        return {'issue': 'no_data', 'actions': []}
    
    signals = [l for l in logs if l.get('signal', False)]
    total_runs = len(logs)
    signal_rate = len(signals) / total_runs if total_runs > 0 else 0
    max_diffs = [l.get('max_abs_diff', 0) for l in signals]
    avg_max_diff = np.mean(max_diffs) if max_diffs else 0
    
    signal_intervals = []
    for l in signals:
        si = l.get('signal_interval')
        if si:
            signal_intervals.append(f"{si['lo']}-{si['hi']}")
    
    from collections import Counter
    interval_counts = Counter(signal_intervals)
    
    issues = []
    actions = []
    
    if signal_rate < 0.2:
        issues.append('low_signal_rate')
        actions.append({
            'id': 'lower_threshold',
            'description': 'Lower signal threshold from 0.08 to 0.05',
            'param': 'threshold', 'current': 0.08, 'proposed': 0.05,
            'expected': 'More signals, but may increase false positives'
        })
    
    if interval_counts:
        top_interval, count = interval_counts.most_common(1)[0]
        if count / len(signals) > 0.7:
            issues.append('overconfident_signals')
            actions.append({
                'id': 'add_uncertainty',
                'description': 'Add temperature-dependent noise to model probs',
                'param': 'model_uncertainty', 'current': 0.0, 'proposed': 0.1,
                'expected': 'Reduce overconfidence, improve calibration'
            })
            actions.append({
                'id': 'bias_correction_strength',
                'description': 'Increase fc_anomaly weight in feature vector (immediate)',
                'param': 'bias_feature_weight', 'current': 1.0, 'proposed': 2.0,
                'expected': 'Better capture forecast bias'
            })
            actions.append({
                'id': 'retrain_with_augmentation',
                'description': 'Retrain with feature noise augmentation (next retrain)',
                'param': 'retrain_augment', 'current': False, 'proposed': True,
                'expected': 'Force model to learn diverse patterns'
            })
    
    if len(interval_counts) == 1:
        issues.append('fixed_prediction')
        actions.append({
            'id': 'retrain_with_augmentation',
            'description': 'Retrain with feature noise augmentation',
            'param': 'retrain_augment', 'current': False, 'proposed': True,
            'expected': 'Force model to learn diverse patterns'
        })
    
    if avg_max_diff > 0.3:
        issues.append('large_gap')
        actions.append({
            'id': 'bias_correction_strength',
            'description': 'Increase fc_anomaly weight in feature vector',
            'param': 'bias_feature_weight', 'current': 1.0, 'proposed': 2.0,
            'expected': 'Better capture forecast bias'
        })
    
    return {
        'timestamp': datetime.now().isoformat(),
        'total_runs': total_runs,
        'signal_rate': signal_rate,
        'avg_max_diff': avg_max_diff,
        'signal_intervals': dict(interval_counts),
        'issues': issues,
        'actions': actions
    }

def select_best_action(analysis):
    if not analysis['actions']:
        return None
    
    # Priority: practical actions first (no rewrite needed)
    priority = {
        'add_uncertainty': 1,           # Quick, no retrain
        'bias_correction_strength': 2,  # Quick feature weight change
        'lower_threshold': 3,           # Simple threshold change
        'retrain_with_augmentation': 4, # Needs retrain
        'ensemble_models': 5            # Needs rewrite
    }
    
    sorted_actions = sorted(analysis['actions'], 
                          key=lambda a: priority.get(a['id'], 99))
    return sorted_actions[0] if sorted_actions else None

def execute_action(action):
    action_id = action['id']
    result = {'action_id': action_id, 'timestamp': datetime.now().isoformat(), 'success': False, 'details': ''}
    
    try:
        if action_id == 'lower_threshold':
            validator_path = BASE_DIR / 'validator.py'
            content = validator_path.read_text()
            new_content = content.replace('threshold = 0.08', f'threshold = {action["proposed"]}')
            validator_path.write_text(new_content)
            result['success'] = True
            result['details'] = f'Updated validator threshold to {action["proposed"]}'
            
        elif action_id == 'add_uncertainty':
            builder_path = BASE_DIR / 'model_builder.py'
            content = builder_path.read_text()
            if 'temperature_uncertainty' not in content:
                noise_code = '''
        # Add temperature-dependent uncertainty
        temp_uncertainty = 0.1 * (1 + abs(fc_mean - 28) / 5)
        probs_mapped += np.random.normal(0, temp_uncertainty, probs_mapped.shape)
        probs_mapped = np.clip(probs_mapped, 0, 1)
        probs_mapped = probs_mapped / probs_mapped.sum(axis=1, keepdims=True)
'''
                content = content.replace(
                    'probs_mapped = apply_calibration(model, feat_vec, calibrators, n_mapped_classes)',
                    f'probs_mapped = apply_calibration(model, feat_vec, calibrators, n_mapped_classes){noise_code}'
                )
                builder_path.write_text(content)
            result['success'] = True
            result['details'] = 'Added temperature-dependent uncertainty to model probs'
            
        elif action_id == 'bias_correction_strength':
            builder_path = BASE_DIR / 'model_builder.py'
            content = builder_path.read_text()
            if 'fc_anomaly * 2' not in content:
                content = content.replace(
                    'fc_anomaly,  # NEW: 14th feature',
                    'fc_anomaly * 2.0,  # Weighted bias correction'
                )
                builder_path.write_text(content)
                result['success'] = True
                result['details'] = 'Doubled fc_anomaly weight in feature vector'
            else:
                result['success'] = True
                result['details'] = 'fc_anomaly already weighted'
                
        elif action_id == 'retrain_with_augmentation':
            retrain_script = BASE_DIR / 'train_with_bias_fix_v2.py'
            content = retrain_script.read_text()
            if 'augmentation' not in content.lower():
                content = content.replace(
                    'X_train, X_cal, y_train, y_cal = train_test_split',
                    '''# Data augmentation: add noise to features
np.random.seed(42)
X_aug = np.vstack([X_train + np.random.normal(0, 0.01, X_train.shape) for _ in range(3)] + [X_train])
y_aug = np.hstack([y_train] * 4)
X_train, X_cal, y_train, y_cal = train_test_split(X_aug, y_aug, test_size=0.2, random_state=42)'''
                )
                retrain_script.write_text(content)
            proc = subprocess.run([sys.executable, str(retrain_script)], 
                                cwd=BASE_DIR, capture_output=True, text=True, timeout=300)
            result['success'] = proc.returncode == 0
            result['details'] = 'Retrain with augmentation ' + ('succeeded' if result['success'] else 'failed: ' + proc.stderr[:200])
            
    except Exception as e:
        result['success'] = False
        result['details'] = f'Error: {str(e)}'
    
    return result

def log_action(analysis, action, result):
    entry = {
        'timestamp': datetime.now().isoformat(),
        'analysis': analysis,
        'selected_action': action,
        'result': result
    }
    with open(ACTION_LOG, 'a') as f:
        f.write(json.dumps(entry) + '\n')

def main():
    print('=== ACTION OPTIMIZER ===')
    logs = load_recent_performance(7)
    print(f'Loaded {len(logs)} validation logs from last 7 days')
    
    analysis = analyze_performance(logs)
    print(f'Signal rate: {analysis["signal_rate"]:.2%}')
    print(f'Avg max diff: {analysis["avg_max_diff"]:.4f}')
    print(f'Signal intervals: {analysis["signal_intervals"]}')
    print(f'Issues: {analysis["issues"]}')
    
    action = select_best_action(analysis)
    if not action:
        print('No action needed')
        return True
    
    print(f'\nSelected: {action["id"]} - {action["description"]}')
    result = execute_action(action)
    print(f'Result: {"SUCCESS" if result["success"] else "FAILED"} - {result["details"]}')
    
    log_action(analysis, action, result)
    return result['success']

if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)
