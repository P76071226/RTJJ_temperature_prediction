#!/usr/bin/env python3
"""
Simple backtest for RJTT temperature prediction pipeline.
Computes Brier score, log score, RPS, directional accuracy, top-interval hit rate,
and simulated trade PnL based on signal rule.
"""

import json
import os
import csv
import math
from datetime import datetime
from pathlib import Path

# ------------------- USER SETTINGS -------------------
ARCHIVE_ROOT = Path("/root/hermes_research/rjtt_project/archive")
SIGNAL_THRESHOLD = 0.08          # 8 percentage points – same as in validator.py
MAX_SPREAD = 0.04                # 4 ¢ max bid‑ask spread (you can adjust)
# ----------------------------------------------------

def normal_cdf(x, mu=0.0, sigma=1.0):
    """CDF of N(mu, sigma^2) using math.erf."""
    return 0.5 * (1 + math.erf((x - mu) / (sigma * math.sqrt(2))))

def interval_prob_from_normal(lo, hi, mu, sigma):
    """P(lo ≤ X ≤ hi) for X ~ N(mu, sigma^2)."""
    return normal_cdf(hi, mu, sigma) - normal_cdf(lo, mu, sigma)

def load_json(path):
    with open(path, 'r') as f:
        return json.load(f)

def compute_scores(market_intervals, model_intervals, actual_temp):
    """
    market_intervals, model_intervals: list of dicts with keys lo, hi, p
    actual_temp: float or int (observed temperature in °C)
    Returns a dict of scores (including extra fields for debugging).
    """
    # Ensure intervals are aligned and sorted by lo
    mkt = sorted(market_intervals, key=lambda x: x['lo'])
    mdl = sorted(model_intervals, key=lambda x: x['lo'])
    assert len(mkt) == len(mdl), "Mismatch in number of intervals"

    # Build arrays for vectorised formulas
    lo = [i['lo'] for i in mkt]
    hi = [i['hi'] for i in mkt]
    p_mkt = [i['p'] for i in mkt]
    p_mdl = [i['p'] for i in mdl]

    # Find which interval contains the actual temperature
    actual_idx = None
    for idx, (l, h) in enumerate(zip(lo, hi)):
        if l <= actual_temp < h:
            actual_idx = idx
            break
    if actual_idx is None:
        # If actual equals the upper bound of the last interval, treat it as inside
        if abs(actual_temp - hi[-1]) < 1e-9:
            actual_idx = len(lo) - 1
        else:
            # Fallback: put into the nearest interval (should not happen with decent binning)
            actual_idx = min(range(len(lo)), key=lambda i: abs(actual_temp - (lo[i]+hi[i])/2))

    # ----- Brier score (multi‑category) -----
    # For each interval, treat it as a binary outcome (1 if actual in that interval else 0)
    brier = sum((p_mdl[i] - (1 if i == actual_idx else 0))**2 for i in range(len(p_mdl))) / len(p_mdl)

    # ----- Log score -----
    # Avoid log(0) by clipping
    eps = 1e-12
    log_score = -math.log(max(p_mdl[actual_idx], eps))

    # ----- Ranked Probability Score (RPS) -----
    # Compute cumulative probabilities
    cum_mkt = []
    cum_mdl = []
    s = 0.0
    for p in p_mkt:
        s += p
        cum_mkt.append(s)
    s = 0.0
    for p in p_mdl:
        s += p
        cum_mdl.append(s)
    # Number of categories - 1
    rps = sum((cum_mdl[i] - cum_mkt[i])**2 for i in range(len(cum_mkt)-1)) / (len(cum_mkt)-1)

    # ----- Directional accuracy -----
    # Use the interval mid‑point as a proxy for the expected temperature
    def mid(i):
        return (lo[i] + hi[i]) / 2.0
    mkt_mid = sum(p_mkt[i] * mid(i) for i in range(len(p_mkt)))   # expected temp under market
    mdl_mid = sum(p_mdl[i] * mid(i) for i in range(len(p_mdl)))   # expected temp under model
    # actual direction relative to market mid‑point
    direction_correct = 1 if ((actual_temp - mkt_mid) * (mdl_mid - mkt_mid) > 0) else 0

    # ----- Top‑interval hit rate -----
    top_idx = p_mdl.index(max(p_mdl))
    top_hit = 1 if top_idx == actual_idx else 0

    return {
        "brier": brier,
        "log_score": log_score,
        "rps": rps,
        "direction_correct": direction_correct,
        "top_hit": top_hit,
        "actual_temp": actual_temp,
        "mkt_mid": mkt_mid,
        "mdl_mid": mdl_mid,
        # The following are kept for debugging but not written to CSV
        "market_probs": p_mkt.copy(),
        "model_probs": p_mdl.copy(),
    }

def simulated_trade(market_intervals, model_intervals, actual_temp, spread_estimate=0.02):
    """
    Very simple P&L model:
      - If |model_prob - market_prob| ≥ SIGNAL_THRESHOLD AND spread ≤ MAX_SPREAD → take a position.
      - Position size = edge (in probability points).
      - Buy if model_prob > market_prob (i.e. we think the interval is under‑priced),
        sell otherwise.
      - Payoff = +edge if the actual temperature falls in that interval, –edge otherwise.
    Returns (pnl, took_trade, edge_used, side).
    """
    # Use the same interval ordering as in compute_scores
    mkt = sorted(market_intervals, key=lambda x: x['lo'])
    mdl = sorted(model_intervals, key=lambda x: x['lo'])
    p_mkt = [i['p'] for i in mkt]
    p_mdl = [i['p'] for i in mdl]

    best_edge = 0.0
    best_idx = -1
    best_side = None   # +1 for buy (model>market), -1 for sell
    for i in range(len(p_mkt)):
        edge = p_mdl[i] - p_mkt[i]
        abs_edge = abs(edge)
        if abs_edge > best_edge:
            best_edge = abs_edge
            best_idx = i
            best_side = 1 if edge > 0 else -1

    took = (best_edge >= SIGNAL_THRESHOLD) and (spread_estimate <= MAX_SPREAD)
    if not took:
        return 0.0, False, 0.0, 0

    # Determine if actual temperature falls in the best interval
    actual_in_best = False
    for i, iv in enumerate(market_intervals):
        if iv['lo'] <= actual_temp < iv['hi']:
            actual_in_best = (i == best_idx)
            break
    if abs(actual_temp - market_intervals[-1]['hi']) < 1e-9 and best_idx == len(market_intervals)-1:
        actual_in_best = True

    pnl = best_edge if actual_in_best else -best_edge
    return pnl, took, best_edge, best_side

def main():
    if not ARCHIVE_ROOT.is_dir():
        print(f"Archive directory not found: {ARCHIVE_ROOT}")
        print("Run the pipeline a few times first so that data gets archived.")
        return

    results = []   # list of dicts for CSV
    folders = sorted(ARCHIVE_ROOT.iterdir(), key=lambda p: p.name)   # chronological

    for folder in folders:
        if not folder.is_dir():
            continue
        # Expect files: market_raw.json, osint_features.json, model_probs.json, validation_log.json
        try:
            market = load_json(folder / "market_raw.json")
            model  = load_json(folder / "model_probs.json")
            validation = load_json(folder / "validation_log.json")
        except Exception as e:
            print(f"Skipping {folder.name}: missing or unreadable JSON ({e})")
            continue

        date_str = market.get("date")
        if not date_str:
            print(f"Skipping {folder.name}: no date in market_raw.json")
            continue

        # ------------------- Get observed temperature for the target date -------------------
        # We will use the observed_temp stored in validation_log (which is from METAR, not ideal)
        # For a proper backtest we should fetch the actual daily max temperature from Open-Meteo,
        # but to keep the backtest self-contained we use the stored observed_temp.
        # If you have the actual observed temperature elsewhere, you can replace this.
        validation_data = load_json(folder / "validation_log.json")
        obs_temp = validation_data.get("observed_temp")
        if obs_temp is None:
            # fallback: use METAR temperature from osint_features
            feat = load_json(folder / "osint_features.json")
            obs_temp = feat.get("metar_temp_c", 0.0)
            print(f"[WARN] Using METAR temp as proxy for {date_str}: {obs_temp}°C")
        else:
            obs_temp = float(obs_temp)

        # ------------------- Compute scores -------------------
        scores = compute_scores(
            market["intervals"],
            model["intervals"],
            obs_temp,
        )
        # Build a dict with only the fields we want to write to CSV
        csv_row = {
            "date": date_str,
            "folder": folder.name,
            "brier": scores["brier"],
            "log_score": scores["log_score"],
            "rps": scores["rps"],
            "direction_correct": scores["direction_correct"],
            "top_hit": scores["top_hit"],
            "actual_temp": scores["actual_temp"],
            "mkt_mid": scores["mkt_mid"],
            "mdl_mid": scores["mdl_mid"],
        }
        scores["date"] = date_str
        scores["folder"] = folder.name

        # ------------------- Optional simulated trade -------------------
        # Estimate a bid‑ask spread; we don't have it in the archive, so we use a fixed
        # representative value (you can change MAX_SPREAD above or improve this later).
        spread_est = 0.02   # 2 ¢ as a typical spread for liquid RJTT markets
        pnl, took, edge, side = simulated_trade(
            market["intervals"],
            model["intervals"],
            obs_temp,
            spread_estimate=spread_est,
        )
        csv_row["took_trade"] = took
        csv_row["pnl"] = pnl
        csv_row["edge_used"] = edge
        csv_row["trade_side"] = side   # +1 = buy model > market, -1 = sell

        results.append(csv_row)
        print(f"[{date_str}] Brier={csv_row['brier']:.4f}  Log={csv_row['log_score']:.3f}  "
              f"RPS={csv_row['rps']:.4f}  DirAcc={csv_row['direction_correct']}  "
              f"TopHit={csv_row['top_hit']}  Trade={took}  PnL={pnl:.3f}")

    # ------------------- Write CSV summary -------------------
    csv_path = Path("/root/hermes_research/rjtt_project/backtest_results.csv")
    fieldnames = [
        "date", "folder", "brier", "log_score", "rps",
        "direction_correct", "top_hit",
        "actual_temp", "mkt_mid", "mdl_mid",
        "took_trade", "pnl", "edge_used", "trade_side",
    ]
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in results:
            writer.writerow(r)

    print(f"\nBack‑test complete. Results written to: {csv_path}")
    if results:
        # Print overall averages
        N = len(results)
        avg_brier = sum(r["brier"] for r in results) / N
        avg_log   = sum(r["log_score"] for r in results) / N
        avg_rps   = sum(r["rps"] for r in results) / N
        dir_acc   = sum(r["direction_correct"] for r in results) / N
        top_hit   = sum(r["top_hit"] for r in results) / N
        trade_cnt = sum(1 for r in results if r["took_trade"])
        avg_pnl   = sum(r["pnl"] for r in results if r["took_trade"]) / max(trade_cnt,1)
        print(f"--- Averages over {N} days ---")
        print(f"Brier score      : {avg_brier:.4f} (lower is better)")
        print(f"Log score        : {avg_log:.3f}  (lower is better)")
        print(f"RPS              : {avg_rps:.4f} (lower is better)")
        print(f"Directional acc. : {dir_acc:.2%}")
        print(f"Top‑interval hit : {top_hit:.2%}")
        print(f"Trades taken     : {trade_cnt}/{N}")
        print(f"Avg PnL per trade: {avg_pnl:.3f} (positive → profitable)")

if __name__ == '__main__':
    main()