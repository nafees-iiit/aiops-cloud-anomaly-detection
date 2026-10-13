"""
AIOps: Unsupervised Anomaly Detection on Real AWS Cloud Server Metrics
=====================================================================
Author : Mohammad Nafees (Ph.D. Scholar, ECE, MNNIT Allahabad)

Problem
-------
Cloud and edge servers are watched by monitoring systems (e.g. AWS CloudWatch)
that record CPU, disk, network and request counts every 5 minutes. Outages and
attacks usually show up first as unusual behaviour in these metrics. Nobody
labels this data in real time, so detection must be UNSUPERVISED.

Data
----
Numenta Anomaly Benchmark (NAB), 'realAWSCloudwatch' collection: 17 real AWS
metrics (EC2 CPU, disk writes, network in, load-balancer requests, RDS database
CPU) with 31 anomaly windows marked by experts. Labels are used ONLY for
evaluation, never for training or for choosing thresholds.

Detectors
---------
1. Static threshold   - distance from the normal mean (what basic alarms do)
2. Rolling z-score    - distance from the last 24 hours' behaviour
3. Isolation Forest   - ML on short-term shape features
4. Autoencoder (MLP)  - neural network that learns to rebuild normal 1-hour windows
"""

import os
import json
import time
import pickle
import urllib.request
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest
from sklearn.neural_network import MLPRegressor
from sklearn.exceptions import ConvergenceWarning
warnings.filterwarnings("ignore", category=ConvergenceWarning)   # keep the output readable

SEED = 42
PROBATION = 0.15      # first 15 % of each series = learning period (NAB convention)
DAY = 288             # 288 x 5 min = 24 hours
WIN = 12              # 12 x 5 min = 1 hour window for the autoencoder
BUDGETS = [0.1, 0.25, 0.5, 1.0, 2.0, 5.0]   # alert budget: % of points allowed to alarm
MAIN_BUDGET = 1.0
os.makedirs("data", exist_ok=True)
os.makedirs("results", exist_ok=True)

# ------------------------------------------------------------------
# 1. Download the data (only once)
# ------------------------------------------------------------------
NAB = "https://raw.githubusercontent.com/numenta/NAB/master/"
if not os.path.exists("data/labels.json"):
    urllib.request.urlretrieve(NAB + "labels/combined_windows.json", "data/labels.json")
labels = {k: v for k, v in json.load(open("data/labels.json")).items()
          if k.startswith("realAWSCloudwatch/")}
for key in labels:
    path = "data/" + os.path.basename(key)
    if not os.path.exists(path):
        print("Downloading", os.path.basename(key))
        urllib.request.urlretrieve(NAB + "data/" + key, path)

series = {}
for key, wins in labels.items():
    df = pd.read_csv("data/" + os.path.basename(key), parse_dates=["timestamp"])
    df["in_window"] = False
    for start, end in wins:
        df.loc[df.timestamp.between(pd.Timestamp(start), pd.Timestamp(end)), "in_window"] = True
    series[os.path.basename(key).replace(".csv", "")] = (df, [(pd.Timestamp(a), pd.Timestamp(b)) for a, b in wins])
print(f"Loaded {len(series)} AWS metrics, {sum(len(w) for _, w in series.values())} labelled anomaly windows, "
      f"{sum(len(d) for d, _ in series.values()):,} data points")


# ------------------------------------------------------------------
# 2. Four anomaly detectors -> one score per time step (higher = stranger)
# ------------------------------------------------------------------
def static_score(x, n_prob):
    mu, sd = x[:n_prob].mean(), x[:n_prob].std() + 1e-9
    return np.abs(x - mu) / sd

def rolling_z_score(x, n_prob):
    s = pd.Series(x)
    mu = s.rolling(DAY, min_periods=WIN).mean().shift(1)        # only the PAST is used
    sd = s.rolling(DAY, min_periods=WIN).std().shift(1)
    return np.nan_to_num(np.abs((s - mu) / (sd + 1e-9)).values)

def features(x, n_prob):
    """Short-term shape of the signal around each point (all causal)."""
    s = pd.Series((x - x[:n_prob].mean()) / (x[:n_prob].std() + 1e-9))
    f = pd.DataFrame({"value": s,
                      "diff": s.diff(),
                      "dev_1h": s - s.rolling(WIN, min_periods=1).mean(),
                      "std_1h": s.rolling(WIN, min_periods=2).std(),
                      "dev_1d": s - s.rolling(DAY, min_periods=WIN).mean().shift(1)})
    return f.fillna(0).values

def isolation_forest_score(x, n_prob):
    F = features(x, n_prob)
    iso = IsolationForest(n_estimators=200, random_state=SEED).fit(F[:n_prob])
    return -iso.score_samples(F), iso

def autoencoder_score(x, n_prob):
    z = (x - x[:n_prob].mean()) / (x[:n_prob].std() + 1e-9)
    W = np.lib.stride_tricks.sliding_window_view(np.r_[np.full(WIN - 1, z[0]), z], WIN)  # window ending at t
    W = W - W.mean(axis=1, keepdims=True)                     # learn the SHAPE, not the level
    ae = MLPRegressor(hidden_layer_sizes=(6,), max_iter=1000, random_state=SEED)
    ae.fit(W[:n_prob], W[:n_prob])                            # rebuild input through a 6-neuron bottleneck
    return ((ae.predict(W) - W) ** 2).mean(axis=1), ae

DETECTORS = ["Static threshold", "Rolling z-score (24 h)", "Isolation Forest", "Autoencoder (MLP)"]


# ------------------------------------------------------------------
# 3. Evaluation: alerts -> events -> detected windows / false alarms
# ------------------------------------------------------------------
def evaluate(df, wins, score, n_prob, budget_pct):
    sc = score[n_prob:]
    thr = np.quantile(sc, 1 - budget_pct / 100)               # unsupervised: fixed alert budget
    alert = np.zeros(len(df), bool); alert[n_prob:] = score[n_prob:] > thr
    # merge alerts closer than 15 minutes into one alert EVENT
    idx = np.flatnonzero(alert); events = []
    for i in idx:
        if events and i - events[-1][1] <= 3: events[-1][1] = i
        else: events.append([i, i])
    ts = df.timestamp.values
    t_prob = df.timestamp.iloc[n_prob]
    scored_wins = [(a, b) for a, b in wins if b >= t_prob]
    detected, delays = 0, []
    for a, b in scored_wins:
        hits = idx[(ts[idx] >= np.datetime64(a)) & (ts[idx] <= np.datetime64(b))]
        if len(hits):
            detected += 1
            centre = np.datetime64(a + (b - a) / 2)              # labelled anomaly = window centre
            delays.append((ts[hits[0]] - centre) / np.timedelta64(1, "m"))
    true_ev = sum(df.in_window.values[e0:e1 + 1].any() for e0, e1 in events)
    days = (df.timestamp.iloc[-1] - t_prob).total_seconds() / 86400
    return dict(windows=len(scored_wins), detected=detected, events=len(events), true_events=true_ev,
                false_events=len(events) - true_ev, days=days, delays=delays, alert=alert)


t0 = time.time()
scores, models = {}, {}
for name, (df, wins) in series.items():
    x = df.value.values.astype(float); n_prob = int(PROBATION * len(x))
    s_iso, iso = isolation_forest_score(x, n_prob)
    s_ae, ae = autoencoder_score(x, n_prob)
    scores[name] = {"Static threshold": static_score(x, n_prob),
                    "Rolling z-score (24 h)": rolling_z_score(x, n_prob),
                    "Isolation Forest": s_iso, "Autoencoder (MLP)": s_ae}
    models[name] = (iso, ae)
print(f"Scored all series with 4 detectors in {time.time()-t0:.1f} s")

curve_rows, main_rows, per_series = [], [], []
for det in DETECTORS:
    for b in BUDGETS:
        agg = dict(windows=0, detected=0, events=0, true_events=0, false_events=0, days=0, delays=[])
        for name, (df, wins) in series.items():
            r = evaluate(df, wins, scores[name][det], int(PROBATION * len(df)), b)
            for k in ["windows", "detected", "events", "true_events", "false_events", "days"]: agg[k] += r[k]
            agg["delays"] += r["delays"]
            if b == MAIN_BUDGET:
                per_series.append({"Metric": name, "Detector": det, "Windows": r["windows"], "Detected": r["detected"],
                                   "False alarm events": r["false_events"]})
        recall = agg["detected"] / agg["windows"]
        precision = agg["true_events"] / max(agg["events"], 1)
        row = {"Detector": det, "Alert budget (%)": b, "Windows detected": agg["detected"],
               "Total windows": agg["windows"], "Detection rate": recall, "Alert precision": precision,
               "F1": 2 * recall * precision / max(recall + precision, 1e-9),
               "False alarms per metric-day": agg["false_events"] / agg["days"],
               "Median alert time vs anomaly (min)": float(np.median(agg["delays"])) if agg["delays"] else np.nan}
        curve_rows.append(row)
        if b == MAIN_BUDGET: main_rows.append(row)

curve = pd.DataFrame(curve_rows); main = pd.DataFrame(main_rows)
curve.to_csv("results/budget_tradeoff.csv", index=False)
main.to_csv("results/detector_comparison.csv", index=False)
pd.DataFrame(per_series).to_csv("results/per_metric_results.csv", index=False)
pd.set_option("display.width", 220)
print(f"\nRESULTS at an alert budget of {MAIN_BUDGET}% of points "
      f"({main['Total windows'].iloc[0]} scored anomaly windows in {len(series)} metrics):")
print(main.drop(columns=["Alert budget (%)"]).to_string(index=False, float_format=lambda v: f"{v:.3f}"))

# Edge-deployment cost of the two ML detectors (one metric)
iso, ae = models["ec2_cpu_utilization_5f5533"]
x = series["ec2_cpu_utilization_5f5533"][0].value.values.astype(float)
F = features(x, int(PROBATION * len(x)))
t1 = time.time(); iso.score_samples(F); t_iso = (time.time() - t1) / len(F)
print(f"\nEdge cost per metric: Isolation Forest {len(pickle.dumps(iso))/1024:.0f} KB, "
      f"{1e6*t_iso:.1f} us/point | Autoencoder {len(pickle.dumps(ae))/1024:.0f} KB")

# ------------------------------------------------------------------
# 4. Figures
# ------------------------------------------------------------------
colors = {"Static threshold": "tab:gray", "Rolling z-score (24 h)": "tab:green",
          "Isolation Forest": "tab:blue", "Autoencoder (MLP)": "tab:red"}
plt.figure(figsize=(7.2, 5))
for det in DETECTORS:
    c = curve[curve.Detector == det]
    plt.plot(c["False alarms per metric-day"], c["Detection rate"] * 100, "o-", color=colors[det], label=det)
    for _, r in c.iterrows():
        if r["Alert budget (%)"] in (0.25, 1.0, 5.0):
            plt.annotate(f"{r['Alert budget (%)']:g}%", (r["False alarms per metric-day"], r["Detection rate"] * 100),
                         fontsize=7, xytext=(3, -9), textcoords="offset points", color=colors[det])
plt.xscale("log"); plt.xlabel("False-alarm events per metric per day (log)")
plt.ylabel("Anomaly windows detected (%)")
plt.title("Detection vs false alarms for different alert budgets (labels = % of points)", fontsize=10)
plt.grid(True, which="both", alpha=0.3); plt.legend(fontsize=8)
plt.tight_layout(); plt.savefig("results/detection_vs_false_alarms.png", dpi=200); plt.close()

fig, ax = plt.subplots(1, 2, figsize=(10, 4))
ax[0].bar(range(4), main["Detection rate"] * 100, color=[colors[d] for d in DETECTORS])
ax[1].bar(range(4), main["Alert precision"] * 100, color=[colors[d] for d in DETECTORS])
for a, t in zip(ax, ["Anomaly windows detected (%)", "Alerts that were real incidents (%)"]):
    a.set_xticks(range(4)); a.set_xticklabels([d.replace(" (", "\n(") for d in DETECTORS], fontsize=8)
    a.set_title(t, fontsize=10); a.grid(axis="y", alpha=0.3)
    for c in a.containers: a.bar_label(c, fmt="%.0f", fontsize=8)
plt.suptitle(f"Detector comparison at {MAIN_BUDGET}% alert budget (17 real AWS metrics)", fontsize=11)
plt.tight_layout(); plt.savefig("results/detector_comparison.png", dpi=200); plt.close()

for name in ["ec2_cpu_utilization_fe7f93", "rds_cpu_utilization_e47b3b", "elb_request_count_8c0756"]:
    df, wins = series[name]; n_prob = int(PROBATION * len(df))
    fig, axs = plt.subplots(3, 1, figsize=(10, 6.2), sharex=True, gridspec_kw={"height_ratios": [2, 1, 1]})
    axs[0].plot(df.timestamp, df.value, lw=0.7, color="k")
    for a, b in wins:
        for axx in axs: axx.axvspan(a, b, color="orange", alpha=0.25)
    axs[0].axvspan(df.timestamp.iloc[0], df.timestamp.iloc[n_prob], color="gray", alpha=0.12)
    axs[0].set_title(f"{name}  (orange = expert-labelled anomaly window, grey = learning period)", fontsize=9)
    for axx, det in zip(axs[1:], ["Rolling z-score (24 h)", "Isolation Forest"]):
        r = evaluate(df, wins, scores[name][det], n_prob, MAIN_BUDGET)
        axx.plot(df.timestamp, scores[name][det], lw=0.6, color=colors[det])
        axx.scatter(df.timestamp[r["alert"]], scores[name][det][r["alert"]], s=8, color="red", zorder=3)
        axx.set_ylabel(det.split(" (")[0], fontsize=8)
    axs[-1].set_xlabel("Time (red dots = alerts)")
    plt.tight_layout(); plt.savefig(f"results/example_{name}.png", dpi=170); plt.close()
print("\nAll results saved in the 'results' folder.")
