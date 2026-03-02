#!python3
"""
Logistic regression model predicting fact recall probability.

Fits three models:
  M1: Baseline recall (calibration data, ~1700 obs)
  M2: + compaction (calibration + compaction data, ~3900 obs)
  M3: Strategy comparison (iterative data, ~960 obs)

Plus XGBoost comparison on M2 features.

Usage:
    python model_recall.py
    python model_recall.py --collect-only   # just export CSV, no model
"""

import json
import re
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

import statsmodels.api as sm
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import cross_val_score
from xgboost import XGBClassifier


# ============================================================================
# CONSTANTS
# ============================================================================

BASE_DIR = Path(".")
COMPACTION_MAP = {"C1": 5, "C2": 25, "C3": 50, "C4": 98}

CATEGORY_ORDER = [
    "single-session-user",
    "single-session-assistant",
    "knowledge-update",
    "single-session-preference",
    "temporal-reasoning",
    "multi-session",
]


# ============================================================================
# DATA COLLECTION
# ============================================================================

def load_fact_meta_v5(density, seed=42):
    """Load fact metadata from v5 context meta files."""
    metaFile = BASE_DIR / f"data/contexts/v5_R4/d{density}_seed{seed}_meta.json"
    if not metaFile.exists():
        return {}
    with open(metaFile) as f:
        meta = json.load(f)
    return {
        fact["fact_id"]: {
            "position_pct": fact["position_pct"],
            "category": fact["question_type"],
            "est_tokens": fact.get("est_tokens", 0),
            "n_turns": fact.get("n_turns", 0),
        }
        for fact in meta["facts"]
    }


def load_fact_meta_v6(convTokens, density=80, seed=42):
    """Load fact metadata from v6 conversation meta files."""
    convDir = BASE_DIR / "data/conversations/v6_R4"
    # Find closest meta file by token count
    candidates = list(convDir.glob(f"d{density}_*_seed{seed}_meta.json"))
    if not candidates:
        return {}
    best = None
    bestDelta = float("inf")
    for c in candidates:
        with open(c) as f:
            m = json.load(f)
        delta = abs(m["est_tokens"] - convTokens)
        if delta < bestDelta:
            bestDelta = delta
            best = m
    if best is None:
        return {}
    return {
        fact["fact_id"]: {
            "position_pct": fact["position_pct"],
            "category": fact["question_type"],
            "est_tokens": fact.get("est_tokens", 0),
            "n_turns": fact.get("n_turns", 0),
        }
        for fact in best["facts"]
    }


def extract_verdicts(judgFile):
    """Extract (fact_id, recalled) pairs from a judgment file."""
    with open(judgFile) as f:
        data = json.load(f)

    verdicts = []
    # Part C format: flat dict with "verdicts" list
    if "verdicts" in data and isinstance(data["verdicts"], list):
        for v in data["verdicts"]:
            verdicts.append((v["fact_id"], v["recalled"]))
    # Part A/B format: batches list
    elif "batches" in data:
        for batch in data["batches"]:
            for v in batch.get("verdicts", []):
                verdicts.append((v["fact_id"], v["recalled"]))
    return verdicts


def collect_calibration(runDir):
    """Collect per-fact outcomes from recall calibration (§3)."""
    rows = []
    judgDir = runDir / "judgments"
    if not judgDir.exists():
        return rows

    for judgFile in sorted(judgDir.glob("d*_bs*.json")):
        # Parse filename: d40_bs5.json
        m = re.match(r"d(\d+)_bs(\d+)\.json", judgFile.name)
        if not m:
            continue
        density = int(m.group(1))
        batchSize = int(m.group(2))

        factMeta = load_fact_meta_v5(density)
        for factId, recalled in extract_verdicts(judgFile):
            meta = factMeta.get(factId, {})
            rows.append({
                "fact_id": factId,
                "recalled": int(recalled),
                "position_pct": meta.get("position_pct", 50),
                "category": meta.get("category", "unknown"),
                "density": density,
                "batch_size": batchSize,
                "compaction_pct": 0,
                "strategy": "none",
                "conv_tokens": 190_000,
                "experiment": "calibration",
            })
    return rows


def collect_compaction(runDir):
    """Collect per-fact outcomes from single-pass compaction (§5)."""
    rows = []
    judgDir = runDir / "judgments"
    if not judgDir.exists():
        return rows

    for judgFile in sorted(judgDir.glob("d*_C*_bs*.json")):
        # Parse filename: d80_C2_bs5.json
        m = re.match(r"d(\d+)_(C\d+)_bs(\d+)\.json", judgFile.name)
        if not m:
            continue
        density = int(m.group(1))
        compLevel = m.group(2)
        batchSize = int(m.group(3))
        compactionPct = COMPACTION_MAP.get(compLevel, 0)

        factMeta = load_fact_meta_v5(density)
        for factId, recalled in extract_verdicts(judgFile):
            meta = factMeta.get(factId, {})
            rows.append({
                "fact_id": factId,
                "recalled": int(recalled),
                "position_pct": meta.get("position_pct", 50),
                "category": meta.get("category", "unknown"),
                "density": density,
                "batch_size": batchSize,
                "compaction_pct": compactionPct,
                "strategy": "none",
                "conv_tokens": 190_000,
                "experiment": "compaction",
            })
    return rows


def collect_strategies(runDirs):
    """Collect per-fact outcomes from iterative strategy benchmark (§6)."""
    rows = []
    for runDir in runDirs:
        # Get conv_tokens from config
        configFile = runDir / "config.json"
        if not configFile.exists():
            continue
        with open(configFile) as f:
            cfg = json.load(f)
        convTokens = cfg["conversation_tokens"]

        factMeta = load_fact_meta_v6(convTokens)
        judgDir = runDir / "judgments"
        if not judgDir.exists():
            continue

        for judgFile in sorted(judgDir.glob("S*_bs*.json")):
            m = re.match(r"(S\d+)_bs(\d+)\.json", judgFile.name)
            if not m:
                continue
            strategy = m.group(1)
            batchSize = int(m.group(2))

            for factId, recalled in extract_verdicts(judgFile):
                meta = factMeta.get(factId, {})
                rows.append({
                    "fact_id": factId,
                    "recalled": int(recalled),
                    "position_pct": meta.get("position_pct", 50),
                    "category": meta.get("category", "unknown"),
                    "density": 80,
                    "batch_size": batchSize,
                    "compaction_pct": 0,
                    "strategy": strategy,
                    "conv_tokens": convTokens,
                    "experiment": "strategy",
                })
    return rows


def collect_all():
    """Collect all data into a single DataFrame."""
    rows = []

    # Part A: calibration
    for d in sorted(BASE_DIR.glob("recall_v5_R4_*")):
        if (d / "judgments").exists():
            r = collect_calibration(d)
            print(f"  Calibration {d.name}: {len(r)} observations")
            rows.extend(r)
            break  # use first (main) run only

    # Part B: compaction
    for d in sorted(BASE_DIR.glob("compaction_v5_R4_*")):
        if (d / "judgments").exists():
            r = collect_compaction(d)
            print(f"  Compaction {d.name}: {len(r)} observations")
            rows.extend(r)
            break

    # Part C: strategies
    stratDirs = sorted(
        [d for d in BASE_DIR.glob("iterative_v6_R4_*")
         if (d / "summary.json").exists()],
        key=lambda d: d.name
    )
    r = collect_strategies(stratDirs)
    print(f"  Strategies ({len(stratDirs)} runs): {len(r)} observations")
    rows.extend(r)

    df = pd.DataFrame(rows)
    print(f"\n  Total: {len(df)} observations, {df['recalled'].sum()} recalled ({df['recalled'].mean()*100:.1f}%)")
    return df


# ============================================================================
# MODELS
# ============================================================================

def fit_model1(df):
    """Model 1: Baseline recall (calibration only)."""
    data = df[df["experiment"] == "calibration"].copy()
    print(f"\n{'='*60}")
    print(f"  MODEL 1: Baseline recall ({len(data)} obs)")
    print(f"{'='*60}")

    # Features
    data["log_density"] = np.log(data["density"])
    data["position_sq"] = data["position_pct"] ** 2

    # Category dummies (drop single-session-user as reference)
    catDummies = pd.get_dummies(data["category"], prefix="cat", drop_first=False)
    refCat = "cat_single-session-user"
    catCols = [c for c in catDummies.columns if c != refCat]
    data = pd.concat([data, catDummies[catCols]], axis=1)

    featureCols = ["log_density", "batch_size", "position_pct", "position_sq"] + catCols
    X = sm.add_constant(data[featureCols].astype(float))
    y = data["recalled"].astype(float)

    model = sm.Logit(y, X).fit(disp=0)
    print(model.summary2())

    yPred = model.predict(X)
    auc = roc_auc_score(y, yPred)
    print(f"\n  AUC: {auc:.4f}")
    print(f"  Pseudo-R²: {model.prsquared:.4f}")

    return model, data, featureCols, auc


def fit_model2(df):
    """Model 2: Baseline + compaction."""
    data = df[df["experiment"].isin(["calibration", "compaction"])].copy()
    print(f"\n{'='*60}")
    print(f"  MODEL 2: + Compaction ({len(data)} obs)")
    print(f"{'='*60}")

    data["log_density"] = np.log(data["density"])
    data["position_sq"] = data["position_pct"] ** 2
    data["compact_x_pos"] = data["compaction_pct"] * data["position_pct"] / 100

    catDummies = pd.get_dummies(data["category"], prefix="cat", drop_first=False)
    refCat = "cat_single-session-user"
    catCols = [c for c in catDummies.columns if c != refCat]
    data = pd.concat([data, catDummies[catCols]], axis=1)

    featureCols = ["log_density", "batch_size", "position_pct", "position_sq",
                   "compaction_pct", "compact_x_pos"] + catCols
    X = sm.add_constant(data[featureCols].astype(float))
    y = data["recalled"].astype(float)

    model = sm.Logit(y, X).fit(disp=0)
    print(model.summary2())

    yPred = model.predict(X)
    auc = roc_auc_score(y, yPred)
    print(f"\n  AUC: {auc:.4f}")
    print(f"  Pseudo-R²: {model.prsquared:.4f}")

    return model, data, featureCols, auc


def fit_model3(df):
    """Model 3: Strategy comparison (iterative data only)."""
    data = df[df["experiment"] == "strategy"].copy()
    if len(data) == 0:
        print("\n  MODEL 3: No strategy data available, skipping.")
        return None, None, None, None

    print(f"\n{'='*60}")
    print(f"  MODEL 3: Strategy comparison ({len(data)} obs)")
    print(f"{'='*60}")

    data["log_conv_tokens"] = np.log(data["conv_tokens"])
    data["position_sq"] = data["position_pct"] ** 2

    # Strategy dummies (drop S1 as reference = worst strategy)
    stratDummies = pd.get_dummies(data["strategy"], prefix="strat", drop_first=False)
    refStrat = "strat_S1"
    stratCols = [c for c in stratDummies.columns if c != refStrat]
    data = pd.concat([data, stratDummies[stratCols]], axis=1)

    catDummies = pd.get_dummies(data["category"], prefix="cat", drop_first=False)
    refCat = "cat_single-session-user"
    catCols = [c for c in catDummies.columns if c != refCat]
    data = pd.concat([data, catDummies[catCols]], axis=1)

    featureCols = ["log_conv_tokens", "position_pct", "position_sq"] + stratCols + catCols
    X = sm.add_constant(data[featureCols].astype(float))
    y = data["recalled"].astype(float)

    model = sm.Logit(y, X).fit(disp=0)
    print(model.summary2())

    yPred = model.predict(X)
    auc = roc_auc_score(y, yPred)
    print(f"\n  AUC: {auc:.4f}")
    print(f"  Pseudo-R²: {model.prsquared:.4f}")

    return model, data, featureCols, auc


# ============================================================================
# XGBOOST COMPARISON
# ============================================================================

def compare_xgboost(df):
    """Compare logistic regression with XGBoost on Model 2 features."""
    data = df[df["experiment"].isin(["calibration", "compaction"])].copy()
    print(f"\n{'='*60}")
    print(f"  XGBOOST COMPARISON ({len(data)} obs)")
    print(f"{'='*60}")

    data["log_density"] = np.log(data["density"])
    data["position_sq"] = data["position_pct"] ** 2
    data["compact_x_pos"] = data["compaction_pct"] * data["position_pct"] / 100

    # Encode category as integer
    catMap = {c: i for i, c in enumerate(CATEGORY_ORDER)}
    data["category_code"] = data["category"].map(catMap).fillna(-1).astype(int)

    featureCols = ["log_density", "batch_size", "position_pct", "position_sq",
                   "compaction_pct", "compact_x_pos", "category_code"]

    X = data[featureCols].values
    y = data["recalled"].values

    # Logistic regression (sklearn for fair comparison)
    from sklearn.linear_model import LogisticRegression
    logReg = LogisticRegression(max_iter=1000, random_state=42)
    logScores = cross_val_score(logReg, X, y, cv=5, scoring="roc_auc")

    # XGBoost
    xgb = XGBClassifier(
        n_estimators=100, max_depth=4, learning_rate=0.1,
        random_state=42, eval_metric="logloss",
        enable_categorical=False,
    )
    xgbScores = cross_val_score(xgb, X, y, cv=5, scoring="roc_auc")

    print(f"\n  Logistic Regression  5-fold AUC: {logScores.mean():.4f} ± {logScores.std():.4f}")
    print(f"  XGBoost              5-fold AUC: {xgbScores.mean():.4f} ± {xgbScores.std():.4f}")
    delta = xgbScores.mean() - logScores.mean()
    print(f"  Delta (XGB - LR): {delta:+.4f}")

    if abs(delta) < 0.02:
        print("  -> Similar performance. Logistic regression sufficient.")
    elif delta > 0.02:
        print("  -> XGBoost better. Consider adding interactions to logistic model.")
    else:
        print("  -> Logistic regression better (unusual). Check XGBoost hyperparams.")

    # Feature importance from XGBoost (fit on full data)
    xgb.fit(X, y)
    importances = dict(zip(featureCols, xgb.feature_importances_))
    print(f"\n  XGBoost feature importances:")
    for feat, imp in sorted(importances.items(), key=lambda x: -x[1]):
        print(f"    {feat:25s} {imp:.4f}")

    return logScores, xgbScores, importances


# ============================================================================
# PLOTS
# ============================================================================

def plot_calibration(models, outputDir):
    """Plot predicted probability vs observed recall rate."""
    fig, axes = plt.subplots(1, len(models), figsize=(5 * len(models), 5), sharey=True)
    if len(models) == 1:
        axes = [axes]

    for idx, (name, model, data, featureCols) in enumerate(models):
        ax = axes[idx]
        if model is None:
            ax.set_title(f"{name}\n(no data)")
            continue

        catDummies = [c for c in data.columns if c.startswith("cat_") or c.startswith("strat_")]
        allCols = [c for c in (["const"] + featureCols) if c in data.columns]
        if "const" not in data.columns:
            tmpX = sm.add_constant(data[featureCols].astype(float))
        else:
            tmpX = data[["const"] + featureCols].astype(float)
        yPred = model.predict(tmpX)
        yTrue = data["recalled"].values

        # Bin predictions into deciles
        nBins = 10
        bins = np.linspace(0, 1, nBins + 1)
        binCenters = []
        binRates = []
        binCounts = []
        for i in range(nBins):
            mask = (yPred >= bins[i]) & (yPred < bins[i + 1])
            if i == nBins - 1:
                mask = (yPred >= bins[i]) & (yPred <= bins[i + 1])
            if mask.sum() > 0:
                binCenters.append((bins[i] + bins[i + 1]) / 2)
                binRates.append(yTrue[mask].mean())
                binCounts.append(mask.sum())

        ax.plot([0, 1], [0, 1], "k--", alpha=0.3, label="Perfect calibration")
        ax.scatter(binCenters, binRates, s=[max(20, c / 2) for c in binCounts],
                   color="#3498db", edgecolor="white", zorder=3)
        ax.set_xlabel("Predicted P(recall)")
        ax.set_title(f"{name}\nAUC={roc_auc_score(yTrue, yPred):.3f}")
        ax.set_xlim(-0.05, 1.05)
        ax.set_ylim(-0.05, 1.05)
        ax.grid(True, alpha=0.2)

    axes[0].set_ylabel("Observed recall rate")
    fig.suptitle("Model Calibration", fontsize=14)
    fig.tight_layout()

    outPath = outputDir / "fig_m1_calibration.png"
    fig.savefig(outPath, dpi=150)
    plt.close(fig)
    print(f"  {outPath}")


def plot_coefficients(models, outputDir):
    """Forest plot of coefficients with 95% CI."""
    fig, axes = plt.subplots(1, len(models), figsize=(6 * len(models), 8))
    if len(models) == 1:
        axes = [axes]

    for idx, (name, model, data, featureCols) in enumerate(models):
        ax = axes[idx]
        if model is None:
            ax.set_title(f"{name}\n(no data)")
            continue

        params = model.params[1:]  # skip intercept
        confInt = model.conf_int().iloc[1:]
        pvals = model.pvalues[1:]
        names = params.index.tolist()

        # Clean names for display
        displayNames = []
        for n in names:
            n = n.replace("cat_single-session-", "cat:ss-")
            n = n.replace("cat_knowledge-update", "cat:know-upd")
            n = n.replace("cat_temporal-reasoning", "cat:temporal")
            n = n.replace("cat_multi-session", "cat:multi-s")
            n = n.replace("strat_", "strat:")
            displayNames.append(n)

        yPos = np.arange(len(names))
        colors = ["#e74c3c" if p < 0.001 else "#f39c12" if p < 0.05 else "#95a5a6"
                  for p in pvals]

        ax.barh(yPos, params.values, xerr=[params.values - confInt.iloc[:, 0].values,
                                            confInt.iloc[:, 1].values - params.values],
                color=colors, edgecolor="white", height=0.7, capsize=3)
        ax.set_yticks(yPos)
        ax.set_yticklabels(displayNames, fontsize=9)
        ax.axvline(0, color="black", linewidth=0.5)
        ax.set_xlabel("Coefficient (log-odds)")
        ax.set_title(name)
        ax.grid(True, alpha=0.2, axis="x")
        ax.invert_yaxis()

    fig.suptitle("Logistic Regression Coefficients (95% CI)\nRed: p<0.001, Orange: p<0.05, Grey: n.s.",
                 fontsize=12)
    fig.tight_layout()

    outPath = outputDir / "fig_m2_coefficients.png"
    fig.savefig(outPath, dpi=150)
    plt.close(fig)
    print(f"  {outPath}")


def plot_xgboost_comparison(logScores, xgbScores, outputDir):
    """Bar chart comparing AUC of logistic regression vs XGBoost."""
    fig, ax = plt.subplots(figsize=(6, 4))

    x = [0, 1]
    means = [logScores.mean(), xgbScores.mean()]
    stds = [logScores.std(), xgbScores.std()]
    colors = ["#3498db", "#2ecc71"]

    bars = ax.bar(x, means, yerr=stds, color=colors, edgecolor="white",
                  width=0.5, capsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels(["Logistic Regression", "XGBoost"])
    ax.set_ylabel("AUC (5-fold CV)")
    ax.set_title("Model Comparison: Logistic Regression vs XGBoost")
    ax.set_ylim(0.5, 1.0)
    ax.grid(True, alpha=0.2, axis="y")

    for bar, mean, std in zip(bars, means, stds):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + std + 0.01,
                f"{mean:.3f}±{std:.3f}", ha="center", fontsize=10)

    fig.tight_layout()
    outPath = outputDir / "fig_m3_xgboost_comparison.png"
    fig.savefig(outPath, dpi=150)
    plt.close(fig)
    print(f"  {outPath}")


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Recall prediction model")
    parser.add_argument("--collect-only", action="store_true",
                        help="Only collect data, export CSV, no model fitting")
    parser.add_argument("--output-dir", type=str, default="figures")
    args = parser.parse_args()

    outputDir = Path(args.output_dir)
    outputDir.mkdir(exist_ok=True)

    print("=" * 60)
    print("  RECALL PREDICTION MODEL")
    print("=" * 60)

    # Step 1: Collect data
    print("\n  Collecting data...")
    df = collect_all()

    csvPath = BASE_DIR / "recall_model_data.csv"
    df.to_csv(csvPath, index=False)
    print(f"  Exported: {csvPath} ({len(df)} rows)")

    if args.collect_only:
        print("\n  --collect-only: stopping here.")
        return

    # Step 2: Model 1 — Baseline
    m1Model, m1Data, m1Features, m1Auc = fit_model1(df)

    # Step 3: Model 2 — + Compaction
    m2Model, m2Data, m2Features, m2Auc = fit_model2(df)

    # Step 4: Model 3 — Strategies
    m3Model, m3Data, m3Features, m3Auc = fit_model3(df)

    # Step 5: XGBoost comparison
    logScores, xgbScores, importances = compare_xgboost(df)

    # Step 6: Plots
    print(f"\n  Generating figures...")
    modelList = [
        ("M1: Baseline", m1Model, m1Data, m1Features),
        ("M2: + Compaction", m2Model, m2Data, m2Features),
    ]
    if m3Model is not None:
        modelList.append(("M3: Strategies", m3Model, m3Data, m3Features))

    plot_calibration(modelList, outputDir)
    plot_coefficients(modelList, outputDir)
    plot_xgboost_comparison(logScores, xgbScores, outputDir)

    # Summary
    print(f"\n{'='*60}")
    print(f"  SUMMARY")
    print(f"{'='*60}")
    print(f"  M1 Baseline:    AUC={m1Auc:.4f}  pseudo-R²={m1Model.prsquared:.4f}")
    print(f"  M2 +Compaction: AUC={m2Auc:.4f}  pseudo-R²={m2Model.prsquared:.4f}")
    if m3Auc:
        print(f"  M3 Strategies:  AUC={m3Auc:.4f}  pseudo-R²={m3Model.prsquared:.4f}")
    print(f"  XGBoost 5-fold: AUC={xgbScores.mean():.4f} ± {xgbScores.std():.4f}")
    print(f"  LogReg  5-fold: AUC={logScores.mean():.4f} ± {logScores.std():.4f}")
    print(f"\n  Done!")


if __name__ == "__main__":
    main()
