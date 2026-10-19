"""Train and evaluate the fraud model.

    python -m snare.train            # full run, writes models/
    python -m snare.train --quick    # small run for a fast check

Evaluation is out-of-time: the model trains on earlier weeks and is scored on
later weeks it never saw, the way it would be used. Accuracy is meaningless
when 0.4% of transactions are fraud (a model that approves everything is
99.6% accurate), so the report uses what a fraud team cares about:

  * fraud caught at a fixed, small rate of honest customers wrongly declined,
  * fraud dollars caught,
  * precision-recall AUC,
  * the same numbers for a hand-written rules engine, as a baseline,
  * and an ablation without the account-graph features, to show what they add.
"""

from __future__ import annotations

import argparse
import gzip
import json
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

from .dataset import Dataset, generate
from .features import CATEGORICAL, FEATURES
from .simulator import DAY

MODELS = Path(__file__).resolve().parent.parent / "models"
GRAPH_FEATURES = ("device_accounts", "address_accounts", "ring_size", "ring_new_accounts")
# The operating point: decline at most 1 in 200 honest purchases.
TARGET_FPR = 0.005

PARAMS = {
    "objective": "binary",
    "learning_rate": 0.05,
    "num_leaves": 63,
    "min_data_in_leaf": 80,
    "feature_fraction": 0.85,
    "bagging_fraction": 0.8,
    "bagging_freq": 1,
    "lambda_l2": 1.0,
    "scale_pos_weight": 4.0,
    "metric": "average_precision",
    "verbosity": -1,
    "num_threads": 0,
    "seed": 13,
}


def split(data: Dataset, days: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    day = data.ts / DAY
    test_start = days * 0.85
    valid_start = days * 0.7
    return day < valid_start, (day >= valid_start) & (day < test_start), day >= test_start


def train_model(data: Dataset, train: np.ndarray, valid: np.ndarray, features: tuple[str, ...] = FEATURES) -> lgb.Booster:
    columns = [FEATURES.index(name) for name in features]
    categorical = [name for name in CATEGORICAL if name in features]
    train_set = lgb.Dataset(data.X[train][:, columns], data.y[train], feature_name=list(features), categorical_feature=categorical, free_raw_data=False)
    valid_set = lgb.Dataset(data.X[valid][:, columns], data.y[valid], reference=train_set)
    return lgb.train(
        PARAMS,
        train_set,
        num_boost_round=1500,
        valid_sets=[valid_set],
        callbacks=[lgb.early_stopping(80, verbose=False)],
    )


def threshold_for_fpr(scores: np.ndarray, labels: np.ndarray, fpr: float) -> float:
    """Lowest score threshold that declines at most `fpr` of honest transactions."""
    honest = np.sort(scores[labels == 0])
    return float(honest[int(np.ceil(len(honest) * (1 - fpr))) - 1])


def report(scores: np.ndarray, data: Dataset, mask: np.ndarray, threshold: float) -> dict:
    y = data.y[mask]
    s = scores
    amount = data.amount[mask]
    kinds = data.fraud_type[mask]
    flagged = s > threshold
    fraud = y == 1
    by_type = {}
    for kind in sorted(set(kinds[fraud])):
        selected = kinds == kind
        by_type[kind] = round(float(flagged[selected].mean()), 4)
    # Attacks caught: at least one transaction of the scenario declined.
    scenarios = data.scenario[mask]
    attack_ids = np.unique(scenarios[fraud])
    caught_attacks = np.mean([flagged[scenarios == sid].any() for sid in attack_ids]) if len(attack_ids) else 0.0
    return {
        "transactions": int(mask.sum()),
        "fraud_transactions": int(fraud.sum()),
        "fraud_rate": round(float(fraud.mean()), 5),
        "pr_auc": round(float(average_precision_score(y, s)), 4),
        "roc_auc": round(float(roc_auc_score(y, s)), 4),
        "false_positive_rate": round(float(flagged[~fraud].mean()), 5),
        "recall": round(float(flagged[fraud].mean()), 4),
        "precision": round(float(flagged[fraud].sum() / max(1, flagged.sum())), 4),
        "dollar_recall": round(float(amount[fraud & flagged].sum() / amount[fraud].sum()), 4),
        "attacks_caught": round(float(caught_attacks), 4),
        "recall_by_type": by_type,
    }


def rules_baseline(data: Dataset, mask: np.ndarray) -> np.ndarray:
    """The kind of rules a team writes before it has a model."""
    X = data.X[mask]
    col = {name: index for index, name in enumerate(FEATURES)}
    amount = np.expm1(X[:, col["amount_log"]])
    new_device = np.nan_to_num(X[:, col["device_known_min_log"]], nan=99) < np.log1p(60)
    rules = (
        (X[:, col["card_count_10m"]] >= 5)
        | (X[:, col["card_small_10m"]] >= 3)
        | ((amount > 500) & new_device)
        | ((X[:, col["foreign"]] > 0) & (amount > 300))
        | (X[:, col["store_speed_kmh_log"]] > np.log1p(900))
    )
    return rules.astype(np.float64)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true", help="small dataset for a fast check")
    args = parser.parse_args()
    customers, days = (6_000, 40) if args.quick else (15_000, 100)

    started = time.time()
    print(f"Simulating {days} days for {customers:,} customers...")
    data = generate(customers=customers, days=days)
    print(f"  {len(data.y):,} transactions, {int(data.y.sum()):,} fraudulent ({data.y.mean():.3%}) in {time.time() - started:.0f}s")

    train, valid, test = split(data, days)
    print("Training LightGBM...")
    model = train_model(data, train, valid)

    valid_scores = model.predict(data.X[valid])
    threshold = threshold_for_fpr(valid_scores, data.y[valid], TARGET_FPR)
    test_scores = model.predict(data.X[test])
    model_report = report(test_scores, data, test, threshold)

    baseline_scores = rules_baseline(data, test)
    baseline_report = report(baseline_scores, data, test, 0.5)

    print("Ablation: without account-graph features...")
    without = tuple(name for name in FEATURES if name not in GRAPH_FEATURES)
    ablation = train_model(data, train, valid, without)
    columns = [FEATURES.index(name) for name in without]
    ablation_threshold = threshold_for_fpr(ablation.predict(data.X[valid][:, columns]), data.y[valid], TARGET_FPR)
    ablation_report = report(ablation.predict(data.X[test][:, columns]), data, test, ablation_threshold)

    importance = dict(zip(model.feature_name(), (round(float(v), 1) for v in model.feature_importance("gain"))))
    metrics = {
        "dataset": {"customers": customers, "days": days, "transactions": len(data.y), "fraud_transactions": int(data.y.sum())},
        "split": {"train_days": f"14-{days * 0.7:.0f}", "valid_days": f"{days * 0.7:.0f}-{days * 0.85:.0f}", "test_days": f"{days * 0.85:.0f}-{days}"},
        "operating_point": {"target_false_positive_rate": TARGET_FPR, "threshold": threshold},
        "model": model_report,
        "rules_baseline": baseline_report,
        "without_graph_features": ablation_report,
        "feature_importance_gain": dict(sorted(importance.items(), key=lambda item: -item[1])),
        "trees": model.num_trees(),
    }

    MODELS.mkdir(exist_ok=True)
    if not args.quick:
        # Gzipped text: a quarter of the size, and small enough for any git host.
        with gzip.open(MODELS / "model.txt.gz", "wt", compresslevel=9) as out:
            out.write(model.model_to_string())
        (MODELS / "metrics.json").write_text(json.dumps(metrics, indent=2))
    _print_summary(metrics)
    print(f"Done in {time.time() - started:.0f}s")


def _print_summary(metrics: dict) -> None:
    def line(name: str, r: dict) -> str:
        return (
            f"  {name:24s} recall {r['recall']:.1%}  dollars {r['dollar_recall']:.1%}  attacks {r['attacks_caught']:.1%}  "
            f"precision {r['precision']:.1%}  honest declined {r['false_positive_rate']:.2%}  PR-AUC {r['pr_auc']:.3f}"
        )

    print("\nTest set (unseen later weeks):")
    print(line("Model", metrics["model"]))
    print(line("Rules baseline", metrics["rules_baseline"]))
    print(line("Model without graph", metrics["without_graph_features"]))
    print("  Recall by attack:", metrics["model"]["recall_by_type"])
    print("  Without graph:   ", metrics["without_graph_features"]["recall_by_type"])


if __name__ == "__main__":
    main()
