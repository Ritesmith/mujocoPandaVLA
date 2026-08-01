#!/usr/bin/env python3
"""DT Orchestrator Task 5: Train DecisionTreeClassifier router.

Reads the codebook JSON (from dt_codebook.py) and trains a sklearn
DecisionTreeClassifier with the spec's regularization settings.

Outputs 4 artifacts:
  1. dt_model.pkl         — pickled model + metadata (for online router)
  2. feature_importance.json — feature importance ranking
  3. tree_structure.txt    — text tree dump (export_text)
  4. training_report.json  — accuracy, CV, confusion matrix, class dist,
                              feature correlation matrix

Spec compliance:
  - DecisionTreeClassifier(max_depth=5, min_samples_leaf=5, criterion="gini",
    class_weight="balanced", random_state=42)
  - StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
  - weak_label samples get sample_weight=0.5
  - Q-value imputation: q1_at_step20=0.0, has_q_value=0 when missing
  - Feature correlation matrix with |correlation|>0.8 warnings

Usage:
    python dt_trainer.py \
        --codebook outputs/dt_orchestrator/codebook.json \
        --output_dir outputs/dt_orchestrator/
"""
import argparse
import json
import pickle
import sys
from datetime import date
from pathlib import Path

import numpy as np
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import (confusion_matrix, classification_report,
                             precision_recall_fscore_support)


def load_codebook(path):
    with open(path) as f:
        return json.load(f)


def main():
    parser = argparse.ArgumentParser(
        description="DT Orchestrator: train DecisionTreeClassifier router")
    parser.add_argument("--codebook", type=str, required=True,
                        help="Path to codebook JSON (from dt_codebook.py)")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Output directory for model + reports")
    parser.add_argument("--suffix", type=str, default="",
                        help="Suffix appended to output filenames "
                             "(e.g. '_v2' → dt_model_v2.pkl, "
                             "feature_importance_v2.json, "
                             "training_report_v2.json). Default: ''")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    suffix = args.suffix

    # Load codebook
    codebook = load_codebook(args.codebook)
    codebook_version = codebook.get("version", "unknown")
    config_set = codebook.get("config_set", [])
    feature_names = codebook.get("feature_names", [])
    all_entries = codebook.get("entries", [])

    # Filter trainable entries (exclude=True are skipped)
    trainable = [e for e in all_entries if not e.get("exclude", False)]
    if not trainable:
        print("ERROR: no trainable entries in codebook", file=sys.stderr)
        sys.exit(1)

    print(f"Codebook: version={codebook_version}, configs={config_set}")
    print(f"  Total entries: {len(all_entries)}, trainable: {len(trainable)}")

    # Build X (features), y (labels), sample_weights
    X = np.array([[e["features"][fn] for fn in feature_names]
                  for e in trainable], dtype=float)
    y = np.array([e["optimal_config"] for e in trainable])
    sample_weights = np.array(
        [0.5 if e.get("weak_label", False) else 1.0 for e in trainable])

    # Check for NaN
    if np.any(np.isnan(X)):
        nan_rows = np.where(np.isnan(X).any(axis=1))[0]
        print(f"WARNING: {len(nan_rows)} entries have NaN features, filling with 0")
        X = np.nan_to_num(X, nan=0.0)

    # Class distribution
    from collections import Counter
    class_dist = Counter(y)
    print(f"\n  Class distribution:")
    for cls, count in sorted(class_dist.items()):
        print(f"    {cls:10s}: {count:3d} ({100*count/len(y):.1f}%)")

    # Feature correlation matrix (P2)
    if X.shape[1] > 1:
        corr_matrix = np.corrcoef(X.T)
        collinear_warnings = []
        for i in range(len(feature_names)):
            for j in range(i + 1, len(feature_names)):
                if abs(corr_matrix[i, j]) > 0.8:
                    collinear_warnings.append({
                        "feature_a": feature_names[i],
                        "feature_b": feature_names[j],
                        "correlation": round(float(corr_matrix[i, j]), 4),
                    })
        if collinear_warnings:
            print(f"\n  WARNING: {len(collinear_warnings)} collinear feature pairs (|r|>0.8):")
            for w in collinear_warnings:
                print(f"    {w['feature_a']} <-> {w['feature_b']}: r={w['correlation']}")
    else:
        corr_matrix = np.array([[1.0]])
        collinear_warnings = []

    # Train DecisionTreeClassifier (spec: max_depth=5, min_samples_leaf=5,
    # class_weight="balanced", random_state=42)
    dt = DecisionTreeClassifier(
        max_depth=5,
        min_samples_leaf=5,
        criterion="gini",
        class_weight="balanced",
        random_state=42,
    )
    dt.fit(X, y, sample_weight=sample_weights)

    # Training accuracy
    y_pred_train = dt.predict(X)
    train_acc = float(np.mean(y_pred_train == y))

    # 5-fold StratifiedKFold CV
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(dt, X, y, cv=cv, scoring="accuracy")

    # Confusion matrix
    labels_sorted = sorted(class_dist.keys())
    cm = confusion_matrix(y, y_pred_train, labels=labels_sorted)

    # Per-class precision/recall/F1
    precision, recall, f1, support = precision_recall_fscore_support(
        y, y_pred_train, labels=labels_sorted, zero_division=0)

    # Feature importance
    importances = dt.feature_importances_
    feat_imp = sorted(zip(feature_names, importances),
                      key=lambda x: -x[1])

    # ---- Save artifacts ----

    # 1. dt_model.pkl (model + metadata for version binding P1-4.5)
    model_payload = {
        "model": dt,
        "metadata": {
            "codebook_version": codebook_version,
            "config_set": config_set,
            "feature_names": feature_names,
            "training_date": str(date.today()),
            "n_train_samples": len(trainable),
            "class_distribution": dict(class_dist),
        },
    }
    model_path = output_dir / f"dt_model{suffix}.pkl"
    with open(model_path, "wb") as f:
        pickle.dump(model_payload, f)
    print(f"\n  Saved model: {model_path}")

    # 2. feature_importance.json
    feat_imp_data = {
        "feature_importances": [
            {"feature": fn, "importance": round(float(imp), 6)}
            for fn, imp in feat_imp
        ],
        "total": round(float(sum(importances)), 6),
    }
    imp_path = output_dir / f"feature_importance{suffix}.json"
    with open(imp_path, "w") as f:
        json.dump(feat_imp_data, f, indent=2)
    print(f"  Saved feature importance: {imp_path}")

    # 3. tree_structure.txt
    tree_text = export_text(dt, feature_names=feature_names)
    tree_path = output_dir / f"tree_structure{suffix}.txt"
    with open(tree_path, "w") as f:
        f.write(tree_text)
    print(f"  Saved tree structure: {tree_path}")

    # 4. training_report.json
    report = {
        "codebook_version": codebook_version,
        "config_set": config_set,
        "feature_names": feature_names,
        "n_trainable": len(trainable),
        "n_excluded": len(all_entries) - len(trainable),
        "training_accuracy": round(train_acc, 4),
        "cv_accuracy": {
            "mean": round(float(cv_scores.mean()), 4),
            "std": round(float(cv_scores.std()), 4),
            "scores": [round(float(s), 4) for s in cv_scores],
        },
        "class_distribution": dict(class_dist),
        "confusion_matrix": {
            "labels": labels_sorted,
            "matrix": cm.tolist(),
        },
        "per_class_metrics": {
            labels_sorted[i]: {
                "precision": round(float(precision[i]), 4),
                "recall": round(float(recall[i]), 4),
                "f1": round(float(f1[i]), 4),
                "support": int(support[i]),
            }
            for i in range(len(labels_sorted))
        },
        "feature_importance": feat_imp_data["feature_importances"],
        "feature_correlation_matrix": {
            "feature_names": feature_names,
            "matrix": [[round(float(corr_matrix[i][j]), 4)
                        for j in range(len(feature_names))]
                       for i in range(len(feature_names))],
            "collinear_warnings": collinear_warnings,
        },
        "model_params": {
            "max_depth": 5,
            "min_samples_leaf": 5,
            "criterion": "gini",
            "class_weight": "balanced",
            "random_state": 42,
        },
    }
    report_path = output_dir / f"training_report{suffix}.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2, default=str)
    print(f"  Saved training report: {report_path}")

    # ---- Print summary ----
    print(f"\n{'='*60}")
    print(f"DT Trainer complete")
    print(f"{'='*60}")
    print(f"  Training accuracy:  {train_acc:.4f}")
    print(f"  CV accuracy:        {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
    print(f"\n  Feature importance (top 3):")
    for fn, imp in feat_imp[:3]:
        print(f"    {fn:25s}: {imp:.4f}")
    print(f"\n  Per-class metrics:")
    for i, cls in enumerate(labels_sorted):
        print(f"    {cls:10s}: P={precision[i]:.3f} R={recall[i]:.3f} "
              f"F1={f1[i]:.3f} (n={support[i]})")
    if collinear_warnings:
        print(f"\n  Collinear warnings: {len(collinear_warnings)}")

    # Root cause insight (spec scenario)
    top_feat = feat_imp[0][0] if feat_imp else "none"
    print(f"\n  Root cause insight:")
    if top_feat in ("q1_at_step20", "early_drift_signal"):
        print(f"    Top feature '{top_feat}' →印证 critic 时序偏差是根因")
    elif top_feat == "dist_change_rate":
        print(f"    Top feature 'dist_change_rate' → 印证时序一致性是 success 关键")
    else:
        print(f"    Top feature '{top_feat}' → 需进一步分析")


if __name__ == "__main__":
    main()
