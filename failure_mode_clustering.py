#!/usr/bin/env python3
"""P0.5 Failure Mode Clustering on D_fail via GMM + BIC.

Loads D_fail.npz, builds 6-dim continuous features, fits GMM with BIC model
selection (k in [3,8]), and outputs a Cluster Table with per-cluster stats
and confident/hesitant failure classification.

GMM features (6-dim continuous):
  1. gripper_width       -- gripper opening (m)
  2. EE_dist_to_target   -- ||EE_pos - target_pos|| (m)
  3. contact_force_peak  -- contact_flag (1=contact, 0=no contact)
  4. action_L2           -- ||action||_2
  5. critic_V59          -- V59 Critic V(s)
  6. logpi_V59           -- log pi_V59(a|s)

Hard keys (NOT in GMM, used as Voronoi routing keys):
  - task_stage (0=grasp, 1=place)
  - gripper_bin (0=closed <0.02m, 1=open >=0.02m)

Output: outputs/csil_plus_plus/failure_clusters.npz + cluster_table.json

Usage:
    python failure_mode_clustering.py
    python failure_mode_clustering.py --d_fail_path data/D_fail.npz --k_range 3 8
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

WORKSPACE = Path(__file__).parent.resolve()
sys.path.insert(0, str(WORKSPACE))

D_FAIL_PATH = WORKSPACE / "data" / "D_fail.npz"
OUTPUT_DIR = WORKSPACE / "outputs" / "csil_plus_plus"
CLUSTERS_PATH = OUTPUT_DIR / "failure_clusters.npz"
TABLE_PATH = OUTPUT_DIR / "cluster_table.json"

TARGET_POS_DEFAULT = np.array([0.5, 0.3, 0.22])


def build_features(data: dict) -> tuple[np.ndarray, dict]:
    """Build 6-dim continuous feature matrix from D_fail data.

    Returns (features (N,6), feature_names list).
    """
    gripper = data["gripper_width"].astype(np.float64)
    ee_pos = data["EE_pos"].astype(np.float64)  # (N, 3)

    # Target position from state_vec indices 9:12.
    state_vec = data["state_vec"].astype(np.float64)
    if state_vec.shape[1] >= 12:
        target_pos = state_vec[:, 9:12]
    else:
        target_pos = np.tile(TARGET_POS_DEFAULT, (len(ee_pos), 1))

    ee_dist = np.linalg.norm(ee_pos - target_pos, axis=1)  # (N,)

    contact = data["contact_flag"].astype(np.float64)
    action_l2 = np.linalg.norm(
        data["action"].astype(np.float64), axis=1)  # (N,)
    critic = data["critic_V59"].astype(np.float64)
    logpi = data["logpi_V59"].astype(np.float64)

    features = np.stack([gripper, ee_dist, contact, action_l2,
                         critic, logpi], axis=1)  # (N, 6)
    names = ["gripper_width", "EE_dist_to_target", "contact_force_peak",
             "action_L2", "critic_V59", "logpi_V59"]
    return features, {"names": names}


def fit_gmm_bic(features: np.ndarray, k_range: tuple[int, int],
                seed: int = 42) -> tuple[GaussianMixture, int, list]:
    """Fit GMM with BIC model selection over k_range.

    Returns (best_gmm, best_k, bic_scores).
    """
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)

    bic_scores = []
    best_bic = np.inf
    best_gmm = None
    best_k = k_range[0]

    for k in range(k_range[0], k_range[1] + 1):
        gmm = GaussianMixture(
            n_components=k,
            covariance_type="diag",
            reg_covar=1e-6,
            max_iter=200,
            n_init=5,
            random_state=seed,
        )
        gmm.fit(features_scaled)
        bic = gmm.bic(features_scaled)
        bic_scores.append({"k": k, "bic": float(bic),
                           "n_iter": int(gmm.n_iter_)})
        print(f"  k={k}: BIC={bic:.1f}  n_iter={gmm.n_iter_}")
        if bic < best_bic:
            best_bic = bic
            best_gmm = gmm
            best_k = k

    print(f"\nBest k={best_k} (BIC={best_bic:.1f})")
    return best_gmm, best_k, bic_scores


def classify_failure(cluster_logpi: float, overall_logpi_median: float) -> str:
    """Classify a cluster as 'confident_failure' or 'hesitant_failure'.

    - confident_failure: logπ close to 0 (high probability) -> V59 confidently
      takes wrong actions
    - hesitant_failure: logπ very negative (low probability) -> V59 doesn't
      know what to do
    """
    if cluster_logpi > overall_logpi_median:
        return "confident_failure"
    return "hesitant_failure"


def build_cluster_table(gmm: GaussianMixture, features: np.ndarray,
                        data: dict, scaler: StandardScaler,
                        best_k: int) -> list[dict]:
    """Build per-cluster summary table."""
    features_scaled = scaler.transform(features)
    labels = gmm.predict(features_scaled)
    overall_logpi_median = float(np.median(data["logpi_V59"]))

    # Gripper binary: 0=closed (<0.02m), 1=open (>=0.02m).
    gripper_bin = (data["gripper_width"] >= 0.02).astype(np.int32)

    table = []
    for c in range(best_k):
        mask = labels == c
        n_c = int(mask.sum())
        if n_c == 0:
            continue

        # Per-cluster stats.
        avg_critic = float(data["critic_V59"][mask].mean())
        avg_logpi = float(data["logpi_V59"][mask].mean())
        avg_gripper = float(data["gripper_width"][mask].mean())
        avg_ee_dist = float(features[mask, 1].mean())
        avg_action_l2 = float(features[mask, 3].mean())
        contact_rate = float(data["contact_flag"][mask].mean())
        avg_final_dist = float(data["final_dist"][mask].mean())

        # Unique episodes in this cluster.
        ep_ids = np.unique(data["ep_id"][mask])
        n_episodes = len(ep_ids)

        # Dominant task_stage and gripper_bin.
        dominant_stage = int(np.median(data["task_stage"][mask]))
        dominant_gripper_bin = int(np.median(gripper_bin[mask]))

        # Failure type classification.
        failure_type = classify_failure(avg_logpi, overall_logpi_median)

        # Human-readable name (heuristic).
        if contact_rate > 0.5 and avg_gripper < 0.02:
            name = "gripped_but_missed"
        elif contact_rate < 0.3:
            name = "lost_contact"
        elif avg_ee_dist > 0.15:
            name = "far_from_target"
        elif avg_logpi > overall_logpi_median:
            name = "confident_wrong"
        else:
            name = "hesitant_unsure"

        entry = {
            "cluster_id": c,
            "name": name,
            "n_transitions": n_c,
            "frac_of_dfail": n_c / len(labels),
            "n_episodes": n_episodes,
            "avg_critic_V59": avg_critic,
            "avg_logpi_V59": avg_logpi,
            "avg_gripper_width": avg_gripper,
            "avg_EE_dist_to_target": avg_ee_dist,
            "avg_action_L2": avg_action_l2,
            "contact_rate": contact_rate,
            "avg_final_dist_cm": avg_final_dist * 100,
            "dominant_task_stage": dominant_stage,
            "dominant_gripper_bin": dominant_gripper_bin,
            "failure_type": failure_type,
        }
        table.append(entry)

    # Sort by size (largest first).
    table.sort(key=lambda x: x["n_transitions"], reverse=True)
    # Re-number after sort.
    for i, entry in enumerate(table):
        entry["cluster_id_sorted"] = i
    return table


def main():
    parser = argparse.ArgumentParser(
        description="P0.5 Failure Mode Clustering on D_fail via GMM + BIC."
    )
    parser.add_argument("--d_fail_path", type=str, default=str(D_FAIL_PATH))
    parser.add_argument("--k_range", type=int, nargs=2, default=[3, 8],
                        help="GMM k range [min, max] (default: 3 8)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_dir", type=str, default=str(OUTPUT_DIR))
    args = parser.parse_args()

    # Load D_fail.
    if not Path(args.d_fail_path).exists():
        print(f"ERROR: D_fail not found at {args.d_fail_path}")
        print("  Run `python collect_d_fail.py --n_episodes 100` first.")
        sys.exit(1)

    print("=" * 60)
    print("P0.5 Failure Mode Clustering (GMM + BIC)")
    print("=" * 60)
    data = dict(np.load(args.d_fail_path, allow_pickle=True))
    n = len(data["action"])
    print(f"Loaded D_fail: {n} transitions from "
          f"{len(np.unique(data['ep_id']))} failure episodes")
    print(f"  Fields: {list(data.keys())}")

    # Build features.
    features, info = build_features(data)
    print(f"\nFeatures ({features.shape[1]}-dim): {info['names']}")
    print(f"  Feature stats (min/max/mean):")
    for i, name in enumerate(info["names"]):
        print(f"    {name:25s}: min={features[:, i].min():8.3f} "
              f"max={features[:, i].max():8.3f} "
              f"mean={features[:, i].mean():8.3f}")

    # Fit GMM with BIC.
    print(f"\n--- GMM BIC Model Selection (k={args.k_range[0]}..{args.k_range[1]}) ---")
    scaler = StandardScaler()
    scaler.fit(features)  # fit here so we can reuse scaler later
    gmm, best_k, bic_scores = fit_gmm_bic(features, args.k_range, args.seed)

    # Build cluster table.
    print(f"\n--- Cluster Table (k={best_k}) ---")
    table = build_cluster_table(gmm, features, data, scaler, best_k)
    print(f"{'ID':>3} {'Name':20s} {'N':>6} {'Frac':>6} {'Eps':>4} "
          f"{'Critic':>7} {'LogPi':>7} {'Contact':>7} {'Dist':>6} "
          f"{'Type':20s}")
    print("-" * 100)
    for entry in table:
        print(f"{entry['cluster_id']:3d} {entry['name']:20s} "
              f"{entry['n_transitions']:6d} {entry['frac_of_dfail']:6.1%} "
              f"{entry['n_episodes']:4d} "
              f"{entry['avg_critic_V59']:7.3f} {entry['avg_logpi_V59']:7.3f} "
              f"{entry['contact_rate']:6.1%} "
              f"{entry['avg_final_dist_cm']:5.1f}cm "
              f"{entry['failure_type']:20s}")

    # Save outputs.
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Save cluster assignments + GMM params.
    features_scaled = scaler.transform(features)
    labels = gmm.predict(features_scaled)
    np.savez_compressed(
        str(CLUSTERS_PATH),
        labels=labels.astype(np.int32),
        cluster_centers=gmm.means_,
        feature_names=np.array(info["names"]),
        scaler_mean=scaler.mean_,
        scaler_scale=scaler.scale_,
        bic_scores=np.array([b["bic"] for b in bic_scores]),
        k_range=np.array(args.k_range),
        best_k=best_k,
    )
    print(f"\nSaved cluster assignments to {CLUSTERS_PATH}")

    # Save cluster table JSON.
    output = {
        "best_k": best_k,
        "bic_scores": bic_scores,
        "feature_names": info["names"],
        "n_total_transitions": n,
        "n_episodes": len(np.unique(data["ep_id"])),
        "clusters": table,
        "overall_logpi_median": float(np.median(data["logpi_V59"])),
    }
    with open(TABLE_PATH, "w") as f:
        json.dump(output, f, indent=2)
    print(f"Saved cluster table to {TABLE_PATH}")

    # Print Voronoi key summary.
    print("\n--- Voronoi Hard Keys Summary ---")
    gripper_bin = (data["gripper_width"] >= 0.02).astype(np.int32)
    task_stages = data["task_stage"].astype(np.int32)
    print(f"  task_stage distribution: "
          f"grasp={int((task_stages==0).sum())} "
          f"place={int((task_stages==1).sum())}")
    print(f"  gripper_bin distribution: "
          f"closed={int((gripper_bin==0).sum())} "
          f"open={int((gripper_bin==1).sum())}")
    print(f"  Combined hard keys: "
          f"{len(np.unique(list(zip(task_stages, gripper_bin))))} unique")


if __name__ == "__main__":
    main()
