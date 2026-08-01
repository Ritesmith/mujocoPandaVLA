"""Diagnostic: compare V59 MLP head weights against V62-V68.

P2 (ensemble averaging) only makes sense if V62-V68 actually differ from
V59 in the trainable MLP head (mlp_extractor.policy_net, action_net, log_std).
V62-V68 all start from V59 and their best_hier is at step 1k (before the first
PPO update at step 2048), so they MAY be byte-identical to V59 in the head.
If so, weight averaging is a NO-OP.

This script reads the SB3 state_dict directly from the zip (no DAPGPPO.load,
no env, no GPU) and compares only the MLP head keys.

Usage:
    /home/w/miniconda3/envs/vla/bin/python -u diagnose_ensemble_weights.py
"""

import gc
import os
import re
import zipfile
from collections import OrderedDict

import torch

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

V59_PATH = "/home/w/vla_workspace/outputs/place_policy_v59/best_hier/best_model.zip"

CANDIDATES = [
    ("V62", "/home/w/vla_workspace/outputs/place_policy_v62/best_hier/best_model.zip"),
    ("V63", "/home/w/vla_workspace/outputs/place_policy_v63/best_hier/best_model.zip"),
    ("V64", "/home/w/vla_workspace/outputs/place_policy_v64/best_hier/best_model.zip"),
    ("V65", "/home/w/vla_workspace/outputs/place_policy_v65/best_hier/best_model.zip"),
    ("V66", "/home/w/vla_workspace/outputs/place_policy_v66/best_hier/best_model.zip"),
    ("V67", "/home/w/vla_workspace/outputs/place_policy_v67/best_hier/best_model.zip"),
    ("V68", "/home/w/vla_workspace/outputs/place_policy_v68/best_hier/best_model.zip"),
]

# MLP head keys we care about for averaging.
# - mlp_extractor.policy_net.{0,2}.{weight,bias}  (Linear+Tanh Sequential)
# - action_net.{weight,bias}                       (final action projection)
# - log_std                                        (learnable action std)
#
# We deliberately EXCLUDE:
#   - features_extractor.* (ResNet-18 backbone, frozen, identical by construction)
#   - mlp_extractor.value_net.* (value head, irrelevant for action sampling)
#   - optimizer state, rollout buffers, etc.
HEAD_KEY_RE = re.compile(
    r"^(mlp_extractor\.policy_net\.\d+\.(weight|bias)"
    r"|action_net\.(weight|bias)"
    r"|log_std)$"
)

IDENTICAL_TOL = 1e-6  # max_abs_diff below this counts as "identical"


def is_head_key(k: str) -> bool:
    return bool(HEAD_KEY_RE.match(k))


def load_state_dict(path: str) -> OrderedDict:
    """Load SB3 policy state_dict directly from the zip (CPU, no env).

    The policy weights live in `policy.pth` (45 MB). The bulk of V59's 5 GB
    is in the `data` file (demos / replay buffer), which we do NOT need.
    """
    with zipfile.ZipFile(path, "r") as archive:
        with archive.open("policy.pth") as f:
            sd = torch.load(f, map_location="cpu", weights_only=False)
    return sd


def extract_head(sd: OrderedDict) -> OrderedDict:
    """Keep only MLP head tensors; drop everything else to free RAM."""
    head = OrderedDict()
    for k in list(sd.keys()):
        if is_head_key(k):
            head[k] = sd[k]
    return head


def compare(a: OrderedDict, b: OrderedDict, label: str) -> dict:
    """Compare two head state_dicts. Returns per-key + aggregate stats."""
    assert set(a.keys()) == set(b.keys()), (
        f"Key mismatch for {label}:\n"
        f"  only in A: {set(a.keys()) - set(b.keys())}\n"
        f"  only in B: {set(b.keys()) - set(a.keys())}"
    )

    per_key = OrderedDict()
    max_abs_overall = 0.0
    sum_abs_overall = 0.0
    n_elems_overall = 0
    sum_sq_overall = 0.0  # for L2 of full difference

    for k in a.keys():
        ta = a[k].float()
        tb = b[k].float()
        assert ta.shape == tb.shape, f"Shape mismatch {k}: {ta.shape} vs {tb.shape}"
        diff = (ta - tb).abs()
        k_max = float(diff.max().item())
        k_mean = float(diff.mean().item())
        k_l2 = float((ta - tb).norm().item())  # L2 norm of difference vector
        k_l2_a = float(ta.norm().item())        # L2 norm of reference (for relative scale)
        per_key[k] = {
            "max_abs": k_max,
            "mean_abs": k_mean,
            "l2_diff": k_l2,
            "l2_ref": k_l2_a,
            "shape": tuple(ta.shape),
        }
        max_abs_overall = max(max_abs_overall, k_max)
        n = ta.numel()
        sum_abs_overall += float(diff.sum().item())
        n_elems_overall += n
        sum_sq_overall += float(((ta - tb) ** 2).sum().item())

    mean_abs_overall = sum_abs_overall / max(n_elems_overall, 1)
    l2_overall = sum_sq_overall ** 0.5
    return {
        "per_key": per_key,
        "max_abs": max_abs_overall,
        "mean_abs": mean_abs_overall,
        "l2_diff": l2_overall,
    }


def fmt(x: float) -> str:
    if x == 0.0:
        return "0.00e+00"
    return f"{x:.3e}"


def main():
    print("=" * 78)
    print("P2 DIAGNOSTIC: V59 vs V62-V68 MLP head weights")
    print("=" * 78)

    # --- Load V59 once, keep only head, free the rest -----------------------
    print(f"\n[load] V59  {V59_PATH}")
    print(f"       size on disk: {os.path.getsize(V59_PATH)/1e9:.2f} GB")
    sd_v59_full = load_state_dict(V59_PATH)
    n_keys_full = len(sd_v59_full)
    head_v59 = extract_head(sd_v59_full)
    n_head = len(head_v59)
    del sd_v59_full
    gc.collect()
    print(f"       total state_dict keys: {n_keys_full}")
    print(f"       MLP head keys kept:    {n_head}")
    print(f"       head keys: {list(head_v59.keys())}")
    total_head_params = sum(t.numel() for t in head_v59.values())
    print(f"       head params: {total_head_params:,}")

    # --- Compare each candidate against V59 ---------------------------------
    results = OrderedDict()
    differing_versions = []

    for name, path in CANDIDATES:
        print(f"\n[load] {name}  {path}")
        print(f"       size on disk: {os.path.getsize(path)/1e6:.1f} MB")
        sd_cand_full = load_state_dict(path)
        head_cand = extract_head(sd_cand_full)
        del sd_cand_full
        gc.collect()

        if set(head_cand.keys()) != set(head_v59.keys()):
            print(f"  !! HEAD KEY SET DIFFERS for {name}:")
            print(f"     only in V59 : {set(head_v59.keys()) - set(head_cand.keys())}")
            print(f"     only in {name}: {set(head_cand.keys()) - set(head_v59.keys())}")
            differing_versions.append(name)
            continue

        cmp = compare(head_v59, head_cand, name)
        results[name] = cmp

        print(f"  --- {name} vs V59 ---")
        print(f"  {'key':<42} {'max_abs':>12} {'mean_abs':>12} "
              f"{'l2_diff':>12} {'l2_ref':>12}")
        for k, st in cmp["per_key"].items():
            print(f"  {k:<42} {fmt(st['max_abs']):>12} {fmt(st['mean_abs']):>12} "
                  f"{fmt(st['l2_diff']):>12} {fmt(st['l2_ref']):>12}")
        print(f"  {'AGGREGATE (all head keys)':<42} "
              f"{fmt(cmp['max_abs']):>12} {fmt(cmp['mean_abs']):>12} "
              f"{fmt(cmp['l2_diff']):>12}")

        if cmp["max_abs"] >= IDENTICAL_TOL:
            differing_versions.append(name)
            print(f"  -> DIFFERS from V59 (max_abs={fmt(cmp['max_abs'])} >= {IDENTICAL_TOL})")
        else:
            print(f"  -> IDENTICAL to V59 (max_abs={fmt(cmp['max_abs'])} < {IDENTICAL_TOL})")

    # --- Conclusion ---------------------------------------------------------
    print("\n" + "=" * 78)
    print("CONCLUSION")
    print("=" * 78)
    print(f"\nMLP head keys compared: {list(head_v59.keys())}")
    print(f"V59 head param count  : {total_head_params:,}")
    print()
    print(f"{'version':<8} {'max_abs_vs_V59':>16} {'mean_abs':>14} {'l2_diff':>14}  verdict")
    for name, cmp in results.items():
        verdict = "DIFFER" if cmp["max_abs"] >= IDENTICAL_TOL else "IDENTICAL"
        print(f"{name:<8} {fmt(cmp['max_abs']):>16} {fmt(cmp['mean_abs']):>14} "
              f"{fmt(cmp['l2_diff']):>14}  {verdict}")

    print()
    if not differing_versions:
        print("IDENTICAL: weight averaging is a NO-OP.")
        print("All V62-V68 share the exact same MLP head weights as V59.")
        print("V62-V68 were saved at step 1k (before first PPO update at 2048),")
        print("so their best_hier is a byte-copy of V59's head. Averaging")
        print("identical weights yields the same weights -> no change.")
        print("P2 weight-averaging cannot improve over V59.")
    else:
        print(f"DIFFER: weight averaging viable. Differing versions: {differing_versions}")
        print("At least one of V62-V68 has different MLP head weights from V59.")
        print("Proceed with weight averaging across the differing models.")

    # --- JSON dump for downstream use --------------------------------------
    out_json = {
        "identical_tol": IDENTICAL_TOL,
        "head_keys": list(head_v59.keys()),
        "v59_head_param_count": total_head_params,
        "comparisons": {
            name: {
                "max_abs": cmp["max_abs"],
                "mean_abs": cmp["mean_abs"],
                "l2_diff": cmp["l2_diff"],
                "per_key": {
                    k: {kk: vv for kk, vv in st.items() if kk != "shape"}
                    for k, st in cmp["per_key"].items()
                },
            }
            for name, cmp in results.items()
        },
        "differing_versions": differing_versions,
        "conclusion": "IDENTICAL" if not differing_versions else "DIFFER",
    }
    out_path = "/home/w/vla_workspace/auto_iter/case_memory/_diag_ensemble_weights.json"
    import json
    with open(out_path, "w") as f:
        json.dump(out_json, f, indent=2)
    print(f"\n[written] {out_path}")


if __name__ == "__main__":
    main()
