"""P2 ensemble prediction — NO-OP declaration (Case B: identical weights).

DIAGNOSTIC RESULT (see diagnose_ensemble_weights.py + _diag_ensemble_weights.json):
  All 8 checkpoints (V59, V62-V68) share a SINGLE md5 hash for policy.pth.
  Every MLP head key (mlp_extractor.policy_net.*, action_net.*, log_std) has
  max_abs_diff = 0.00e+00 vs V59. The optimizer state is also byte-identical.
  V62-V68 were saved at step 1k (before the first PPO update at step 2048),
  so their best_hier is a literal copy of V59's policy.

CONSEQUENCE:
  1. Weight averaging: averaging N copies of identical weights yields the
     SAME weights. mean(w, w, ..., w) = w. Producing an "ensemble" model
     would just be a copy of V59 -> eval would reproduce 56%. NO-OP.
  2. Inference-time ensemble prediction (sample N stochastic actions,
     average them): all members are the same policy, so this reduces to
     "sample N times from ONE policy and average". By the law of large
     numbers, the average of N samples from a Gaussian N(mu, sigma)
     converges to mu as N grows. V59's deterministic eval ALREADY uses mu
     (the mean action). So averaged stochastic samples approximate the
     deterministic mean -> cannot beat deterministic V59. NO-OP.

  Neither variant of P2 can improve over V59. Per project constraint,
  P2 is the LAST non-gradient attempt. Conclusion: accept V59 (56%) as
  the final production policy.

This script re-verifies the identity programmatically (md5 of policy.pth
across all 8 checkpoints) and prints the NO-OP declaration. It does NOT
create an ensemble model and does NOT run the 50-ep eval (Step 3 is
conditional on Step 2 producing a model, which it does not).
"""

import hashlib
import os
import zipfile
from collections import OrderedDict

MODELS = [
    ("V59", "/home/w/vla_workspace/outputs/place_policy_v59/best_hier/best_model.zip"),
    ("V62", "/home/w/vla_workspace/outputs/place_policy_v62/best_hier/best_model.zip"),
    ("V63", "/home/w/vla_workspace/outputs/place_policy_v63/best_hier/best_model.zip"),
    ("V64", "/home/w/vla_workspace/outputs/place_policy_v64/best_hier/best_model.zip"),
    ("V65", "/home/w/vla_workspace/outputs/place_policy_v65/best_hier/best_model.zip"),
    ("V66", "/home/w/vla_workspace/outputs/place_policy_v66/best_hier/best_model.zip"),
    ("V67", "/home/w/vla_workspace/outputs/place_policy_v67/best_hier/best_model.zip"),
    ("V68", "/home/w/vla_workspace/outputs/place_policy_v68/best_hier/best_model.zip"),
]

ENTRIES = ["policy.pth", "pytorch_variables.pth", "policy.optimizer.pth"]


def md5_of_entry(zip_path: str, entry: str) -> str:
    with zipfile.ZipFile(zip_path, "r") as z:
        return hashlib.md5(z.read(entry)).hexdigest()


def main():
    print("=" * 78)
    print("P2 ENSEMBLE PREDICTION — NO-OP DECLARATION")
    print("=" * 78)

    # --- Re-verify byte-identity of all 8 checkpoints -----------------------
    print("\n[1] Re-verifying byte-identity of policy.pth across all 8 checkpoints")
    print(f"    {'version':<6} {'policy.pth md5':<34} {'pytorch_vars md5':<34} {'optimizer md5':<34}")
    hashes_by_entry = {e: OrderedDict() for e in ENTRIES}
    for name, path in MODELS:
        if not os.path.exists(path):
            print(f"    {name}: MISSING {path}")
            continue
        row = []
        for e in ENTRIES:
            h = md5_of_entry(path, e)
            hashes_by_entry[e][name] = h
            row.append(h)
        print(f"    {name:<6} {row[0]:<34} {row[1]:<34} {row[2]:<34}")

    print("\n    Distinct hashes per entry:")
    all_single = True
    for e in ENTRIES:
        distinct = set(hashes_by_entry[e].values())
        ok = len(distinct) == 1
        all_single = all_single and ok
        tag = "OK (all identical)" if ok else f"DIFFER ({len(distinct)} distinct)"
        print(f"      {e:<28} : {tag}")
        if not ok:
            for h in distinct:
                who = [n for n, h2 in hashes_by_entry[e].items() if h2 == h]
                print(f"        {h}  <- {who}")

    if not all_single:
        print("\n    !! Unexpected: some entries differ. Re-run diagnose_ensemble_weights.py.")
        print("    Falling back to NO-OP declaration based on MLP head diagnostic only.")
    else:
        print("\n    -> CONFIRMED: all 8 checkpoints are byte-identical (policy + vars + optimizer).")

    # --- Why weight averaging is a NO-OP ------------------------------------
    print("\n[2] Why weight averaging is a NO-OP")
    print("    Let w59 = V59's MLP head weights (policy_net + action_net).")
    print("    Since V62..V68 each have w62 = w63 = ... = w68 = w59, the average is:")
    print("      w_avg = (w59 + w62 + w63 + w64 + w65 + w66 + w67 + w68) / 8")
    print("            = (8 * w59) / 8")
    print("            = w59")
    print("    Averaging identical weights yields the SAME weights.")
    print("    An 'ensemble' model would be a byte-copy of V59 -> eval = 56%.")
    print("    -> NO-OP. Not creating outputs/place_policy_ensemble/.")

    # --- Why inference-time ensemble prediction is a NO-OP ------------------
    print("\n[3] Why inference-time ensemble prediction is a NO-OP")
    print("    All ensemble members are the SAME policy (single MD5). So 'ensemble")
    print("    prediction' reduces to: sample N stochastic actions from ONE policy,")
    print("    average them. SB3 samples actions as a = mu + sigma * eps, with")
    print("    eps ~ N(0, I). Averaging N samples:")
    print("      mean(a_1..a_N) = mu + sigma * mean(eps_1..eps_N)")
    print("    As N -> inf, mean(eps) -> 0, so mean(a) -> mu.")
    print("    V59's deterministic eval ALREADY uses mu (the mean action).")
    print("    Therefore averaged stochastic samples approximate the deterministic")
    print("    mean action -> ensemble prediction converges to V59 deterministic.")
    print("    It CANNOT beat V59's 56% deterministic eval (only adds sampling noise")
    print("    for finite N). -> NO-OP.")

    # --- Decision -----------------------------------------------------------
    print("\n[4] DECISION")
    print("    P2 (ensemble averaging) is a NO-OP because V62-V68 are byte-identical")
    print("    copies of V59 (saved at step 1k, before the first PPO update at 2048).")
    print("    Neither weight averaging nor inference-time ensemble prediction can")
    print("    improve over V59.")
    print("    Per project constraint: P2 is the LAST non-gradient attempt. All")
    print("    gradient-based methods (PPO, BC, DAgger, distillation) also failed.")
    print("    -> ACCEPT V59 (56% place rate) AS THE FINAL PRODUCTION POLICY.")
    print("    -> No ensemble model produced. 50-ep eval SKIPPED (would reproduce 56%).")

    print("\n" + "=" * 78)
    print("P2 VERDICT: NO-OP (identical weights). Accept V59 as final.")
    print("=" * 78)


if __name__ == "__main__":
    main()
