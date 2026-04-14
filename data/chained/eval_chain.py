# eval_chain.py — Chain Duplicate Detection Evaluator
#
# This tests specifically for the Union-Find chaining problem:
# When A≈B and B≈C but A≉C, does the service incorrectly merge A and C?
#
# Usage:
#   python eval_chain.py --threshold 0.90
#   python eval_chain.py --all
#   python eval_chain.py --all --url http://localhost:8000
#
# Requires:
#   chain_ground_truth.json in the same directory
#   pip install requests

import json
import argparse
import requests
from collections import defaultdict
from itertools import combinations

CHAIN_GT_PATH = "chain_ground_truth.json"
DEFAULT_URL   = "http://localhost:8000"
THRESHOLDS    = [0.80, 0.85, 0.90, 0.92, 0.95]


# ── Load ground truth ──────────────────────────────────────────────────────────
def load_ground_truth(path: str):
    with open(path) as f:
        records = json.load(f)

    # prompt_id → full record
    gt_map = {r["prompt_id"]: r for r in records}

    # cluster_id → list of records
    clusters = defaultdict(list)
    for r in records:
        clusters[r["cluster_id"]].append(r)

    return gt_map, dict(clusters)


# ── Fetch service results ──────────────────────────────────────────────────────
def fetch_service_clusters(base_url: str, threshold: float) -> list[set]:
    url  = f"{base_url}/api/analysis/duplicates?threshold={threshold}"
    resp = requests.get(url, timeout=120)
    resp.raise_for_status()
    data = resp.json()

    cluster_list = data["clusters"] if isinstance(data, dict) else data
    return [{p["prompt_id"] for p in c["prompts"]} for c in cluster_list]


# ── Pair helpers ───────────────────────────────────────────────────────────────
def to_pairs(members) -> set[frozenset]:
    return {frozenset(p) for p in combinations(members, 2)}


# ── Core chain eval ────────────────────────────────────────────────────────────
def evaluate_chains(service_clusters: list[set], gt_map: dict, gt_clusters: dict) -> dict:

    # Map prompt_id → which service cluster it landed in
    pid_to_service_cluster: dict[str, int] = {}
    for idx, sc in enumerate(service_clusters):
        for pid in sc:
            pid_to_service_cluster[pid] = idx

    # ── Per-cluster analysis ───────────────────────────────────────────────────
    cluster_results = []
    over_merge_cases  = []   # Union-Find chained A+C without A≈C directly
    under_merge_cases = []   # Hop that should have clustered but didn't

    for cid, members in gt_clusters.items():
        orig_rec  = next(r for r in members if r["chain_position"] == 0)
        orig_id   = orig_rec["prompt_id"]
        chain_type = orig_rec["chain_type"]

        orig_service_cluster = pid_to_service_cluster.get(orig_id)

        per_hop = []
        for rec in sorted(members, key=lambda r: r["chain_position"]):
            if rec["chain_position"] == 0:
                continue

            pid              = rec["prompt_id"]
            should_cluster   = rec["should_cluster"]
            bridge_risk      = rec["bridge_risk"]
            service_cluster  = pid_to_service_cluster.get(pid)

            # Did this hop land in the same service cluster as the original?
            grouped_with_orig = (
                service_cluster is not None
                and orig_service_cluster is not None
                and service_cluster == orig_service_cluster
            )

            outcome = None
            if should_cluster and grouped_with_orig:
                outcome = "TP"
            elif should_cluster and not grouped_with_orig:
                outcome = "FN"
                under_merge_cases.append({
                    "cluster_id":     cid,
                    "original_id":    orig_id,
                    "hop_prompt_id":  pid,
                    "hop_label":      rec["hop_label"],
                    "chain_type":     chain_type,
                })
            elif not should_cluster and grouped_with_orig and bridge_risk:
                outcome = "OVER_MERGE"   # chaining pulled in a node that shouldn't be there
                over_merge_cases.append({
                    "cluster_id":    cid,
                    "original_id":   orig_id,
                    "hop_prompt_id": pid,
                    "hop_label":     rec["hop_label"],
                    "chain_type":    chain_type,
                    "note":          "hop4 reachable via hop3 bridge — Union-Find chaining",
                })
            elif not should_cluster and not grouped_with_orig:
                outcome = "TN"   # correctly excluded far-drift hops

            per_hop.append({
                "hop_label":        rec["hop_label"],
                "chain_position":   rec["chain_position"],
                "should_cluster":   should_cluster,
                "bridge_risk":      bridge_risk,
                "grouped_with_orig": grouped_with_orig,
                "outcome":          outcome,
            })

        cluster_results.append({
            "cluster_id":  cid,
            "chain_type":  chain_type,
            "original_id": orig_id,
            "hops":        per_hop,
        })

    # ── Aggregate counts by hop label ──────────────────────────────────────────
    hop_counts = defaultdict(lambda: defaultdict(int))
    for cr in cluster_results:
        for hop in cr["hops"]:
            hop_counts[hop["hop_label"]][hop["outcome"]] += 1

    # ── Overall TP/FP/FN for should_cluster hops ──────────────────────────────
    all_hops = [h for cr in cluster_results for h in cr["hops"]]

    tp = sum(1 for h in all_hops if h["outcome"] == "TP")
    fn = sum(1 for h in all_hops if h["outcome"] == "FN")
    tn = sum(1 for h in all_hops if h["outcome"] == "TN")
    om = sum(1 for h in all_hops if h["outcome"] == "OVER_MERGE")

    def safe_div(a, b): return a / b if b else 0.0

    precision = safe_div(tp, tp + om)   # of all clustered hops, how many should be
    recall    = safe_div(tp, tp + fn)   # of all should-cluster hops, how many caught
    f1        = safe_div(2 * precision * recall, precision + recall)

    return {
        "counts": {
            "total_chains":    len(gt_clusters),
            "total_hops":      len(all_hops),
            "TP":              tp,
            "FN":              fn,
            "TN":              tn,
            "OVER_MERGE":      om,
        },
        "metrics": {
            "precision": precision,
            "recall":    recall,
            "f1":        f1,
        },
        "hop_breakdown":    dict(hop_counts),
        "over_merge_cases": over_merge_cases,
        "under_merge_cases": under_merge_cases[:10],   # cap for display
    }


# ── Pretty print ───────────────────────────────────────────────────────────────
def print_result(threshold: float, result: dict):
    c = result["counts"]
    m = result["metrics"]

    print(f"\n{'─'*64}")
    print(f"  threshold = {threshold}")
    print(f"{'─'*64}")
    print(f"  Chains evaluated : {c['total_chains']}")
    print(f"  Total hops       : {c['total_hops']}")
    print()
    print(f"  TP  (correctly grouped with original)  : {c['TP']}")
    print(f"  FN  (should cluster, missed)            : {c['FN']}")
    print(f"  TN  (correctly excluded far-drift hops) : {c['TN']}")
    print(f"  OVER_MERGE (chained in via bridge node) : {c['OVER_MERGE']}")
    print()
    print(f"  Precision : {m['precision']:.1%}  (TP / TP+OVER_MERGE)")
    print(f"  Recall    : {m['recall']:.1%}  (TP / TP+FN)")
    print(f"  F1        : {m['f1']:.1%}")

    # Per-hop breakdown
    print(f"\n  HOP BREAKDOWN")
    print(f"  {'Hop':<10}  {'TP':>4}  {'FN':>4}  {'TN':>4}  {'OVER_MERGE':>10}")
    print(f"  {'─'*10}  {'─'*4}  {'─'*4}  {'─'*4}  {'─'*10}")
    for hop in ["hop1", "hop2", "hop3", "hop4"]:
        counts = result["hop_breakdown"].get(hop, {})
        tp_ = counts.get("TP", 0)
        fn_ = counts.get("FN", 0)
        tn_ = counts.get("TN", 0)
        om_ = counts.get("OVER_MERGE", 0)
        print(f"  {hop:<10}  {tp_:>4}  {fn_:>4}  {tn_:>4}  {om_:>10}")

    # Over-merge detail (the interesting failure mode)
    if result["over_merge_cases"]:
        print(f"\n  ⚠  OVER-MERGE CASES (Union-Find chaining pulled in far-drift hops):")
        for case in result["over_merge_cases"]:
            print(f"     cluster={case['cluster_id']}  "
                  f"original={case['original_id']}  "
                  f"hop={case['hop_label']}  "
                  f"type={case['chain_type']}")
            print(f"     → {case['note']}")
    else:
        print(f"\n  ✓  No over-merge cases detected at this threshold")

    # Under-merge sample
    if result["under_merge_cases"]:
        print(f"\n  MISSED (under-merge) sample:")
        for case in result["under_merge_cases"][:3]:
            print(f"     cluster={case['cluster_id']}  "
                  f"{case['hop_label']} of {case['original_id']}  "
                  f"chain={case['chain_type']}")


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Evaluate chained duplicate detection")
    parser.add_argument("--url",       default=DEFAULT_URL)
    parser.add_argument("--threshold", type=float, help="Single threshold to evaluate")
    parser.add_argument("--all",       action="store_true", help="Run all thresholds")
    parser.add_argument("--json",      action="store_true", help="Dump full results to JSON")
    args = parser.parse_args()

    gt_map, gt_clusters = load_ground_truth(CHAIN_GT_PATH)

    chain_types = defaultdict(int)
    for members in gt_clusters.values():
        chain_types[members[0]["chain_type"]] += 1
    print(f"Chain ground truth loaded: {len(gt_clusters)} chains")
    for k, v in sorted(chain_types.items()):
        print(f"  {k}: {v} chains")

    thresholds = THRESHOLDS if args.all else [args.threshold or 0.90]

    all_results = {}
    for t in thresholds:
        print(f"\nFetching service results at threshold={t} ...", end=" ", flush=True)
        try:
            service_clusters = fetch_service_clusters(args.url, t)
            print(f"{len(service_clusters)} clusters returned")
        except Exception as e:
            print(f"FAILED: {e}")
            continue

        result = evaluate_chains(service_clusters, gt_map, gt_clusters)
        print_result(t, result)
        all_results[t] = result

    # Summary table
    if len(all_results) > 1:
        print(f"\n{'═'*64}")
        print(f"  CHAIN EVAL SUMMARY")
        print(f"{'═'*64}")
        print(f"  {'Threshold':>10}  {'Precision':>10}  {'Recall':>8}  {'F1':>8}  "
              f"{'TP':>4}  {'FN':>4}  {'TN':>4}  {'OVER_MERGE':>10}")
        print(f"  {'─'*10}  {'─'*10}  {'─'*8}  {'─'*8}  "
              f"{'─'*4}  {'─'*4}  {'─'*4}  {'─'*10}")
        for t, r in sorted(all_results.items()):
            m = r["metrics"]
            c = r["counts"]
            print(f"  {t:>10.2f}  "
                  f"{m['precision']:>10.1%}  "
                  f"{m['recall']:>8.1%}  "
                  f"{m['f1']:>8.1%}  "
                  f"{c['TP']:>4}  "
                  f"{c['FN']:>4}  "
                  f"{c['TN']:>4}  "
                  f"{c['OVER_MERGE']:>10}")

        print(f"\n  Legend:")
        print(f"    TP         = hop correctly grouped with original")
        print(f"    FN         = hop should have grouped but didn't (under-merge)")
        print(f"    TN         = far-drift hop correctly excluded")
        print(f"    OVER_MERGE = far-drift hop wrongly pulled in via chain bridge")
        print(f"    Precision  = TP / (TP + OVER_MERGE)")
        print(f"    Recall     = TP / (TP + FN)")

    if args.json:
        out = "chain_eval_results.json"
        with open(out, "w") as f:
            json.dump({str(k): v for k, v in all_results.items()}, f, indent=2)
        print(f"\nFull results written to {out}")


if __name__ == "__main__":
    main()
