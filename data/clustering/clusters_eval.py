# eval.py — Duplicate Detection Evaluator
#
# Usage:
#   python eval.py --threshold 0.90
#   python eval.py --threshold 0.85 --url http://localhost:8000
#   python eval.py --all   # runs all thresholds: 0.80, 0.85, 0.90, 0.92, 0.95
#
# Requires:
#   pip install requests
#   ground_truth.json in the same directory

import json
import argparse
import requests
from collections import defaultdict
from itertools import combinations

GROUND_TRUTH_PATH = "ground_truth.json"
DEFAULT_URL       = "http://localhost:8000"
THRESHOLDS        = [0.80, 0.85, 0.90, 0.92, 0.95]


# ── Load ground truth ──────────────────────────────────────────────────────────
def load_ground_truth(path: str):
    with open(path) as f:
        records = json.load(f)

    # prompt_id → cluster_id
    gt_map = {r["prompt_id"]: r["cluster_id"] for r in records}

    # cluster_id → set of prompt_ids  (only clusters with >1 member)
    cluster_members = defaultdict(set)
    for r in records:
        cluster_members[r["cluster_id"]].add(r["prompt_id"])
    gt_clusters = {cid: members for cid, members in cluster_members.items()
                   if len(members) > 1}

    return gt_map, gt_clusters


# ── Fetch service results ──────────────────────────────────────────────────────
def fetch_service_clusters(base_url: str, threshold: float) -> list[set]:
    url = f"{base_url}/api/analysis/duplicates?threshold={threshold}"
    resp = requests.get(url, timeout=120)
    resp.raise_for_status()
    data = resp.json()

    # API returns {"clusters": [...], "latency": {...}}
    cluster_list = data["clusters"] if isinstance(data, dict) else data

    clusters = []
    for cluster in cluster_list:
        member_ids = {p["prompt_id"] for p in cluster["prompts"]}
        clusters.append(member_ids)
    return clusters


# ── Pair helpers ───────────────────────────────────────────────────────────────
def to_pairs(members: set) -> set[frozenset]:
    return {frozenset(pair) for pair in combinations(members, 2)}


# ── Core eval ─────────────────────────────────────────────────────────────────
def evaluate(service_clusters: list[set], gt_map: dict, gt_clusters: dict) -> dict:

    # ── Cluster-level TP / FP ─────────────────────────────────────────────────
    tp_clusters, fp_clusters = [], []
    detected_gt_ids = set()

    for sc in service_clusters:
        # Which ground truth cluster does each member belong to?
        gt_ids = {gt_map[pid] for pid in sc if pid in gt_map}

        if len(gt_ids) == 1:
            # All members point to the same ground truth cluster → TP
            tp_clusters.append(sc)
            detected_gt_ids.add(next(iter(gt_ids)))
        else:
            # Members span multiple ground truth clusters (or unknowns) → FP
            fp_clusters.append(sc)
            # Still mark any real clusters that were partially detected
            detected_gt_ids.update(gt_ids - {None})

    # ── Cluster-level FN ──────────────────────────────────────────────────────
    fn_clusters = {cid: members for cid, members in gt_clusters.items()
                   if cid not in detected_gt_ids}

    # ── Pair-level ────────────────────────────────────────────────────────────
    # Ground truth pairs = all pairs within each gt cluster
    gt_pairs: set[frozenset] = set()
    for members in gt_clusters.values():
        gt_pairs.update(to_pairs(members))

    # Service pairs = all pairs within each service cluster
    service_pairs: set[frozenset] = set()
    for sc in service_clusters:
        service_pairs.update(to_pairs(sc))

    tp_pairs = gt_pairs & service_pairs
    fp_pairs = service_pairs - gt_pairs
    fn_pairs = gt_pairs - service_pairs

    # ── Metrics ───────────────────────────────────────────────────────────────
    def safe_div(a, b): return a / b if b else 0.0

    n_tp_c  = len(tp_clusters)
    n_fp_c  = len(fp_clusters)
    n_fn_c  = len(fn_clusters)
    n_tp_p  = len(tp_pairs)
    n_fp_p  = len(fp_pairs)
    n_fn_p  = len(fn_pairs)

    c_precision = safe_div(n_tp_c, n_tp_c + n_fp_c)
    c_recall    = safe_div(n_tp_c, n_tp_c + n_fn_c)
    c_f1        = safe_div(2 * c_precision * c_recall, c_precision + c_recall)

    p_precision = safe_div(n_tp_p, n_tp_p + n_fp_p)
    p_recall    = safe_div(n_tp_p, n_tp_p + n_fn_p)
    p_f1        = safe_div(2 * p_precision * p_recall, p_precision + p_recall)

    return {
        "counts": {
            "service_clusters":   len(service_clusters),
            "gt_clusters":        len(gt_clusters),
            "gt_pairs":           len(gt_pairs),
            "tp_clusters":        n_tp_c,
            "fp_clusters":        n_fp_c,
            "fn_clusters":        n_fn_c,
            "tp_pairs":           n_tp_p,
            "fp_pairs":           n_fp_p,
            "fn_pairs":           n_fn_p,
        },
        "metrics": {
            "cluster_precision":  c_precision,
            "cluster_recall":     c_recall,
            "cluster_f1":         c_f1,
            "pair_precision":     p_precision,
            "pair_recall":        p_recall,
            "pair_f1":            p_f1,
        },
        "detail": {
            "tp_clusters":        [sorted(c) for c in tp_clusters],
            "fp_clusters":        [sorted(c) for c in fp_clusters],
            "fn_clusters":        {cid: sorted(m) for cid, m in fn_clusters.items()},
        },
    }


# ── Pretty print ───────────────────────────────────────────────────────────────
def print_result(threshold: float, result: dict):
    c = result["counts"]
    m = result["metrics"]

    print(f"\n{'─'*60}")
    print(f"  threshold = {threshold}")
    print(f"{'─'*60}")
    print(f"  Service clusters found : {c['service_clusters']}")
    print(f"  Ground truth clusters  : {c['gt_clusters']}")
    print()
    print(f"  CLUSTER-LEVEL")
    print(f"    TP : {c['tp_clusters']}")
    print(f"    FP : {c['fp_clusters']}")
    print(f"    FN : {c['fn_clusters']}")
    print(f"    Precision : {m['cluster_precision']:.1%}")
    print(f"    Recall    : {m['cluster_recall']:.1%}")
    print(f"    F1        : {m['cluster_f1']:.1%}")
    print()
    print(f"  PAIR-LEVEL  (ground truth pairs: {c['gt_pairs']})")
    print(f"    TP : {c['tp_pairs']}")
    print(f"    FP : {c['fp_pairs']}")
    print(f"    FN : {c['fn_pairs']}")
    print(f"    Precision : {m['pair_precision']:.1%}")
    print(f"    Recall    : {m['pair_recall']:.1%}")
    print(f"    F1        : {m['pair_f1']:.1%}")

    # FP detail
    if result["detail"]["fp_clusters"]:
        print(f"\n  FALSE POSITIVE CLUSTERS ({c['fp_clusters']}):")
        for fp in result["detail"]["fp_clusters"][:5]:   # cap at 5
            print(f"    {fp}")
        if c["fp_clusters"] > 5:
            print(f"    ... and {c['fp_clusters'] - 5} more")

    # FN detail (missed clusters)
    if result["detail"]["fn_clusters"]:
        fn_items = list(result["detail"]["fn_clusters"].items())
        print(f"\n  MISSED CLUSTERS / FN ({c['fn_clusters']}):")
        for cid, members in fn_items[:5]:
            print(f"    cluster_id={cid}: {members}")
        if c["fn_clusters"] > 5:
            print(f"    ... and {c['fn_clusters'] - 5} more")


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Evaluate duplicate detection service")
    parser.add_argument("--url",       default=DEFAULT_URL, help="Base URL of the service")
    parser.add_argument("--threshold", type=float,          help="Single threshold to evaluate")
    parser.add_argument("--all",       action="store_true", help="Run all thresholds")
    parser.add_argument("--json",      action="store_true", help="Also dump full JSON results")
    args = parser.parse_args()

    gt_map, gt_clusters = load_ground_truth(GROUND_TRUTH_PATH)
    print(f"Ground truth loaded: {len(gt_clusters)} duplicate clusters, "
          f"{sum(len(v) for v in gt_clusters.values())} member prompts")

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

        result = evaluate(service_clusters, gt_map, gt_clusters)
        print_result(t, result)
        all_results[t] = result

    # Summary table across thresholds
    if len(all_results) > 1:
        print(f"\n{'═'*60}")
        print(f"  SUMMARY")
        print(f"{'═'*60}")
        print(f"  {'Threshold':>10}  {'C-P':>6}  {'C-R':>6}  {'C-F1':>6}  "
              f"{'P-P':>6}  {'P-R':>6}  {'P-F1':>6}")
        print(f"  {'─'*10}  {'─'*6}  {'─'*6}  {'─'*6}  {'─'*6}  {'─'*6}  {'─'*6}")
        for t, r in sorted(all_results.items()):
            m = r["metrics"]
            print(f"  {t:>10.2f}  "
                  f"{m['cluster_precision']:>6.1%}  "
                  f"{m['cluster_recall']:>6.1%}  "
                  f"{m['cluster_f1']:>6.1%}  "
                  f"{m['pair_precision']:>6.1%}  "
                  f"{m['pair_recall']:>6.1%}  "
                  f"{m['pair_f1']:>6.1%}")

    if args.json:
        out = "eval_results.json"
        with open(out, "w") as f:
            json.dump({str(k): v for k, v in all_results.items()}, f, indent=2)
        print(f"\nFull results written to {out}")


if __name__ == "__main__":
    main()