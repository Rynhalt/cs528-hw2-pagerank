#!/usr/bin/env python3
import argparse
import statistics
import time
import importlib.util
from graph import build_graph_from_outlinks
from pagerank import pagerank_iterative, top_k
from typing import Dict, List


def percentile(sorted_vals: List[int], p: float) -> float:
    """Linear interpolation percentile. p in [0,1]."""
    if not sorted_vals:
        return 0.0
    if p <= 0:
        return float(sorted_vals[0])
    if p >= 1:
        return float(sorted_vals[-1])

    k = (len(sorted_vals) - 1) * p
    f = int(k)
    c = min(f + 1, len(sorted_vals) - 1)
    if f == c:
        return float(sorted_vals[f])

    return float(sorted_vals[f] + (k - f) * (sorted_vals[c] - sorted_vals[f]))


def compute_stats(counts: List[int]) -> Dict[str, float]:
    s = sorted(counts)
    n = len(s)
    return {
        "avg": (sum(s) / n) if n else 0.0,
        "median": float(statistics.median(s)) if n else 0.0,
        "min": float(s[0]) if n else 0.0,
        "max": float(s[-1]) if n else 0.0,
        # quintile cutpoints (20/40/60/80 percentiles)
        "q20": percentile(s, 0.20),
        "q40": percentile(s, 0.40),
        "q60": percentile(s, 0.60),
        "q80": percentile(s, 0.80),
    }


def load_gcs_client_module(path: str = "gcs-client.py"):
    spec = importlib.util.spec_from_file_location("gcs_client_mod", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load module from {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bucket", required=True, help="GCS bucket name (no gs://)")
    ap.add_argument("--n", type=int, default=None, help="Expected number of pages (e.g. 20000)")
    ap.add_argument("--max-files", type=int, default=None, help="Debug: only read first K files")
    ap.add_argument("--gcs-client-path", default="gcs-client.py", help="Path to gcs-client.py")
    ap.add_argument("--workers", type=int, default=32, help="Download worker threads (if supported by gcs-client)")
    ap.add_argument("--max-iters", type=int, default=200)
    ap.add_argument("--min-iters", type=int, default=10)
    args = ap.parse_args()

    gcs = load_gcs_client_module(args.gcs_client_path)

    t0 = time.time()
    graph_counts = gcs.load_link_counts_from_gcs(
        bucket_name=args.bucket,
        expected_n=args.n,
        max_files=args.max_files,
    )
    t1 = time.time()

    out_stats = compute_stats(graph_counts.out_counts)
    in_stats = compute_stats(graph_counts.in_counts)

    print("=== Basic Link Statistics ===")
    print(f"Pages (n): {graph_counts.n}")
    if args.max_files is not None:
        print(f"Note: max-files={args.max_files} (debug mode)")

    print("\nOutgoing link counts stats:")
    print(out_stats)

    print("\nIncoming link counts stats:")
    print(in_stats)

    print("\nTiming:")
    print(f"Load+parse seconds: {t1 - t0:.3f}")

    # Build graph (requires graph_counts.out_links)
    t2 = time.time()
    g = build_graph_from_outlinks(graph_counts.n, graph_counts.out_links)
    t3 = time.time()

    t4 = time.time()
    res = pagerank_iterative(g, max_iters=args.max_iters, min_iters=args.min_iters)
    top5 = top_k(res.pr, k=5)
    t5 = time.time()

    print("\n=== PageRank ===")
    print(f"Edges: {g.num_edges}")
    print(f"Iterations: {res.iterations}")
    print(f"Final sum(PR): {res.final_sum:.6f}")
    print(f"Final rel sum change: {res.final_rel_change:.6f}")
    print("Top 5 pages by PR:")
    for pid, score in top5:
        print(f"{pid}.html\t{score:.8f}")

    print("\n=== Timing ===")
    print(f"Load+parse seconds: {t1 - t0:.3f}")
    print(f"Build graph seconds: {t3 - t2:.3f}")
    print(f"PageRank seconds: {t5 - t4:.3f}")
    print(f"Total seconds: {(t5 - t0):.3f}")


if __name__ == "__main__":
    main()