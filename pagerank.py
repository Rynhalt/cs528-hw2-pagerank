#!/usr/bin/env python3
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

from graph import Graph


@dataclass
class PageRankResult:
    pr: List[float]
    iterations: int
    final_sum: float
    final_rel_change: float


def pagerank_iterative(
    g: Graph,
    damping: float = 0.85,
    base: float = 0.15,
    rel_sum_tol: float = 0.005,   # dif from previous 0.5%
    max_iters: int = 200,
    min_iters: int = 1,
    handle_dangling: bool = True,
) -> PageRankResult:
    """
    Original iterative PageRank algorithm (random surfer) as specified in HW2:

      PR_new(A) = base/n + damping * sum_{T -> A} PR(T) / C(T)

    Stopping criterion (as in prompt):
      stop when relative change in sum(PR) <= rel_sum_tol between iterations.
    """
    n = g.n
    if n <= 0:
        return PageRankResult(pr=[], iterations=0, final_sum=0.0, final_rel_change=0.0)

    # Initialize uniformly
    pr = [1.0 / n] * n
    prev_sum = sum(pr)

    teleport = base / n

    for it in range(1, max_iters + 1):
        # We handle page with no outgoing links
        # When we hit a dead end, we distribute the "deadend"'s PR  uniformly to other nodes
        dangling_mass = 0.0
        if handle_dangling:
            for t in range(n):
                if g.out_degree[t] == 0:
                    dangling_mass += pr[t]
            dangling_add = damping * (dangling_mass / n)
        else:
            dangling_add = 0.0

        pr_new = [0.0] * n

        # Core update per node A using incoming links
        for a in range(n):
            acc = 0.0
            for t in g.in_links[a]:
                c = g.out_degree[t]
                if c > 0:
                    acc += pr[t] / c
            pr_new[a] = teleport + damping * acc + dangling_add
            #page rank update

        new_sum = sum(pr_new)
        rel_change = abs(new_sum - prev_sum) / prev_sum if prev_sum != 0 else float("inf")

        pr = pr_new
        prev_sum = new_sum

        if it >= min_iters and rel_change <= rel_sum_tol:
            return PageRankResult(pr=pr, iterations=it, final_sum=new_sum, final_rel_change=rel_change)

    # If we hit max_iters
    return PageRankResult(pr=pr, iterations=max_iters, final_sum=prev_sum, final_rel_change=rel_change)


def top_k(pr: List[float], k: int = 5) -> List[Tuple[int, float]]:
    """
    Return top-k (node_id, pagerank) sorted descending by pagerank.
    """
    idx = list(range(len(pr)))
    idx.sort(key=lambda i: pr[i], reverse=True)
    return [(i, pr[i]) for i in idx[:k]]