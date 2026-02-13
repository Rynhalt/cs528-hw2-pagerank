#!/usr/bin/env python3
from __future__ import annotations

from dataclasses import dataclass
from array import array
from typing import Iterable, List, Sequence, Tuple


@dataclass
class Graph:
    """
    Directed graph for PageRank.

    n: number of nodes (0..n-1)
    in_links[a]: array of nodes t such that t -> a
    out_degree[t]: number of outgoing edges from t (C(t))
    """
    n: int
    in_links: List[array]
    out_degree: List[int]

    @property
    def num_edges(self) -> int:
        return sum(len(x) for x in self.in_links)


def build_graph_from_outlinks(n: int, out_links: Sequence[Sequence[int]]) -> Graph:
    """
    Build Graph from an out adjacency list.

    out_links[t] = iterable of a targets.
    """
    in_links: List[array] = [array("I") for _ in range(n)]
    out_degree: List[int] = [0] * n

    for t in range(n):
        targets = out_links[t]
        out_degree[t] = len(targets)
        for a in targets:
            if 0 <= a < n:
                in_links[a].append(t)

    return Graph(n=n, in_links=in_links, out_degree=out_degree)


def build_graph_from_edges(n: int, edges: Iterable[Tuple[int, int]]) -> Graph:
    """
    Build Graph from an edge list iterable (t, a) meaning t -> a.
    """
    in_links: List[array] = [array("I") for _ in range(n)]
    out_degree: List[int] = [0] * n

    for t, a in edges:
        if 0 <= t < n and 0 <= a < n:
            in_links[a].append(t)
            out_degree[t] += 1

    return Graph(n=n, in_links=in_links, out_degree=out_degree)