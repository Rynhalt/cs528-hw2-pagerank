
import math

from graph import build_graph_from_edges
from pagerank import pagerank_iterative, top_k


def assert_close(a: float, b: float, tol: float = 1e-6):
    assert abs(a - b) <= tol, f"{a} not close to {b} (tol={tol})"


def test_cycle_graph_equal_ranks():
    """
    3-cycle: 0->1->2->0
    By symmetry, all nodes should have equal rank.
    """
    n = 3
    edges = [(0, 1), (1, 2), (2, 0)]
    g = build_graph_from_edges(n, edges)

    res = pagerank_iterative(g, handle_dangling=True, min_iters=10, max_iters=200)

    # ranks should be ~equal
    assert_close(res.pr[0], res.pr[1], tol=1e-6)
    assert_close(res.pr[1], res.pr[2], tol=1e-6)

    # non-negative
    assert all(x >= 0.0 for x in res.pr)


def test_star_graph_center_highest():
    """
    Star-in graph: 1..k all point to 0.
    Expect node 0 to have the highest PageRank.
    """
    n = 6
    edges = [(1, 0), (2, 0), (3, 0), (4, 0), (5, 0)]
    g = build_graph_from_edges(n, edges)

    res = pagerank_iterative(g, handle_dangling=True, min_iters=10, max_iters=200)
    winners = top_k(res.pr, k=1)
    top_node, top_score = winners[0]

    assert top_node == 0
    # check that center node beats every leaf
    for i in range(1, n):
        assert res.pr[0] > res.pr[i]


def test_sum_reasonable_when_handling_dangling():
    """
    If you redistribute dangling mass, sum(PR) should stay close to 1.
    (This depends on your implementation choices; adjust tolerance if needed.)
    """
    n = 4
    # Includes a dangling node (3 has no outlinks)
    edges = [(0, 1), (1, 2), (2, 1)]
    g = build_graph_from_edges(n, edges)

    res = pagerank_iterative(g, handle_dangling=True, min_iters=10, max_iters=200)

    s = sum(res.pr)
    #check that dangling nodes are handled
    assert abs(s - 1.0) < 1e-3