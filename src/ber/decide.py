"""From pair probabilities to final matches.

Two rules, both tuned on the validation fold:

1. Assignment. Every target belongs to at most one S1 entity (true for all
   7.64M training pairs), so each target is kept only for the S1 where it
   scores highest. This removes the classic double-merge false positive, a
   near-duplicate target claimed by two similar S1 records.
2. Threshold. A pair survives if p >= t. F0.5 weights precision 2x, so the
   best t sits well above 0.5; the grid search finds it directly on the
   macro per-entity metric, singletons included.
"""

from __future__ import annotations

import numpy as np
import polars as pl

from .metrics import macro_f


def assign(pairs: pl.DataFrame) -> pl.DataFrame:
    """Keep, for each target, only its highest-probability S1."""
    return pairs.sort("p", descending=True).unique("it", keep="first")


def predict_pairs(pairs: pl.DataFrame, threshold: float, use_assignment: bool = True) -> pl.DataFrame:
    p = assign(pairs) if use_assignment else pairs
    return p.filter(pl.col("p") >= threshold).select("i1", "it")


def _poisson_binomial(P: np.ndarray) -> np.ndarray:
    """Row-wise distribution of the number of successes. P: (n, m) -> (n, m + 1)."""
    n, m = P.shape
    dist = np.zeros((n, m + 1))
    dist[:, 0] = 1.0
    for j in range(m):
        p = P[:, j:j + 1]
        dist[:, 1:] = dist[:, 1:] * (1 - p) + dist[:, :-1] * p
        dist[:, 0] *= 1 - P[:, j]
    return dist


def expected_f_select(pairs: pl.DataFrame, beta: float = 0.5, p_floor: float = 0.02, max_m: int = 12) -> pl.DataFrame:
    """Per-S1 expected-F_beta optimal prediction set (Lewis; Jansche 2007).

    After the per-target assignment, each S1's candidates are sorted by p. Under
    independent Bernoulli labels the optimal set is the top k for some k in
    0..m (k = 0 predicts no match), so E[F(top-k)] is computed exactly for every
    k from the Poisson-binomial distributions of true links inside and outside
    the top k, and the best k is kept. Candidates with p < p_floor are treated
    as certain non-matches. Assumes calibrated p; true links outside the
    candidate set are ignored.
    """
    b2 = beta * beta
    ranked = (
        assign(pairs).filter(pl.col("p") >= p_floor)
        .sort(["i1", "p"], descending=[False, True])
        .with_columns(pl.int_range(pl.len()).over("i1").alias("_k"))
        .filter(pl.col("_k") < max_m)
    )
    wide = ranked.group_by("i1").agg(pl.col("p"), pl.col("it"))
    keep_i1, keep_it = [], []
    for m, grp in wide.with_columns(pl.col("p").list.len().alias("m")).group_by("m"):
        m = int(m[0])
        P = np.array(grp["p"].to_list(), dtype=np.float64)          # (n, m), sorted desc
        best_k = np.zeros(len(P), np.int64)
        best_v = _poisson_binomial(P)[:, 0]                          # k = 0: all false
        a = np.arange(m + 1)[:, None]
        b = np.arange(m + 1)[None, :]
        for k in range(1, m + 1):
            inside = _poisson_binomial(P[:, :k])                     # (n, k + 1)
            outside = _poisson_binomial(P[:, k:])                    # (n, m - k + 1)
            aa, bb = a[: k + 1], b[:, : m - k + 1]
            f = (1 + b2) * aa / (b2 * (aa + bb) + k)                 # (k + 1, m - k + 1)
            v = np.einsum("na,ab,nb->n", inside, f, outside)
            better = v > best_v
            best_v = np.where(better, v, best_v)
            best_k = np.where(better, k, best_k)
        its = grp["it"].to_list()
        for i1, row, k in zip(grp["i1"].to_list(), its, best_k):
            if k:
                keep_i1.extend([i1] * int(k))
                keep_it.extend(row[: int(k)])
    return pl.DataFrame({"i1": keep_i1, "it": keep_it}, schema={"i1": pl.Int32, "it": pl.Int32})


def tune_threshold(pairs: pl.DataFrame, truth: pl.DataFrame, s1_ids: pl.Series, grid=None, use_assignment: bool = True) -> tuple[float, float, list]:
    grid = grid if grid is not None else np.round(np.arange(0.20, 0.96, 0.025), 3)
    base = assign(pairs) if use_assignment else pairs
    curve = []
    for t in grid:
        pred = base.filter(pl.col("p") >= t).select("i1", "it")
        curve.append((float(t), macro_f(pred, truth, s1_ids)))
    best_t, best_f = max(curve, key=lambda x: x[1])
    return best_t, best_f, curve
