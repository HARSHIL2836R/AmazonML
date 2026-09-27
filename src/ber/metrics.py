"""Challenge metric and blocking diagnostics.

F_beta per Source-1 entity, macro-averaged over all Source-1 entities,
singletons included. With tp matches predicted correctly, np predicted and
nt true:

    F_beta = (1 + b^2) * tp / (b^2 * nt + np)      (tp > 0)
    1.0 if nt == np == 0, else 0.0 when tp == 0.
"""

from __future__ import annotations

import polars as pl

BETA = 0.5


def per_entity_f(pred: pl.DataFrame, truth: pl.DataFrame, s1_ids: pl.Series, beta: float = BETA) -> pl.DataFrame:
    """pred, truth: (i1, it) pairs. s1_ids: every evaluated i1. Returns i1, tp, n_pred, n_true, f."""
    b2 = beta * beta
    base = pl.DataFrame({"i1": s1_ids.cast(pl.Int32)})
    n_pred = pred.group_by("i1").agg(pl.len().alias("n_pred"))
    n_true = truth.group_by("i1").agg(pl.len().alias("n_true"))
    tp = pred.join(truth, on=["i1", "it"]).group_by("i1").agg(pl.len().alias("tp"))
    df = (
        base.join(n_pred, on="i1", how="left")
        .join(n_true, on="i1", how="left")
        .join(tp, on="i1", how="left")
        .fill_null(0)
    )
    return df.with_columns(
        pl.when((pl.col("n_true") == 0) & (pl.col("n_pred") == 0))
        .then(1.0)
        .when(pl.col("tp") == 0)
        .then(0.0)
        .otherwise((1 + b2) * pl.col("tp") / (b2 * pl.col("n_true") + pl.col("n_pred")))
        .alias("f")
    )


def macro_f(pred: pl.DataFrame, truth: pl.DataFrame, s1_ids: pl.Series, beta: float = BETA) -> float:
    return float(per_entity_f(pred, truth, s1_ids, beta)["f"].mean())


def blocking_report(cands: pl.DataFrame, truth: pl.DataFrame, s1_ids: pl.Series, n_targets: int) -> dict:
    """Recall ceiling and size of a candidate set restricted to `s1_ids`."""
    ids = pl.DataFrame({"i1": s1_ids.cast(pl.Int32)})
    c = cands.select("i1", "it").join(ids, on="i1")
    t = truth.join(ids, on="i1")
    hit = c.join(t, on=["i1", "it"]).height
    per = ids.join(c.group_by("i1").agg(pl.len().alias("n")), on="i1", how="left").fill_null(0)["n"]
    oracle = macro_f(c.join(t, on=["i1", "it"]), t, s1_ids)
    return {
        "pairs": c.height,
        "cand_per_s1_mean": round(float(per.mean()), 3),
        "cand_per_s1_p50": float(per.median()),
        "cand_per_s1_p95": float(per.quantile(0.95)),
        "pair_recall": round(hit / max(t.height, 1), 5),
        "reduction_ratio": 1 - c.height / (len(s1_ids) * n_targets),
        "oracle_macro_f05": round(oracle, 5),
    }
