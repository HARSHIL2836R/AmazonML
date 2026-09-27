"""Candidate generation, stage 2: supervised meta-blocking.

Stage 1 is generous on purpose (~20 pairs per S1 at ~98% recall), because
each target contributes its top S1 records and a quarter of targets match
nothing. This stage scores every stage-1 pair with a small LightGBM model
that only sees retrieval-graph features: channel cosines, ranks in both
directions, and how far the pair sits from the best score of its S1 and of
its target. It is the supervised meta-blocking idea (Papadakis et al.): the
blocking graph itself carries enough signal to discard most comparisons
before any string is compared.

The survivors, capped at `max_per_s1` per S1 entity, are the candidate set
written to candidate_pairs.tsv and the only pairs the matcher scores.
"""

from __future__ import annotations

import lightgbm as lgb
import numpy as np
import polars as pl

PRUNE_FEATURES = [
    "name_cos", "addr_cos", "ret_score",
    "name_fwd", "name_rev", "addr_fwd", "addr_rev", "blend_fwd", "blend_rev", "same_block",
    "s1_n_cands", "t_n_cands", "s1_gap", "t_gap", "s1_rank", "t_rank", "t_name_gap", "t_addr_gap",
    "s1_src_n_cands", "s1_src_rank", "s1_src_gap", "s1_top2_gap", "t_top2_gap",
]


def _matrix(df: pl.DataFrame) -> np.ndarray:
    return df.select([pl.col(c).cast(pl.Float32) for c in PRUNE_FEATURES]).to_numpy()


def train_pruner(train: pl.DataFrame, seed: int, threads: int) -> lgb.Booster:
    params = dict(
        objective="binary", learning_rate=0.1, num_leaves=63, min_data_in_leaf=100,
        feature_fraction=0.9, bagging_fraction=0.8, bagging_freq=1, seed=seed,
        num_threads=threads, verbose=-1,
    )
    ds = lgb.Dataset(_matrix(train), label=train["y"].to_numpy(), feature_name=PRUNE_FEATURES)
    return lgb.train(params, ds, num_boost_round=200)


def score(pairs: pl.DataFrame, model: lgb.Booster, chunk: int = 5_000_000) -> pl.DataFrame:
    p = np.concatenate([
        model.predict(_matrix(pairs.slice(s, chunk))) for s in range(0, pairs.height, chunk)
    ]) if pairs.height else np.zeros(0)
    return pairs.with_columns(pl.Series("p_prune", p.astype(np.float32)))


def threshold_for_recall(scored: pl.DataFrame, keep_share: float) -> float:
    """Lowest p_prune that keeps `keep_share` of the positives stage 1 found."""
    pos = np.sort(scored.filter(pl.col("y") == 1)["p_prune"].to_numpy())
    if len(pos) == 0:
        return 0.0
    return float(pos[int((1.0 - keep_share) * len(pos))])


def select(scored: pl.DataFrame, threshold: float, max_per_s1: int) -> pl.DataFrame:
    """Apply the threshold and the per-S1 cap."""
    return (
        scored.filter(pl.col("p_prune") >= threshold)
        .with_columns(pl.col("p_prune").rank("ordinal", descending=True).over("i1").alias("_r"))
        .filter(pl.col("_r") <= max_per_s1)
        .drop("_r")
    )
