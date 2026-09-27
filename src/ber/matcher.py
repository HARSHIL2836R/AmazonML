"""Pairwise matcher: LightGBM on string, token and context features."""

from __future__ import annotations

import lightgbm as lgb
import numpy as np
import polars as pl

from .config import MatcherConfig


def _matrix(df: pl.DataFrame, cols: list[str]) -> np.ndarray:
    return df.select([pl.col(c).cast(pl.Float32) for c in cols]).to_numpy()


def train_matcher(train: pl.DataFrame, valid: pl.DataFrame, cols: list[str], cfg: MatcherConfig, seed: int, threads: int, log=print) -> lgb.Booster:
    params = dict(
        objective="binary", learning_rate=cfg.learning_rate, num_leaves=cfg.num_leaves,
        min_data_in_leaf=cfg.min_child_samples, feature_fraction=cfg.feature_fraction,
        bagging_fraction=cfg.bagging_fraction, bagging_freq=1, seed=seed,
        num_threads=threads, verbose=-1,
    )
    dtr = lgb.Dataset(_matrix(train, cols), label=train["y"].to_numpy(), feature_name=cols)
    dva = lgb.Dataset(_matrix(valid, cols), label=valid["y"].to_numpy(), reference=dtr)
    model = lgb.train(
        params, dtr, num_boost_round=cfg.n_estimators, valid_sets=[dva],
        callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(0)],
    )
    log(f"  matcher: {model.best_iteration} trees, valid logloss {model.best_score['valid_0']['binary_logloss']:.5f}")
    return model


def predict(model: lgb.Booster, df: pl.DataFrame, cols: list[str]) -> np.ndarray:
    return model.predict(_matrix(df, cols), num_iteration=model.best_iteration).astype(np.float32)


def importance(model: lgb.Booster, top: int = 25) -> list[tuple[str, float]]:
    gain = model.feature_importance("gain")
    names = model.feature_name()
    order = np.argsort(-gain)[:top]
    total = gain.sum()
    return [(names[i], round(float(gain[i] / total), 4)) for i in order]
