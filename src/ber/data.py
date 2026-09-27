"""Loading the challenge TSVs, parquet caching, folds and world subsampling.

Conventions used across the package:

- `s1` is the Source-1 table, `tgt` is Sources 2 and 3 concatenated. Both keep
  the raw `entity_id` string; pair tables refer to rows by integer position
  (`i1` into s1, `it` into tgt) so that ~10^8 candidate pairs fit in memory.
- `truth` is the exploded ground truth: one row per (source1_entity_id, entity_id).
"""

from __future__ import annotations

from pathlib import Path

import polars as pl

from .config import Config

TEXT_COLS = ["entity_id", "business_name", "business_address", "country"]


def read_tsv(path: Path) -> pl.DataFrame:
    # quote_char=None: names contain quotes; the files are plain TSV.
    return pl.read_csv(path, separator="\t", quote_char=None, infer_schema=False)


def _cached(cfg: Config, name: str, src: Path) -> pl.DataFrame:
    out = cfg.cache_dir / f"{name}.parquet"
    if not out.exists():
        out.parent.mkdir(parents=True, exist_ok=True)
        read_tsv(src).write_parquet(out)
    return pl.read_parquet(out)


def load_source(cfg: Config, split: str, k: int) -> pl.DataFrame:
    df = _cached(cfg, f"{split}_s{k}", cfg.data_dir / split / f"{split}_source{k}.tsv")
    return df.select(TEXT_COLS)


def load_split(cfg: Config, split: str) -> tuple[pl.DataFrame, pl.DataFrame]:
    s1 = load_source(cfg, split, 1)
    tgt = pl.concat([load_source(cfg, split, 2), load_source(cfg, split, 3)])
    return s1, tgt


def load_truth(cfg: Config) -> pl.DataFrame:
    gt = _cached(cfg, "train_gt", cfg.data_dir / "train" / "train_ground_truth.tsv")
    return (
        gt.with_columns(pl.col("matched_entity_ids").fill_null("").str.split(","))
        .explode("matched_entity_ids", empty_as_null=True)
        .drop_nulls()
        .filter(pl.col("matched_entity_ids") != "")
        .rename({"matched_entity_ids": "entity_id"})
    )


def id_bucket(col: str, n: int) -> pl.Expr:
    """Deterministic bucket from the numeric part of an entity id (ids look random)."""
    return pl.col(col).str.slice(3).cast(pl.Int64) % n


def add_folds(s1: pl.DataFrame, n_folds: int) -> pl.DataFrame:
    return s1.with_columns(id_bucket("entity_id", n_folds).cast(pl.Int8).alias("fold"))


def sample_world(
    s1: pl.DataFrame, tgt: pl.DataFrame, truth: pl.DataFrame | None, frac: float
) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame | None]:
    """Shrink the problem while keeping its shape.

    Keeps a `frac` share of Source-1 entities with all their true matches, plus
    the same share of every other target (distractors and targets whose S1 was
    dropped), so the matched/unmatched mix per S1 stays as in the full data.
    Candidate density is lower than at full scale, so blocking recall measured
    on a small world is optimistic.
    """
    if frac >= 1.0:
        return s1, tgt, truth
    buckets = 10_000
    keep = int(frac * buckets)
    s1 = s1.filter(id_bucket("entity_id", buckets) < keep)
    if truth is None:
        tgt = tgt.filter(id_bucket("entity_id", buckets) < keep)
        return s1, tgt, None
    truth = truth.join(s1.select(pl.col("entity_id").alias("source1_entity_id")), on="source1_entity_id")
    matched = truth.select("entity_id")
    others = tgt.join(matched, on="entity_id", how="anti").filter(id_bucket("entity_id", buckets) < keep)
    tgt = pl.concat([tgt.join(matched, on="entity_id"), others])
    return s1, tgt, truth


def truth_positions(s1: pl.DataFrame, tgt: pl.DataFrame, truth: pl.DataFrame) -> pl.DataFrame:
    """Ground truth as (i1, it) row positions."""
    return (
        truth.join(s1.select(pl.col("entity_id").alias("source1_entity_id"), pl.int_range(pl.len(), dtype=pl.Int32).alias("i1")), on="source1_entity_id")
        .join(tgt.select("entity_id", pl.int_range(pl.len(), dtype=pl.Int32).alias("it")), on="entity_id")
        .select("i1", "it")
    )
