"""End-to-end orchestration with fingerprinted on-disk caching of every stage.

Fold roles on the training split (folds come from the S1 entity id, see data.py):

    folds 1-3   train the pruner and the matcher
    fold 4      picks the pruner threshold, early-stops the matcher and tunes
                the match threshold
    fold 0      untouched until the end; every reported score comes from it

Artifacts land in `artifacts/<split>[_w<frac>]/`, models in
`artifacts/models/train[_w<frac>]/`, and the shared script dictionary in
`artifacts/models/script_dict.json`. `predict` always uses the full-scale models.
"""

from __future__ import annotations

import dataclasses
import hashlib
import inspect
import json
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import polars as pl

from . import blocking, data, decide, features, lexicon, matcher, normalize, prune, translit
from .config import Config
from .metrics import blocking_report, macro_f, per_entity_f
from .normalize import NON_LATIN, normalize_records
from .translit import ScriptDictionary

T0 = time.time()


def log(msg: str) -> None:
    print(f"[{time.time() - T0:7.0f}s] {msg}", flush=True)


def _script_dict_path(cfg: Config) -> Path:
    """Shared by every run: fit once from the training pairs outside the val fold."""
    d = cfg.work_dir / "models"
    d.mkdir(parents=True, exist_ok=True)
    return d / "script_dict.json"


def _models_dir(cfg: Config) -> Path:
    """Pruner, matcher and thresholds, one folder per training run, so a quick
    `--world 0.1` run never overwrites the full-scale models `predict` uses."""
    tag = "train" if cfg.world_frac >= 1.0 else f"train_w{cfg.world_frac:g}"
    d = cfg.work_dir / "models" / tag
    d.mkdir(parents=True, exist_ok=True)
    return d


# --------------------------------------------------------------------------
# Cache fingerprints
# --------------------------------------------------------------------------
# Every cached parquet has a sibling `.fp` file holding a hash of the code,
# settings and inputs that produced it. A mismatch recomputes that stage, so a
# changed blocking setting or a retrained pruner can never be paired with
# stale candidates. Only what can change a stage's output goes into its hash.

def _hash(*parts) -> str:
    return hashlib.sha1(json.dumps(parts, sort_keys=True, default=str).encode()).hexdigest()[:16]


def _code(*objs) -> list[str]:
    return [_hash(inspect.getsource(o)) for o in objs]


def _file(path: Path) -> str:
    return hashlib.sha1(Path(path).read_bytes()).hexdigest()[:16]


def _fresh(path: Path, fp: str) -> bool:
    fp_path = path.with_name(path.name + ".fp")
    return path.exists() and fp_path.exists() and fp_path.read_text() == fp


def _stamp(path: Path, fp: str) -> None:
    path.with_name(path.name + ".fp").write_text(fp)


def norm_fingerprint(cfg: Config, split: str) -> str:
    return _hash("norm", split, cfg.world_frac, cfg.n_folds,
                 _code(normalize, lexicon, translit, data), _file(_script_dict_path(cfg)))


RETRIEVAL_FIELDS = [f.name for f in dataclasses.fields(Config().blocking) if f.name not in ("prune_keep_share", "max_per_s1", "chunk_rows")]


def retrieval_fingerprint(cfg: Config, split: str) -> list:
    return [norm_fingerprint(cfg, split), {k: getattr(cfg.blocking, k) for k in RETRIEVAL_FIELDS},
            _code(blocking, features.context_features)]


# --------------------------------------------------------------------------
# Stage 0: load + normalize
# --------------------------------------------------------------------------

def fit_script_dictionary(cfg: Config) -> ScriptDictionary:
    """Learn the native-script dictionary from training pairs outside the validation fold."""
    s1, tgt = data.load_split(cfg, "train")
    truth = data.load_truth(cfg)
    s1 = data.add_folds(s1, cfg.n_folds).filter(pl.col("fold") != cfg.val_fold)
    native = tgt.filter(pl.col("business_name").str.contains(NON_LATIN)).select("entity_id", pl.col("business_name").alias("n2"))
    pairs = (
        truth.join(native, on="entity_id")
        .join(s1.select(pl.col("entity_id").alias("source1_entity_id"), pl.col("business_name").alias("n1")), on="source1_entity_id")
    )
    sd = ScriptDictionary.fit(pairs["n1"].to_list(), pairs["n2"].to_list())
    log(f"script dictionary: {len(sd.table):,} entries from {pairs.height:,} native-script pairs")
    return sd


def _normalize_chunked(df: pl.DataFrame, sd: ScriptDictionary, chunk: int = 1_000_000) -> pl.DataFrame:
    parts = []
    for s in range(0, df.height, chunk):
        parts.append(normalize_records(df.slice(s, chunk), sd))
        log(f"  normalized {min(s + chunk, df.height):,}/{df.height:,}")
    return pl.concat(parts, rechunk=True)


def prepare(cfg: Config, split: str):
    """Normalized S1/target tables (+ ground-truth positions for train)."""
    d = cfg.run_dir(split)
    f1, fT, fY = d / "s1n.parquet", d / "tgtn.parquet", d / "truth_pos.parquet"
    sd_path = _script_dict_path(cfg)
    if sd_path.exists():
        sd = ScriptDictionary.load(sd_path)
    elif split == "train":
        sd = fit_script_dictionary(cfg)
        sd.save(sd_path)
    else:
        raise FileNotFoundError(f"{sd_path} missing: run `train` first")

    fp = norm_fingerprint(cfg, split)
    if _fresh(f1, fp) and _fresh(fT, fp) and (split != "train" or _fresh(fY, fp)):
        s1n, tgtn = pl.read_parquet(f1), pl.read_parquet(fT)
        return s1n, tgtn, (pl.read_parquet(fY) if split == "train" else None)

    # Normalize 1M rows at a time and one source at a time, freeing raw text as
    # we go: the regex chain materializes intermediate string columns, and the
    # full split does not fit in memory twice.
    if cfg.world_frac < 1.0:
        s1, tgt = data.load_split(cfg, split)
        truth = data.load_truth(cfg) if split == "train" else None
        s1, tgt, _ = data.sample_world(s1, tgt, truth, cfg.world_frac)
        del truth
        s1n = _normalize_chunked(s1, sd)
        del s1
        tgtn = _normalize_chunked(tgt, sd)
        del tgt
    else:
        s1n = _normalize_chunked(data.load_source(cfg, split, 1), sd)
        tgtn = pl.concat([_normalize_chunked(data.load_source(cfg, split, k), sd) for k in (2, 3)], rechunk=True)
    log(f"{split}: S1={s1n.height:,} targets={tgtn.height:,}")
    if split == "train":
        s1n = data.add_folds(s1n, cfg.n_folds)
    s1n.write_parquet(f1)
    tgtn.write_parquet(fT)
    _stamp(f1, fp)
    _stamp(fT, fp)
    tp = None
    if split == "train":
        truth = data.load_truth(cfg).join(s1n.select(pl.col("entity_id").alias("source1_entity_id")), on="source1_entity_id")
        tp = data.truth_positions(s1n, tgtn, truth)
        del truth
        tp.write_parquet(fY)
        _stamp(fY, fp)
    log(f"normalized {split}")
    return s1n, tgtn, tp


# --------------------------------------------------------------------------
# Stage 1 + 2: candidate generation
# --------------------------------------------------------------------------

def stage1(cfg: Config, split: str, s1n: pl.DataFrame, tgtn: pl.DataFrame, tp: pl.DataFrame | None,
           keep_s1: np.ndarray | None = None) -> pl.DataFrame:
    """Retrieval + context features, one country at a time to bound memory.

    `keep_s1` (train only) drops pairs of S1 entities that no later step uses,
    after the context features have seen the full competition.
    """
    path = cfg.run_dir(split) / "stage1.parquet"
    fp = _hash("stage1", retrieval_fingerprint(cfg, split),
               _hash(np.sort(keep_s1).tolist()) if keep_s1 is not None else None)
    if _fresh(path, fp):
        return pl.read_parquet(path)
    blocks = (blocking.s1_blocks(s1n), blocking.target_blocks(s1n, tgtn))
    n_missing = int((blocks[1] == "").sum())
    log(f"  targets without a block after state inference: {n_missing:,} ({n_missing / max(tgtn.height, 1):.2%})")
    keep = pl.DataFrame({"i1": keep_s1.astype(np.int32)}) if keep_s1 is not None else None
    parts = []
    tgt_src = tgtn["src"].to_numpy()
    for country in sorted(s1n["country"].unique().to_list()):
        pairs = blocking.retrieve(s1n, tgtn, cfg.blocking, cfg.threads, log=log, countries=[country], blocks=blocks)
        pairs = features.context_features(pairs, tgt_src)
        if tp is not None:
            pairs = pairs.join(tp.with_columns(pl.lit(1, pl.Int8).alias("y")), on=["i1", "it"], how="left").with_columns(pl.col("y").fill_null(0))
        if keep is not None:
            pairs = pairs.join(keep, on="i1")
        parts.append(pairs)
    pairs = pl.concat(parts).sort("i1")
    pairs.write_parquet(path)
    _stamp(path, fp)
    return pairs


def candidates(cfg: Config, split: str, s1n: pl.DataFrame, tgtn: pl.DataFrame, pruner: lgb.Booster, threshold: float) -> pl.DataFrame:
    """Stage 1 + stage 2 per country, keeping only the pruned candidates in memory."""
    path = cfg.run_dir(split) / "candidates.parquet"
    fp = _hash("candidates", retrieval_fingerprint(cfg, split), _code(prune, features.sibling_features),
               _file(_models_dir(cfg) / "pruner.txt"), threshold, cfg.blocking.max_per_s1)
    if _fresh(path, fp):
        return pl.read_parquet(path)
    blocks = (blocking.s1_blocks(s1n), blocking.target_blocks(s1n, tgtn))
    tgt_src = tgtn["src"].to_numpy()
    parts = []
    for country in sorted(s1n["country"].unique().to_list()):
        pairs = features.context_features(
            blocking.retrieve(s1n, tgtn, cfg.blocking, cfg.threads, log=log, countries=[country], blocks=blocks), tgt_src
        )
        kept = prune.select(prune.score(pairs, pruner), threshold, cfg.blocking.max_per_s1)
        log(f"  [{country}] stage 1 {pairs.height:,} -> candidates {kept.height:,}")
        parts.append(kept)
        del pairs
    cands = features.sibling_features(pl.concat(parts).sort("i1"), tgtn)
    cands.write_parquet(path)
    _stamp(path, fp)
    return cands


def _with_fold(pairs: pl.DataFrame, s1n: pl.DataFrame) -> pl.DataFrame:
    folds = s1n.select(pl.int_range(pl.len(), dtype=pl.Int32).alias("i1"), "fold")
    return pairs.join(folds, on="i1", how="left")


def _sample_s1(pairs: pl.DataFrame, max_s1: int, seed: int) -> pl.DataFrame:
    ids = pairs["i1"].unique()
    if len(ids) > max_s1:
        ids = ids.sample(max_s1, seed=seed)
    return pairs.join(pl.DataFrame({"i1": ids}), on="i1")


# --------------------------------------------------------------------------
# Features in chunks
# --------------------------------------------------------------------------

def featurize(pairs: pl.DataFrame, s1n: pl.DataFrame, tgtn: pl.DataFrame, idf: features.NameIdf, chunk: int = 1_000_000) -> pl.DataFrame:
    out = [features.string_features(pairs.slice(s, chunk), s1n, tgtn, idf) for s in range(0, pairs.height, chunk)]
    return pl.concat(out) if out else pairs


# --------------------------------------------------------------------------
# Train
# --------------------------------------------------------------------------

def entity_subsets(cfg: Config, s1n: pl.DataFrame):
    """Fold roles and the S1 entities each role uses.

    Evaluates on at most max_eval_s1 entities of the val and tune folds and
    trains on at most max_train_s1 entities of the other folds. Returns
    (tune_fold, train_folds, val_ids, tune_ids, keep), where keep is every S1
    row whose stage-1 pairs are worth storing.
    """
    tune_fold = (cfg.val_fold + cfg.n_folds - 1) % cfg.n_folds
    train_folds = [f for f in range(cfg.n_folds) if f not in (cfg.val_fold, tune_fold)]
    rows = s1n.select(pl.int_range(pl.len(), dtype=pl.Int32).alias("i1"), "fold")
    pick = lambda df, n: df.sample(min(n, df.height), seed=cfg.seed)["i1"]
    val_ids = pick(rows.filter(pl.col("fold") == cfg.val_fold), cfg.max_eval_s1).sort()
    tune_ids = pick(rows.filter(pl.col("fold") == tune_fold), cfg.max_eval_s1).sort()
    train_ids = pick(rows.filter(pl.col("fold").is_in(train_folds)), cfg.matcher.max_train_s1).sort()
    keep = np.concatenate([val_ids.to_numpy(), tune_ids.to_numpy(), train_ids.to_numpy()])
    return tune_fold, train_folds, val_ids, tune_ids, keep


def train(cfg: Config) -> dict:
    s1n, tgtn, tp = prepare(cfg, "train")
    tune_fold, train_folds, val_ids, tune_ids, keep = entity_subsets(cfg, s1n)
    pairs = _with_fold(stage1(cfg, "train", s1n, tgtn, tp, keep_s1=keep), s1n)
    tp_f = _with_fold(tp, s1n)
    report = {"config": {k: str(v) for k, v in vars(cfg).items()}}

    report["stage1_val"] = blocking_report(pairs, tp, val_ids, tgtn.height)
    log(f"stage 1 (val fold): {report['stage1_val']}")

    # ---- pruner
    tr = pairs.filter(pl.col("fold").is_in(train_folds))
    pruner = prune.train_pruner(tr, cfg.seed, cfg.threads)
    pruner.save_model(str(_models_dir(cfg) / "pruner.txt"))
    pairs = prune.score(pairs, pruner)
    thr = prune.threshold_for_recall(pairs.filter(pl.col("fold") == tune_fold), cfg.blocking.prune_keep_share)
    cands = features.sibling_features(prune.select(pairs, thr, cfg.blocking.max_per_s1), tgtn)
    del pairs
    report["prune_threshold"] = thr
    report["candidates_val"] = blocking_report(cands, tp, val_ids, tgtn.height)
    log(f"candidates (val fold), p_prune >= {thr:.4f}: {report['candidates_val']}")

    # ---- matcher
    idf = features.NameIdf(s1n, tgtn)
    tr = featurize(cands.filter(pl.col("fold").is_in(train_folds)), s1n, tgtn, idf)
    tu = featurize(cands.filter(pl.col("fold") == tune_fold), s1n, tgtn, idf)
    va = featurize(cands.filter(pl.col("fold") == cfg.val_fold), s1n, tgtn, idf)
    del cands
    cols = features.feature_columns(tr)
    log(f"features: {len(cols)} columns, train pairs {tr.height:,}")
    model = matcher.train_matcher(tr, tu, cols, cfg.matcher, cfg.seed, cfg.threads, log=log)
    model.save_model(str(_models_dir(cfg) / "matcher.txt"), num_iteration=model.best_iteration)
    report["importance"] = matcher.importance(model)
    log(f"top features: {report['importance'][:12]}")

    tu = tu.with_columns(pl.Series("p", matcher.predict(model, tu, cols)))
    va = va.with_columns(pl.Series("p", matcher.predict(model, va, cols)))
    t_best, f_tune, _ = decide.tune_threshold(tu, tp_f.filter(pl.col("fold") == tune_fold).select("i1", "it"), tune_ids)
    pred = decide.predict_pairs(va, t_best)
    val_truth = tp_f.join(pl.DataFrame({"i1": val_ids}), on="i1").select("i1", "it")
    f_val = macro_f(pred, val_truth, val_ids)
    f_val_noassign = macro_f(decide.predict_pairs(va, t_best, use_assignment=False), val_truth, val_ids)
    _, f_val_oracle_t, curve = decide.tune_threshold(va, val_truth, val_ids)
    per = per_entity_f(pred, val_truth, val_ids).join(
        s1n.select(pl.int_range(pl.len(), dtype=pl.Int32).alias("i1"), "country"), on="i1"
    )
    report.update({
        "match_threshold": t_best,
        "macro_f05_tune_fold": f_tune,
        "macro_f05_val": f_val,
        "macro_f05_val_without_assignment": f_val_noassign,
        "macro_f05_val_best_threshold_in_hindsight": f_val_oracle_t,
        "val_by_country": {r[0]: round(r[1], 5) for r in per.group_by("country").agg(pl.col("f").mean()).iter_rows()},
        "val_singletons": {
            "share": float((per["n_true"] == 0).mean()),
            "f": float(per.filter(pl.col("n_true") == 0)["f"].mean()),
            "f_non_singletons": float(per.filter(pl.col("n_true") > 0)["f"].mean()),
        },
        "threshold_curve_val": curve,
    })
    (_models_dir(cfg) / "decision.json").write_text(json.dumps({"match_threshold": t_best, "prune_threshold": thr, "features": cols}, indent=1))
    (cfg.run_dir("train") / "report.json").write_text(json.dumps(report, indent=1, default=str))
    log(f"VAL macro F0.5 = {f_val:.5f} (threshold {t_best}, tuned on fold {tune_fold}); "
        f"without assignment {f_val_noassign:.5f}; by country {report['val_by_country']}")
    va.write_parquet(cfg.run_dir("train") / "val_scored.parquet")
    return report


# --------------------------------------------------------------------------
# Predict
# --------------------------------------------------------------------------

def _write_lists(path: Path, s1n: pl.DataFrame, tgtn: pl.DataFrame, pairs: pl.DataFrame, col: str) -> None:
    ids = (
        pairs.select("i1", "it")
        .join(tgtn.select(pl.int_range(pl.len(), dtype=pl.Int32).alias("it"), pl.col("entity_id").alias("t")), on="it")
        .sort("i1", "t")
        .group_by("i1", maintain_order=True).agg(pl.col("t").str.join(","))
    )
    out = (
        s1n.select(pl.int_range(pl.len(), dtype=pl.Int32).alias("i1"), pl.col("entity_id").alias("source1_entity_id"))
        .join(ids, on="i1", how="left")
        .select("source1_entity_id", pl.col("t").fill_null("").alias(col))
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    out.write_csv(path, separator="\t", quote_style="never")
    log(f"wrote {path} ({out.height:,} rows, {pairs.height:,} ids)")


def predict(cfg: Config) -> None:
    md = _models_dir(cfg)
    dec = json.loads((md / "decision.json").read_text())
    pruner = lgb.Booster(model_file=str(md / "pruner.txt"))
    model = lgb.Booster(model_file=str(md / "matcher.txt"))
    s1n, tgtn, _ = prepare(cfg, "test")
    cands = candidates(cfg, "test", s1n, tgtn, pruner, dec["prune_threshold"])
    log(f"test candidates: {cands.height:,} ({cands.height / s1n.height:.2f}/S1)")
    _write_lists(cfg.output_dir / "candidate_pairs.tsv", s1n, tgtn, cands, "candidate_entity_ids")

    idf = features.NameIdf(s1n, tgtn)
    scored = []
    for s in range(0, cands.height, 1_000_000):
        ch = features.string_features(cands.slice(s, 1_000_000), s1n, tgtn, idf)
        scored.append(ch.select("i1", "it").with_columns(pl.Series("p", model.predict(ch.select([pl.col(c).cast(pl.Float32) for c in dec["features"]]).to_numpy()).astype(np.float32))))
        log(f"  scored {min(s + 1_000_000, cands.height):,}/{cands.height:,}")
    scored = pl.concat(scored)
    scored.write_parquet(cfg.run_dir("test") / "test_scored.parquet")
    matches = decide.predict_pairs(scored, dec["match_threshold"])
    log(f"test matches: {matches.height:,} ({matches.height / s1n.height:.2f}/S1), "
        f"S1 with no match: {1 - matches['i1'].n_unique() / s1n.height:.3f}")
    _write_lists(cfg.output_dir / "matching_results.tsv", s1n, tgtn, matches, "matched_entity_ids")
