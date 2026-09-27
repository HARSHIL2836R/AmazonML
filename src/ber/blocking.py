"""Candidate generation, stage 1: partitioned sparse retrieval.

Three measurements on the training data shape this module:

1. Matched records agree on state. US pairs agree 100% when both sides carry a
   state; India agrees 98.8%, and the disagreements are Telangana against
   Andhra Pradesh (the 2014 split, Hyderabad addresses), so those two codes
   share one block. Every S1 record has a state; 3.8% of targets do not.
2. Cost is concentrated in a few features. Over the US, the trigram "and" is 6%
   of the name product and "st", "rd", "dr", "ave" plus state tokens are over
   70% of the address product. Dropping features whose document frequency is
   above 1% of the block keeps 98% of name features and 14% of the cost.
3. About 4.6% of true pairs share no name word (invented DBA names, domains)
   and are linked only by the address, so name and address are retrieved as
   separate channels instead of one blended vector.

Layout:

    block      (country, state) with ap/tg merged. Targets with no detected
               state get one inferred from their address tokens when a token is
               almost always seen with one state in Source 1 ("bordeaux" -> naq).
               Targets still without a state query every S1 of their country
               on the name channel.
    channels   name:  TF-IDF over character 3-grams of `name_nospace`
               addr:  TF-IDF over address tokens and house numbers ("#123")
               blend: both blocks, weighted w / (1 - w); the best single
                      ranking (reverse top-1 alone finds ~96% of true pairs)
    directions forward (each S1 keeps its top k targets) and reverse (each
               target keeps its top k S1 records); the reverse direction uses
               the at-most-one-S1-per-target structure of the data.

The union of all channel/direction lists is scored exactly on both channels
and handed to `prune.py`.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np
import polars as pl
import scipy.sparse as sp
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize as l2_normalize
from sparse_dot_topn import sp_matmul_topn

from .config import BlockingConfig

MERGED_STATES = {"tg": "ap"}  # Telangana/Andhra Pradesh share a block


# --------------------------------------------------------------------------
# Block keys
# --------------------------------------------------------------------------

def _first_state(col: str) -> pl.Expr:
    first = pl.col(col).str.split(" ").list.first().fill_null("")
    return first.replace(MERGED_STATES)


def s1_blocks(s1n: pl.DataFrame) -> pl.Series:
    return s1n.select(_first_state("state").alias("block")).to_series()


def target_blocks(s1n: pl.DataFrame, tgtn: pl.DataFrame, min_count: int = 20, purity: float = 0.95) -> pl.Series:
    """Block per target; stateless targets get a state from address tokens that pin one down in S1."""
    s1_tok = (
        s1n.select("country", _first_state("state").alias("block"), pl.col("addr").str.split(" ").alias("tok"))
        .explode("tok")
        .filter(pl.col("tok").str.len_chars() > 2)
    )
    purity_tbl = (
        s1_tok.group_by("country", "tok", "block").agg(pl.len().alias("n"))
        .with_columns(pl.col("n").sum().over("country", "tok").alias("tot"))
        .filter((pl.col("n") == pl.col("n").max().over("country", "tok")) & (pl.col("tot") >= min_count))
        .filter(pl.col("n") / pl.col("tot") >= purity)
        .select("country", "tok", pl.col("block").alias("inferred"), (pl.col("n") / pl.col("tot")).alias("p"))
    )
    del s1_tok
    block = tgtn.select(_first_state("state").alias("block")).to_series()
    missing = (
        tgtn.select("country", pl.col("addr").str.split(" ").alias("tok"))
        .with_row_index("_r").filter(block == "")
    )
    votes = (
        missing.explode("tok").join(purity_tbl, on=["country", "tok"])
        .group_by("_r", "inferred").agg(pl.col("p").sum().alias("score"))
        .sort("score", descending=True).group_by("_r").first()
    )
    arr = block.to_numpy().astype(object)
    arr[votes["_r"].to_numpy()] = votes["inferred"].to_numpy()
    return pl.Series("block", arr, dtype=pl.String)


# --------------------------------------------------------------------------
# Vectors
# --------------------------------------------------------------------------

def _addr_docs(df: pl.DataFrame) -> list[str]:
    return df.select(
        pl.concat_str(
            [pl.col("addr"), pl.col("addr_nums").str.replace_all(r"(\d+)", "#${1}")], separator=" "
        )
    ).to_series().to_list()


def _vectorize(docs: list[str], analyzer, max_df_frac: float, min_cap: int) -> sp.csr_matrix:
    """TF-IDF, then drop features more frequent than the cap and re-normalize rows."""
    kw = dict(ngram_range=(3, 3), analyzer="char") if analyzer == "char3" else dict(analyzer=str.split)
    vec = TfidfVectorizer(min_df=2, sublinear_tf=True, dtype=np.float32, norm=None, **kw)
    try:
        X = vec.fit_transform(docs)
    except ValueError:  # empty vocabulary
        return sp.csr_matrix((len(docs), 1), dtype=np.float32)
    df = np.bincount(X.indices, minlength=X.shape[1])
    cap = max(int(max_df_frac * len(docs)), min_cap)
    keep = np.flatnonzero(df <= cap)
    return l2_normalize(X[:, keep].tocsr(), copy=False).astype(np.float32)


@dataclass
class _Lists:
    i1: list
    it: list
    score: list
    rank: list

    def add(self, a, b, s, r):
        self.i1.append(a); self.it.append(b); self.score.append(s); self.rank.append(r)

    def frame(self, score_name: str, rank_name: str) -> pl.DataFrame:
        cat = lambda xs, dt: np.concatenate(xs).astype(dt) if xs else np.zeros(0, dt)
        return pl.DataFrame({
            "i1": cat(self.i1, np.int32), "it": cat(self.it, np.int32),
            score_name: cat(self.score, np.float32), rank_name: cat(self.rank, np.int8),
        }).unique(["i1", "it"], keep="first")


def _topk(A: sp.csr_matrix, BT: sp.csr_matrix, k: int, min_score: float, threads: int):
    """Row-wise top-k of A @ BT (BT is the transposed index, CSR). Returns (row, col, score, rank)."""
    if A.shape[0] == 0 or BT.shape[1] == 0 or k <= 0:
        e = np.zeros(0, np.int32)
        return e, e, np.zeros(0, np.float32), np.zeros(0, np.int8)
    C = sp_matmul_topn(A, BT, top_n=k, threshold=min_score, sort=True, n_threads=threads)
    counts = np.diff(C.indptr)
    rows = np.repeat(np.arange(C.shape[0], dtype=np.int32), counts)
    ranks = (np.arange(C.nnz) - np.repeat(C.indptr[:-1], counts)).astype(np.int8)
    return rows, C.indices.astype(np.int32), C.data.astype(np.float32), ranks


def _pair_cos(X1: sp.csr_matrix, XT: sp.csr_matrix, r: np.ndarray, c: np.ndarray, chunk: int = 500_000) -> np.ndarray:
    out = np.empty(len(r), np.float32)
    for s in range(0, len(r), chunk):
        out[s:s + chunk] = np.asarray(X1[r[s:s + chunk]].multiply(XT[c[s:s + chunk]]).sum(axis=1)).ravel()
    return out


# --------------------------------------------------------------------------
# Retrieval
# --------------------------------------------------------------------------

CHANNELS = ("name", "addr", "blend")


def retrieve(
    s1n: pl.DataFrame, tgtn: pl.DataFrame, cfg: BlockingConfig, threads: int, log=print,
    countries: list[str] | None = None, blocks: tuple[pl.Series, pl.Series] | None = None,
) -> pl.DataFrame:
    """Stage-1 candidate pairs.

    Returns one row per (i1, it), positions into the full s1n / tgtn, with for
    each channel c in {name, addr, blend}: `{c}_fwd` (rank of the target in the
    S1's list, -1 if absent) and `{c}_rev` (rank of the S1 in the target's list,
    -1 if absent); exact `name_cos` and `addr_cos`; and `same_block` (False for
    the stateless fallback). `countries` restricts the work to those labels;
    `blocks` passes precomputed (s1_blocks, target_blocks).
    """
    t0 = time.time()
    b1, bT = blocks if blocks is not None else (s1_blocks(s1n), target_blocks(s1n, tgtn))
    keys1 = pl.DataFrame({"country": s1n["country"], "block": b1}).with_row_index("_i")
    keysT = pl.DataFrame({"country": tgtn["country"], "block": bT}).with_row_index("_i")
    if countries is not None:
        keys1 = keys1.filter(pl.col("country").is_in(countries))
        keysT = keysT.filter(pl.col("country").is_in(countries))
    g1 = {(r[0], r[1]): np.asarray(r[2], np.int64) for r in keys1.group_by("country", "block").agg("_i").iter_rows()}
    gT = {(r[0], r[1]): np.asarray(r[2], np.int64) for r in keysT.group_by("country", "block").agg("_i").iter_rows()}
    cols = ["name_nospace", "addr", "addr_nums"]
    s1v, tgtv = s1n.select(cols), tgtn.select(cols)

    parts = []
    for key in sorted(g1, key=lambda k: -len(g1[k])):
        if key[1] == "" or key not in gT:
            continue
        parts.append(_retrieve_block(s1v, tgtv, g1[key], gT[key], cfg, threads, same_block=True))
    # Fallback: stateless targets query every S1 of their country, name channel only.
    for (country, block), it in gT.items():
        if block != "":
            continue
        i1 = keys1.filter(pl.col("country") == country)["_i"].to_numpy().astype(np.int64)
        if len(i1):
            parts.append(_retrieve_block(s1v, tgtv, i1, it, cfg, threads, same_block=False))

    pairs = pl.concat(parts, how="diagonal_relaxed") if parts else pl.DataFrame()
    n1 = keys1.height
    log(f"  retrieval {countries or 'all'}: {pairs.height:,} pairs ({pairs.height / max(n1, 1):.1f}/S1) in {time.time() - t0:.0f}s")
    return pairs


def _retrieve_block(s1v: pl.DataFrame, tgtv: pl.DataFrame, g1: np.ndarray, gT: np.ndarray, cfg: BlockingConfig, threads: int, same_block: bool) -> pl.DataFrame:
    a, b = s1v[g1], tgtv[gT]
    n1 = a.height
    mats = {
        "name": _vectorize(a["name_nospace"].to_list() + b["name_nospace"].to_list(), "char3", cfg.name_max_df, cfg.min_df_cap),
    }
    if same_block:
        mats["addr"] = _vectorize(_addr_docs(a) + _addr_docs(b), "tokens", cfg.addr_max_df, cfg.min_df_cap)
        w = cfg.blend_name_weight
        mats["blend"] = sp.hstack(
            [mats["name"] * np.float32(np.sqrt(w)), mats["addr"] * np.float32(np.sqrt(1 - w))], format="csr"
        )

    frames = []
    for ch, X in mats.items():
        X1, XT = X[:n1], X[n1:]
        k_fwd = getattr(cfg, f"k_{ch}_fwd") if same_block else 0
        k_rev = getattr(cfg, f"k_{ch}_rev")
        fwd, rev = _Lists([], [], [], []), _Lists([], [], [], [])
        if k_fwd > 0:
            XTt = XT.T.tocsr()
            for s in range(0, X1.shape[0], cfg.chunk_rows):
                r, c, v, k = _topk(X1[s:s + cfg.chunk_rows], XTt, k_fwd, cfg.min_score, threads)
                fwd.add(g1[r + s], gT[c], v, k)
            del XTt
        X1t = X1.T.tocsr()
        for s in range(0, XT.shape[0], cfg.chunk_rows):
            r, c, v, k = _topk(XT[s:s + cfg.chunk_rows], X1t, k_rev, cfg.min_score, threads)
            rev.add(g1[c], gT[r + s], v, k)
        del X1t
        f = fwd.frame(f"{ch}_s", f"{ch}_fwd").drop(f"{ch}_s")
        rv = rev.frame(f"{ch}_s", f"{ch}_rev").drop(f"{ch}_s")
        frames.append(f.join(rv, on=["i1", "it"], how="full", coalesce=True))

    pairs = frames[0]
    for f in frames[1:]:
        pairs = pairs.join(f, on=["i1", "it"], how="full", coalesce=True)

    # Exact cosines on every channel for every pair (block-local row positions).
    pos1 = np.full(g1.max() + 1, -1, np.int64); pos1[g1] = np.arange(n1)
    posT = np.full(gT.max() + 1, -1, np.int64); posT[gT] = np.arange(b.height)
    r = pos1[pairs["i1"].to_numpy()]
    c = posT[pairs["it"].to_numpy()]
    cols = {}
    for ch in ("name", "addr"):
        if ch in mats:
            cols[f"{ch}_cos"] = _pair_cos(mats[ch][:n1], mats[ch][n1:], r, c)
    rank_cols = [f"{ch}_{d}" for ch in CHANNELS for d in ("fwd", "rev")]
    pairs = pairs.with_columns(
        *[pl.Series(k, v) for k, v in cols.items()],
        pl.lit(same_block).alias("same_block"),
    )
    return pairs.with_columns([pl.col(c).fill_null(-1).cast(pl.Int8) if c in pairs.columns else pl.lit(-1, pl.Int8).alias(c) for c in rank_cols])
