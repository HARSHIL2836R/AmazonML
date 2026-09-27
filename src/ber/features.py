"""Pairwise features for (S1, target) candidate pairs.

Three families:

    context   computed once over the whole pair table from retrieval scores:
              how this pair ranks among the S1's candidates and among the
              target's S1 options. Cheap, and the only features that see
              competition between candidates.
    string    rapidfuzz similarities on normalized name and address fields,
              computed chunk by chunk (rapidfuzz.process.cpdist, all cores).
    token     IDF-weighted word overlap on names, legal-form agreement,
              house-number agreement, state agreement, flags.

Country is deliberately not a feature: France has no training labels, and a
country indicator would only let the model learn US/India-specific offsets.
"""

from __future__ import annotations

import numpy as np
import polars as pl
import scipy.sparse as sp
from rapidfuzz import fuzz, process
from rapidfuzz.distance import JaroWinkler, Levenshtein
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize as l2_normalize

RETRIEVAL_COLS = ["name_cos", "addr_cos", "name_fwd", "name_rev", "addr_fwd", "addr_rev", "same_block"]

S1_FIELDS = ["name_core", "name_nospace", "name_full", "legal", "addr", "addr_nums", "state", "addr_null"]
T_FIELDS = S1_FIELDS + ["name_alt", "is_domain", "is_native", "src"]


def _margin_to_second(p: pl.DataFrame, group_col: str, rank_col: str, score_col: str, out_col: str) -> pl.DataFrame:
    """Gap between a group's best and second-best score, broadcast to every row
    of the group. A group of size 1 has no runner-up, so its margin is just the
    winning score itself (maximal, uncontested)."""
    top = p.filter(pl.col(rank_col) == 1).select(group_col, pl.col(score_col).alias("_top"))
    second = p.filter(pl.col(rank_col) == 2).select(group_col, pl.col(score_col).alias("_second"))
    m = top.join(second, on=group_col, how="left").with_columns(
        (pl.col("_top") - pl.col("_second").fill_null(0.0)).alias(out_col)
    ).select(group_col, out_col)
    return p.join(m, on=group_col, how="left")


def context_features(pairs: pl.DataFrame, tgt_src: np.ndarray) -> pl.DataFrame:
    """Competition features from retrieval scores, over the full pair table.

    `tgt_src` is the target table's `src` column (S2/S3), position-aligned with
    `it`; it lets the per-source listwise features tell "best S3 candidate for
    this S1" apart from "best candidate overall", which matters because an S1
    can legitimately take several S2 and several S3 matches at once.
    """
    p = pairs.with_columns(
        pl.col("name_cos").fill_null(0.0),
        pl.col("addr_cos").fill_null(0.0),
    ).with_columns((0.6 * pl.col("name_cos") + 0.4 * pl.col("addr_cos")).alias("ret_score"))
    p = p.with_columns(pl.Series("_src", tgt_src[p["it"].to_numpy()]))
    p = p.with_columns(
        pl.len().over("i1").alias("s1_n_cands").cast(pl.Int16),
        pl.len().over("it").alias("t_n_cands").cast(pl.Int16),
        (pl.col("ret_score").max().over("i1") - pl.col("ret_score")).alias("s1_gap"),
        (pl.col("ret_score").max().over("it") - pl.col("ret_score")).alias("t_gap"),
        pl.col("ret_score").rank("ordinal", descending=True).over("i1").cast(pl.Int16).alias("s1_rank"),
        pl.col("ret_score").rank("ordinal", descending=True).over("it").cast(pl.Int16).alias("t_rank"),
        (pl.col("name_cos").max().over("it") - pl.col("name_cos")).alias("t_name_gap"),
        (pl.col("addr_cos").max().over("it") - pl.col("addr_cos")).alias("t_addr_gap"),
        pl.len().over(["i1", "_src"]).alias("s1_src_n_cands").cast(pl.Int16),
        (pl.col("ret_score").max().over(["i1", "_src"]) - pl.col("ret_score")).alias("s1_src_gap"),
        pl.col("ret_score").rank("ordinal", descending=True).over(["i1", "_src"]).cast(pl.Int16).alias("s1_src_rank"),
    )
    p = _margin_to_second(p, "i1", "s1_rank", "ret_score", "s1_top2_gap")
    p = _margin_to_second(p, "it", "t_rank", "ret_score", "t_top2_gap")
    return p.drop("_src")


class NameIdf:
    """Word-level TF-IDF per country, for IDF-weighted name overlap."""

    def __init__(self, s1n: pl.DataFrame, tgtn: pl.DataFrame):
        self.vec = {}
        for country in set(s1n["country"].unique()) | set(tgtn["country"].unique()):
            docs = (
                s1n.filter(pl.col("country") == country)["name_core"].to_list()
                + tgtn.filter(pl.col("country") == country)["name_core"].to_list()
            )
            v = TfidfVectorizer(analyzer=str.split, min_df=1, dtype=np.float32, norm=None, use_idf=True, smooth_idf=True)
            v.fit(docs)
            self.vec[country] = v

    def overlap(self, country: np.ndarray, a: list[str], b: list[str]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Returns (cosine, share of a's IDF mass found in b, share of b's found in a)."""
        cos = np.zeros(len(a), np.float32)
        ca = np.zeros(len(a), np.float32)
        cb = np.zeros(len(a), np.float32)
        for c in np.unique(country):
            idx = np.flatnonzero(country == c)
            v = self.vec.get(c)
            if v is None:
                continue
            A = v.transform([a[i] for i in idx]); B = v.transform([b[i] for i in idx])
            An = l2_normalize(A); Bn = l2_normalize(B)
            cos[idx] = np.asarray(An.multiply(Bn).sum(axis=1)).ravel()
            A2 = A.multiply(A); B2 = B.multiply(B)
            Ab = A2.multiply(B > 0); Ba = B2.multiply(A > 0)
            with np.errstate(invalid="ignore", divide="ignore"):
                ca[idx] = np.nan_to_num(np.asarray(Ab.sum(axis=1)).ravel() / np.asarray(A2.sum(axis=1)).ravel())
                cb[idx] = np.nan_to_num(np.asarray(Ba.sum(axis=1)).ravel() / np.asarray(B2.sum(axis=1)).ravel())
        return cos, ca, cb

    def rarest_token_match(self, country: np.ndarray, a: list[str], b: list[str]) -> np.ndarray:
        """1.0 if a's single highest-idf (most distinguishing) token also occurs
        in b, else 0.0 (0.0 too when a has no known token). A near-twin business
        on the same street usually differs in exactly this token (the France
        "Defense" vs "Medi" case), so this is a sharper, threshold-free
        complement to the continuous IDF-overlap-share features above."""
        out = np.zeros(len(a), np.float32)
        for c in np.unique(country):
            idx = np.flatnonzero(country == c)
            v = self.vec.get(c)
            if v is None:
                continue
            A = v.transform([a[i] for i in idx]).tocsr()
            B = v.transform([b[i] for i in idx]).tocsr()
            counts = np.diff(A.indptr)
            if not (counts > 0).any():
                continue
            row_idx = np.repeat(np.arange(A.shape[0]), counts)
            order = np.lexsort((-A.data, row_idx))
            row_sorted = row_idx[order]
            first = np.concatenate(([True], row_sorted[1:] != row_sorted[:-1]))
            top_rows = row_sorted[first]
            top_cols = A.indices[order][first]
            match = np.asarray(B[top_rows, top_cols]).ravel() > 0
            out[idx[top_rows]] = match.astype(np.float32)
        return out


def _cp(scorer, a, b, workers):
    return process.cpdist(a, b, scorer=scorer, workers=workers, dtype=np.float32) / 100.0


def _cp_norm(scorer, a, b, workers):
    return process.cpdist(a, b, scorer=scorer.normalized_similarity, workers=workers, dtype=np.float32)


def _set_feats(a: list[str], b: list[str]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """For space-separated sets: (jaccard, a subset of b, both non-empty)."""
    jac = np.zeros(len(a), np.float32); sub = np.zeros(len(a), np.float32); both = np.zeros(len(a), np.float32)
    for i, (x, y) in enumerate(zip(a, b)):
        if x and y:
            sx, sy = set(x.split()), set(y.split())
            inter = len(sx & sy)
            jac[i] = inter / len(sx | sy)
            sub[i] = float(sx <= sy)
            both[i] = 1.0
    return jac, sub, both


def string_features(chunk: pl.DataFrame, s1n: pl.DataFrame, tgtn: pl.DataFrame, idf: NameIdf, workers: int = -1) -> pl.DataFrame:
    """Features for one chunk of pairs (columns i1, it plus retrieval/context columns)."""
    A = s1n.select(S1_FIELDS + ["country"])[chunk["i1"].to_numpy()]
    B = tgtn.select(T_FIELDS)[chunk["it"].to_numpy()]
    an, bn = A["name_core"].to_list(), B["name_core"].to_list()
    ans, bns = A["name_nospace"].to_list(), B["name_nospace"].to_list()
    aa, ba = A["addr"].to_list(), B["addr"].to_list()
    balt = B["name_alt"].to_list()

    f = {}
    f["n_ratio"] = _cp(fuzz.ratio, an, bn, workers)
    f["n_tset"] = _cp(fuzz.token_set_ratio, an, bn, workers)
    f["n_tsort"] = _cp(fuzz.token_sort_ratio, an, bn, workers)
    f["n_partial"] = _cp(fuzz.partial_ratio, an, bn, workers)
    f["n_nospace_ratio"] = _cp(fuzz.ratio, ans, bns, workers)
    f["n_nospace_jw"] = _cp_norm(JaroWinkler, ans, bns, workers)
    f["n_lev"] = _cp_norm(Levenshtein, an, bn, workers)
    f["n_full_ratio"] = _cp(fuzz.ratio, A["name_full"].to_list(), B["name_full"].to_list(), workers)
    f["n_alt_tset"] = _cp(fuzz.token_set_ratio, an, balt, workers)
    country_a = A["country"].to_numpy()
    cos, ca, cb = idf.overlap(country_a, an, bn)
    f["n_idf_cos"], f["n_idf_a_in_b"], f["n_idf_b_in_a"] = cos, ca, cb
    f["n_rare_a_in_b"] = idf.rarest_token_match(country_a, an, bn)
    f["n_rare_b_in_a"] = idf.rarest_token_match(country_a, bn, an)

    f["a_ratio"] = _cp(fuzz.ratio, aa, ba, workers)
    f["a_tset"] = _cp(fuzz.token_set_ratio, aa, ba, workers)
    f["a_partial"] = _cp(fuzz.partial_ratio, aa, ba, workers)

    jac, sub, both = _set_feats(A["addr_nums"].to_list(), B["addr_nums"].to_list())
    f["num_jac"], f["num_a_in_b"], f["num_both"] = jac, sub, both
    first_a = A["addr_nums"].str.split(" ").list.first()
    first_b = B["addr_nums"].str.split(" ").list.first()
    f["num_first_eq"] = ((first_a == first_b) & (first_a != "")).fill_null(False).to_numpy().astype(np.float32)

    jac, sub, both = _set_feats(A["legal"].to_list(), B["legal"].to_list())
    f["legal_jac"], f["legal_both"] = jac, both
    f["legal_any_b"] = (B["legal"] != "").to_numpy().astype(np.float32)

    jac, _, both = _set_feats(A["state"].to_list(), B["state"].to_list())
    f["state_eq"], f["state_both"] = jac, both

    f["n_len_a"] = A["name_core"].str.count_matches(" ").to_numpy().astype(np.float32) + 1
    f["n_len_b"] = B["name_core"].str.count_matches(" ").to_numpy().astype(np.float32) + 1
    f["b_addr_null"] = B["addr_null"].to_numpy().astype(np.float32)
    f["b_domain"] = B["is_domain"].to_numpy().astype(np.float32)
    f["b_native"] = B["is_native"].to_numpy().astype(np.float32)
    f["b_alias"] = (B["name_alt"] != "").to_numpy().astype(np.float32)
    f["b_src3"] = (B["src"] == 3).to_numpy().astype(np.float32)
    f["first_tok_eq"] = (
        (A["name_core"].str.split(" ").list.first() == B["name_core"].str.split(" ").list.first())
        .fill_null(False).to_numpy().astype(np.float32)
    )
    return chunk.hstack(pl.DataFrame(f))


def sibling_features(cands: pl.DataFrame, tgtn: pl.DataFrame, sim_threshold: float = 0.85, workers: int = -1, chunk: int = 2_000_000) -> pl.DataFrame:
    """Same-source sibling signal within each S1's surviving candidates.

    A target with a near-identical same-source neighbour among the same S1's
    candidates is either a second location of a legitimately multi-matched
    business, or a spurious near-duplicate splitting the S1's precision. The
    two cases are told apart by whether that neighbour independently ranks
    this S1 as its own best retrieval pick (`t_rank == 1`, already computed by
    `context_features`): agreement from an unrelated target is the closest
    thing to the transitivity signal of full clustering, without its recall
    collapse. Built only from name text and `t_rank`, so it has no dependency
    on the trained matcher and cannot leak.
    """
    if cands.height == 0:
        return cands.with_columns(pl.lit(0.0, pl.Float32).alias("sib_max_sim"), pl.lit(0, pl.Int16).alias("sib_agree_count"))
    meta = tgtn.select(pl.int_range(pl.len(), dtype=pl.Int32).alias("it"), pl.col("name_core"), pl.col("src"))
    c = cands.select("i1", "it", "t_rank").join(meta, on="it")
    left = c.select("i1", "src", pl.col("it").alias("it_a"), pl.col("name_core").alias("name_a"))
    right = c.select("i1", "src", pl.col("it").alias("it_b"), pl.col("name_core").alias("name_b"), pl.col("t_rank").alias("rank_b"))
    joined = left.join(right, on=["i1", "src"]).filter(pl.col("it_a") != pl.col("it_b"))
    if joined.height == 0:
        return cands.with_columns(pl.lit(0.0, pl.Float32).alias("sib_max_sim"), pl.lit(0, pl.Int16).alias("sib_agree_count"))
    sims = [
        _cp(fuzz.token_set_ratio, joined.slice(s, chunk)["name_a"].to_list(), joined.slice(s, chunk)["name_b"].to_list(), workers)
        for s in range(0, joined.height, chunk)
    ]
    joined = joined.with_columns(pl.Series("sim", np.concatenate(sims)))
    agg = (
        joined.group_by(["i1", "it_a"])
        .agg(
            pl.col("sim").max().alias("sib_max_sim"),
            ((pl.col("sim") >= sim_threshold) & (pl.col("rank_b") == 1)).sum().cast(pl.Int16).alias("sib_agree_count"),
        )
        .rename({"it_a": "it"})
    )
    return cands.join(agg, on=["i1", "it"], how="left").with_columns(
        pl.col("sib_max_sim").fill_null(0.0), pl.col("sib_agree_count").fill_null(0)
    )


def feature_columns(df: pl.DataFrame) -> list[str]:
    drop = {"i1", "it", "y", "fold", "p_prune", "p"}
    return [c for c in df.columns if c not in drop]
