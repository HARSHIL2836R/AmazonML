import numpy as np
import polars as pl
import pytest

from ber.features import NameIdf, context_features, sibling_features, string_features


def _pairs(rows):
    return pl.DataFrame(rows, schema={"i1": pl.Int32, "it": pl.Int32}, orient="row")


def test_context_features_margin_to_second_no_competitor():
    # i1=1 has a single candidate: its own score is the margin (no runner-up).
    pairs = pl.DataFrame({
        "i1": [0, 0, 1], "it": [0, 1, 0],
        "name_cos": [0.9, 0.9, 0.5], "addr_cos": [0.8, 0.1, 0.5],
    }, schema={"i1": pl.Int32, "it": pl.Int32, "name_cos": pl.Float32, "addr_cos": pl.Float32})
    out = context_features(pairs, np.array([2, 3], dtype=np.int8))
    row = out.filter((pl.col("i1") == 1) & (pl.col("it") == 0)).row(0, named=True)
    assert row["s1_top2_gap"] == row["ret_score"]
    top = out.filter((pl.col("i1") == 0) & (pl.col("it") == 0)).row(0, named=True)
    runner = out.filter((pl.col("i1") == 0) & (pl.col("it") == 1)).row(0, named=True)
    assert top["s1_top2_gap"] == pytest.approx(top["ret_score"] - runner["ret_score"], abs=1e-6)
    assert runner["s1_top2_gap"] == top["s1_top2_gap"]  # broadcast to the whole group


def test_context_features_per_source_rank():
    # Two S2 and one S3 candidate for the same S1: source-scoped rank must not
    # be shadowed by the cross-source winner.
    pairs = pl.DataFrame({
        "i1": [0, 0, 0], "it": [0, 1, 2],
        "name_cos": [0.9, 0.4, 0.3], "addr_cos": [0.9, 0.4, 0.3],
    }, schema={"i1": pl.Int32, "it": pl.Int32, "name_cos": pl.Float32, "addr_cos": pl.Float32})
    out = context_features(pairs, np.array([2, 2, 3], dtype=np.int8)).sort("it")
    assert out["s1_src_rank"].to_list() == [1, 2, 1]
    assert out["s1_src_n_cands"].to_list() == [2, 2, 1]


def test_sibling_features_agreement_requires_own_top_pick():
    tgtn = pl.DataFrame({
        "name_core": ["acme robotics downtown", "acme robotics uptown", "unrelated business"],
        "src": [2, 2, 3],
    }, schema={"name_core": pl.String, "src": pl.Int8})
    cands = pl.DataFrame({
        "i1": [0, 0, 0], "it": [0, 1, 2], "t_rank": [1, 1, 1],
    }, schema={"i1": pl.Int32, "it": pl.Int32, "t_rank": pl.Int16})
    out = sibling_features(cands, tgtn, sim_threshold=0.7).sort("it")
    # it=0 and it=1 are near-identical same-source siblings that both pick this S1.
    assert out["sib_agree_count"].to_list() == [1, 1, 0]
    assert out["sib_max_sim"][2] == 0.0  # no same-source sibling for it=2

    # Agreement is asymmetric: it counts a sibling whose OWN top pick is this
    # S1, not whether this row itself is a top pick. it=1 no longer picks this
    # S1 first (t_rank=2), so it=0 loses its one agreeing sibling, but it=1
    # still gets credit from it=0, which still ranks this S1 first.
    cands_disagree = cands.with_columns(pl.Series("t_rank", [1, 2, 1], dtype=pl.Int16))
    out2 = sibling_features(cands_disagree, tgtn, sim_threshold=0.7).sort("it")
    assert out2["sib_agree_count"].to_list() == [0, 1, 0]


def test_sibling_features_empty_input():
    tgtn = pl.DataFrame({"name_core": ["x"], "src": [2]}, schema={"name_core": pl.String, "src": pl.Int8})
    cands = pl.DataFrame({"i1": [], "it": [], "t_rank": []}, schema={"i1": pl.Int32, "it": pl.Int32, "t_rank": pl.Int16})
    out = sibling_features(cands, tgtn)
    assert out.height == 0
    assert {"sib_max_sim", "sib_agree_count"} <= set(out.columns)


def _bilingual_s1_tgt():
    s1n = pl.DataFrame({
        "entity_id": ["S1-1"], "country": ["France"],
        "name_core": ["etablissements defense"], "name_nospace": ["etablissementsdefense"],
        "name_full": ["etablissements defense sasu"], "legal": ["sasu"],
        "addr": ["route de bordeaux petit piquey lege cap ferret"], "addr_nums": ["121"],
        "state": ["naq"], "addr_null": [False],
    })
    tgtn = pl.DataFrame({
        "entity_id": ["S2-1", "S2-2"], "country": ["France", "France"],
        "name_core": ["etablissements medi", "etablissements defense groupe"],
        "name_nospace": ["etablissementsmedi", "etablissementsdefensegroupe"],
        "name_full": ["etablissements medi sas", "etablissements defense groupe sasu"],
        "legal": ["sas", "sasu"],
        "addr": [
            "route de bordeaux petit piquey lege cap ferret",
            "route de bordeaux petti piquey lege cap ferret",
        ],
        "addr_nums": ["47", "125"], "state": ["naq", "naq"], "addr_null": [False, False],
        "name_alt": ["", ""], "is_domain": [False, False], "is_native": [False, False], "src": [2, 2],
    })
    return s1n, tgtn


def test_rarest_token_match_separates_same_street_near_twins():
    """The France same-street pattern from docs/eda.md: address alone can't
    tell 'Établissements Defense SASU' apart from a same-street 'Medi', but
    the rarest (most distinguishing) name token can."""
    s1n, tgtn = _bilingual_s1_tgt()
    idf = NameIdf(s1n, tgtn)
    pairs = _pairs([(0, 0), (0, 1)])
    out = string_features(pairs, s1n, tgtn, idf, workers=1).sort("it")
    assert out["a_tset"][0] > 0.9  # same street either way
    assert out["n_rare_a_in_b"].to_list() == [0.0, 1.0]  # "defense" only matches the true target
