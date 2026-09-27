import polars as pl
import pytest

from ber.metrics import macro_f, per_entity_f


def _pairs(rows):
    return pl.DataFrame(rows, schema={"i1": pl.Int32, "it": pl.Int32}, orient="row")


def test_problem_statement_example():
    # predicted [47, 193, 812], truth [47, 812] -> P = 2/3, R = 1, F0.5 = 0.714
    pred = _pairs([(1, 47), (1, 193), (1, 812)])
    truth = _pairs([(1, 47), (1, 812)])
    assert macro_f(pred, truth, pl.Series([1])) == pytest.approx(0.7142857, abs=1e-6)


def test_singletons():
    ids = pl.Series([1, 2])
    truth = _pairs([])
    # entity 1: correctly empty -> 1.0; entity 2: predicted a match for a singleton -> 0.0
    pred = _pairs([(2, 5)])
    f = per_entity_f(pred, truth, ids).sort("i1")["f"].to_list()
    assert f == [1.0, 0.0]


def test_missed_everything_scores_zero():
    truth = _pairs([(1, 9)])
    assert macro_f(_pairs([]), truth, pl.Series([1])) == 0.0
