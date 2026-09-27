import random
import sys
from pathlib import Path

import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from simulate_french_validation import (  # noqa: E402
    build_hard_negatives,
    build_positive_targets,
    drop_house_number,
    swap_or_drop_legal_form,
    swap_street_abbrev,
)


def _s1():
    return pl.DataFrame({
        "entity_id": ["S1-001", "S1-002", "S1-003"],
        "business_name": ["Etablissements Defense SASU", "Etablissements Medi SAS", "Ecole Sport Union"],
        "business_address": [
            "121 Route de Bordeaux Petit Piquey, Lege-Cap-Ferret",
            "121 Route de Bordeaux Petit Piquey, Lege-Cap-Ferret",
            "5 Rue Jean Bart, Lille",
        ],
        "country": ["France"] * 3,
    })


def test_swap_street_abbrev_stays_in_documented_pairs():
    rng = random.Random(0)
    out = swap_street_abbrev("121 Route de Bordeaux", rng)
    assert out in ("121 Route de Bordeaux", "121 Rte de Bordeaux")


def test_drop_house_number_only_touches_leading_digits():
    rng = random.Random(0)
    for _ in range(20):
        out = drop_house_number("121 Route de Bordeaux", rng)
        assert out in ("121 Route de Bordeaux", "Route de Bordeaux")


def test_swap_or_drop_legal_form_keeps_word_count_bounded():
    rng = random.Random(0)
    out = swap_or_drop_legal_form("Etablissements Defense SASU", rng)
    # either dropped (2 words), swapped (3 words), or (no-op, 3 words) - never grows unboundedly
    assert 2 <= len(out.split()) <= 3


def test_build_positive_targets_one_per_s1_and_traceable():
    s1_fr = _s1()
    pos = build_positive_targets(s1_fr, seed=1)
    assert pos.height == s1_fr.height
    assert set(pos["_match"].to_list()) == set(s1_fr["entity_id"].to_list())
    assert pos["entity_id"].n_unique() == pos.height  # synthetic ids don't collide


def test_build_hard_negatives_only_for_shared_streets_and_swaps_names():
    s1_fr = _s1()
    neg = build_hard_negatives(s1_fr, seed=2)
    # S1-001 and S1-002 share a street; S1-003 is alone on Rue Jean Bart -> no negative for it.
    assert neg.height == 2
    names = " ".join(neg["business_name"].to_list()).lower()
    # each negative borrows the OTHER same-street entity's name, never its own.
    assert "sport" not in names  # S1-003's family never appears (different street)


def test_build_hard_negatives_no_shared_street_is_empty():
    s1_fr = pl.DataFrame({
        "entity_id": ["S1-001", "S1-002"],
        "business_name": ["A", "B"],
        "business_address": ["1 Rue A, Lille", "2 Rue B, Lille"],
        "country": ["France"] * 2,
    })
    neg = build_hard_negatives(s1_fr, seed=0)
    assert neg.height == 0
