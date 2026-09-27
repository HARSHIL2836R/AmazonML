import polars as pl

from ber.normalize import normalize_records
from ber.translit import ScriptDictionary


def _norm(name, address="", country="US"):
    df = pl.DataFrame({
        "entity_id": ["S2-1"], "business_name": [name], "business_address": [address], "country": [country],
    })
    return normalize_records(df, ScriptDictionary()).row(0, named=True)


def test_legal_forms_and_dots():
    r = _norm("Maid Yoga! L.L.C.")
    assert r["name_core"] == "maid yoga"
    assert r["legal"] == "llc"


def test_private_limited_canonical():
    r = _norm("Shri JACK PRECAST PRIVATE LIMITED - 5521374295")
    assert r["name_core"] == "jack precast"
    assert r["legal"] == "ltd pvt"


def test_leetspeak_and_accents():
    assert _norm("J0hnson  Fréight")["name_core"] == "johnson freight"


def test_domain_names():
    r = _norm("-- riede1americanpony.com")
    assert r["is_domain"] and r["name_nospace"] == "riedelamericanpony"


def test_alias_keeps_part_after_marker():
    r = _norm("Noviorbi DBA Noble & Co")
    assert r["name_core"] == "noble and"
    assert r["name_alt"] == "noviorbi"


def test_name_starting_with_aka_is_not_split():
    assert _norm("Aka (India) Estate Private Limited", country="India")["name_core"] == "aka estate"


def test_us_address_state_and_street_type():
    r = _norm("X", "GREENSBORO, NC, 19 1/2 STARDUST TRAIL")
    assert r["state"] == "nc"
    assert r["addr"] == "greensboro stardust trl"
    assert r["addr_nums"] == "19 1 2"


def test_india_native_state_and_labels():
    r = _norm("X", "HOUSE NO. D/205, SANGRAMPUR ROAD, BULDHANA, महाराष्ट्र", country="India")
    assert r["state"] == "mh"
    assert r["addr_nums"] == "205"
    assert "sangrampur rd" in r["addr"]


def test_france_department_maps_to_region():
    r = _norm("X", "27 R. JEAN BART, LILLE, Nord", country="France")
    assert r["state"] == "hdf"
    assert r["addr"] == "rue jean bart lille"


def test_unknown_country_passes_through():
    r = _norm("Acme GmbH", "Hauptstrasse 5, Berlin", country="Germany")
    assert r["name_core"] == "acme" and r["state"] == "" and r["addr_nums"] == "5"


def test_null_address():
    assert _norm("X", None)["addr_null"]
