"""Record normalization: raw name/address strings -> comparable fields.

Almost everything runs as polars string expressions (Rust, multi-threaded),
because the pipeline touches ~12M records per split. The only Python-level
loop is the script transliteration, which runs on the ~1M names that
contain non-Latin characters.

Output columns added by `normalize_records`:

    name_full   canonical name, legal forms canonicalized ("pvt ltd"), honorifics dropped
    name_core   name_full without legal forms and country words
    name_alt    core of the part before an alias marker ("X dba Y" -> X), else ""
    name_nospace name_core with spaces removed (matches domain-style names)
    legal       space-joined sorted legal-form codes found in the name
    is_domain   name looked like "johnsonfreight.com"
    is_native   name contained non-Latin script
    addr        normalized address tokens (state removed), "" if missing
    addr_nums   space-joined numbers in the address, leading zeros stripped
    state       space-joined state/region codes detected in the address
    addr_null   address missing or a null placeholder
"""

from __future__ import annotations

import re
import unicodedata

import polars as pl

from . import lexicon as lx
from .translit import ScriptDictionary

NON_LATIN = r"[^\x00-\x7FÀ-ɏ‘-‟°º]"


def strip_accents_py(s: str) -> str:
    """Python twin of `_strip_accents`, used to build lookup keys."""
    return "".join(c for c in unicodedata.normalize("NFKD", s) if unicodedata.category(c) != "Mn")


def _strip_accents(e: pl.Expr) -> pl.Expr:
    return e.str.normalize("NFKD").str.replace_all(r"\p{Mn}", "")


def _tokens_space_padded(e: pl.Expr) -> pl.Expr:
    """Turn "a b c" into "  a  b  c  " so literal " tok " replacements never overlap."""
    return pl.lit("  ") + e.str.replace_all(" ", "  ") + pl.lit("  ")


def _unpad(e: pl.Expr) -> pl.Expr:
    return e.str.replace_all(r"\s+", " ").str.strip_chars()


def _replace_tokens(e: pl.Expr, mapping: dict[str, str]) -> pl.Expr:
    """Whole-token replacement for a space-separated token string."""
    items = [(k, v) for k, v in mapping.items() if k != v]
    if not items:
        return e
    pats = [f" {k} " for k, _ in items]
    reps = [f" {v} " if v else " " for _, v in items]
    return _unpad(_tokens_space_padded(e).str.replace_many(pats, reps))


def _leet(e: pl.Expr) -> pl.Expr:
    """Undo digit-for-letter substitutions next to letters ("j0hnson", "5ervices")."""
    for digit, letter in (("0", "o"), ("1", "l"), ("3", "e"), ("4", "a"), ("5", "s"), ("7", "t")):
        e = e.str.replace_all(f"([a-z]){digit}", f"${{1}}{letter}")
        e = e.str.replace_all(f"{digit}([a-z])", f"{letter}${{1}}")
    return e


_ALIAS_RE = "|".join(re.escape(m).replace(r"\ ", r"\s+") for m in lx.ALIAS_MARKERS)
_TLD_RE = "|".join(re.escape(t) for t in lx.TLDS)
_LEGAL_RE = r"\b(?:" + "|".join(sorted(set(lx.LEGAL_FORMS.values()), key=len, reverse=True)) + r")\b"
_HONORIFIC_RE = r"^(?:(?:" + "|".join(sorted(lx.HONORIFICS, key=len, reverse=True)) + r")\s+)+"
_COUNTRY_RE = r"\b(?:" + "|".join(lx.COUNTRY_WORDS) + r")\b"


def _clean_name(e: pl.Expr) -> pl.Expr:
    """Shared cleanup for one name fragment (after alias split)."""
    e = e.str.replace_all(r"\b(\d+)(st|nd|rd|th)\b", "${1}")      # ordinals, protect from leet
    e = e.str.replace_all(r"\.", "")                                 # l.l.c. -> llc, pvt. -> pvt
    e = e.str.replace_all("&", " and ").str.replace_all(r"[@$]", "a")
    e = _leet(e)
    e = e.str.replace_all(r"[^a-z0-9]+", " ").str.strip_chars()
    e = e.str.replace_all(r"\b(m s)\b", "")
    e = e.str.replace_all(_HONORIFIC_RE, "")
    e = _replace_tokens(e, lx.LEGAL_FORMS)
    return e


def _core(e: pl.Expr) -> pl.Expr:
    return _unpad(e.str.replace_all(_LEGAL_RE, " ").str.replace_all(_COUNTRY_RE, " "))


def normalize_names(df: pl.DataFrame, script_dict: ScriptDictionary | None) -> pl.DataFrame:
    raw = pl.col("business_name").fill_null("")
    df = df.with_columns(raw.str.contains(NON_LATIN).alias("is_native"))
    if script_dict is not None:
        df = df.with_columns(
            pl.when(pl.col("is_native"))
            .then(raw.map_batches(script_dict.transform_series, return_dtype=pl.String))
            .otherwise(raw)
            .alias("_n")
        )
    else:
        df = df.with_columns(raw.alias("_n"))

    n = _strip_accents(pl.col("_n").str.replace_all("Â", "").str.to_lowercase())
    n = n.str.replace_all(r"\(?\bid\s*:?\s*\d+\)?", " ")             # "(ID: 81649)"
    n = n.str.replace_all(r"\s-\s*\d{7,}", " ").str.replace_all(r"\d{7,}", " ")  # phone numbers
    n = n.str.replace_all(r"#\d+", " ")
    df = df.with_columns(n.alias("_n"))

    # Alias split: the Source-1 name follows the marker.
    # A non-empty prefix is required, so an S1 name that starts with "Aka" survives.
    parts = pl.col("_n").str.extract_groups(rf"^(.+?)\s+(?:{_ALIAS_RE})\b[:\s]*(.+)$")
    df = df.with_columns(
        pl.coalesce(parts.struct.field("2"), pl.col("_n")).alias("_main"),
        parts.struct.field("1").fill_null("").alias("_alt"),
    )

    # Domain-style names: "johnsonfreight.com", "@choiceadvisory", "#shanksnewhold".
    dom = pl.col("_main").str.extract(
        rf"^[^a-z0-9]*(?:(?:the|shri|sri|smt|dr|mr|mrs|ms)\s+)?([a-z0-9\-]+)\.(?:{_TLD_RE})\b", 1
    )
    df = df.with_columns(
        dom.is_not_null().alias("is_domain"),
        pl.coalesce(dom, pl.col("_main")).alias("_main"),
    )

    full = _clean_name(pl.col("_main"))
    alt = _clean_name(pl.col("_alt"))
    df = df.with_columns(full.alias("name_full"), _core(alt).alias("name_alt"))
    core = _core(pl.col("name_full"))
    df = df.with_columns(
        pl.when(core.str.len_chars() > 0).then(core).otherwise(pl.col("name_full")).alias("name_core"),
        pl.col("name_full").str.extract_all(_LEGAL_RE).list.unique().list.sort().list.join(" ").alias("legal"),
    )
    df = df.with_columns(pl.col("name_core").str.replace_all(" ", "").alias("name_nospace"))
    return df.drop("_n", "_main", "_alt")


def _state_regex(table: dict[str, str]) -> tuple[str, dict[str, str]]:
    keys = {}
    for k, v in table.items():
        nk = re.sub(r"[^\wऀ-෿]+", " ", strip_accents_py(k.lower())).strip()
        keys[nk] = v
    alt = "|".join(re.escape(k) for k in sorted(keys, key=len, reverse=True))
    return rf"(?:^|,)\s*({alt})\s*(?:,|$)", keys


_STATE_RX = {c: _state_regex(t) for c, t in lx.STATE_TABLES.items()}


def _normalize_addr_one_country(df: pl.DataFrame, country: str) -> pl.DataFrame:
    a = pl.col("business_address").fill_null("")
    a = _strip_accents(a.str.replace_all("Â", "").str.to_lowercase())
    # Component-level cleanup keeps commas so state detection sees whole components.
    a = a.str.replace_all(r"[^\wऀ-෿,]+", " ").str.replace_all(r"\s*,\s*", ",")
    a = a.str.replace_all(r"\b(?:null|none|nil)\b", " ").str.replace_all(r"\bn a\b", " ")
    df = df.with_columns(a.alias("_a"))

    rx = _STATE_RX.get(country.lower())
    if rx is not None:
        pattern, keys = rx
        # Pad commas so adjacent state components ("wi,tomah") both match.
        padded = pl.lit(",") + pl.col("_a").str.replace_all(",", ",,") + pl.lit(",")
        found = padded.str.extract_all(pattern)
        df = df.with_columns(
            found.list.eval(
                pl.element().str.replace_all(r"^,\s*|\s*,$", "").replace_strict(keys, default=None)
            ).list.drop_nulls().list.unique().list.sort().list.join(" ").alias("state"),
            padded.str.replace_all(pattern, ",").alias("_a"),
        )
    else:
        df = df.with_columns(pl.lit("").alias("state"))

    t = pl.col("_a").str.replace_all(",", " ")
    t = t.str.replace_all(r"\b(\d+)(?:st|nd|rd|th|er|eme|e)\b", "${1}")  # 1st, 2eme -> 1, 2
    t = t.str.replace_all(r"(\d)([a-z])", "${1} ${2}").str.replace_all(r"([a-z])(\d)", "${1} ${2}")
    t = t.str.replace_all(r"_+", " ").str.replace_all(r"\s+", " ").str.strip_chars()
    mapping = dict(lx.STREET_TYPES) | dict(lx.CITY_ALIASES) | {k: "" for k in lx.ADDRESS_LABELS}
    if country.lower() == "france":
        mapping |= lx.FR_STREET_TYPES
    t = _replace_tokens(t, mapping)
    df = df.with_columns(t.alias("_a"))
    df = df.with_columns(
        pl.col("_a").str.extract_all(r"\b\d+\b")
        .list.eval(pl.element().str.strip_chars_start("0").replace("", "0"))
        .list.join(" ").alias("addr_nums"),
        pl.col("_a").str.replace_all(r"\b\d+\b", " ").str.replace_all(r"\s+", " ").str.strip_chars().alias("addr"),
    )
    return df.drop("_a")


def normalize_addresses(df: pl.DataFrame) -> pl.DataFrame:
    df = df.with_columns(pl.int_range(pl.len(), dtype=pl.UInt32).alias("_row"))
    out = [
        _normalize_addr_one_country(part, country)
        for (country,), part in df.group_by("country", maintain_order=True)
    ]
    df = pl.concat(out).sort("_row").drop("_row")
    return df.with_columns(
        ((pl.col("addr").str.len_chars() == 0) & (pl.col("addr_nums").str.len_chars() == 0)).alias("addr_null")
    )


def normalize_records(df: pl.DataFrame, script_dict: ScriptDictionary | None = None) -> pl.DataFrame:
    """Add all normalized columns. Keeps entity_id and country; drops raw text."""
    df = df.with_columns(pl.col("entity_id").str.slice(1, 1).cast(pl.Int8).alias("src"))
    df = normalize_names(df, script_dict)
    df = normalize_addresses(df)
    return df.drop("business_name", "business_address")
