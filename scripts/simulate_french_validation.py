"""Simulated French validation set (docs/research.md roadmap item 8).

France has zero training labels, so the matcher's decision threshold and its
cross-country feature transfer are both unverified there before the
leaderboard. This script rebuilds a labelled French validation set by
applying the noise operators the challenge documents to real French
Source-1 test records:

    - legal-form swaps and drops among SARL/SAS/SASU/SA/SCI/EURL/SNC/ETS
      (ps.md "Name variations"; drop rate inside the 31-46% range
      docs/eda.md measures for other countries' legal forms)
    - French street-type abbreviation swaps: Route/Rte, Boulevard/Bd,
      Rue/R., Chemin/Chem, ... (ps.md "Address variations";
      docs/eda.md "Street types use French abbreviations")
    - accent injection, case changes, doubled spaces (docs/eda.md "Names")
    - bracketed/junk tokens: "--", "(ID: 12345)", "#392" (docs/eda.md)
    - dropped house numbers (ps.md "missing components")

Plus same-street hard negatives: Source-1 entities that share a normalized
street address (same normalized state + number-free address, the same key
`blocking.py` partitions on) are each paired with a noised copy of a
DIFFERENT same-street entity's name at their own address -- the exact
"different business, same street, similar name" pattern docs/eda.md
documents for France. These are never added to the ground truth, so a model
that merges them is correctly penalized; they are what makes this a
*validation* set rather than just more positives.

This can only create pseudo-labels from the noise grammar the task
description and docs/eda.md actually document; it cannot reproduce noise
they do not list (docs/research.md, "France without labels").

Usage:
    python scripts/simulate_french_validation.py --data-dir student_resource/dataset
    python scripts/simulate_french_validation.py --data-dir student_resource/dataset --score

Writes sim_source1/2/3.tsv and sim_ground_truth.tsv under <data-dir>/sim/, in
the same schema as the real splits, so the existing pipeline (normalize ->
blocking -> pruner -> matcher -> decide) scores it unmodified. `--score`
additionally runs that scoring pass with the trained models under
<work-dir>/models/train and reports macro F0.5 at the trained global
threshold versus a threshold re-tuned on this French set alone -- if they
differ by much, the global threshold is likely miscalibrated for France.
"""

from __future__ import annotations

import argparse
import json
import random
import re
from pathlib import Path

import numpy as np
import polars as pl

from ber.data import TEXT_COLS, read_tsv

FR_LEGAL_FORMS = ["sarl", "sas", "sasu", "sa", "sci", "eurl", "snc", "ets"]
FR_LEGAL_DISPLAY = {
    "sarl": "SARL", "sas": "SAS", "sasu": "SASU", "sa": "SA", "sci": "SCI",
    "eurl": "EURL", "snc": "SNC", "ets": "Etablissements",
}
LEGAL_DROP_RATE = 0.35  # inside the 31-46% range docs/eda.md reports for LLC/Inc/Limited

# Bidirectional street abbreviation swaps (ps.md; docs/eda.md's France section).
STREET_PAIRS = [
    ("Route", "Rte"), ("Boulevard", "Bd"), ("Rue", "R."), ("Chemin", "Chem"),
    ("Allee", "All"), ("Impasse", "Imp"), ("Cours", "Crs"), ("Residence", "Res"),
    ("Place", "Pl"), ("Avenue", "Av"),
]

ACCENT_MAP = str.maketrans({"a": "à", "e": "é", "i": "î", "o": "ô", "u": "ù"})
JUNK_TEMPLATES = ["--", "<<", "#{n}", "(ID: {n})", "- {n}"]


def _maybe(rng: random.Random, p: float) -> bool:
    return rng.random() < p


def swap_or_drop_legal_form(name: str, rng: random.Random) -> str:
    tokens = name.split()
    present = [i for i, t in enumerate(tokens) if t.strip(".,").lower() in FR_LEGAL_FORMS]
    if present:
        i = present[0]
        if _maybe(rng, LEGAL_DROP_RATE):
            del tokens[i]
        else:
            current = tokens[i].strip(".,").lower()
            choices = [f for f in FR_LEGAL_FORMS if f != current] or FR_LEGAL_FORMS
            tokens[i] = FR_LEGAL_DISPLAY[rng.choice(choices)]
    elif _maybe(rng, 0.5):
        tokens.append(FR_LEGAL_DISPLAY[rng.choice(FR_LEGAL_FORMS)])
    return " ".join(tokens)


def swap_street_abbrev(addr: str, rng: random.Random) -> str:
    for long, short in STREET_PAIRS:
        if re.search(rf"\b{re.escape(long)}\b", addr, re.IGNORECASE):
            if _maybe(rng, 0.6):
                return re.sub(rf"\b{re.escape(long)}\b", short, addr, count=1, flags=re.IGNORECASE)
            return addr
        if re.search(rf"\b{re.escape(short)}\b", addr, re.IGNORECASE):
            if _maybe(rng, 0.6):
                return re.sub(rf"\b{re.escape(short)}\b", long, addr, count=1, flags=re.IGNORECASE)
            return addr
    return addr


def add_accent_and_case_noise(text: str, rng: random.Random) -> str:
    if _maybe(rng, 0.3):
        text = text.translate(ACCENT_MAP)
    if _maybe(rng, 0.2):
        text = text.upper()
    elif _maybe(rng, 0.2):
        text = text.lower()
    if _maybe(rng, 0.25):
        spaces = [i for i, c in enumerate(text) if c == " "]
        if spaces:
            i = rng.choice(spaces)
            text = text[:i] + " " + text[i:]
    return text


def add_junk(text: str, rng: random.Random) -> str:
    if not _maybe(rng, 0.15):
        return text
    tok = rng.choice(JUNK_TEMPLATES).format(n=rng.randint(10_000, 999_999))
    return f"{tok} {text}" if rng.random() < 0.5 else f"{text} {tok}"


def drop_house_number(addr: str, rng: random.Random) -> str:
    if _maybe(rng, 0.3):
        return re.sub(r"^\s*\d+\w*\s*", "", addr)
    return addr


def noise_record(name: str, addr: str, rng: random.Random) -> tuple[str, str]:
    """Apply the documented operators to one (name, address) pair."""
    name = swap_or_drop_legal_form(name or "", rng)
    name = add_accent_and_case_noise(name, rng)
    name = add_junk(name, rng)
    addr = swap_street_abbrev(addr or "", rng)
    addr = add_accent_and_case_noise(addr, rng)
    addr = drop_house_number(addr, rng)
    return name, addr


def build_positive_targets(s1_fr: pl.DataFrame, seed: int) -> pl.DataFrame:
    """One noised, labelled target per French S1 entity."""
    rng = random.Random(seed)
    names, addrs, ids, matches = [], [], [], []
    for i, row in enumerate(s1_fr.iter_rows(named=True)):
        n, a = noise_record(row["business_name"], row["business_address"], rng)
        names.append(n); addrs.append(a); ids.append(f"S2-F{i:07d}"); matches.append(row["entity_id"])
    return pl.DataFrame({
        "entity_id": ids, "business_name": names, "business_address": addrs,
        "country": ["France"] * len(ids), "_match": matches,
    })


def build_hard_negatives(s1_fr: pl.DataFrame, seed: int) -> pl.DataFrame:
    """For S1 entities sharing a normalized street, pair each with a noised
    copy of a DIFFERENT same-street entity's name at its own address. Never
    added to the ground truth: these are the near-twins the matcher must
    reject even though blocking will retrieve them as candidates."""
    from ber.normalize import normalize_addresses

    rng = random.Random(seed)
    addr_df = normalize_addresses(s1_fr.select("entity_id", "business_address", "country"))
    name_by_id = dict(zip(s1_fr["entity_id"].to_list(), s1_fr["business_name"].to_list()))
    addr_by_id = dict(zip(s1_fr["entity_id"].to_list(), s1_fr["business_address"].to_list()))
    groups = (
        addr_df.filter(pl.col("addr") != "")
        .group_by("state", "addr").agg(pl.col("entity_id"))
        .filter(pl.col("entity_id").list.len() >= 2)
    )

    ids, names, addrs = [], [], []
    i = 0
    for members in groups["entity_id"].to_list():
        for e in members:
            e2 = rng.choice([m for m in members if m != e])
            n, _ = noise_record(name_by_id[e2], addr_by_id[e2], rng)
            _, a = noise_record(name_by_id[e], addr_by_id[e], rng)
            ids.append(f"S3-F{i:07d}"); names.append(n); addrs.append(a)
            i += 1
    return pl.DataFrame({
        "entity_id": ids, "business_name": names, "business_address": addrs,
        "country": ["France"] * len(ids),
    })


def _score(data_dir: Path, work_dir: Path) -> int:
    import lightgbm as lgb

    from ber import data, decide, features, pipeline
    from ber.config import Config
    from ber.metrics import macro_f

    cfg = Config(data_dir=data_dir, work_dir=work_dir)
    md = cfg.work_dir / "models" / "train"
    if not (md / "matcher.txt").exists():
        print(f"no trained model at {md}; run `python -m ber train` first")
        return 1
    dec = json.loads((md / "decision.json").read_text())
    pruner = lgb.Booster(model_file=str(md / "pruner.txt"))
    model = lgb.Booster(model_file=str(md / "matcher.txt"))

    s1n, tgtn, _ = pipeline.prepare(cfg, "sim")
    gt = read_tsv(cfg.data_dir / "sim" / "sim_ground_truth.tsv")
    tp = data.truth_positions(s1n, tgtn, gt)

    cands = pipeline.candidates(cfg, "sim", s1n, tgtn, pruner, dec["prune_threshold"])
    idf = features.NameIdf(s1n, tgtn)
    scored = []
    for s in range(0, cands.height, 1_000_000):
        ch = features.string_features(cands.slice(s, 1_000_000), s1n, tgtn, idf)
        p = model.predict(ch.select([pl.col(c).cast(pl.Float32) for c in dec["features"]]).to_numpy())
        scored.append(ch.select("i1", "it").with_columns(pl.Series("p", p.astype(np.float32))))
    scored = pl.concat(scored) if scored else cands.select("i1", "it").with_columns(pl.lit(0.0, pl.Float32).alias("p"))

    s1_ids = pl.Series("i1", list(range(s1n.height)), dtype=pl.Int32)
    f_global = macro_f(decide.predict_pairs(scored, dec["match_threshold"]), tp, s1_ids)
    t_fr, f_fr, _ = decide.tune_threshold(scored, tp, s1_ids)
    print("simulated French validation, macro F0.5:")
    print(f"  at the trained global threshold ({dec['match_threshold']}): {f_global:.5f}")
    print(f"  at a France-only tuned threshold ({t_fr:.3f}): {f_fr:.5f}")
    if f_fr - f_global > 0.01:
        print("  -> the global threshold looks miscalibrated for France; consider a country-specific one")
    else:
        print("  -> the global threshold transfers reasonably to this simulated French set")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", type=Path, default=Path("student_resource/dataset"))
    ap.add_argument("--work-dir", type=Path, default=Path("artifacts"))
    ap.add_argument("--seed", type=int, default=13)
    ap.add_argument("--score", action="store_true", help="also score with the trained models under <work-dir>/models/train")
    args = ap.parse_args()

    s1 = read_tsv(args.data_dir / "test" / "test_source1.tsv")
    s1_fr = s1.filter(pl.col("country") == "France")
    if s1_fr.height == 0:
        print("no French Source-1 records found under --data-dir; nothing to simulate")
        return 1
    print(f"French Source-1 records: {s1_fr.height:,}")

    pos = build_positive_targets(s1_fr, args.seed)
    neg = build_hard_negatives(s1_fr, args.seed + 1)
    print(f"positives: {pos.height:,}  same-street hard negatives: {neg.height:,}")

    out = args.data_dir / "sim"
    out.mkdir(parents=True, exist_ok=True)
    s1_fr.select(TEXT_COLS).write_csv(out / "sim_source1.tsv", separator="\t")
    pos.select(TEXT_COLS).write_csv(out / "sim_source2.tsv", separator="\t")
    (neg.select(TEXT_COLS) if neg.height else pl.DataFrame({c: [] for c in TEXT_COLS})).write_csv(out / "sim_source3.tsv", separator="\t")
    pos.select(pl.col("_match").alias("source1_entity_id"), pl.col("entity_id").alias("matched_entity_ids")).write_csv(
        out / "sim_ground_truth.tsv", separator="\t"
    )
    print(f"wrote {out}/sim_source{{1,2,3}}.tsv and sim_ground_truth.tsv")

    return _score(args.data_dir, args.work_dir) if args.score else 0


if __name__ == "__main__":
    raise SystemExit(main())
