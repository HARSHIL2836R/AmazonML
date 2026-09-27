"""Learned token dictionary for names written in Indic scripts.

About 23% of Indian Source-2 names (13% in Source 3) are phonetic renderings
of English words in Devanagari, Tamil, Kannada and other scripts:
"इंडो प्रोडक्ट्स प्राइवेट लिमिटेड" is "Indo Products Private Limited". A generic
transliterator (anyascii) gets close ("imdo prodkts praivet limited") but not
close enough for token matching.

The generator renders word by word, so when a native name and its matched
Source-1 name have the same token count the tokens align by position. Counting
those alignments over the training pairs gives a native->English dictionary.
Tokens never seen in training fall back to anyascii.
"""

from __future__ import annotations

import json
import re
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path

import polars as pl
from anyascii import anyascii

_NON_LATIN = re.compile(r"[^\x00-\x7FÀ-ɏ‘-‟]")
_SPLIT = re.compile(r"[\s.,\-()\[\]&/'\"!#:;]+")


def _latin_tokens(name: str) -> list[str]:
    s = "".join(c for c in unicodedata.normalize("NFKD", name.lower()) if unicodedata.category(c) != "Mn")
    s = s.replace(".", "")
    return re.findall(r"[a-z0-9]+", s)


def _native_tokens(name: str) -> list[str]:
    return [t for t in _SPLIT.split(name) if t]


class ScriptDictionary:
    def __init__(self, table: dict[str, str] | None = None):
        self.table: dict[str, str] = table or {}

    @classmethod
    def fit(cls, latin_names: list[str], native_names: list[str], min_count: int = 2, min_share: float = 0.5):
        """Fit from aligned (Source-1 Latin name, matched native-script name) pairs."""
        counts: dict[str, Counter] = defaultdict(Counter)
        for lat, nat in zip(latin_names, native_names):
            lt = _latin_tokens(lat)
            nt = _native_tokens(nat)
            if len(lt) != len(nt):
                continue
            for a, b in zip(nt, lt):
                if _NON_LATIN.search(a):
                    counts[a][b] += 1
        table = {}
        for tok, c in counts.items():
            best, n = c.most_common(1)[0]
            total = sum(c.values())
            if n >= min_count and n / total >= min_share:
                table[tok] = best
        return cls(table)

    def translate(self, name: str) -> str:
        if not _NON_LATIN.search(name):
            return name
        out = []
        for tok in _native_tokens(name):
            if not _NON_LATIN.search(tok):
                out.append(tok)
            else:
                out.append(self.table.get(tok) or anyascii(tok).lower())
        return " ".join(out)

    def transform_series(self, s: pl.Series) -> pl.Series:
        return pl.Series(s.name, [self.translate(x) if x is not None else None for x in s.to_list()], dtype=pl.String)

    def coverage(self, native_names: list[str]) -> float:
        toks = [t for n in native_names for t in _native_tokens(n) if _NON_LATIN.search(t)]
        return sum(t in self.table for t in toks) / max(len(toks), 1)

    def save(self, path: Path) -> None:
        Path(path).write_text(json.dumps(self.table, ensure_ascii=False), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> "ScriptDictionary":
        return cls(json.loads(Path(path).read_text(encoding="utf-8")))
