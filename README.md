# Business entity resolution (Amazon ML Challenge 2026)

For every Source-1 business record, find the Source-2 and Source-3 records that describe the same business. The problem statement is in [ps.md](ps.md). What the data looks like, with numbers, is in [docs/eda.md](docs/eda.md). Methods from the literature, and the order in which to try them, are in [docs/research.md](docs/research.md).

## Pipeline

```
raw TSV ─► normalize ─► stage 1: retrieval ─► stage 2: pruning ─► matcher ─► decision
            (polars)     (sparse top-k,        (LightGBM on          (LightGBM on   (1 S1 per target,
                          state blocks)         retrieval graph)      pair features)  F0.5 threshold)
                                                      │                                   │
                                            candidate_pairs.tsv               matching_results.tsv
```

| module | what it does |
|---|---|
| `src/ber/normalize.py` | names: legal forms, honorifics, alias markers (`dba`, `f/k/a`), domains, leetspeak, IDs and phone numbers. Addresses: state detection (US, India incl. native script, France), street types, house numbers |
| `src/ber/translit.py` | native-script to English token dictionary learned from aligned training pairs (1,347 entries, full coverage of held-out tokens) |
| `src/ber/lexicon.py` | static word lists: legal forms, street types, state/region tables |
| `src/ber/blocking.py` | stage 1: TF-IDF top-k inside (country, state) blocks on three channels (name char-3grams, address tokens, blend), in both directions |
| `src/ber/prune.py` | stage 2: supervised meta-blocking. It keeps the stage-1 pairs that a small model on retrieval scores and ranks rates as plausible. Its output is `candidate_pairs.tsv` |
| `src/ber/features.py` | pair features: rapidfuzz similarities, IDF-weighted name overlap, house-number and state agreement, competition context |
| `src/ber/matcher.py` | LightGBM classifier |
| `src/ber/decide.py` | each target kept only for its best S1, then a threshold tuned for macro F0.5 |
| `src/ber/metrics.py` | the challenge metric, plus blocking recall and candidates per S1 |
| `src/ber/pipeline.py` | orchestration, caching, fold roles |

Folds come from the numeric part of the S1 entity id. Folds 1-3 train the models. Fold 4 picks the pruning threshold, early-stops the matcher and tunes the match threshold. Fold 0 is only scored.

## Running it

```bash
uv venv .venv --python 3.11
```

```bash
uv pip install --python .venv/Scripts/python.exe -r requirements.txt -e .
```

The data goes in `student_resource/dataset/{train,test}/` as shipped. Then:

```bash
python -m ber train --world 0.1
```

```bash
python -m ber train
```

```bash
python -m ber predict
```

```bash
python student_resource/utils/validate_submission.py --matching output/matching_results.tsv --candidate output/candidate_pairs.tsv --test-dir student_resource/dataset/test
```

```bash
python scripts/make_submission.py --team <team_name>
```

- `--world 0.1` trains on 10% of Source-1 entities with their matches and 10% of all other targets. It runs in minutes and is meant for iteration. Candidate density is lower than at full scale, so every number it reports is optimistic.
- Stage outputs are cached in `artifacts/<split>[_w<frac>]/`. Each cache has a `.fp` fingerprint of the code, settings and models that produced it, and a mismatch recomputes only the affected stages:
  - editing `normalize.py`, `lexicon.py` or `translit.py` redoes everything
  - a retrieval setting, `blocking.py` or `context_features` redoes retrieval (about 13 minutes per split)
  - a retrained pruner, or a change to `prune_keep_share` or `max_per_s1`, redoes the test candidates
  - string features and the matcher are always recomputed, so iterating on them takes about 4 minutes for `train` and 3 for `predict`

  `--fresh` still wipes a split's folder.
- Models go to `artifacts/models/train/` for full training and `artifacts/models/train_w0.1/` for the slice, so a quick slice run never replaces the models `predict` uses.
- Set `PYTHONUTF8=1` on Windows so log lines with native-script text print.
- Memory: the machine this was built on had about 6 GB free, so pairs are stored as int32 row positions and features are computed in 1M-pair chunks.

```bash
python -m pytest -q tests
```

## Results

Numbers are from fold 0, which no model or threshold saw (`artifacts/train*/report.json`). The full run uses 200K fold-0 S1 entities for scoring. The 10% world is shown for scale only: with a tenth of the competitors, every number there is optimistic.

| | 10% world | full train |
|---|---|---|
| stage 1: pairs per S1 / pair recall | 22.1 / 0.984 | 14.1 / 0.967 |
| candidates (pruned) per S1: mean, p50, p95 | 4.63, 4, 8 | 4.59, 4, 8 |
| candidate pair recall | 0.979 | 0.962 |
| oracle macro F0.5 of the candidate set | 0.993 | 0.987 |
| **macro F0.5** | **0.981** | **0.971** |
| US / India | 0.984 / 0.976 | 0.977 / 0.962 |
| singletons (5.5% of S1): mean F0.5 | 0.976 | 0.969 |
| match threshold (tuned on fold 4) | 0.725 | 0.70 |
| wall-clock, `train` | 2 min | 19 min |

On the full run, the threshold chosen on fold 4 scores within 0.00003 of the best threshold in hindsight on fold 0. The assignment rule adds 0.00002. The per-S1 expected-F0.5 rule (`decide.expected_f_select`) ties the global threshold, because 94% of candidate probabilities are below 0.1 or above 0.9.

Where the full-scale loss of 2.9 points sits:

- **Blocking, 1.35 points.** 3.8% of true pairs never reach the matcher: 3.3% are lost at retrieval and 0.5% at pruning. A perfect matcher on these candidates would score 0.987.
- **Matching, 1.5 points.** These are mostly near-twin businesses: the same name with no address, or the same building with a one-word name difference.

On the test split, `predict` takes 27 minutes and writes 9.54M candidates (5.51 per S1) and 5.66M matches (3.27 per S1). Test has 5.8 targets per S1 against 4.7 in train, so its candidate lists run longer than on validation. The official validator passes with `--check-ids`. France looks in line with the labelled countries, though no score for it can be measured before the leaderboard:

| country | S1 | candidates / S1 | matches / S1 | S1 with no match | pairs with 0.1 < p < 0.9 |
|---|---|---|---|---|---|
| France | 259,452 | 5.37 | 3.26 | 7.0% | 7.7% |
| India | 809,986 | 5.52 | 3.22 | 6.4% | 9.0% |
| US | 663,106 | 5.55 | 3.33 | 5.8% | 8.3% |

`docs/research.md` ends with the ordered list of what to try next.
