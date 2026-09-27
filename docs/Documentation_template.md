# ML Challenge 2026: Business Entity Resolution Solution

**Team Name:** [Your Team Name]  
**Team Members:** [List all team members]  
**Submission Date:** [Date]

---

## 1. Executive Summary

Source-2/3 records are retrieved against Source 1 by sparse TF-IDF search inside (country, state) blocks, and a small gradient-boosted model then prunes that list to about 4.6 candidates per Source-1 entity. A LightGBM matcher scores the survivors from string, token and competition features. Each target is kept only for its best-scoring Source-1 entity, and a threshold tuned directly on the macro F0.5 metric decides the rest. Two ideas carry most of the weight: the at-most-one-S1-per-target structure of the data is used in both retrieval and decision, and native-script Indian names are translated through a dictionary learned from the training pairs.

---

## 2. Methodology

### 2.1 Problem Analysis

Full measurements are in `docs/eda.md`. The facts that shaped the design:

- **Each Source-2/3 record belongs to at most one Source-1 entity.** Train has 7,638,365 positive pairs and exactly as many distinct targets. No pair crosses countries.
- **26% of Source-2/3 records match nothing, and 5.6% of Source-1 entities have no match.** Both groups are where F0.5 precision is lost.
- **The noise is synthetic and enumerable.** It covers legal-form swaps, honorifics, leetspeak, injected accents, bracketed IDs and phone numbers, word shuffles, domain renderings (`johnsonfreight.com`), alias constructions (`X dba Y`), invented trade names, and Indian names written phonetically in native scripts (23.5% of Indian Source-2 names).
- **4.6% of true pairs share no name word.** The address is the only link, so blocking cannot rely on names alone.
- **State agrees on matched pairs.** US agrees 100% of the time and India 98.8%, where every disagreement is Telangana against Andhra Pradesh. State is therefore a near-lossless partition key.
- **France is test-only and unlabelled.** It has three regions and about eighteen cities, and near-identical template names share streets.

### 2.2 Solution Strategy

**Approach Type:** Blocking (retrieval + learned pruning) + feature-based classifier + constrained decision  
**Core Innovation:** retrieval in both directions exploits the one-S1-per-target structure; supervised meta-blocking shrinks the candidate set to near the true link count; a native-script dictionary is learned from the training pairs by positional alignment.

---

## 3. Candidate Generation (Blocking)

Two stages. `candidate_pairs.tsv` is the output of the second, and it is the exact set the matcher scores.

**Stage 1, retrieval (`src/ber/blocking.py`).**

- **Blocks:** (country, state), with Telangana and Andhra Pradesh merged. A target with no parseable state gets one from address tokens that almost always co-occur with a single state in Source 1 (for example `bordeaux` maps to Nouvelle-Aquitaine). Targets still without a state (3.8%) search every Source-1 record of their country on the name channel.
- **Channels:**
  - name: TF-IDF over character 3-grams of the normalized name with spaces removed
  - address: TF-IDF over address tokens and house numbers
  - blend: both blocks together, weighted 0.6 / 0.4
- **Frequency cap:** features present in more than 1% of a block's records are dropped. This removed 86% of the multiply-add work at no measurable recall cost. A tighter cap lost recall quickly.
- **Directions:** each S1 keeps its top 3 / 4 / 3 targets per channel. Each target keeps its top 2 / 1 / 2 S1 records. The reverse direction reflects the data's structure: a target's true S1 is usually its best-scoring S1.

**Stage 2, pruning (`src/ber/prune.py`).** A LightGBM model sees only retrieval-graph features: channel cosines, ranks in both directions, the gap to the best score of the pair's S1 and of its target, and candidate counts. This is supervised meta-blocking (Papadakis et al.). Its threshold keeps 99.5% of the true pairs stage 1 found, measured on a tuning fold, and each S1 keeps at most 12 pairs.

- **Blocking keys used:** (country, state) partition; name character 3-grams; address tokens and house numbers
- **Candidate pairs generated (full train, held-out fold):** stage 1 gives 14.1 per S1 at 96.7% pair recall. After pruning: 4.59 per S1 (median 4, 95th percentile 8) at 96.2% recall. The oracle macro F0.5 of that candidate set is 0.987. Retrieval over the whole training split (2.2M × 10.3M) takes 13 minutes on 16 CPU threads.
- **How true matches were kept:** the address and blend channels catch pairs that share no name word. The reverse direction guarantees every target a slot with its best S1. The stateless fallback covers missing addresses. The pruning threshold is set by recall, not by a size target.

---

## 4. Matching Model

**Features used (50):**

- **Name:** rapidfuzz ratio, token-set, token-sort and partial ratio; Levenshtein and Jaro-Winkler on the space-free name; IDF-weighted word cosine and containment in both directions; similarity to the alias part of `X dba Y`; first-token equality; token counts
- **Address:** ratio, token-set and partial ratio on normalized addresses; house-number Jaccard, containment and first-number equality; state agreement
- **Other:** legal-form agreement; flags for missing address, domain-style name, native script, alias and source; the retrieval cosines and ranks; competition context (rank and score gap within the S1's candidates and within the target's S1 options)

**Model type:** LightGBM binary classifier (MIT licence), 127 leaves, learning rate 0.05, early stopping on the tuning fold.  
**Threshold selection method:** each target is assigned to its highest-probability S1 (the exact solution of the at-most-one constraint for fixed scores). The probability threshold is chosen by grid search on macro F0.5 over the tuning fold, singletons included, and reported on an untouched validation fold. A per-entity expected-F0.5 rule (Lewis; Jansche 2007) was also implemented. It tied the global threshold, because the probabilities are calibrated and bimodal.

Country is deliberately not a feature, so the model has nothing US- or India-specific to learn for France.

---

## 5. Results & Error Analysis

- **F_0.5 Score (macro, full train, held-out fold of 200K S1 entities):** 0.971 (US 0.977, India 0.962; singletons 0.969). On a 10% world with a tenth of the competitors it is 0.981, which shows how much of the difficulty is density. Of the 2.9 points lost, 1.35 come from true pairs that never reach the matcher and 1.5 from matching errors.
- **Common false positives (wrong merges):** a target with an identical name and no address, attached to the wrong one of several same-name businesses; same-building businesses with near-identical names (`Sanghvi Export` against `Sanghvi Chemicals`, same floor and road).
- **Common false negatives (missed matches):** invented trade names whose address differs slightly (house number altered, city alias); names with several added or dropped words and a missing address. 56% of missed pairs never reached the candidate set.

---

## 6. Conclusion

Most of the score comes from normalization and retrieval that respect how the data was generated. The one-S1-per-target structure turned blocking into a mostly reverse-direction search and made the pruned candidate set nearly as small as the true link count. The remaining loss is split between blocking recall and genuinely ambiguous same-name records, and France is the part of the test set this validation cannot measure.

---

## Appendix

### A. Code Artefacts

`code/business_entity_resolution/` (see its README):

- `src/ber/`: normalize, translit, lexicon, blocking, prune, features, matcher, decide, metrics, pipeline
- `python -m ber train` fits the script dictionary, pruner, matcher and thresholds on the training split
- `python -m ber predict` writes `output/candidate_pairs.tsv` and `output/matching_results.tsv`
- `tests/`: metric and normalization unit tests

### B. Additional Results

Blocking sweep, recall against candidates per S1 (10% world, all folds):

| name fwd, addr fwd, blend fwd, name rev, addr rev, blend rev | candidates / S1 | pair recall |
|---|---|---|
| 0, 0, 0, 0, 0, 1 | 7.5 | 0.940 |
| 0, 0, 0, 1, 0, 1 | 9.3 | 0.971 |
| 3, 4, 3, 2, 1, 2 (used) | 22.2 | 0.983 |
| 6, 4, 8, 2, 2, 3 | 34.2 | 0.985 |
