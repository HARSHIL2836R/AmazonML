# Research: blocking, matching and decision rules for the Business Entity Resolution challenge

Date: 2026-09-26 (baseline blocking measurements, the speed analysis and the remaining verifications added 2026-09-27)

The question: which published methods for candidate generation, pairwise matching, constrained post-processing, cross-script name matching, F-beta decision rules and business-name normalization are worth adding to a feature + LightGBM baseline for the Amazon ML Challenge 2026 "Business Entity Resolution" task, given its specifics. Those specifics are three sources (a deduplicated reference S1 and two dirty sources S2/S3 with about 10M records on the test side), a many-to-one link structure (each S2/S3 record matches at most one S1 record), a per-S1 macro F0.5 metric that counts singletons, a scored candidate file where smaller candidate sets rank higher, a zero-shot French test country, a licence rule (MIT or Apache-2.0, at most 8B parameters), a no-external-lookup rule, and a laptop with about 6 GB of free RAM, 16 CPU threads and a 4 GB RTX 3050.

Conventions used below:

- Every number with a link comes from that primary source. Table and figure numbers refer to the source.
- **(derived)** marks arithmetic or reasoning done here from the competition's own statistics, not taken from a paper.
- **(estimate)** marks a runtime or memory figure that has not been measured on this hardware. Treat these as order-of-magnitude guesses to be replaced by a timing run.
- Numbers attributed to the baseline come from runs of this repo's pipeline (`src/ber`) on a 10% train world, reported on 2026-09-27. The feature-cost table in Section 1 comes from a sampling script run for this report on the test files.
- Licences were read from the repository's LICENSE file (via the GitHub API) or from the Hugging Face model card metadata. "No licence file" means the repository grants no licence; the method can still be re-implemented from the paper, but the code cannot be copied into an MIT/Apache deliverable. This is not legal advice.

---

## 1. Blocking and candidate generation

### What the primary sources report

**Top-k TF/IDF blocking (Sparkly).** [Paulsen, Govind and Doan (PVLDB 2023)](https://www.vldb.org/pvldb/vol16/p1507-paulsen.pdf) index the smaller table with Lucene, tokenize the concatenated blocking attributes into character 3-grams, score with BM25, and for every record of the larger table keep the top k records of the smaller one. Against 8 blockers on 15 datasets it reached equal or higher recall with a much smaller output. Lucene makes the top-k search fast with block-max WAND, a branch-and-bound method that skips most of the documents sharing a term with the query (Section 3.2). The design choices they tested, each backed by an experiment in the paper:

- Top-k beats a similarity threshold, because on noisy data the scores of gold matches spread across the whole [0, 1] range, so a threshold either kills matches or blows up the output (Section 5, Figure 7).
- Probing from the larger table gives higher recall for the same k than probing from the smaller one, and running top-k in both directions raised runtime a lot for a minimal recall gain (Section 3.3).
- idf matters a lot and tf does not on short attributes such as names (Section 5).
- 3-grams are a good default; 2-grams and 4-grams were worst, and 2-grams were also slowest (Section 4.4).
- TFIDF-cosine was better than BM25 on many datasets (Section 5), so plain TF-IDF cosine is a reasonable stand-in for BM25.

Recall and output size, from their Table 2 (Sparkly Manual):

| k | Recall range over 15 datasets | Example: Songs (1M records, self-join) |
|---|---|---|
| 10 | 92.5 to 100% | 10.0M pairs, 96.3% recall |
| 20 | 96.4 to 100% | 20.0M pairs, 97.9% |
| 50 | 98.7 to 100% | 50.0M pairs, 99.3% |

At scale (their Table 3), Sparkly Auto on MusicBrainz (20M records) reached recall 95% at k = 10 and 98% at k = 50, and blocked 26M WDC records in 130 minutes on 30 AWS nodes. The code is [BSD-3-Clause](https://github.com/anhaidgroup/sparkly), and it assumes a Spark cluster.

For a laptop, the same computation is a sparse matrix product with per-row top-k. [sparse_dot_topn](https://github.com/ing-bank/sparse_dot_topn) (Apache-2.0) implements it; its README reports up to 6 times faster than the naive product on two 20k x 193k TF-IDF matrices with top 10 per row on 8 cores, and exposes `top_n`, `threshold`, `n_threads` and `density`.

**Token blocking, block cleaning and meta-blocking (JedAI).** The [survey by Papadakis et al. (ACM CSUR 2020)](https://arxiv.org/abs/1905.06167) reports that Block Purging (dropping oversized blocks, i.e. very frequent tokens) and Block Filtering (keeping each entity only in its smallest blocks) raise pair quality and reduction ratio by orders of magnitude while leaving pair completeness almost unchanged, and that Block Filtering is usually applied after Block Purging with ratio r = 50% (Section 4.3). Among comparison-cleaning (meta-blocking) methods, Reciprocal CNP and Reciprocal WNP trade a little recall for much higher precision, and BLAST and supervised meta-blocking dominate them. [Generalized Supervised Meta-blocking (PVLDB 2022)](https://arxiv.org/abs/2204.08801) trains the pruning classifier on as few as 50 labelled pairs. Against its plain binary-classifier pruning (BCl), the WNP pruning rule loses only 0.2% recall, while RWNP gives up 7.2% recall for 68.5% more precision. Implementations: [JedAI Toolkit](https://github.com/scify/JedAIToolkit) and [pyJedAI](https://github.com/AI-team-UoA/pyJedAI), both Apache-2.0.

**A head-to-head benchmark of all three families.** [Papadakis et al., "Benchmarking Filtering Techniques for Entity Resolution" (arXiv 2202.12521, ICDE 2023)](https://arxiv.org/abs/2202.12521) fine-tunes every method to reach pair completeness PC >= 0.9 and ranks them by pair quality (precision) on 10 real datasets plus 7 synthetic Febrl datasets of 10K to 2M person records with typo noise. Its conclusions (Section VII):

- The fine-tuned Standard Blocking workflow (token blocking + cleaning + meta-blocking) had the best average precision rank (2.2), with kNN-Join second (3.2) and DeepBlocker last among the top methods (9.0), in the schema-agnostic setting.
- Cardinality thresholds (top-k per record) beat similarity thresholds, and for top-k methods the candidate count is linear: |C| = k x min(|E1|, |E2|).
- LSH does poorly: MinHash, hyperplane and cross-polytope LSH cut the brute-force pair count by only 48%, 89% and 91% on average, against 99% for the best similarity join. MinHash LSH ran out of memory on the largest real dataset.
- Syntactic representations beat semantic (pre-trained embedding) ones for filtering; the semantic ones won in only 2 cases, which the authors attribute to out-of-vocabulary, domain-specific terms.
- Scale: on 2M synthetic records the blocking workflows took 2.4 to 3.5 hours, kNN-Join with character bigrams needed more than 30 hours already at 1M, the default kNN-Join with 5-grams and K = 5 did scale to 2M, and FAISS was fastest at 1.3 hours.

**Sorted neighbourhood.** [Christen's indexing survey (TKDE 2012)](http://cs.anu.edu.au/people/Peter.Christen/publications/christen2011indexing.pdf) evaluates it on Febrl-style name/address data. Methods that cap block or window size keep precision but lose recall as data grows; threshold-based methods keep recall but generate candidates faster than linearly (Section 5). The sort key is built from the leading characters of a field, and this challenge's noise model attacks exactly that region: junk prefixes ("--", "<<"), honorifics (Shri, Smt, The), leetspeak and word reordering **(derived)**.

**Dense-embedding ANN blocking.**

- [DeepBlocker (Thirumuruganathan et al., PVLDB 2021)](https://www.vldb.org/pvldb/vol14/p2459-thirumuruganathan.pdf), BSD-3-Clause [code](https://github.com/qcri/DeepBlocker): fastText + self-supervised tuple embeddings + FAISS. The Autoencoder variant was best on structured and dirty data. Sparkly beat it on all 15 datasets, for example CSSR 2.5% against 10% at 98% recall on Amazon-Google. On MusicBrainz 10M, DeepBlocker's Autoencoder took 691 minutes against 132 and 61 minutes for the two Sparkly variants, and reached 40% recall at k = 50 against their 94% and 98%; the Hybrid variant ran out of memory at 5M records ([Sparkly, Section 4.5 and Table 3](https://www.vldb.org/pvldb/vol16/p1507-paulsen.pdf)).
- Off-the-shelf sentence encoders: [Zeakis et al. (PVLDB 2023)](https://arxiv.org/abs/2304.12329) found S-GTR-T5 the best pre-trained model for blocking. On the Febrl synthetic series its recall with 10 neighbours fell from 0.962 at 10K records to 0.800 at 2M; S-MiniLM fell from 0.750 to 0.450 and fastText from 0.901 to 0.415 (Section 5.1.1). Pre-trained embeddings without fine-tuning degrade with scale.
- Supervised contrastive bi-encoder: [SC-Block (Brinkmann, Shraga and Bizer, ESWC 2024)](https://arxiv.org/abs/2303.03132), BSD-3-Clause [code](https://github.com/wbsg-uni-mannheim/SC-Block), fine-tunes RoBERTa-base with a supervised contrastive loss and searches with FAISS. With 5 neighbours it had the best average recall (81.2%, Table 3). On its largest benchmark (100K x 2M product offers, Table 4) it reached 89.5% recall with 5M candidates at k = 50, while whitespace BM25 at k = 200 reached 95.5% with 20M. The pipeline gain it reports comes from smaller candidate sets: a cross-encoder pipeline dropped from 30 hours to 8 hours.
- Industry case: in the [Ditto paper's employer-matching case study (Li et al., PVLDB 2021)](https://arxiv.org/abs/2004.00584) (Section 5, Tables 8 and 9), matching 62.5K public employer records (name, address, city, state, zipcode, phone) against 788K internal ones, the union of zipcode blocking and TF-IDF top-20 on name + address produced 10.65M candidates, and matching all of them took 22,823 s. Replacing that with Sentence-BERT top-10 cut the end-to-end time from 6.49 to 1.69 hours. The final F1 was 96.53.

**Auto-EM.** [Zhao and He (WWW 2019)](https://www.microsoft.com/en-us/research/wp-content/uploads/2019/04/Auto-EM.pdf) is a matcher, not a blocker: it pre-trains attribute-type detectors and attribute-level name-variation models on synonym lists from Microsoft's production knowledge base, then fine-tunes them on a new task. The [code](https://github.com/henryzhao5852/AutoEM) is MIT, but the README states that the training data is proprietary and cannot be released, and no trained weights are published. Pre-training on alias pairs from an external knowledge base would also collide with the no-external-data rule. The 7.64M train pairs already are this task's alias list.

### Making sparse top-k fast: purging, prefix filtering and sparse ANN

The cost of an inverted-index or sparse-matrix top-k over TF-IDF vectors is the sum, over features, of (S1 records containing the feature) x (targets containing it). That grows with the square of the data, and a handful of frequent features dominate it. Three bodies of work attack exactly that term.

- **Block purging and max-df pruning.** Dropping the largest blocks, which in token blocking are the most frequent tokens, is the first step of every JedAI workflow and costs almost no recall ([survey, Section 4.3](https://arxiv.org/abs/1905.06167)). In TF-IDF terms this is a max-df cap; the dropped features are the ones with the lowest idf and therefore the smallest share of the cosine.
- **Prefix filtering.** Sort each record's features by global rarity. For a similarity threshold t, two records can only reach t if they share a feature within a short prefix of their rarest features, so only prefixes need to be indexed or probed. [Mann, Augsten and Bouros (PVLDB 2016)](https://www.vldb.org/pvldb/vol9/p636-mann.pdf) compared seven set-similarity-join algorithms and concluded that the prefix filter does most of the work: the simple AllPairs algorithm is still competitive, candidate verification is cheap (often 2 or fewer element comparisons, never more than 18), and the more elaborate filters did not pay for themselves. [All-Pairs (Bayardo, Ma and Srikant, WWW 2007)](http://www.bayardo.org/ps/www2007.pdf) applies the same idea to weighted cosine vectors by indexing only part of each vector and adding the unindexed part during verification; it was 2 to 15 times faster than LSH for cosine similarity on DBLP while, unlike LSH, being exact. The threshold versions do not directly give a per-record top-k; the [top-k set similarity join of Xiao et al. (ICDE 2009)](https://cgi.cse.unsw.edu.au/~lxue/paper/icde09_chuan.pdf) returns the k best pairs overall, not k per record. For per-record top-k, the practical adaptation is query truncation: probe with only each target's p rarest features, then re-score the retrieved pairs with the full vectors **(derived)**. With non-negative TF-IDF weights, the truncated dot product is a lower bound on the full cosine, so this is the filter-then-verify pattern of the join literature **(derived)**.
- **Dynamic pruning and sparse ANN.** Lucene's block-max WAND skips most postings during top-k search (used by Sparkly, above). [Seismic (Bruch, Nardini, Rulli and Venturini, SIGIR 2024)](https://arxiv.org/abs/2404.18812) statically prunes each inverted list, groups it into blocks with summary vectors, and evaluates only promising blocks. On learned sparse embeddings of MS MARCO it reports sub-millisecond queries, one to two orders of magnitude faster than earlier inverted-index methods, and faster than the graph-based winners of the NeurIPS 2023 BigANN sparse track. The [code](https://github.com/TusKANNy/seismic) is MIT (LICENSE.md) with a README request to cite the papers; PyPI ships only a Linux wheel (`pyseismic-lsr` 0.5.2), so Windows needs a Rust source build. It has not been evaluated on character n-gram TF-IDF vectors, whose statistics differ from learned sparse embeddings.

Library options on this machine, from PyPI and GitHub metadata:

| Library | Licence | Windows wheel | What it offers |
|---|---|---|---|
| [sparse_dot_topn](https://github.com/ing-bank/sparse_dot_topn) 1.2.0 | Apache-2.0 | yes | Exact sparse product with per-row top-n; the multiply-adds are those of the full product whatever n is (derived) |
| [tantivy-py](https://github.com/quickwit-oss/tantivy-py) 0.26.2 | MIT | yes | Lucene-like engine in Rust; BM25 top-k with dynamic pruning. Throughput on 3-gram name queries unmeasured. |
| [faiss-cpu](https://github.com/facebookresearch/faiss) 1.15.1 | MIT | yes | Dense IVF/HNSW/PQ; needs an encoder first |
| [hnswlib](https://github.com/nmslib/hnswlib) 0.8.0 | Apache-2.0 | no (source only, needs a C++ compiler) | Dense HNSW |
| [PyNNDescent](https://github.com/lmcinnes/pynndescent) 0.6.0 | BSD-2-Clause | pure Python (numba) | Approximate kNN graph that accepts sparse input with cosine |
| [Seismic](https://github.com/TusKANNy/seismic) 0.5.2 | MIT | no (Linux wheel only) | Sparse ANN with static pruning |

### Blocking comparison table

| Method | Recall and candidates reported | Scale tested | Source |
|---|---|---|---|
| Baseline TF-IDF, name char 3-grams + address tokens + state token (this repo) | reverse top-1: 0.961 recall at 7.8 candidates/S1; + forward top-3: 0.962 at 8.2; forward top-20 + reverse top-3: 0.980 at 33.5 (oracle macro F0.5 0.994); 42 min with sparse_dot_topn | 10% train world: 220K S1 x 1.72M targets | baseline run, 2026-09-27 |
| All-Pairs (partial indexing + verification, exact threshold join) | exact; 2 to 15x faster than LSH for cosine at the same threshold | DBLP and Google query data | [Bayardo 2007](http://www.bayardo.org/ps/www2007.pdf) |
| Sparkly (BM25 on char 3-grams, top-k from the larger table) | 92.5 to 100% recall at k = 10; 98.7 to 100% at k = 50; Songs 1M: 96.3% with 10.0M pairs at k = 10 | up to 26M records; 20M with recall reported (95% at k = 10, 98% at k = 50) | [Paulsen 2023](https://www.vldb.org/pvldb/vol16/p1507-paulsen.pdf), Tables 2 and 3 |
| JedAI parameter-free workflow (PBW) | 74.5 to 100% recall; output up to 4.2B pairs | 15 datasets (up to 1M records) | same, Table 2 |
| JedAI default workflow (DBW) | 84.7% recall or better; output up to 454.5M pairs | same | same, Table 2 |
| Fine-tuned Standard Blocking workflow (token blocking + purging + filtering + meta-blocking) | best average precision rank (2.2) at PC >= 0.9; 3.5 h on 2M records | 10 real + Febrl synthetic up to 2M | [Papadakis 2022/2023](https://arxiv.org/abs/2202.12521) |
| kNN-Join (cosine on char q-grams, top-k) | precision rank 3.2; candidates = k x smaller side; 5-gram K = 5 default scales to 2M | same | same |
| MinHash / hyperplane / cross-polytope LSH | only 48% / 89% / 91% fewer pairs than brute force; MinHash OOM on the largest real set | same | same |
| FAISS / SCANN on pre-trained embeddings | lower precision than the best workflow and kNN-Join (schema-based average rank 7.2 against 4.3 and 4.5); FAISS fastest (2M records in 1.3 h) | same | same |
| DeepBlocker (Autoencoder, Hybrid) | lower recall than Sparkly at equal size on 15 datasets; e.g. 10% vs 2.5% CSSR at 98% recall; 40% recall at k = 50 on MusicBrainz 10M vs 94 to 98% | Hybrid OOM at 5M; Autoencoder 691 min on 10M | [Paulsen 2023](https://www.vldb.org/pvldb/vol16/p1507-paulsen.pdf); [Thirumuruganathan 2021](https://www.vldb.org/pvldb/vol14/p2459-thirumuruganathan.pdf) |
| S-GTR-T5 / S-MiniLM kNN, no fine-tuning | recall with 10 neighbours 0.962 to 0.800 (S-GTR-T5) and 0.750 to 0.450 (S-MiniLM) from 10K to 2M records | 2M Febrl synthetic | [Zeakis 2023](https://arxiv.org/abs/2304.12329) |
| SC-Block (fine-tuned RoBERTa bi-encoder) | 81.2% average recall at k = 5; 89.5% with 5M candidates at k = 50 on 100K x 2M (BM25: 95.5% with 20M at k = 200) | 2M | [Brinkmann 2024](https://arxiv.org/abs/2303.03132), Tables 3 and 4 |
| TF-IDF top-20 + zipcode union (Ditto case study) | 10.65M candidates for 62.5K x 788K; SBERT top-10 then cut matching time 3.8x | 0.8M | [Li 2021](https://arxiv.org/abs/2004.00584), Section 5 |

### Feasibility on this problem and hardware

The link structure changes which side should be probed. Each S2/S3 record has at most one correct S1, while an S1 can have up to 11 correct targets. Probing from the target side means each target only needs its single S1 in the top k, which is the direction Sparkly found gives higher recall **(derived)**. Candidate budgets on the test set, with 10.0M targets and 1.73M S1 records **(derived)**:

| Scheme | Pairs | Candidates per S1 |
|---|---|---|
| Target-side top-1 | 10.0M | 5.8 |
| Target-side top-2 | 20.0M | 11.6 |
| Target-side top-3 | 30.0M | 17.3 |
| S1-side top-5 per source | 17.3M | 10 (but cannot recover an S1 with 6 S3 matches) |
| S1-side top-10 per source | 34.6M | 20 |

The irreducible floor is the true link count, about 3.5 per S1 on train. About 26% of targets are distractors and still consume k candidates each under pure top-k; a learned pruner (the supervised meta-blocking idea, trained on this problem's millions of labels instead of 50) can drop pairs whose blocking features say "no match" before the candidate file is written **(derived)**. Whether such a cascaded pruner counts as part of blocking or of the matcher under the competition's candidate-file rule should be confirmed with the organizers.

### What the baseline measurements say

The baseline's numbers on the 10% train world (220K S1, 1.72M targets) support the target-side design and expose two things to fix.

- **The forward pass buys almost nothing at small k.** Reverse top-1 alone gives 0.961 recall; adding forward top-3 gives 0.962. In `src/ber/blocking.py` each direction is a separate full sparse product, and sparse_dot_topn's cost does not depend on `top_n`, so the forward pass roughly doubles blocking time. Reverse top-3 without any forward pass has not been measured yet and is the obvious next data point **(derived)**.
- **Candidates per S1 are inflated by this world's mix.** Reverse top-1 yields one pair per target, so 7.8 per S1 is simply 1.72M / 220K. Full train has 4.7 targets per S1 and test 5.8, so the same scheme would give about 5.8 per S1 on test before any pruning **(derived)**.
- **Recall at fixed k will be lower at full scale.** The 10% world holds a tenth of the S1 records, so each target competes against about a tenth of the S1 neighbours it will face at full scale. Recall at k measured there is an upper bound for full scale **(derived)**. Measuring recall at 5%, 10% and 20% worlds and extrapolating the trend, or running one full country once the product is fast, would give the real figure.
- **The 0.980 union has room to prune.** Its oracle macro F0.5 of 0.994 means the remaining blocking misses cost at most 0.6 points, so the candidate-size criterion now matters more than squeezing out recall.

### Why the product is slow, and how much each fix buys

The multiply-add count of the product is the sum over features of (S1 rows with the feature) x (target rows with it). The document-frequency fractions of features do not depend on sample size, so the count scales with N1 x NT. Full train is (2.2M x 10.3M) / (220K x 1.72M), about 60 times the 10% world, so the measured 42 minutes becomes roughly 40 hours for both directions; test (1.73M x 10.0M) is about 46 times, roughly 32 hours **(derived; assumes the same country mix and feature distribution)**. Fitting 30 minutes therefore needs a reduction of about 80x with both directions, or about 40x with the reverse pass only.

To see where the multiply-adds go, I sampled 150K S1 and 300K S2+S3 records per country from the test files, built name character 3-grams (after stripping common legal forms and honorifics) plus address word tokens plus a last-field state token, and computed each feature's share of the product from its document-frequency fractions (script in the session scratchpad, not in the repo). The features approximate but do not equal the baseline's normalization.

| | US | India |
|---|---|---|
| Share of product carried by the top 10 features | 26% | 46% |
| Share carried by the top 100 features | 64% | 82% |
| Most expensive features | address `street`, `road`, `drive`, `avenue`; name 3-grams `ent`, `ter`, `ers` | address `no` alone 23.5%; `road`, `floor`, `nagar`, the state name |
| Speed-up from a max-df cap at 1% (share of targets left with fewer than 3 features) | 10.7x (0.13%) | 31.5x (0.45%) |
| Max-df cap at 0.5% | 31x (0.92%) | 78x (1.83%) |
| Max-df cap at 0.2% | 115x (6.5%) | 259x (7.6%) |
| Query truncation to each target's 8 rarest features | 34x | 95x |
| Cap at 0.5% plus 8-feature truncation | 55x | 155x |
| Pair reduction from partitioning on the exact raw last address field (state) | 58x | 33x |

These are multiply-add reductions, not measured wall-clock times, and they say nothing about recall; the recall cost has to be measured on the 10% world **(estimate)**. They do show that the problem is not the size of the data but a few hundred features, most of them address tokens that carry almost no identifying information. A cap at 1% or 0.5% removes nearly all the work while leaving well under 2% of targets thin. The state token is the clearest case: as a feature it multiplies every same-state pair; as a partition key it divides the work by the number of states.

A plan that the cost model says fits in 30 minutes, to be confirmed step by step on the 10% world:

1. Drop the forward product, or derive forward lists from a reverse top-3 to top-5 result, once reverse-only recall is measured.
2. Remove the state token from the vectors and partition by normalized state instead. Records whose state is missing or unparseable (the null-address 3%, native-script states the normalizer misses) go through a country-wide pass with stricter pruning.
3. Apply a max-df cap of 0.5% to 1% to address tokens and name 3-grams, computed within each partition. `TfidfVectorizer` has a `max_df` argument, and the per-feature cost is `np.diff(X1.tocsc().indptr) * np.diff(XT.tocsc().indptr)`, which lets the cap be chosen from a work budget instead of guessed. Express the cap as a fraction, not a count, so that a setting tuned on the 10% world means the same thing at full scale **(derived)**.
4. Keep every record's 3 to 4 rarest features even if they exceed the cap, which is the per-record guarantee prefix filtering relies on, so no record goes dark.
5. Optionally truncate queries to their 8 rarest features, retrieve a slightly larger k' (say 10), and re-score those pairs with the full vectors before keeping the top k. Norms must come from the full vectors so the truncated score stays a lower bound.

Dense ANN is the fallback, not the first move. It needs an encoder pass over about 7.5M records per country on the 4 GB GPU plus index construction; at an assumed 2,000 to 5,000 short strings per second for a small encoder on the RTX 3050, encoding alone takes 25 to 60 minutes per country **(estimate)**, which already exceeds the budget. Zeakis et al. show that pre-trained encoders lose recall as the index grows, and on SC-Block's largest set its fine-tuned encoder reached 89.5% recall with 5M candidates where whitespace BM25 reached 95.5% with 20M. The sparse route keeps the representation that already reaches 0.96 to 0.98 recall and removes only the work that contributes least to the score **(derived)**.

Memory: a character 3-gram TF-IDF matrix for 5M short names is roughly 100 to 150M non-zeros, about 1 to 1.5 GB in CSR with float32 values and int32 indices **(estimate)**. It fits in 6 GB only if built per country and per source and multiplied in chunks. A dense index of 10M x 384 float16 vectors is 7.7 GB **(derived)** and does not fit; FAISS IVF-PQ codes (tens of bytes per vector) would be needed if a dense channel is added.

**Verdict:** keep target-side TF-IDF top-k, and make it fast with a reverse-only product, a state partition and a max-df cap with a per-record rare-feature guarantee (cost model: well over 40x) before considering tantivy or dense ANN; skip LSH and sorted neighbourhood, and add a fine-tuned dense channel only if a recall audit shows residual misses concentrated in cross-script or DBA names.

---

## 2. Pairwise matchers

### What the primary sources report

**Magellan-style similarity features + tree ensemble.** [py_entitymatching](https://github.com/anhaidgroup/py_entitymatching) (BSD-3-Clause) generates per-attribute string similarities and trains a random forest or similar. On clean bibliographic data it comes close to deep matchers; on product data, and above all on long product text, it falls far behind. As reported in the DeepMatcher paper and tabulated in [Ditto's Table 10](https://arxiv.org/abs/2004.00584), Magellan scores 92.3 F1 on DBLP-Scholar, 71.9 on Walmart-Amazon, 49.1 on Amazon-Google and 43.6 on Abt-Buy. On [WDC Products](https://arxiv.org/abs/2301.09521) (Table 3) it stays between 30.6 and 41.6 F1 in every setting, against 65.5 to 87.8 for fine-tuned RoBERTa. [Papadakis et al. (2023)](https://arxiv.org/abs/2307.01231) argue that most popular ER benchmarks are close to linearly separable, which makes small gaps between classical and deep matchers on those benchmarks hard to interpret.

This challenge's records are short name and address strings with synthetic, enumerable noise operators, which is closer to the structured benchmarks (and to the employer case in Ditto) than to WDC product titles **(derived)**. The WDC Products paper adds one relevant observation: raising the share of hard corner-case negatives generally cost the matchers more precision than recall (Section 5.2). The French same-street template names are that kind of negative.

**DeepMatcher.** [Mudgal et al. (SIGMOD 2018)](https://github.com/anhaidgroup/deepmatcher), BSD-3-Clause. RNN and attention over fastText embeddings. Reported F1: 62.8 Abt-Buy, 94.7 DBLP-Scholar, 69.3 Amazon-Google, 66.9 Walmart-Amazon ([Ditto Table 10](https://arxiv.org/abs/2004.00584)). Ditto measured 2.30 ms per pair for it on WDC (Table 11).

**Ditto.** [Li et al. (PVLDB 2021)](https://arxiv.org/abs/2004.00584), [code](https://github.com/megagonlabs/ditto) Apache-2.0. A cross-encoder over serialized records ("COL name VAL ... COL addr VAL ...") on RoBERTa-base or DistilBERT, plus domain-knowledge span tagging, summarization and data augmentation. Reported F1 (Table 5): Abt-Buy 89.33, DBLP-Scholar 95.6, Amazon-Google 75.58, Walmart-Amazon 86.76. On WDC Products under Peeters et al.'s setup, 84.90 ([Peeters 2025, Table 4](https://arxiv.org/abs/2310.11244)). In the employer case the domain knowledge was a tagger for the first number in the address and the last 4 digits of the phone, which is the same signal this challenge's house numbers and phone suffixes carry. Inference: 1.80 to 2.11 ms per pair on WDC and 6.78 to 8.01 ms on ER-Magellan (Table 11); the experiments ran on one V100 per run of an AWS p3.8xlarge (experimental setup).

**HierMatcher.** [Fu et al. (IJCAI 2020)](https://www.ijcai.org/Proceedings/2020/0507.pdf), code in [icip-cas/EntityMatcher](https://github.com/icip-cas/EntityMatcher) with **no licence file**; built on DeepMatcher with fastText English Wikipedia vectors. F1 (Table 2): DBLP-Scholar 95.3, Walmart-Amazon 81.6, Amazon-Google 74.9. No Abt-Buy result.

**Unicorn.** [Tu et al. (SIGMOD 2023)](https://nantang.github.io/research/pubs/unicorn.pdf) (author-hosted copy of the [ACM version](https://dl.acm.org/doi/abs/10.1145/3588938)). One DeBERTa-base encoder (139M parameters) plus an 8M-parameter mixture-of-experts layer, trained jointly on 20 datasets across 7 matching tasks. Entity-matching F1 (Table 3, Unicorn / Unicorn++): Walmart-Amazon 86.89 / 86.93, DBLP-Scholar 95.64 / 96.22, iTunes-Amazon 96.43 / 98.18, Fodors-Zagats 100 / 97.67, Beer 90.32 / 87.5; Ditto's numbers on the same splits are 86.76, 95.6, 97.06, 100 and 94.37. No Abt-Buy or WDC Products result. Code at [ruc-datalab/Unicorn](https://github.com/ruc-datalab/Unicorn) has **no licence file**; the pre-trained checkpoint [RUC-DataLab/unicorn-plus-v1](https://huggingface.co/RUC-DataLab/unicorn-plus-v1) is marked MIT.

**AnyMatch.** [Zhang et al. (2024)](https://arxiv.org/abs/2409.04073). Fine-tunes GPT-2 (124M parameters) for zero-shot matching with leave-one-dataset-out transfer. Zero-shot F1 (Table 2): Abt-Buy 86.05, DBLP-Scholar 90.59, Amazon-Google 55.08, Walmart-Amazon 61.51, WDC 63.31, mean over 9 datasets 81.96, which the authors put within 4.4% of MatchGPT with GPT-4. Throughput (Table 3): 693,999 tokens/s on 4 x A100, 25 times Jellyfish-13B. [Code](https://github.com/Jantory/anymatch) has **no licence file**; GPT-2 weights are MIT.

**Jellyfish.** [Zhang et al. (2023)](https://arxiv.org/abs/2312.01678). All three checkpoints, [7B](https://huggingface.co/NECOUDBFM/Jellyfish-7B), [8B](https://huggingface.co/NECOUDBFM/Jellyfish-8B) and [13B](https://huggingface.co/NECOUDBFM/Jellyfish-13B), are **CC-BY-NC-4.0**, which disqualifies them. Model-card F1 on Abt-Buy (unseen): 86.06 / 88.84 / 89.58.

**MatchGPT (LLM prompting).** [Peeters, Steiner and Bizer (EDBT 2025)](https://arxiv.org/abs/2310.11244). Best zero-shot F1 per model (Table 4):

| Model | WDC Products | Abt-Buy | Amazon-Google | DBLP-Scholar |
|---|---|---|---|---|
| GPT-4 | 89.61 | 95.78 | 76.38 | 89.82 |
| Llama 3.1 (70B) | 83.67 | 89.84 | 73.99 | 86.32 |
| Mixtral | 53.37 | 82.20 | 40.98 | 77.75 |
| Fine-tuned RoBERTa | 77.53 | 91.21 | 79.27 | 93.88 |
| Ditto | 84.90 | 91.31 | 80.07 | 94.31 |

The same table shows why LLMs are interesting for France: PLM matchers trained on one dataset and applied to WDC Products lost 22 to 61 F1 (RoBERTa) and 36 to 56 F1 (Ditto), while LLMs need no task data. None of the tested LLMs is usable here: GPT-4 is an API, Llama is under Meta's licence, and Mixtral is far above 8B parameters. The [MatchGPT repository](https://github.com/wbsg-uni-mannheim/MatchGPT) has no licence file.

**Fine-tuned small LLMs.** [Steiner, Peeters and Bizer (2024)](https://arxiv.org/abs/2409.08185) report that fine-tuning Llama 3.1 8B raised its F1 by 17.31 points on average, while fine-tuning hurt cross-domain transfer. Llama 3.1 is under Meta's community licence. Permissive models at or under 8B do exist, per their Hugging Face metadata:

- [Qwen2.5-1.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct), Apache-2.0, 1.54B parameters.
- [Qwen2.5-7B-Instruct](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct), Apache-2.0, 7.62B.
- [Phi-3.5-mini-instruct](https://huggingface.co/microsoft/Phi-3.5-mini-instruct), MIT, 3.82B.
- [Qwen2.5-3B-Instruct](https://huggingface.co/Qwen/Qwen2.5-3B-Instruct) is under an "other" licence, and [Qwen3-8B](https://huggingface.co/Qwen/Qwen3-8B) lists 8.19B parameters, over the cap.

No entity-matching results for these models were found in the sources read.

**Small cross-encoders.** [cross-encoder/ms-marco-MiniLM-L6-v2](https://huggingface.co/cross-encoder/ms-marco-MiniLM-L6-v2) (Apache-2.0, 22.7M) is listed by [Sentence-Transformers](https://www.sbert.net/docs/cross_encoder/pretrained_models.html) at 1,800 docs/s on MS MARCO passages (the page does not state the hardware); the L2 and L4 variants are listed at 4,100 and 2,500. [multilingual-e5-small](https://huggingface.co/intfloat/multilingual-e5-small) (MIT, 118M, most of it a 250K-token embedding table) is the multilingual option of similar compute. No ER benchmark numbers are published on these model cards. In the SC-Block pipeline a RoBERTa-base cross-encoder reached 75.9% F1 on the largest WDC-B set and took 27.9k seconds for 5M candidates on an RTX A6000 ([Brinkmann 2024, Table 5](https://arxiv.org/abs/2303.03132)).

### Matcher comparison table

"30M-pair cost" is the time to score 30M short pairs on the RTX 3050 laptop GPU or the 16-thread CPU **(estimate)**, extrapolated from the per-pair or per-token figures above where a source gives one.

| Name | Licence (code / weights) | Params | 4 GB GPU feasible | Reported F1 | 30M-pair cost | Source |
|---|---|---|---|---|---|---|
| Similarity features + LightGBM (Magellan-style) | LightGBM MIT; rapidfuzz MIT; py_entitymatching BSD-3 | tree model | CPU only | DBLP-Scholar 92.3, Walmart-Amazon 71.9, Amazon-Google 49.1, Abt-Buy 43.6; WDC Products 30.6 to 41.6 | under 1 h on CPU (estimate) | [Ditto T.10](https://arxiv.org/abs/2004.00584), [WDC Products T.3](https://arxiv.org/abs/2301.09521) |
| DeepMatcher | BSD-3 | small RNN (not reported) | yes | Abt-Buy 62.8, DBLP-Scholar 94.7, Amazon-Google 69.3, Walmart-Amazon 66.9 | 2.3 ms/pair on V100, about 19 V100-hours | [Ditto T.10, T.11](https://arxiv.org/abs/2004.00584) |
| Ditto (RoBERTa-base / DistilBERT) | Apache-2.0 / MIT, Apache-2.0 | 125M / 66M | training yes with fp16 and short sequences (estimate) | Abt-Buy 89.33, DBLP-Scholar 95.6, Amazon-Google 75.58, Walmart-Amazon 86.76, WDC Products 84.90 | 1.8 to 2.1 ms/pair on V100 (15 to 18 V100-hours); 1 to 2 days on the 3050 (estimate) | [Li 2021](https://arxiv.org/abs/2004.00584), [Peeters 2025](https://arxiv.org/abs/2310.11244) |
| HierMatcher | no licence file | fastText-based (not reported) | yes | DBLP-Scholar 95.3, Walmart-Amazon 81.6, Amazon-Google 74.9 | not reported | [Fu 2020](https://www.ijcai.org/Proceedings/2020/0507.pdf) |
| Unicorn | no licence file / MIT | 139M + 8M MoE | likely, fp16 (estimate) | Walmart-Amazon 86.89, DBLP-Scholar 95.64, iTunes-Amazon 96.43 | not reported; similar to Ditto per pair (estimate) | [Tu 2023, T.3](https://nantang.github.io/research/pubs/unicorn.pdf) |
| AnyMatch | no licence file / GPT-2 MIT | 124M | yes | zero-shot Abt-Buy 86.05, DBLP-Scholar 90.59, Amazon-Google 55.08, WDC 63.31 | about 174k tokens/s per A100; roughly a day on the 3050 (estimate) | [Zhang 2024](https://arxiv.org/abs/2409.04073) |
| Jellyfish 7B / 8B / 13B | CC-BY-NC-4.0 (excluded) | 7 to 13B | no | Abt-Buy 86.06 / 88.84 / 89.58 | 26,721 tokens/s for 13B on one A100 | [HF cards](https://huggingface.co/NECOUDBFM/Jellyfish-8B), [AnyMatch T.3](https://arxiv.org/abs/2409.04073) |
| MatchGPT prompting, GPT-4 / Llama 3.1 70B | proprietary / Llama licence (excluded) | very large / 70B | no | WDC 89.61 / 83.67; Abt-Buy 95.78 / 89.84 | GPT-4 0.68 to 2.19 s per request | [Peeters 2025](https://arxiv.org/abs/2310.11244), Tables 4 and 9 |
| Fine-tuned Llama 3.1 8B | Llama 3.1 licence (excluded) | 8B | no | +17.31 F1 average from fine-tuning | not reported | [Steiner 2024](https://arxiv.org/abs/2409.08185) |
| Qwen2.5-1.5B / Phi-3.5-mini as a yes/no matcher | Apache-2.0 / MIT | 1.5B / 3.8B | 1.5B in fp16 fits; 3.8B only 4-bit (estimate) | none found | days to weeks; subset use only (estimate) | [HF metadata](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct) |
| Small cross-encoder (MiniLM-L6 / multilingual-e5-small) | Apache-2.0 / MIT | 23M / 118M | yes, including fine-tuning | no ER results on cards; RoBERTa-base CE 75.9 on WDC-B | 1 to 4 h on the 3050 at sequence length 48 to 64 (estimate) | [SBERT table](https://www.sbert.net/docs/cross_encoder/pretrained_models.html), [SC-Block T.5](https://arxiv.org/abs/2303.03132) |

**Verdict:** keep feature + LightGBM as the main matcher and spend effort on features that separate same-street template names; a small permissively licensed cross-encoder, scored only on the uncertain band and fed back as a LightGBM feature, is the one neural addition that fits the hardware, and every published LLM matcher is excluded by licence, size or throughput.

---

## 3. Post-processing under the at-most-one-S1-per-target constraint

### What the primary sources report

[Papadakis, Efthymiou, Thanos and Hassanzadeh (EDBT 2022)](https://arxiv.org/abs/2112.14030) compare bipartite matching algorithms for clean-clean ER, where each entity matches at most one entity on the other side. Macro-averaged over their similarity graphs (Table 4):

| Algorithm | Precision | Recall | F-measure |
|---|---|---|---|
| Király's Clustering (KRC; adapts a linear-time 3/2 approximation to maximum stable marriage) | 0.696 | 0.597 | 0.619 |
| Unique Mapping Clustering (UMC) | 0.645 | 0.628 | 0.618 |
| Exact Clustering (mutual best match) | 0.735 | 0.544 | 0.591 |
| Best Match Clustering | 0.631 | 0.582 | 0.586 |
| Connected Components | 0.801 | 0.403 | 0.490 |

UMC sorts edges by weight and accepts an edge only if neither endpoint is already matched. It is the most balanced method and is equivalent to the CLIP clustering of Saeedi et al. for multi-source ER. Connected components (plain transitivity) has the highest precision and by far the lowest recall. The Hungarian algorithm was excluded from their study for its cost.

LLM work points the same way from another angle. [Wang et al. (COLING 2025)](https://arxiv.org/abs/2405.16884) show that letting the matcher see the competing candidates ("selecting" one record from a list) beats independent pairwise matching by 16.02% F1 on average over 8 datasets.

### What applies here

The challenge's constraint is weaker than clean-clean ER. It is many-to-one: a target has at most one S1, but an S1 can take many targets from the same source. UMC, Hungarian and stable-marriage methods all enforce one-to-one and would forbid an S1 from receiving its 3 or 4 legitimate S2 duplicates, so they should not be used as written **(derived)**.

Under the many-to-one constraint, the assignment decomposes by target. For each target, only the highest-probability S1 can be kept, and which S1 that is does not depend on any other target's choice. So "per-target argmax, then threshold" is the exact constrained assignment for a fixed set of pairwise scores, with no Hungarian step needed **(derived)**. What the literature adds is the value of context: the pairwise score should know about the competition. Features that carry it into LightGBM **(derived)**:

- rank of this S1 among the target's candidates, and the score margin to the runner-up S1;
- rank of this target among the S1's candidates from the same source;
- number of near-identical targets (S2-S2 or S3-S3 similarity above a high cut) whose own argmax is this S1, which is the transitivity signal without the recall collapse of connected components.

Correlation clustering and connected components are designed for dirty ER without anchors. Here S1 already supplies one anchor per cluster, so full clustering adds cost without adding structure **(derived)**.

**Verdict:** per-target argmax (exact for this constraint) plus listwise rank, margin and sibling-agreement features in a second LightGBM pass; do not use one-to-one bipartite matching.

---

## 4. Cross-script and multilingual name matching

### Transliteration libraries

| Library | Licence | What it does | Notes |
|---|---|---|---|
| [anyascii](https://github.com/anyascii/anyascii) | ISC | Context-free, per-character ASCII replacement for practically all scripts | README examples: महासमुंद gives `mhasmumd` (conventional Mahasamund), ಬೆಂಗಳೂರು gives `bemgluru`, கன்னியாகுமரி gives `knniyakumri`. Inherent vowels are dropped, so output is a consonant skeleton. |
| [indic_transliteration](https://github.com/indic-transliteration/indic_transliteration_py) | MIT | Scheme-based conversion between Brahmic scripts and ITRANS, IAST, etc. | Deterministic; renders inherent vowels, which English loanwords usually lack (derived). |
| [IndicXlit](https://github.com/AI4Bharat/IndicXlit) | MIT (code and models, per the [paper](https://arxiv.org/abs/2205.03018)) | 11M-parameter transformer; a separate Indic-to-Roman v1.0 model is released | The paper evaluates only Roman-to-Indic; it reports named entities as the hardest category. Needs fairseq, which is awkward on Windows. Indic-to-Roman accuracy not verified. |
| [uroman](https://github.com/isi-nlp/uroman) | MIT-style text plus an attribution request for publications (GitHub reports NOASSERTION) | Universal romanizer | Check whether the attribution clause is acceptable. |
| [Unidecode](https://github.com/avian2/unidecode) | GPL-2.0 | Per-character transliteration | Excluded by licence. |
| [aksharamukha-python](https://github.com/virtualvinodh/aksharamukha-python) | AGPL-3.0 | Script conversion | Excluded by licence. |

The baseline's learned token dictionary (`artifacts/script_dict.json`) held 1,347 entries at the time of writing, in scripts including Malayalam, Telugu, Devanagari, Gujarati, Kannada, Tamil and Bengali. A dictionary learned from train pairs is exact for words it has seen and blind to everything else. A person or brand name that appears only in test (for example a new "shyam" or "jain" variant) needs a fallback **(derived)**. The generator transliterates English words phonetically, so an English word and its Indic rendering share their consonant sequence far more reliably than their vowels (प्रोडक्ट्स "products" versus `prodkts`-like anyascii output) **(derived)**. A consonant-skeleton key applied to both sides after anyascii, with c/k/q and similar sound classes merged, is a cheap OOV fallback and a blocking key **(derived; untested)**.

### Phonetic encodings

- [jellyfish](https://github.com/jamesturk/jellyfish) (MIT) provides Soundex, Metaphone, NYSIIS and Match Rating; all are designed for English spelling.
- Indic Soundex, [described by its author Santhosh Thottingal](https://thottingal.in/blog/2009/07/26/indicsoundex/), groups phonetically similar letters of all Indic scripts into shared codes (the five velar stops ka, kha, ga, gha, nga get one code), so the same name written in Malayalam and in Gujarati produces the same key. The reference implementation, [libindic soundex](https://github.com/libindic/soundex), is **LGPL-3.0**; the algorithm itself is a character table and can be re-implemented, with a Latin-letter column added for this task.
- [indic-namematch](https://pypi.org/project/indic-namematch/) (MIT, 2026) scores Latin-script Indian person names across initials, surname-first order, honorifics and spelling variants such as Lakshmi/Laxmi. It is a small, recent, non-peer-reviewed package aimed at identity documents, not at native-script business names.
- Soundex variants for Indian names exist in the academic literature, for example [Gujarati](https://www.researchgate.net/publication/269672873_Improvement_of_Soundex_Algorithm_for_Indian_Language_Based_on_Phonetic_Matching) and [Indian personal names](https://www.researchgate.net/publication/316891126_Need_for_Customized_Soundex_based_Algorithm_on_Indian_Names_for_Phonetic_Matching); I did not read their full text and found no released code.

No permissively licensed, maintained phonetic encoder built for native-script Indian business names was found. The practical route is the consonant-skeleton key above or a re-implemented Indic Soundex table, either of which is a few dozen lines.

### Multilingual encoders

| Model | Licence | Params | Relevant property |
|---|---|---|---|
| [LaBSE](https://huggingface.co/sentence-transformers/LaBSE) | Apache-2.0 | 471M | Maps 109 languages into one vector space (model card) |
| [multilingual-e5-small](https://huggingface.co/intfloat/multilingual-e5-small) | MIT | 118M | 12 layers, 384 dims, so cheap to run |
| [paraphrase-multilingual-MiniLM-L12-v2](https://huggingface.co/sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2) | Apache-2.0 | 118M | Multilingual paraphrase model |
| [MuRIL](https://arxiv.org/abs/2103.10730) ([weights](https://huggingface.co/google/muril-base-cased)) | Apache-2.0 | BERT-base architecture | Pre-trained on 17 Indian languages with translated and transliterated segment pairs; the abstract reports gains on transliterated test sets. An MLM encoder, not a sentence encoder, so it needs fine-tuning. |
| [bge-m3](https://huggingface.co/BAAI/bge-m3) | MIT | XLM-RoBERTa-large based (count not in card metadata) | Dense, sparse and multi-vector multilingual retrieval |

I found no primary source that measures any of these on English words written phonetically in Indic scripts, which is the exact case here. The measurement is cheap: about 23% of Indian S2 names in train are in native script, so cosine similarity of true pairs against same-city non-matches can be computed on a sample in minutes **(derived)**. Zeakis et al.'s scale result (Section 1) says that without fine-tuning, embedding recall falls sharply as the index grows, so any encoder used for blocking should be fine-tuned on the train pairs.

**Verdict:** keep the learned dictionary, add an anyascii consonant-skeleton fallback for out-of-dictionary tokens (ISC licence, no model), and test a fine-tuned multilingual-e5-small only if the recall audit shows cross-script misses the fallback cannot fix.

---

## 5. Thresholds and decision rules for per-entity macro F0.5

### What the primary sources report

- **Global threshold.** [Lipton, Elkan and Naryanaswamy (2014)](https://arxiv.org/abs/1402.1892) prove that with calibrated probabilities the F1-optimal threshold is half the best achievable F1, and note that their results generalize to other F-beta. The same marginal argument gives the F-beta version **(derived)**: writing F = (1+β²)TP / (TP + FP + β²·P) with P the fixed number of true links, adding a prediction with probability s adds (1+β²)s to the numerator and 1 to the denominator, which helps if and only if s > F*/(1+β²). For β = 0.5 the threshold is 0.8·F*, so a model reaching F0.5 = 0.9 should cut calibrated probabilities near 0.72.
- **Per-instance F under independence.** Lewis's theorem, as restated by [Waegeman et al. (JMLR 2014)](https://arxiv.org/abs/1310.4849), shows that when the labels are independent the expected-F-optimal set always consists of the highest-probability items or no items at all, so only m + 1 candidate sets need checking. Chai (2005) and Jansche (2007) compute the exact optimum in O(m³) and O(m⁴); [Ye, Chai, Lee and Chieu (ICML 2012)](https://arxiv.org/abs/1206.4625) do it in O(m²).
- **Dependent labels.** [Dembczyński et al.'s GFM (NIPS 2011)](https://papers.nips.cc/paper/4389-an-exact-algorithm-for-f-measure-maximization) is exact for any joint distribution but needs m² + 1 parameters of it. The evidence on whether this matters in practice is mixed: Ye et al. summarize the NIPS paper as finding independence-based methods at least as good on its practical datasets, while the later JMLR version by Waegeman et al. proves a high worst-case regret for them and reports cases on real multi-label data where they were suboptimal.
- **Empty prediction.** Waegeman et al. define F with 0/0 = 1, which is precisely this challenge's singleton rule: predicting nothing for an S1 with no links scores 1.
- **Which approach to trust.** Ye et al. find threshold tuning on held-out data (empirical utility maximization) less sensitive to a misspecified model, while the decision-theoretic approach did better on rare classes and in a domain-adaptation scenario. France is such a domain shift.

### What applies here

The challenge metric is the per-instance (example-based) F0.5 of multi-label classification, with each S1 as an instance and its candidate targets as labels. After the per-target argmax of Section 3, an S1 keeps m candidates, usually under 10, so the exact rule is cheap **(derived)**:

1. Sort the S1's candidates by calibrated probability p1 >= ... >= pm.
2. For each k from 0 to m, compute E[F0.5(top-k)]. The number of true links inside the top k and outside it follow Poisson-binomial distributions, each computed by an O(m²) dynamic programme, and F0.5 = 1.25a / (0.25(a + b) + k) for a true links inside and b outside.
3. For k = 0, the expected score is the probability that all m candidates are false, which is the empty-prediction value.
4. Predict the k with the highest expected score.

The asymmetry this rule exploits is large. For an S1 with 4 true links, one extra false positive drops F0.5 from 1.0 to 0.833, while one missed link drops it only to 0.9375 **(derived)**. A global threshold cannot express that the cost of the fifth prediction depends on how many good ones the S1 already has.

Two caveats. The at-most-one-S1 constraint and near-duplicate siblings make the labels dependent, which is the case GFM covers and the independence rule does not. The rule is also only as good as the calibration, and French probabilities cannot be calibrated on French labels.

**Verdict:** calibrate LightGBM per country (isotonic on held-out US and India pairs), implement the per-S1 expected-F0.5 rule with the empty option, compare it against a single tuned threshold on a held-out split that includes singletons, and keep whichever wins; for France prefer the tuned threshold unless a simulated French validation set (Section 6) says otherwise.

---

## 6. Business-name and address specifics

**Legal-form stripping.** [cleanco](https://github.com/psolin/cleanco) (MIT) strips organization-type terms and suggests countries from them. Its term data includes the French forms `sarl`, `sas`, `sasu` and `sci` and the Indian `pvt. ltd.`. The README advises running `basename()` twice because names can carry two suffixes. The noise model's suffix swaps (LLC/Inc/Ltd/Pvt/Private Limited; SARL/SAS/SCI) mean the legal form is noise for matching but can still be a weak feature when both sides carry one and they disagree **(derived)**.

**Address parsing: libpostal.** [libpostal](https://github.com/openvenues/libpostal) and its Python binding [pypostal](https://github.com/openvenues/pypostal) are MIT. Facts from the README:

- The parser is a CRF trained on more than 1 billion addresses built from OpenStreetMap and OpenAddresses, with 99.45% accuracy on held-out data.
- `expand_address` normalizes abbreviations across languages.
- pypostal exposes `near_dupe_hashes`, whose docstring describes it as a record-linkage blocking function over name, address and geographic qualifiers, and `is_name_duplicate` / `is_street_duplicate` helpers.
- The default model is about 1.8 GB and the Senzing alternative about 2.2 GB; the data files are downloaded from S3 at build time.
- On Windows the build needs MSYS2/MinGW and then a separate build of the Python extension against the Microsoft toolchain.

Two concerns follow. First, it runs fully offline after installation, but its model encodes knowledge distilled from external gazetteers (OSM, OpenAddresses), so whether that counts as "external data" under the no-lookup rule is a rules question, not a technical one. Second, the model plus the 2 GB of data takes a third of the free RAM and a quarter of the free disk. [usaddress](https://github.com/datamade/usaddress) (MIT) covers only the US. [deepparse](https://github.com/GRAAL-Research/deepparse) is LGPL-3.0.

**What the train pairs already provide.** The address noise is enumerable: St/Street, R./Rue, BD/Boulevard, state code versus full name versus native script, and alternate city names (Bombay/Mumbai, Poona/Pune). Every one of these can be mined from aligned token pairs in the 7.64M positive train pairs, the same way the script dictionary was learned, with no external data and no licence question **(derived)**. That also covers the city alias problem, which a parser does not solve.

**France without labels.** The French noise operators are listed in the task description, and the generator is synthetic. Applying those operators to test S1 French records creates labelled French pairs:

- legal-form swaps among SARL, SAS and SCI;
- R./Rue and BD/Boulevard abbreviations;
- accents, case changes and doubled spaces;
- bracketed tokens;
- dropped house numbers.

Same-street records with different template names give hard negatives for free. The result is a French validation set for threshold choice and a check on feature transfer. It cannot reproduce operators the task description does not list **(derived)**.

**Verdict:** mine normalization rules and alias tables from train pairs, use cleanco's French and Indian term lists for suffix stripping, and do not add libpostal unless the organizers confirm it is allowed and a measured address-feature gap justifies 2 GB of RAM.

### Rule questions to put to the organizers

1. Does a learned candidate pruner count as blocking, so that `candidate_pairs.tsv` is the pruner's output?
2. Do BSD-3-Clause libraries (scikit-learn, Magellan) count against the MIT/Apache-2.0 rule, or does it apply to model weights only?
3. Is libpostal's pre-trained model, built from OSM/OpenAddresses, "external data"?
4. Is transductive use of the unlabeled test records allowed (pseudo-labelling confident French pairs, fitting TF-IDF idf on test)?

---

## Recommended roadmap

### Status on 2026-09-27

Three items below are built and measured on the 10% train world (220K S1, 1.72M targets, fold 0 held out):

- **Item 1, fast blocking.** Done in `src/ber/blocking.py`: (country, state) partitions with Telangana and Andhra Pradesh merged, state inference for stateless targets, a 1% max-df cap within each partition, and three channels (name, address, blend) in both directions. Retrieval fell from 2,531 s to 41 to 47 s. Recall of the union went up, from 0.980 to 0.984. A tighter cap costs recall and buys little time: 0.3% gives 0.954 recall and 0.1% gives 0.894, at 37 s and 35 s.
- **Item 5, learned pruner.** Done in `src/ber/prune.py`. It cuts 22.1 candidates per S1 at 0.984 recall down to 4.63 per S1 (95th percentile 8) at 0.979 recall. The oracle macro F0.5 of the candidate set is 0.993.
- **Item 4, expected-F0.5 rule.** Built as `decide.expected_f_select`. It ties the tuned global threshold (0.98093 against 0.98108) and does not beat it. The LightGBM probabilities are already calibrated within about 0.02 per decile, and 94% of candidates score below 0.1 or above 0.9. With so few uncertain pairs, the decision rule has little to work with.

The baseline scores macro F0.5 0.981 on the held-out fold of the 10% world (US 0.984, India 0.976). At full scale (2.2M S1 × 10.3M targets) it scores 0.971 (US 0.977, India 0.962). Two numbers change most at full density:

- **Stage-1 recall** falls from 0.984 to 0.967.
- **Retrieval scores separate less well** on the pruned candidates: `ret_score` AUC drops from 0.94 to 0.81, because the survivors are near-twins. The matcher shifts its weight to house-number agreement.

Full-scale retrieval takes 13 minutes for train, which settles item 1. Item 2 has its answer too: recall at full competition is 0.967, and blocking costs 1.35 of the 2.9 lost points. That makes items 3, 6 and 10 the next candidates.

**Items 3, 6 and 8, 2026-09-27: implemented, not yet measured.** No dataset was available in the environment this code was written in, so the numbers below are still to be produced by running `python -m ber train --world 0.1` and, for item 8, `scripts/simulate_french_validation.py --score` against a trained model.

- **Item 3, listwise/competition features.** `features.context_features` gained per-source rank/gap (`s1_src_rank`, `s1_src_gap`, `s1_src_n_cands`: is this the best S3 candidate for this S1, separate from the best overall) and a top-1/top-2 score margin broadcast to every row of a group (`s1_top2_gap`, `t_top2_gap`: was this candidate a clear winner or a close call). A new `features.sibling_features`, run once on the pruned candidate set, adds `sib_max_sim` (closest same-source name match among this S1's other candidates) and `sib_agree_count` (how many of those siblings *independently* rank this S1 as their own best retrieval pick, via the already-computed `t_rank`) -- the transitivity signal from Section 3's literature review, without a full clustering pass. All of it is built from retrieval scores, name text and `t_rank`, none of it from the trained matcher's own output, so cross-fitting was not needed to avoid leakage. The new columns were also added to `prune.PRUNE_FEATURES`, since the pruner sees the same competition.
- **Item 6, hard-negative token features.** `NameIdf.rarest_token_match` finds each name's single highest-idf (most distinguishing) token and checks whether it occurs on the other side -- a binary, threshold-free complement to the existing continuous IDF-overlap-share features, aimed at exactly the France same-street case (Section "France" in docs/eda.md): `a_tset` alone cannot tell "Établissements Defense" from "Établissements Medi" on the same street, but the rarest-token flag can (verified in `tests/test_features.py::test_rarest_token_match_separates_same_street_near_twins`). Phone-suffix agreement and an explicit legal-form-disagreement flag from the original item-6 list were deliberately skipped: docs/eda.md documents the digit suffixes normalize.py strips (`(ID: 81649)`, phone-like runs) as decorative junk, not a shared identifier the way Ditto's case study had one, so a phone feature would carry no signal here; and `legal_disagree` would just be the AND of two features (`legal_jac`, `legal_both`) already in the set, which a depth-2 tree split already captures.
- **Item 8, simulated French validation set.** `scripts/simulate_french_validation.py` applies the documented operators (legal-form swap/drop, French street-abbreviation swap, accent/case/space noise, junk tokens, dropped house numbers) to real French S1 test records, plus same-street hard negatives built by swapping names between S1 entities that share a normalized street -- the same near-twin pattern docs/eda.md shows for France. It writes a `sim` split in the challenge's own file format, so the existing pipeline scores it unmodified; `--score` compares macro F0.5 at the trained global threshold against a France-only tuned threshold. The generator functions are unit-tested (`tests/test_simulate_french_validation.py`); the `--score` path reuses already-tested pipeline functions but has not itself been run, for lack of French test data in this environment.

### Original ordering

Ordered by expected payoff per hour on top of the feature + LightGBM baseline. Payoffs are **(estimate)** in macro F0.5 points and have not been measured; the first experiment exists to replace these guesses with numbers.

1. **Make full-scale blocking fit in 30 minutes.** On the 10% world, measure reverse-only top-3 (no forward product), then add in turn: a state partition with a country-wide fallback for stateless records, a max-df cap (0.5% to 1%, chosen from the per-feature cost `nnz_col(X1) * nnz_col(XT)`) with each record's 3 to 4 rarest features always kept, and optionally 8-feature query truncation with full-vector re-scoring. Record recall and wall-clock at each step. Payoff: without it, the current pipeline needs roughly 30 to 40 hours per full run (derived from the 42-minute measurement); the cost model predicts 40x to over 100x less work. Cost: 3 to 5 hours.
2. **Recall at full-scale competition.** Repeat the reverse top-k recall measurement at 5%, 10% and 20% worlds (or one full country once step 1 lands) and split the misses by country, native-script versus Latin name, null address and DBA name. Payoff: the 10% world's 0.961 to 0.980 is an upper bound, and the split says whether the cross-script or dense-channel items below are worth doing. Cost: 2 hours after step 1.
3. **Per-target argmax plus listwise features.** Add rank, margin to the runner-up, per-source rank within the S1 and sibling agreement, then retrain. Payoff: +1 to 3 points, mostly precision on same-street template names, which F0.5 weights most. Cost: 2 to 3 hours.
4. **Per-S1 expected-F0.5 decision rule.** Isotonic calibration per country, the exact top-k rule with the empty option, compared against one tuned threshold on a held-out split with singletons. Payoff: +0.5 to 2 points, concentrated on singletons and on S1s with many links. Cost: 2 hours.
5. **Learned candidate pruner.** A small LightGBM on blocking-only features (TF-IDF score, forward and reverse rank, margin to the runner-up, token overlap counts) that shrinks the wide union (0.980 recall at 33.5 per S1, oracle macro F0.5 0.994) to something like 6 to 10 candidates per S1 before `candidate_pairs.tsv` is written. Payoff: better standing on the candidate-size criterion and a faster matcher, at a recall cost to be measured. Cost: 3 hours, and it depends on rule question 1.
6. **Hard-negative features for template names.** IDF-weighted mass of the differing tokens, a "rare token only on one side" flag, house number and phone-suffix agreement (Ditto's domain-knowledge signal), and legal-form disagreement. Train on the blocker's own top-k negatives. Payoff: +1 to 2 points, largest in France if it transfers. Cost: 3 to 4 hours.
7. **Cross-script fallback.** anyascii consonant-skeleton keys (or a re-implemented Indic Soundex table) for out-of-dictionary tokens, used both as an extra blocking probe and as a similarity feature; measure recall on native-script Indian S2 names before and after. Payoff: +0.3 to 1 point overall, larger for the Indian subset. Cost: 2 to 3 hours.
8. **Simulated French validation set.** Apply the documented French noise operators to French S1 records to create pseudo-labelled pairs and same-street negatives; use it to choose the French threshold and to check which features transfer. Payoff: protects the zero-shot country from a badly placed threshold, which could cost several points on French S1s. Cost: 4 to 6 hours.
9. **Small cross-encoder on the uncertain band.** Fine-tune multilingual-e5-small or MiniLM as a cross-encoder on train pairs (fp16, sequence length about 64), score only pairs whose LightGBM probability sits in an uncertain band, and feed the score back as a feature. Payoff: uncertain; the benchmarks show large gains over classical features on textual data but this data is short and structured. Cost: 8 to 12 hours including inference.
10. **Fine-tuned dense bi-encoder channel.** Train SC-Block style on train pairs and search with FAISS IVF-PQ; union with the TF-IDF candidates. Do this only if experiment 2 leaves more than about 1% of links outside the candidates, concentrated in DBA or cross-script names. Payoff: recovers the residual that character n-grams cannot. Cost: 8 or more hours, plus memory pressure.

---

## References

1. Paulsen, Govind, Doan. Sparkly: A Simple yet Surprisingly Strong TF/IDF Blocker for Entity Matching. PVLDB 16(6), 2023. https://www.vldb.org/pvldb/vol16/p1507-paulsen.pdf. Code: https://github.com/anhaidgroup/sparkly
2. Papadakis, Fisichella, Schoger, Mandilaras, Augsten, Nejdl. Benchmarking Filtering Techniques for Entity Resolution (arXiv 2202.12521; ICDE 2023). https://arxiv.org/abs/2202.12521
3. Papadakis, Skoutas, Thanos, Palpanas. A Survey of Blocking and Filtering Techniques for Entity Resolution. ACM CSUR 2020. https://arxiv.org/abs/1905.06167
4. Gagliardelli, Papadakis, Simonini, Bergamaschi, Palpanas. Generalized Supervised Meta-blocking. PVLDB 15(9), 2022. https://arxiv.org/abs/2204.08801
5. Christen. A Survey of Indexing Techniques for Scalable Record Linkage and Deduplication. IEEE TKDE 2012. http://cs.anu.edu.au/people/Peter.Christen/publications/christen2011indexing.pdf
6. Thirumuruganathan et al. Deep Learning for Blocking in Entity Matching: A Design Space Exploration. PVLDB 14, 2021. https://www.vldb.org/pvldb/vol14/p2459-thirumuruganathan.pdf. Code: https://github.com/qcri/DeepBlocker
7. Zeakis, Papadakis, Skoutas, Koubarakis. Pre-trained Embeddings for Entity Resolution: An Experimental Analysis. PVLDB 16(9), 2023. https://arxiv.org/abs/2304.12329
8. Brinkmann, Shraga, Bizer. SC-Block: Supervised Contrastive Blocking within Entity Resolution Pipelines. ESWC 2024. https://arxiv.org/abs/2303.03132. Code: https://github.com/wbsg-uni-mannheim/SC-Block
9. Li, Li, Suhara, Doan, Tan. Deep Entity Matching with Pre-Trained Language Models (Ditto). PVLDB 14, 2021. https://arxiv.org/abs/2004.00584. Code: https://github.com/megagonlabs/ditto
10. Mudgal et al. Deep Learning for Entity Matching: A Design Space Exploration (DeepMatcher). SIGMOD 2018. Code: https://github.com/anhaidgroup/deepmatcher
11. Konda et al. Magellan: Toward Building Entity Matching Management Systems. PVLDB 2016. Code: https://github.com/anhaidgroup/py_entitymatching
12. Fu et al. Hierarchical Matching Network for Heterogeneous Entity Resolution (HierMatcher). IJCAI 2020. https://www.ijcai.org/Proceedings/2020/0507.pdf. Code: https://github.com/icip-cas/EntityMatcher
13. Tu et al. Unicorn: A Unified Multi-tasking Model for Supporting Matching Tasks in Data Integration. SIGMOD 2023 (PACMMOD 1(1)). https://dl.acm.org/doi/abs/10.1145/3588938. Author copy: https://nantang.github.io/research/pubs/unicorn.pdf. Code: https://github.com/ruc-datalab/Unicorn. Weights: https://huggingface.co/RUC-DataLab/unicorn-plus-v1
14. Zhang, Groth, Calixto, Schelter. AnyMatch: Efficient Zero-Shot Entity Matching with a Small Language Model. 2024. https://arxiv.org/abs/2409.04073. Code: https://github.com/Jantory/anymatch
15. Zhang et al. Jellyfish: A Large Language Model for Data Preprocessing. 2023. https://arxiv.org/abs/2312.01678. Model cards: https://huggingface.co/NECOUDBFM/Jellyfish-7B, https://huggingface.co/NECOUDBFM/Jellyfish-8B, https://huggingface.co/NECOUDBFM/Jellyfish-13B
16. Peeters, Steiner, Bizer. Entity Matching using Large Language Models. EDBT 2025. https://arxiv.org/abs/2310.11244. Code: https://github.com/wbsg-uni-mannheim/MatchGPT
17. Steiner, Peeters, Bizer. Fine-tuning Large Language Models for Entity Matching. 2024. https://arxiv.org/abs/2409.08185
18. Peeters, Der, Bizer. WDC Products: A Multi-Dimensional Entity Matching Benchmark. EDBT 2024. https://arxiv.org/abs/2301.09521
19. Papadakis, Kirielle, Christen, Palpanas. A Critical Re-evaluation of Benchmark Datasets for (Deep) Learning-Based Matching Algorithms. 2023. https://arxiv.org/abs/2307.01231
20. Wang et al. Match, Compare, or Select? An Investigation of Large Language Models for Entity Matching. COLING 2025. https://arxiv.org/abs/2405.16884
21. Papadakis, Efthymiou, Thanos, Hassanzadeh. Bipartite Graph Matching Algorithms for Clean-Clean Entity Resolution: An Empirical Evaluation. EDBT 2022. https://arxiv.org/abs/2112.14030
22. Lipton, Elkan, Naryanaswamy. Optimal Thresholding of Classifiers to Maximize F1 Measure. ECML PKDD 2014. https://arxiv.org/abs/1402.1892
23. Ye, Chai, Lee, Chieu. Optimizing F-measures: A Tale of Two Approaches. ICML 2012. https://arxiv.org/abs/1206.4625
24. Dembczyński, Waegeman, Cheng, Hüllermeier. An Exact Algorithm for F-Measure Maximization. NIPS 2011. https://papers.nips.cc/paper/4389-an-exact-algorithm-for-f-measure-maximization
25. Waegeman, Dembczyński, Jachnik, Cheng, Hüllermeier. On the Bayes-Optimality of F-Measure Maximizers. JMLR 2014. https://arxiv.org/abs/1310.4849
26. Madhani et al. Aksharantar: Open Indic-language Transliteration Datasets and Models for the Next Billion Users (IndicXlit). Findings of EMNLP 2023. https://arxiv.org/abs/2205.03018. Code: https://github.com/AI4Bharat/IndicXlit
27. Khanuja et al. MuRIL: Multilingual Representations for Indian Languages. 2021. https://arxiv.org/abs/2103.10730
28. anyascii. https://github.com/anyascii/anyascii
29. indic_transliteration. https://github.com/indic-transliteration/indic_transliteration_py
30. uroman. https://github.com/isi-nlp/uroman
31. libindic soundex. https://github.com/libindic/soundex
32. jellyfish (phonetic encodings). https://github.com/jamesturk/jellyfish
33. sparse_dot_topn. https://github.com/ing-bank/sparse_dot_topn
34. JedAI Toolkit. https://github.com/scify/JedAIToolkit. pyJedAI: https://github.com/AI-team-UoA/pyJedAI
35. libpostal. https://github.com/openvenues/libpostal. pypostal: https://github.com/openvenues/pypostal
36. cleanco. https://github.com/psolin/cleanco
37. usaddress. https://github.com/datamade/usaddress. deepparse: https://github.com/GRAAL-Research/deepparse
38. Sentence-Transformers cross-encoder model table. https://www.sbert.net/docs/cross_encoder/pretrained_models.html
39. Hugging Face model cards used for licence and parameter counts: https://huggingface.co/sentence-transformers/LaBSE, https://huggingface.co/intfloat/multilingual-e5-small, https://huggingface.co/sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2, https://huggingface.co/BAAI/bge-m3, https://huggingface.co/google/muril-base-cased, https://huggingface.co/cross-encoder/ms-marco-MiniLM-L6-v2, https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct, https://huggingface.co/Qwen/Qwen2.5-3B-Instruct, https://huggingface.co/Qwen/Qwen2.5-7B-Instruct, https://huggingface.co/Qwen/Qwen3-8B, https://huggingface.co/microsoft/Phi-3.5-mini-instruct
40. Mann, Augsten, Bouros. An Empirical Evaluation of Set Similarity Join Techniques. PVLDB 9(9), 2016. https://www.vldb.org/pvldb/vol9/p636-mann.pdf
41. Bayardo, Ma, Srikant. Scaling Up All Pairs Similarity Search. WWW 2007. http://www.bayardo.org/ps/www2007.pdf
42. Xiao, Wang, Lin, Shang. Top-k Set Similarity Joins. ICDE 2009. https://cgi.cse.unsw.edu.au/~lxue/paper/icde09_chuan.pdf
43. Bruch, Nardini, Rulli, Venturini. Efficient Inverted Indexes for Approximate Retrieval over Learned Sparse Representations (Seismic). SIGIR 2024. https://arxiv.org/abs/2404.18812. Code: https://github.com/TusKANNy/seismic
44. Zhao, He. Auto-EM: End-to-end Fuzzy Entity-Matching using Pre-trained Deep Models and Transfer Learning. WWW 2019. https://www.microsoft.com/en-us/research/wp-content/uploads/2019/04/Auto-EM.pdf. Code: https://github.com/henryzhao5852/AutoEM
45. Thottingal. Phonetic Comparison Algorithm for Indian Languages (Indic Soundex). 2009. https://thottingal.in/blog/2009/07/26/indicsoundex/
46. indic-namematch. https://pypi.org/project/indic-namematch/ (source: https://github.com/kiranshivaraju/indic-namematch)
47. Soundex variants for Indian names (not read in full): https://www.researchgate.net/publication/269672873_Improvement_of_Soundex_Algorithm_for_Indian_Language_Based_on_Phonetic_Matching and https://www.researchgate.net/publication/316891126_Need_for_Customized_Soundex_based_Algorithm_on_Indian_Names_for_Phonetic_Matching
48. Search and ANN libraries: tantivy-py https://github.com/quickwit-oss/tantivy-py, hnswlib https://github.com/nmslib/hnswlib, PyNNDescent https://github.com/lmcinnes/pynndescent
49. LightGBM (MIT). https://github.com/microsoft/LightGBM. RapidFuzz (MIT): https://github.com/rapidfuzz/RapidFuzz. FAISS (MIT): https://github.com/facebookresearch/faiss. Splink (MIT, Fellegi-Sunter linkage; README claims about a million records on a laptop in around a minute): https://github.com/moj-analytical-services/splink
