"""All tunable knobs in one place. CLI flags override the defaults."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class BlockingConfig:
    # Stage 1, retrieval inside (country, state) blocks; see blocking.py.
    # Chosen from a sweep on a 10% world: ~22 pairs/S1 at 98.3% pair recall.
    k_name_fwd: int = 3               # top targets per S1 on the name channel
    k_addr_fwd: int = 4               # top targets per S1 on the address channel
    k_blend_fwd: int = 3              # blended channel: name and address in one vector
    k_name_rev: int = 2               # top S1 per target on the name channel
    k_addr_rev: int = 1               # top S1 per target on the address channel
    k_blend_rev: int = 2
    blend_name_weight: float = 0.6    # share of the blended squared norm given to the name
    name_max_df: float = 0.01         # drop char-3grams in more than 1% of a block's records
    addr_max_df: float = 0.01         # same for address tokens
    min_df_cap: int = 50              # small blocks never prune below this document frequency
    min_score: float = 0.05           # retrieval scores below this are dropped
    chunk_rows: int = 200_000         # query rows per sparse top-k call
    # Stage 2, pruning (supervised meta-blocking): the threshold keeps this share
    # of the true pairs stage 1 found (measured on the tuning fold), then at most
    # max_per_s1 pairs per S1 entity. The survivors are candidate_pairs.tsv.
    prune_keep_share: float = 0.995
    max_per_s1: int = 12


@dataclass
class MatcherConfig:
    num_leaves: int = 127
    learning_rate: float = 0.05
    n_estimators: int = 600
    min_child_samples: int = 50
    feature_fraction: float = 0.8
    bagging_fraction: float = 0.8
    max_train_s1: int = 400_000       # S1 entities sampled to train the matcher


@dataclass
class Config:
    data_dir: Path = Path("student_resource/dataset")
    work_dir: Path = Path("artifacts")
    output_dir: Path = Path("output")
    n_folds: int = 5
    val_fold: int = 0
    world_frac: float = 1.0           # <1 shrinks train for fast iteration (see data.sample_world)
    max_eval_s1: int = 200_000        # S1 entities scored from each of the val and tune folds
    threads: int = 14
    seed: int = 13
    blocking: BlockingConfig = field(default_factory=BlockingConfig)
    matcher: MatcherConfig = field(default_factory=MatcherConfig)

    @property
    def cache_dir(self) -> Path:
        return self.work_dir / "cache"

    def run_dir(self, split: str) -> Path:
        tag = split if self.world_frac >= 1.0 else f"{split}_w{self.world_frac:g}"
        d = self.work_dir / tag
        d.mkdir(parents=True, exist_ok=True)
        return d
