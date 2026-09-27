"""Command line: `python -m ber {train,predict,all} [options]`.

    python -m ber train --world 0.1     # fast iteration on a 10% world
    python -m ber train                 # full training split
    python -m ber predict               # test split -> output/*.tsv
    python -m ber all                   # train then predict

Stage outputs are cached under artifacts/; delete a run directory (or pass
--fresh) to recompute it after changing normalization or blocking.
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from . import pipeline
from .config import Config


def main() -> None:
    ap = argparse.ArgumentParser(prog="ber")
    ap.add_argument("command", choices=["train", "predict", "all"])
    ap.add_argument("--data-dir", type=Path, default=Config.data_dir)
    ap.add_argument("--work-dir", type=Path, default=Config.work_dir)
    ap.add_argument("--output-dir", type=Path, default=Config.output_dir)
    ap.add_argument("--world", type=float, default=1.0, help="train on a share of S1 entities (see data.sample_world)")
    ap.add_argument("--threads", type=int, default=Config.threads)
    ap.add_argument("--fresh", action="store_true", help="drop cached stage outputs for the chosen split(s)")
    args = ap.parse_args()

    cfg = Config(data_dir=args.data_dir, work_dir=args.work_dir, output_dir=args.output_dir,
                 world_frac=args.world, threads=args.threads)
    splits = {"train": ["train"], "predict": ["test"], "all": ["train", "test"]}[args.command]
    if args.fresh:
        for split in splits:
            # predict always runs at full scale, whatever --world says
            frac = cfg.world_frac if split == "train" else 1.0
            shutil.rmtree(Config(work_dir=cfg.work_dir, world_frac=frac).run_dir(split), ignore_errors=True)
    if args.command in ("train", "all"):
        pipeline.train(cfg)
    if args.command in ("predict", "all"):
        cfg.world_frac = 1.0
        pipeline.predict(cfg)


if __name__ == "__main__":
    main()
