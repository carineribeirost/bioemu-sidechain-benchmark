#!/usr/bin/env python3
"""Protein ensemble generation pipeline — orchestrator."""

import argparse
import yaml
import sys

from pipeline.utils import log
from pipeline.input import run_stage0
from pipeline.bioemu import run_stage1, run_stage2
from pipeline.prepare import run_stage3, run_stage5
from pipeline.gromacs import run_stage4, run_stage7
from pipeline.flowpacker import run_stage6
from pipeline.molprobity import run_stage8


STAGES = {0: ("Input preparation", run_stage0),
           1: ("BioEmu backbone sampling", run_stage1),
           2: ("Frame extraction", run_stage2),
           3: ("Mutate to glycine", run_stage3),
           4: ("Energy minimization (vacuum)", run_stage4),
           5: ("Restore sequence", run_stage5),
           6: ("FlowPacker sidechain repacking", run_stage6),
           7: ("NVT MD refinement", run_stage7),
           8: ("Structure quality scoring", run_stage8)}


def main():
    parser = argparse.ArgumentParser(
        description="Protein ensemble generation pipeline")
    parser.add_argument("config", help="Path to config.yaml")
    parser.add_argument("--stage", type=int, default=0,
                        help="Start from this stage (default: 0)")
    parser.add_argument("--only", action="store_true",
                        help="Run only the specified stage")
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    start = args.stage
    end = args.stage + 1 if args.only else max(STAGES.keys()) + 1

    for num in range(start, end):
        if num not in STAGES:
            continue
        name, fn = STAGES[num]
        log.info(f"{'=' * 60}")
        log.info(f"Stage {num}: {name}")
        log.info(f"{'=' * 60}")
        fn(config)

    log.info("Pipeline complete.")


if __name__ == "__main__":
    main()
