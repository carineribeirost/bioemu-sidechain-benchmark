#!/usr/bin/env python3
"""
FlowPacker vs HPacker comparison.

Runs HPacker on the same backbone structures used by FlowPacker,
refines both with the same GROMACS NVT protocol, and generates
a side-by-side MolProbity quality report.
"""

import os
import sys
import csv
import glob
import shutil
from pathlib import Path

import yaml

# Add project root to path for pipeline imports
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from pipeline.utils import log
from pipeline.hpacker import run_hpacker
from pipeline.gromacs import nvt_single, gen_nvt_mdp
from pipeline.molprobity import score_structures


def run_nvt_on_pdbs(pdbs: list[str], nvt_dir: str,
                    config: dict) -> list[str]:
    """Run GROMACS NVT refinement on a list of PDBs.

    Reuses the same logic as Stage 7 but with arbitrary input.
    """
    gmx_cfg = config["gromacs"]
    ff = gmx_cfg["forcefield"]
    water = gmx_cfg["water_model"]
    nvt_cfg = gmx_cfg["nvt"]
    threads = gmx_cfg.get("threads", 4)

    os.makedirs(nvt_dir, exist_ok=True)

    em_mdp = str(ROOT / "mdp" / "em_solvated.mdp")
    nvt_mdp = os.path.join(nvt_dir, "nvt.mdp")
    gen_nvt_mdp(config, nvt_mdp)

    temp = nvt_cfg.get("temperature", 300)
    dt = nvt_cfg.get("dt", 0.002)
    nsteps = nvt_cfg.get("nsteps", 25000)
    log.info(
        f"NVT refinement: ff={ff}, water={water}, "
        f"T={temp}K, dt={dt}ps, nsteps={nsteps}")

    ok = 0
    fail = 0
    out_paths = []

    for pdb in pdbs:
        base = os.path.basename(pdb).replace("final_", "")
        idx = base.replace(".pdb", "")
        out_name = f"receptor_{idx}.pdb"
        out = os.path.join(nvt_dir, out_name)

        if os.path.exists(out):
            log.info(f"  Already refined: {out_name}, skipping")
            out_paths.append(out)
            ok += 1
            continue

        work = os.path.join(nvt_dir, f"work_{idx}")
        os.makedirs(work, exist_ok=True)
        shutil.copy2(pdb, os.path.join(work, os.path.basename(pdb)))

        log.info(f"  NVT: {os.path.basename(pdb)}...")
        if nvt_single(
                os.path.join(work, os.path.basename(pdb)),
                out, work, ff, water, em_mdp, nvt_mdp,
                nvt_cfg, threads):
            out_paths.append(out)
            ok += 1
            shutil.rmtree(work, ignore_errors=True)
        else:
            fail += 1

    log.info(f"NVT complete: {ok}/{ok + fail} structures refined")
    return out_paths


def write_report(all_results: list[dict], report_path: str) -> None:
    """Write comparison report TSV with per-group averages."""
    fields = ["structure", "stage", "clashscore",
              "rama_favored", "rama_allowed", "rama_outlier", "n_res"]

    with open(report_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, delimiter="\t")
        writer.writeheader()
        writer.writerows(all_results)

    log.info("\n" + "=" * 70)
    log.info("COMPARISON REPORT")
    log.info("=" * 70)

    labels = []
    for r in all_results:
        if r["stage"] not in labels:
            labels.append(r["stage"])

    for label in labels:
        rows = [r for r in all_results if r["stage"] == label]
        cs = [r["clashscore"] for r in rows
              if r["clashscore"] != "N/A"]
        fv = [r["rama_favored"] for r in rows
              if r["rama_favored"] != "N/A"]
        ol = [r["rama_outlier"] for r in rows
              if r["rama_outlier"] != "N/A"]
        avg_cs = sum(cs) / len(cs) if cs else 0
        avg_fv = sum(fv) / len(fv) if fv else 0
        avg_ol = sum(ol) / len(ol) if ol else 0
        log.info(
            f"  {label:25s}  N={len(rows):2d}  "
            f"clash={avg_cs:6.1f}  "
            f"rama_fav={avg_fv:5.1f}%  "
            f"rama_out={avg_ol:4.1f}%")

    log.info("=" * 70)
    log.info(f"Full report: {report_path}")


def main():
    cfg_path = str(ROOT / "config.yaml")
    with open(cfg_path, "r") as f:
        config = yaml.safe_load(f)

    name = config["protein"]["name"]
    out_dir = config["output_dir"]
    base_dir = os.path.join(out_dir, name)

    fp_dir = os.path.join(base_dir, "flowpacker")
    nvt_fp_dir = os.path.join(base_dir, "nvt")
    hp_dir = os.path.join(base_dir, "hpacker")
    nvt_hp_dir = os.path.join(base_dir, "nvt_hpacker")
    cmp_dir = os.path.join(base_dir, "comparison")
    os.makedirs(cmp_dir, exist_ok=True)

    # --- Step 1: Run HPacker on the same FlowPacker inputs ---
    fp_pdbs = sorted(glob.glob(os.path.join(fp_dir, "final_*.pdb")))
    if not fp_pdbs:
        log.error(f"No FlowPacker PDBs in {fp_dir}. "
                  f"Run the main pipeline first.")
        sys.exit(1)

    log.info(f"Found {len(fp_pdbs)} FlowPacker structures")
    log.info("--- Step 1: HPacker sidechain repacking ---")
    hp_pdbs = run_hpacker(config, fp_dir, hp_dir)

    # --- Step 2: GROMACS NVT on HPacker outputs ---
    if hp_pdbs:
        log.info("--- Step 2: GROMACS NVT on HPacker structures ---")
        nvt_hp_pdbs = run_nvt_on_pdbs(hp_pdbs, nvt_hp_dir, config)
    else:
        log.error("No HPacker outputs, skipping NVT")
        nvt_hp_pdbs = []

    # --- Step 3: Score all groups with MolProbity ---
    log.info("--- Step 3: MolProbity scoring ---")
    all_results = []

    log.info(f"Scoring {len(fp_pdbs)} FlowPacker (pre-MD)...")
    all_results.extend(score_structures(fp_pdbs, "flowpacker"))

    if hp_pdbs:
        log.info(f"Scoring {len(hp_pdbs)} HPacker (pre-MD)...")
        all_results.extend(score_structures(hp_pdbs, "hpacker"))

    nvt_fp_pdbs = sorted(
        glob.glob(os.path.join(nvt_fp_dir, "receptor_*.pdb")))
    if nvt_fp_pdbs:
        log.info(
            f"Scoring {len(nvt_fp_pdbs)} FlowPacker+NVT (post-MD)...")
        all_results.extend(
            score_structures(nvt_fp_pdbs, "flowpacker_nvt"))

    if nvt_hp_pdbs:
        log.info(
            f"Scoring {len(nvt_hp_pdbs)} HPacker+NVT (post-MD)...")
        all_results.extend(
            score_structures(nvt_hp_pdbs, "hpacker_nvt"))

    # --- Step 4: Generate report ---
    report_path = os.path.join(cmp_dir, "report.tsv")
    write_report(all_results, report_path)


if __name__ == "__main__":
    main()
