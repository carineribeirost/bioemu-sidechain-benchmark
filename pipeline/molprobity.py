"""Stage 8: Structure quality scoring with reduce + probe + Biopython."""

import os
import glob
import subprocess
import math
import csv
import tempfile

from Bio.PDB import PDBParser
from Bio.PDB.Polypeptide import is_aa
from Bio.PDB.vectors import calc_dihedral

from pipeline.utils import log


def calc_clashscore(pdb_path: str) -> float | None:
    """Calculate clashscore using reduce + probe.

    Returns clashscore (clashes per 1000 atoms) or None on failure.
    """
    try:
        result = subprocess.run(["reduce", "-Quiet", "-Trim", pdb_path],
                                capture_output=True, text=True)
        trimmed = result.stdout

        result = subprocess.run(["reduce", "-Quiet", "-BUILD", "-"],
                                input=trimmed,
                                capture_output=True, text=True)
        reduced = result.stdout
        if not reduced.strip():
            log.warning(
                f"reduce produced no output for {os.path.basename(pdb_path)}")
            return None

        with tempfile.NamedTemporaryFile(mode="w", suffix=".pdb",
                                         delete=False) as tmp:
            tmp.write(reduced)
            tmp_path = tmp.name

        try:
            result = subprocess.run(
                ["probe", "-4H", "-U", "-Quiet", "-self",
                 "ALL", tmp_path],
                capture_output=True, text=True)
        finally:
            os.unlink(tmp_path)

        # Probe format :src->targ:type:atom1:atom2:gap:...
        clashes = set()
        for line in result.stdout.strip().split("\n"):
            if not line.strip():
                continue
            fields = line.split(":")
            if len(fields) >= 5 and fields[2].strip() == "bo":
                pair = tuple(sorted([fields[3].strip(),
                                     fields[4].strip()]))
                clashes.add(pair)

        n_atoms = sum(1 for l in reduced.split("\n")
                      if l.startswith("ATOM") or l.startswith("HETATM"))
        if n_atoms == 0:
            return None

        return round((len(clashes) / n_atoms) * 1000, 2)

    except Exception as e:
        log.warning(
            f"Clashscore failed for {os.path.basename(pdb_path)}: {e}")
        return None


def calc_rama(pdb_path: str) -> dict | None:
    """Calculate Ramachandran statistics using Biopython.

    Returns dict with favored_pct, allowed_pct, outlier_pct, n_res.
    """
    try:
        parser = PDBParser(QUIET=True)
        structure = parser.get_structure("s", pdb_path)

        angles = []
        for model in structure:
            for chain in model:
                residues = [r for r in chain if is_aa(r, standard=True)]
                if not residues:
                    continue
                for i, res in enumerate(residues):
                    try:
                        n = res["N"].get_vector()
                        ca = res["CA"].get_vector()
                        c = res["C"].get_vector()
                        phi = None
                        psi = None
                        if i > 0:
                            try:
                                c_prev = residues[i - 1]["C"].get_vector()
                                phi = math.degrees(
                                    calc_dihedral(c_prev, n, ca, c))
                            except KeyError:
                                pass
                        if i < len(residues) - 1:
                            try:
                                n_next = residues[i + 1]["N"].get_vector()
                                psi = math.degrees(
                                    calc_dihedral(n, ca, c, n_next))
                            except KeyError:
                                pass
                        if phi is not None and psi is not None:
                            angles.append((phi, psi))
                    except KeyError:
                        continue
            break

        if not angles:
            return None

        fav = sum(1 for p, s in angles if _is_favored(p, s))
        alw = sum(1 for p, s in angles
                  if not _is_favored(p, s) and _is_allowed(p, s))
        out = len(angles) - fav - alw
        total = len(angles)

        return {"favored_pct": round(100 * fav / total, 1),
                "allowed_pct": round(100 * alw / total, 1),
                "outlier_pct": round(100 * out / total, 1),
                "n_res": total}

    except Exception as e:
        log.warning(
            f"Ramachandran failed for {os.path.basename(pdb_path)}: {e}")
        return None


def _is_favored(phi: float, psi: float) -> bool:
    """Check if phi/psi falls in a favored Ramachandran region."""
    if -100 < phi < -40 and -67 < psi < -7:
        return True
    if -170 < phi < -50 and 90 < psi < 180:
        return True
    if -170 < phi < -50 and -180 < psi < -120:
        return True
    if 40 < phi < 100 and 10 < psi < 70:
        return True
    return False


def _is_allowed(phi: float, psi: float) -> bool:
    """Check if phi/psi falls in an allowed Ramachandran region."""
    if -180 < phi < -20 and -100 < psi < 10:
        return True
    if -180 < phi < -20 and 50 < psi < 180:
        return True
    if -180 < phi < -20 and -180 < psi < -100:
        return True
    if 20 < phi < 140 and -30 < psi < 100:
        return True
    return False


def score_structures(pdb_paths: list[str], label: str) -> list[dict]:
    """Score a list of PDB files and return results."""
    results = []
    for pdb_path in pdb_paths:
        name = os.path.basename(pdb_path)
        clash = calc_clashscore(pdb_path)
        rama = calc_rama(pdb_path)

        row = {"structure": name,
               "stage": label,
               "clashscore": clash if clash is not None else "N/A",
               "rama_favored": rama["favored_pct"] if rama else "N/A",
               "rama_allowed": rama["allowed_pct"] if rama else "N/A",
               "rama_outlier": rama["outlier_pct"] if rama else "N/A",
               "n_res": rama["n_res"] if rama else "N/A"}
        results.append(row)

        clash_str = f"{clash:.1f}" if clash is not None else "N/A"
        rama_str = (f"fav={rama['favored_pct']}% out={rama['outlier_pct']}%"
                    if rama else "N/A")
        log.info(f"  {name}: clash={clash_str}, {rama_str}")

    return results


def run_stage8(config: dict) -> str:
    """Stage 8: Score structure quality.

    Scores both pre-NVT (FlowPacker) and post-NVT structures.
    Returns path to scores TSV file.
    """
    protein = config["protein"]
    name = protein["name"]
    out_dir = config["output_dir"]

    mp_cfg = config.get("molprobity", {})
    if not mp_cfg.get("enabled", True):
        log.info("Stage 8: MolProbity scoring disabled, skipping")
        return ""

    score_dir = os.path.join(out_dir, name, "molprobity")
    tsv = os.path.join(score_dir, "scores.tsv")

    if os.path.exists(tsv):
        log.info("Stage 8: scores.tsv already exists, skipping")
        return tsv

    os.makedirs(score_dir, exist_ok=True)
    results = []

    fp_dir = os.path.join(out_dir, name, "flowpacker")
    fp_pdbs = sorted(glob.glob(os.path.join(fp_dir, "final_*.pdb")))
    if fp_pdbs:
        log.info(
            f"Scoring {len(fp_pdbs)} FlowPacker structures (pre-NVT)...")
        results.extend(score_structures(fp_pdbs, "flowpacker"))

    nvt_dir = os.path.join(out_dir, name, "nvt")
    nvt_pdbs = sorted(
        glob.glob(os.path.join(nvt_dir, "receptor_*.pdb")))
    if nvt_pdbs:
        log.info(
            f"Scoring {len(nvt_pdbs)} NVT-refined structures (post-NVT)...")
        results.extend(score_structures(nvt_pdbs, "nvt"))

    if not results:
        log.warning("No structures found to score")
        return ""

    fields = ["structure", "stage", "clashscore", "rama_favored",
              "rama_allowed", "rama_outlier", "n_res"]
    with open(tsv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, delimiter="\t")
        writer.writeheader()
        writer.writerows(results)

    for label in ["flowpacker", "nvt"]:
        rows = [r for r in results if r["stage"] == label]
        if not rows:
            continue
        cs = [r["clashscore"] for r in rows if r["clashscore"] != "N/A"]
        fv = [r["rama_favored"] for r in rows if r["rama_favored"] != "N/A"]
        ol = [r["rama_outlier"] for r in rows if r["rama_outlier"] != "N/A"]
        avg_cs = sum(cs) / len(cs) if cs else 0
        avg_fv = sum(fv) / len(fv) if fv else 0
        avg_ol = sum(ol) / len(ol) if ol else 0
        log.info(f"  {label} avg: clashscore={avg_cs:.1f}, "
                 f"rama_favored={avg_fv:.1f}%, rama_outlier={avg_ol:.1f}%")

    log.info(f"Stage 8 complete: scores saved to {tsv}")
    return tsv
