"""HPacker sidechain repacking (alternative to FlowPacker)."""

import os
import glob
import subprocess
from pathlib import Path

from pipeline.utils import log


def run_hpacker(config: dict, input_dir: str,
                output_dir: str) -> list[str]:
    """Run HPacker sidechain repacking on backbone PDBs.

    Args:
        config: Pipeline configuration dict.
        input_dir: Directory with backbone PDBs (final_*.pdb).
        output_dir: Directory for repacked PDBs.

    Returns list of paths to repacked PDB files.
    """
    existing = sorted(glob.glob(os.path.join(output_dir, "final_*.pdb")))
    if existing:
        log.info(
            f"HPacker: {len(existing)} repacked PDBs already exist, "
            f"skipping")
        return existing

    pdbs = sorted(glob.glob(os.path.join(input_dir, "final_*.pdb")))
    if not pdbs:
        raise FileNotFoundError(
            f"No backbone PDBs in {input_dir}")

    root = Path(__file__).resolve().parent.parent
    hp_env = root / "envs" / "hpacker-env"

    os.makedirs(output_dir, exist_ok=True)

    script = """
import sys
from hpacker import HPacker

in_pdb = sys.argv[1]
out_pdb = sys.argv[2]

hp = HPacker(in_pdb)
hp.reconstruct_sidechains(num_refinement_iterations=5)
hp.write_pdb(out_pdb)
"""

    log.info(f"Running HPacker on {len(pdbs)} structures...")
    out_paths = []
    ok = 0
    fail = 0

    for pdb in pdbs:
        name = os.path.basename(pdb)
        dest = os.path.join(output_dir, name)

        if os.path.exists(dest):
            log.info(f"  Already repacked: {name}, skipping")
            out_paths.append(dest)
            ok += 1
            continue

        in_abs = str(Path(pdb).resolve())
        out_abs = str(Path(dest).resolve())

        cmd = ["uv", "run", "--project", str(hp_env),
               "python", "-c", script, in_abs, out_abs]

        result = subprocess.run(
            cmd, capture_output=True, text=True)

        if result.returncode != 0:
            log.error(
                f"  HPacker failed for {name}:\n"
                f"{result.stderr[-500:]}")
            fail += 1
        elif os.path.exists(dest):
            out_paths.append(dest)
            ok += 1
            log.info(f"  Repacked: {name}")
        else:
            log.error(f"  HPacker produced no output for {name}")
            fail += 1

    log.info(
        f"HPacker complete: {ok}/{ok + fail} structures repacked")
    if fail > 0:
        log.warning(f"{fail} structures failed HPacker repacking")

    return out_paths
