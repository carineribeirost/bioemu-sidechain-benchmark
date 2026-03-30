"""Stages 1-2: BioEmu backbone sampling and frame extraction."""

import os
import subprocess
import glob
from pathlib import Path

from pipeline.utils import log


def run_stage1(config: dict) -> str:
    """Stage 1: Run BioEmu to generate backbone samples.

    Returns path to output directory containing samples.xtc and topology.pdb.
    """
    protein = config["protein"]
    emu_cfg = config["bioemu"]
    name = protein["name"]
    out_dir = config["output_dir"]

    emu_dir = os.path.join(out_dir, name, "bioemu")
    xtc = os.path.join(emu_dir, "samples.xtc")

    if os.path.exists(xtc):
        log.info(
            f"Stage 1: samples.xtc already exists at {emu_dir}, skipping")
        return emu_dir

    fasta = os.path.join(out_dir, name, "input", "sequence.fasta")
    if not os.path.exists(fasta):
        raise FileNotFoundError(
            f"FASTA not found: {fasta}. Run Stage 0 first.")

    os.makedirs(emu_dir, exist_ok=True)

    ver_map = {"v1.0": "bioemu-v1.0",
               "v1.1": "bioemu-v1.1",
               "v1.2": "bioemu-v1.2"}
    model = ver_map.get(emu_cfg.get("model_version", "v1.2"),
                        "bioemu-v1.2")
    n_samples = emu_cfg.get("num_samples", 10)
    do_filter = emu_cfg.get("filter_samples", True)

    root = Path(__file__).resolve().parent.parent
    emu_env = root / "envs" / "bioemu-env"

    cmd = ["uv", "run", "--project", str(emu_env),
           "python", "-m", "bioemu.sample",
           str(Path(fasta).resolve()),
           str(n_samples),
           str(Path(emu_dir).resolve()),
           "--model_name", model,
           "--filter_samples", str(do_filter)]

    env = os.environ.copy()
    try:
        import torch
        if torch.cuda.is_available():
            try:
                torch.zeros(1).cuda()
                log.info("GPU detected and usable")
            except RuntimeError:
                log.warning("GPU incompatible, falling back to CPU.")
                env["CUDA_VISIBLE_DEVICES"] = ""
                env["JAX_PLATFORMS"] = "cpu"
        else:
            log.info("No GPU detected, running on CPU")
            env["CUDA_VISIBLE_DEVICES"] = ""
            env["JAX_PLATFORMS"] = "cpu"
    except ImportError:
        env["CUDA_VISIBLE_DEVICES"] = ""
        env["JAX_PLATFORMS"] = "cpu"

    log.info(f"Running BioEmu: {n_samples} samples, model={model}")
    log.info("This may take a while depending on sequence length...")

    result = subprocess.run(cmd, capture_output=True, text=True, env=env)

    if result.returncode != 0:
        log.error(f"BioEmu failed:\n{result.stderr[-2000:]}")
        raise RuntimeError("BioEmu sampling failed. See error above.")

    if result.stdout:
        log.info(result.stdout[-500:])

    if not os.path.exists(xtc):
        raise FileNotFoundError(
            f"BioEmu completed but samples.xtc not found at {xtc}. "
            f"Check BioEmu output.")

    log.info(f"BioEmu sampling complete: {emu_dir}")
    return emu_dir


def run_stage2(config: dict) -> list[str]:
    """Stage 2: Extract individual frames from BioEmu trajectory.

    Returns list of paths to extracted PDB files.
    """
    protein = config["protein"]
    name = protein["name"]
    out_dir = config["output_dir"]
    emu_dir = os.path.join(out_dir, name, "bioemu")

    existing = sorted(glob.glob(os.path.join(emu_dir, "backbone_*.pdb")))
    if existing:
        log.info(
            f"Stage 2: {len(existing)} backbone PDBs already exist, skipping")
        return existing

    xtc = os.path.join(emu_dir, "samples.xtc")
    top = os.path.join(emu_dir, "topology.pdb")

    if not os.path.exists(xtc):
        raise FileNotFoundError(
            f"samples.xtc not found: {xtc}. Run Stage 1 first.")
    if not os.path.exists(top):
        raise FileNotFoundError(
            f"topology.pdb not found: {top}. Run Stage 1 first.")

    root = Path(__file__).resolve().parent.parent
    emu_env = root / "envs" / "bioemu-env"

    script = f"""
import mdtraj as md
import os

xtc = "{Path(xtc).resolve()}"
top = "{Path(top).resolve()}"
out_dir = "{Path(emu_dir).resolve()}"

traj = md.load(xtc, top=top)
print(f"Loaded {{traj.n_frames}} frames from trajectory")

for i in range(traj.n_frames):
    frame = traj[i]
    out = os.path.join(out_dir, f"backbone_{{i:03d}}.pdb")
    frame.save_pdb(out)

print(f"Extracted {{traj.n_frames}} frames to {{out_dir}}")
"""

    cmd = ["uv", "run", "--project", str(emu_env),
           "python", "-c", script]

    log.info("Extracting frames from trajectory...")
    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        log.error(f"Frame extraction failed:\n{result.stderr}")
        raise RuntimeError("Frame extraction failed. See error above.")

    if result.stdout:
        log.info(result.stdout.strip())

    extracted = sorted(glob.glob(os.path.join(emu_dir, "backbone_*.pdb")))
    log.info(f"Stage 2 complete: {len(extracted)} frames extracted")
    return extracted
