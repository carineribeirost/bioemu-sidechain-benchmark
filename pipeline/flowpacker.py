"""Stage 6: Sidechain repacking with FlowPacker."""

import os
import glob
import shutil
import subprocess
import yaml
from pathlib import Path

from pipeline.utils import log


def gen_fp_config(test_path: str, checkpoint: str, num_steps: int,
                  n_samples: int, cfg_dir: str, cfg_name: str) -> str:
    """Generate a FlowPacker YAML config and save it."""
    config = {"mode": "vf",
              "data": {"data": "bc40",
                       "train_path": "",
                       "cluster_path": "",
                       "test_path": test_path,
                       "min_length": 40,
                       "max_length": 512,
                       "edge_type": "knn",
                       "max_radius": 16.0,
                       "max_neighbors": 30},
              "ckpt": checkpoint,
              "conf_ckpt": None,
              "sample": {"batch_size": 1,
                         "n_samples": n_samples,
                         "use_ema": True,
                         "eps": 2.0e-3,
                         "save_trajectory": False,
                         "coeff": 5.0,
                         "num_steps": num_steps}}

    os.makedirs(cfg_dir, exist_ok=True)
    cfg_path = os.path.join(cfg_dir, f"{cfg_name}.yaml")
    with open(cfg_path, "w") as f:
        yaml.dump(config, f, sort_keys=False, default_flow_style=False)

    log.info(f"FlowPacker config saved: {cfg_path}")
    return cfg_name


def run_stage6(config: dict) -> list[str]:
    """Stage 6: Run FlowPacker sidechain repacking.

    Returns list of paths to repacked PDB files.
    """
    protein = config["protein"]
    fp_cfg = config["flowpacker"]
    name = protein["name"]
    out_dir = config["output_dir"]

    repack_dir = os.path.join(out_dir, name, "flowpacker")
    existing = sorted(glob.glob(os.path.join(repack_dir, "final_*.pdb")))
    if existing:
        log.info(
            f"Stage 6: {len(existing)} repacked PDBs already exist, skipping")
        return existing

    root = Path(__file__).resolve().parent.parent
    repo = (root / fp_cfg["repo_path"]).resolve()
    checkpoint = str(repo / fp_cfg["checkpoint"])
    repack_env = root / "envs" / "flowpacker-env"

    bb_dir = os.path.join(out_dir, name, "final_backbones")
    bb_dir = str(Path(bb_dir).resolve())

    pdbs = sorted(glob.glob(os.path.join(bb_dir, "final_*.pdb")))
    if not pdbs:
        raise FileNotFoundError(
            f"No final backbone PDBs in {bb_dir}. Run Stage 5 first.")

    cfg_dir = str(repo / "config" / "inference")
    cfg_name = gen_fp_config(bb_dir, checkpoint,
                             fp_cfg.get("num_steps", 10),
                             fp_cfg.get("n_samples", 1),
                             cfg_dir, name)

    cmd = ["uv", "run", "--project", str(repack_env),
           "python", "sampler_pdb.py",
           cfg_name, cfg_name,
           "--use_gt_masks", "True"]

    env = os.environ.copy()
    env["PYTHONPATH"] = str(repo)

    log.info(f"Running FlowPacker on {len(pdbs)} structures...")
    result = subprocess.run(cmd, capture_output=True, text=True,
                            cwd=str(repo), env=env)

    if result.returncode != 0:
        log.error(f"FlowPacker failed:\n{result.stderr[-2000:]}")
        raise RuntimeError(
            "FlowPacker sidechain repacking failed. See error above.")

    if result.stdout:
        for line in result.stdout.strip().split("\n")[-5:]:
            log.info(line)

    samples_dir = repo / "samples" / cfg_name / "run_1"
    if not samples_dir.exists():
        raise FileNotFoundError(
            f"FlowPacker output not found at {samples_dir}. "
            f"Check FlowPacker logs.")

    os.makedirs(repack_dir, exist_ok=True)
    out_paths = []
    for pdb in sorted(samples_dir.glob("*.pdb")):
        dest = os.path.join(repack_dir, pdb.name)
        shutil.copy2(str(pdb), dest)
        out_paths.append(dest)

    log.info(
        f"Stage 6 complete: {len(out_paths)} structures with sidechains")
    return out_paths
