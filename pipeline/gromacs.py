"""Stages 4 and 7: GROMACS energy minimization and NVT MD refinement."""

import os
import re
import glob
import subprocess
import shutil
from pathlib import Path

from pipeline.utils import log


def run_gmx(cmd: list[str], stdin_text: str | None = None,
            cwd: str | None = None) -> subprocess.CompletedProcess:
    """Run a GROMACS command, raising on failure."""
    result = subprocess.run(cmd, input=stdin_text,
                            capture_output=True, text=True, cwd=cwd)
    if result.returncode != 0:
        raise RuntimeError(
            f"GROMACS command failed: {' '.join(cmd)}\n"
            f"stderr: {result.stderr[-1000:]}")
    return result


def gen_em_mdp(mdp_path: str, emtol: float, nsteps: int,
               coulombtype: str = "cutoff") -> None:
    """Generate energy minimization MDP file."""
    content = f"""\
integrator      = steep
emtol           = {emtol}
emstep          = 0.01
nsteps          = {nsteps}
nstxout         = 0
nstvout         = 0
nstenergy       = 500
nstlog          = 500
cutoff-scheme   = Verlet
ns-type         = grid
nstlist         = 10
rcoulomb        = 1.0
rvdw            = 1.0
coulombtype     = {coulombtype}
pbc             = xyz
constraints     = none
define          = -DPOSRES
"""
    with open(mdp_path, "w") as f:
        f.write(content)


def minimize_single(pdb_path: str, out_path: str,
                    ff: str, mdp: str,
                    restr_fc: int = 50) -> bool:
    """Run vacuum energy minimization on a single GLY structure.

    Returns True on success, False on failure.
    """
    work = os.path.dirname(os.path.abspath(pdb_path))
    base = os.path.splitext(os.path.basename(pdb_path))[0]
    mdp = os.path.abspath(mdp)
    out_path = os.path.abspath(out_path)

    proc_gro = f"{base}_processed.gro"
    topol = f"topol_{base}.top"
    box_gro = f"{base}_boxed.gro"
    em_tpr = f"em_{base}.tpr"
    pdb_file = os.path.basename(pdb_path)

    try:
        run_gmx(["gmx", "pdb2gmx",
                 "-f", pdb_file, "-o", proc_gro, "-p", topol,
                 "-ff", ff, "-water", "none", "-ignh"], cwd=work)

        fc = str(restr_fc)
        posre = os.path.join(work, "posre.itp")
        if os.path.exists(posre):
            with open(posre, "r") as f:
                content = f.read()
            lines = []
            for line in content.split("\n"):
                parts = line.split()
                if len(parts) == 5 and parts[1] == "1":
                    try:
                        int(parts[0])
                        float(parts[2])
                        parts[2] = parts[3] = parts[4] = fc
                        line = "  ".join(parts)
                    except (ValueError, IndexError):
                        pass
                lines.append(line)
            with open(posre, "w") as f:
                f.write("\n".join(lines))

        run_gmx(["gmx", "editconf",
                 "-f", proc_gro, "-o", box_gro,
                 "-c", "-d", "1.0", "-bt", "cubic"], cwd=work)

        run_gmx(["gmx", "grompp",
                 "-f", mdp, "-c", box_gro, "-r", box_gro,
                 "-p", topol, "-o", em_tpr,
                 "-maxwarn", "10"], cwd=work)

        run_gmx(["gmx", "mdrun",
                 "-v", "-deffnm", f"em_{base}",
                 "-ntmpi", "1"], cwd=work)

        run_gmx(["gmx", "trjconv",
                 "-s", em_tpr, "-f", f"em_{base}.gro",
                 "-o", out_path],
                stdin_text="0\n", cwd=work)

        for pat in [f"{base}_processed.gro", f"{base}_boxed.gro",
                    f"em_{base}.*", f"topol_{base}.*", "posre.itp",
                    "mdout.mdp", "#*"]:
            for f in glob.glob(os.path.join(work, pat)):
                if os.path.abspath(f) != out_path:
                    os.remove(f)

        return True

    except RuntimeError as e:
        log.error(
            f"Minimization failed for {os.path.basename(pdb_path)}: {e}")
        return False


def run_stage4(config: dict) -> list[str]:
    """Stage 4: Energy minimization in vacuum for each GLY structure.

    Returns list of paths to minimized PDB files.
    """
    protein = config["protein"]
    gmx_cfg = config["gromacs"]
    name = protein["name"]
    out_dir = config["output_dir"]
    ff = gmx_cfg["forcefield"]

    gly_dir = os.path.join(out_dir, name, "gly")

    existing = sorted(
        glob.glob(os.path.join(gly_dir, "minimized_backbone_*_gly.pdb")))
    if existing:
        log.info(
            f"Stage 4: {len(existing)} minimized PDBs already exist, skipping")
        return existing

    pdbs = sorted(
        glob.glob(os.path.join(gly_dir, "backbone_*_gly.pdb")))
    if not pdbs:
        raise FileNotFoundError(
            f"No GLY PDBs found in {gly_dir}. Run Stage 3 first.")

    em_vac = gmx_cfg.get("em_vacuum", {})
    em_steps = em_vac.get("nsteps", 5000)
    emtol = em_vac.get("emtol", 100)
    vac_fc = em_vac.get("restraint_fc", 50)

    custom_mdp = os.path.join(gly_dir, "em_vacuum.mdp")
    gen_em_mdp(custom_mdp, emtol, em_steps, coulombtype="cutoff")

    log.info(
        f"Running vacuum EM with ff={ff}, emtol={emtol}, "
        f"nsteps={em_steps}, restraint_fc={vac_fc}")

    ok = 0
    fail = 0
    out_paths = []

    for pdb in pdbs:
        base = os.path.basename(pdb)
        out_name = f"minimized_{base}"
        out = os.path.join(gly_dir, out_name)

        if os.path.exists(out):
            log.info(f"Already minimized: {out_name}, skipping")
            out_paths.append(out)
            ok += 1
            continue

        log.info(f"Minimizing {base}...")
        if minimize_single(pdb, out, ff, custom_mdp, vac_fc):
            out_paths.append(out)
            ok += 1
        else:
            fail += 1

    log.info(f"Stage 4 complete: {ok}/{ok + fail} structures minimized")
    if fail > 0:
        log.warning(f"{fail} structures failed minimization")

    return out_paths


def gen_nvt_mdp(config: dict, mdp_path: str) -> None:
    """Generate NVT MDP file from config parameters."""
    nvt = config["gromacs"]["nvt"]
    temp = nvt.get("temperature", 300)
    dt = nvt.get("dt", 0.002)
    nsteps = nvt.get("nsteps", 25000)
    tcoupl = nvt.get("tcoupl", "V-rescale")
    tau_t = nvt.get("tau_t", 0.1)

    content = f"""\
define          = -DPOSRES
integrator      = md
dt              = {dt}
nsteps          = {nsteps}
nstxout         = 500
nstvout         = 500
nstenergy       = 500
nstlog          = 500
continuation    = no
gen-vel         = yes
gen-temp        = {temp}
gen-seed        = -1
cutoff-scheme   = Verlet
ns-type         = grid
nstlist         = 10
rcoulomb        = 1.0
rvdw            = 1.0
coulombtype     = PME
pbc             = xyz
tcoupl          = {tcoupl}
tc-grps         = Protein Non-Protein
tau-t           = {tau_t} {tau_t}
ref-t           = {temp} {temp}
pcoupl          = no
constraints     = h-bonds
constraint-algorithm = LINCS
"""
    with open(mdp_path, "w") as f:
        f.write(content)


def gen_npt_mdp(config: dict, mdp_path: str) -> None:
    """Generate NPT MDP file from config parameters."""
    npt = config["gromacs"]["npt"]
    temp = npt.get("temperature", 300)
    dt = npt.get("dt", 0.002)
    nsteps = npt.get("nsteps", 200000)
    tcoupl = npt.get("tcoupl", "V-rescale")
    tau_t = npt.get("tau_t", 0.1)
    pcoupl = npt.get("pcoupl", "Parrinello-Rahman")
    pressure = npt.get("pressure", 1.0)
    tau_p = npt.get("tau_p", 2.0)

    content = f"""\
define          = -DPOSRES
integrator      = md
dt              = {dt}
nsteps          = {nsteps}
nstxout         = 500
nstvout         = 500
nstenergy       = 500
nstlog          = 500
continuation    = yes
gen-vel         = no
cutoff-scheme   = Verlet
ns-type         = grid
nstlist         = 10
rcoulomb        = 1.0
rvdw            = 1.0
coulombtype     = PME
pbc             = xyz
tcoupl          = {tcoupl}
tc-grps         = Protein Non-Protein
tau-t           = {tau_t} {tau_t}
ref-t           = {temp} {temp}
pcoupl          = {pcoupl}
pcoupltype      = isotropic
tau-p           = {tau_p}
ref-p           = {pressure}
compressibility = 4.5e-5
refcoord-scaling = com
constraints     = h-bonds
constraint-algorithm = LINCS
"""
    with open(mdp_path, "w") as f:
        f.write(content)


def nvt_single(pdb_path: str, out_path: str, work: str,
               ff: str, water: str, em_mdp: str,
               nvt_mdp: str, gmx_cfg: dict, threads: int,
               npt_mdp: str | None = None) -> bool:
    """Run NVT (and optionally NPT) refinement on a single structure.

    Returns True on success, False on failure.
    """
    base = os.path.splitext(os.path.basename(pdb_path))[0]
    pdb_file = os.path.basename(pdb_path)
    em_mdp = os.path.abspath(em_mdp)
    nvt_mdp = os.path.abspath(nvt_mdp)
    out_path = os.path.abspath(out_path)

    nvt_cfg = gmx_cfg.get("nvt", {})
    box_cfg = gmx_cfg.get("box", {})
    ion_cfg = gmx_cfg.get("ions", {})
    em_solv = gmx_cfg.get("em_solvated", {})

    restr_fc = nvt_cfg.get("restraint_fc", 5)
    em_fc = str(em_solv.get("restraint_fc", 500))
    box_type = box_cfg.get("type", "dodecahedron")
    box_clear = str(box_cfg.get("clearance", 1.0))
    ion_conc = str(ion_cfg.get("concentration", 0.15))
    ion_pname = ion_cfg.get("pname", "NA")
    ion_nname = ion_cfg.get("nname", "CL")

    try:
        run_gmx(["gmx", "pdb2gmx",
                 "-f", pdb_file,
                 "-o", f"{base}_processed.gro",
                 "-p", f"topol_{base}.top",
                 "-ff", ff, "-water", water, "-ignh"], cwd=work)

        topol = os.path.join(work, f"topol_{base}.top")

        posre_def = os.path.join(work, "posre.itp")
        posre_heavy = os.path.join(work, f"posre_heavy_{base}.itp")
        if os.path.exists(posre_def):
            with open(posre_def, "r") as f:
                lines = f.readlines()
            with open(posre_heavy, "w") as f:
                for line in lines:
                    parts = line.split()
                    if len(parts) == 5 and parts[1] == "1":
                        try:
                            int(parts[0])
                            float(parts[2])
                            parts[2] = parts[3] = parts[4] = em_fc
                            f.write("  ".join(parts) + "\n")
                            continue
                        except (ValueError, IndexError):
                            pass
                    f.write(line)

        # CA restraints for NVT
        run_gmx(["gmx", "make_ndx",
                 "-f", f"{base}_processed.gro",
                 "-o", f"ca_{base}.ndx"],
                stdin_text="a CA\nq\n", cwd=work)

        run_gmx(["gmx", "genrestr",
                 "-f", f"{base}_processed.gro",
                 "-n", f"ca_{base}.ndx",
                 "-o", f"posre_ca_{base}.itp",
                 "-fc", str(restr_fc), str(restr_fc), str(restr_fc)],
                stdin_text="CA\n", cwd=work)

        _patch_posre(topol, f"posre_heavy_{base}.itp")

        run_gmx(["gmx", "editconf",
                 "-f", f"{base}_processed.gro",
                 "-o", f"{base}_boxed.gro",
                 "-c", "-d", box_clear, "-bt", box_type], cwd=work)

        run_gmx(["gmx", "solvate",
                 "-cp", f"{base}_boxed.gro", "-cs", "spc216.gro",
                 "-o", f"{base}_solv.gro",
                 "-p", f"topol_{base}.top"], cwd=work)

        run_gmx(["gmx", "grompp",
                 "-f", em_mdp, "-c", f"{base}_solv.gro",
                 "-r", f"{base}_solv.gro",
                 "-p", f"topol_{base}.top",
                 "-o", f"ions_{base}.tpr",
                 "-maxwarn", "10"], cwd=work)

        run_gmx(["gmx", "genion",
                 "-s", f"ions_{base}.tpr",
                 "-o", f"{base}_ions.gro",
                 "-p", f"topol_{base}.top",
                 "-pname", ion_pname, "-nname", ion_nname,
                 "-neutral", "-conc", ion_conc],
                stdin_text="SOL\n", cwd=work)

        run_gmx(["gmx", "grompp",
                 "-f", em_mdp, "-c", f"{base}_ions.gro",
                 "-r", f"{base}_ions.gro",
                 "-p", f"topol_{base}.top",
                 "-o", f"em_{base}.tpr",
                 "-maxwarn", "10"], cwd=work)

        run_gmx(["gmx", "mdrun",
                 "-v", "-deffnm", f"em_{base}",
                 "-ntmpi", "1", "-ntomp", str(threads)], cwd=work)

        run_gmx(["gmx", "trjconv",
                 "-s", f"em_{base}.tpr", "-f", f"em_{base}.gro",
                 "-o", f"em_{base}_centered.gro",
                 "-pbc", "mol", "-ur", "compact", "-center"],
                stdin_text="Protein\nSystem\n", cwd=work)

        _patch_posre(topol, f"posre_ca_{base}.itp")

        run_gmx(["gmx", "grompp",
                 "-f", nvt_mdp,
                 "-c", f"em_{base}_centered.gro",
                 "-r", f"em_{base}_centered.gro",
                 "-p", f"topol_{base}.top",
                 "-o", f"nvt_{base}.tpr",
                 "-maxwarn", "10"], cwd=work)

        run_gmx(["gmx", "mdrun",
                 "-v", "-deffnm", f"nvt_{base}",
                 "-ntmpi", "1", "-ntomp", str(threads)], cwd=work)

        # Determine which phase provides the final structure
        final_prefix = f"nvt_{base}"

        # Optional NPT equilibration
        if npt_mdp is not None:
            npt_mdp_abs = os.path.abspath(npt_mdp)
            npt_cfg = gmx_cfg.get("npt", {})
            npt_fc = npt_cfg.get("restraint_fc", 5)

            # Generate CA restraints for NPT (may differ from NVT)
            run_gmx(["gmx", "genrestr",
                     "-f", f"{base}_processed.gro",
                     "-n", f"ca_{base}.ndx",
                     "-o", f"posre_ca_npt_{base}.itp",
                     "-fc", str(npt_fc), str(npt_fc), str(npt_fc)],
                    stdin_text="CA\n", cwd=work)

            _patch_posre(topol, f"posre_ca_npt_{base}.itp")

            run_gmx(["gmx", "grompp",
                     "-f", npt_mdp_abs,
                     "-c", f"nvt_{base}.gro",
                     "-r", f"nvt_{base}.gro",
                     "-p", f"topol_{base}.top",
                     "-t", f"nvt_{base}.cpt",
                     "-o", f"npt_{base}.tpr",
                     "-maxwarn", "10"], cwd=work)

            run_gmx(["gmx", "mdrun",
                     "-v", "-deffnm", f"npt_{base}",
                     "-ntmpi", "1", "-ntomp", str(threads)], cwd=work)

            final_prefix = f"npt_{base}"

        run_gmx(["gmx", "trjconv",
                 "-s", f"{final_prefix}.tpr",
                 "-f", f"{final_prefix}.gro",
                 "-o", out_path,
                 "-pbc", "mol", "-ur", "compact", "-center"],
                stdin_text="Protein\nProtein\n", cwd=work)

        for pat in [f"{base}_processed.gro", f"{base}_boxed.gro",
                    f"{base}_solv.gro", f"{base}_ions.gro",
                    f"ions_{base}.*", f"em_{base}*", f"nvt_{base}*",
                    f"npt_{base}*", f"topol_{base}.*", f"posre*{base}*",
                    f"heavy_{base}*", f"ca_{base}*",
                    "posre.itp", "mdout.mdp", "#*"]:
            for f in glob.glob(os.path.join(work, pat)):
                if os.path.abspath(f) != out_path:
                    os.remove(f)

        return True

    except RuntimeError as e:
        log.error(f"MD refinement failed for {base}: {e}")
        return False


def _patch_posre(topol_path: str, posre_itp: str) -> None:
    """Replace POSRES include in topology with a different restraint file."""
    with open(topol_path, "r") as f:
        content = f.read()

    inc = f'#include "{posre_itp}"'
    if "#ifdef POSRES" in content:
        content = re.sub(r'#ifdef POSRES\n#include "[^"]*"',
                         f"#ifdef POSRES\n{inc}", content)
    else:
        content += f"\n#ifdef POSRES\n{inc}\n#endif\n"

    with open(topol_path, "w") as f:
        f.write(content)


def run_stage7(config: dict) -> list[str]:
    """Stage 7: NVT MD refinement with explicit solvent.

    Returns list of paths to refined PDB files.
    """
    protein = config["protein"]
    gmx_cfg = config["gromacs"]
    name = protein["name"]
    out_dir = config["output_dir"]
    ff = gmx_cfg["forcefield"]
    water = gmx_cfg["water_model"]
    nvt_cfg = gmx_cfg["nvt"]
    threads = gmx_cfg.get("threads", 4)

    nvt_dir = os.path.join(out_dir, name, "nvt")

    existing = sorted(
        glob.glob(os.path.join(nvt_dir, "receptor_*.pdb")))
    if existing:
        log.info(
            f"Stage 7: {len(existing)} NVT-refined PDBs exist, skipping")
        return existing

    fp_dir = os.path.join(out_dir, name, "flowpacker")
    pdbs = sorted(glob.glob(os.path.join(fp_dir, "final_*.pdb")))
    if not pdbs:
        raise FileNotFoundError(
            f"No FlowPacker PDBs in {fp_dir}. Run Stage 6 first.")

    os.makedirs(nvt_dir, exist_ok=True)

    em_solv = gmx_cfg.get("em_solvated", {})
    em_mdp = os.path.join(nvt_dir, "em_solvated.mdp")
    gen_em_mdp(em_mdp,
               emtol=em_solv.get("emtol", 1000),
               nsteps=em_solv.get("nsteps", 50000),
               coulombtype="PME")
    nvt_mdp = os.path.join(nvt_dir, "nvt.mdp")
    gen_nvt_mdp(config, nvt_mdp)

    npt_cfg = gmx_cfg.get("npt", {})
    npt_enabled = npt_cfg.get("enabled", False)
    npt_mdp = None
    if npt_enabled:
        npt_mdp = os.path.join(nvt_dir, "npt.mdp")
        gen_npt_mdp(config, npt_mdp)

    temp = nvt_cfg.get("temperature", 300)
    dt = nvt_cfg.get("dt", 0.002)
    nsteps = nvt_cfg.get("nsteps", 25000)
    phases = "NVT + NPT" if npt_enabled else "NVT"
    log.info(f"Running {phases}: ff={ff}, water={water}, "
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
            log.info(f"Already refined: {out_name}, skipping")
            out_paths.append(out)
            ok += 1
            continue

        struct_work = os.path.join(nvt_dir, f"work_{idx}")
        os.makedirs(struct_work, exist_ok=True)
        shutil.copy2(pdb, os.path.join(struct_work,
                                       os.path.basename(pdb)))

        log.info(f"MD refinement: {os.path.basename(pdb)}...")
        if nvt_single(os.path.join(struct_work, os.path.basename(pdb)),
                      out, struct_work, ff, water,
                      em_mdp, nvt_mdp, gmx_cfg, threads,
                      npt_mdp=npt_mdp):
            out_paths.append(out)
            ok += 1
            shutil.rmtree(struct_work, ignore_errors=True)
        else:
            fail += 1

    log.info(f"Stage 7 complete: {ok}/{ok + fail} structures refined")
    if fail > 0:
        log.warning(f"{fail} structures failed NVT refinement")

    return out_paths
