"""Stages 3 and 5: Mutate to glycine and restore original sequence."""

import os
import glob
from collections import defaultdict

from Bio.PDB import PDBParser, PDBIO, Residue
from Bio import SeqIO
from Bio.Data.IUPACData import protein_letters_1to3

from pipeline.utils import log


def mutate_to_gly(in_pdb: str, out_pdb: str) -> None:
    """Replace all residues with GLY, keeping only backbone atoms."""
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("structure", in_pdb)

    res_idx = 1
    for model in structure:
        for chain in model:
            new_residues = []
            for residue in list(chain):
                if residue.id[0] != " ":
                    continue
                atoms = [atom.copy() for atom in residue
                         if atom.get_name() in ("N", "CA", "C", "O")]
                new_res = Residue.Residue(
                    (" ", res_idx, " "), "GLY", residue.segid)
                for atom in atoms:
                    new_res.add(atom)
                new_residues.append(new_res)
                res_idx += 1

            for residue in list(chain):
                chain.detach_child(residue.id)
            for residue in new_residues:
                chain.add(residue)

    io = PDBIO()
    io.set_structure(structure)
    io.save(out_pdb)


def run_stage3(config: dict) -> list[str]:
    """Stage 3: Mutate all backbone PDBs to glycine.

    Returns list of paths to GLY PDB files.
    """
    protein = config["protein"]
    name = protein["name"]
    out_dir = config["output_dir"]

    gly_dir = os.path.join(out_dir, name, "gly")
    existing = sorted(glob.glob(os.path.join(gly_dir, "backbone_*_gly.pdb")))
    if existing:
        log.info(f"Stage 3: {len(existing)} GLY PDBs already exist, skipping")
        return existing

    bio_dir = os.path.join(out_dir, name, "bioemu")
    pdbs = sorted(glob.glob(os.path.join(bio_dir, "backbone_*.pdb")))
    if not pdbs:
        raise FileNotFoundError(
            f"No backbone PDBs found in {bio_dir}. Run Stage 2 first.")

    os.makedirs(gly_dir, exist_ok=True)

    out_paths = []
    for pdb in pdbs:
        base = os.path.basename(pdb).replace(".pdb", "_gly.pdb")
        out = os.path.join(gly_dir, base)
        log.info(f"Mutating {os.path.basename(pdb)} -> {base}")
        mutate_to_gly(pdb, out)
        out_paths.append(out)

    log.info(f"Stage 3 complete: {len(out_paths)} GLY structures")
    return out_paths


def restore_sequence(in_pdb: str, out_pdb: str, resnames: list[str]) -> None:
    """Replace GLY residue names with correct AA names from FASTA."""
    res_blocks = defaultdict(list)

    with open(in_pdb, "r") as f:
        for line in f:
            if line.startswith("ATOM"):
                chain_id = line[21]
                res_id = int(line[22:26])
                res_blocks[(chain_id, res_id)].append(line)

    lines = []
    idx = 1
    sorted_blocks = sorted(res_blocks.items(), key=lambda x: x[0][1])

    for i, (uid, atom_lines) in enumerate(sorted_blocks):
        if i >= len(resnames):
            raise ValueError(
                f"Too many residues in {in_pdb}: "
                f"got >{i + 1}, expected {len(resnames)}")
        new_name = resnames[i]
        for line in atom_lines:
            new_line = (line[:17] + f"{new_name:>3}" +
                        line[20:22] + f"{idx:>4}" + line[26:])
            lines.append(new_line)
        idx += 1

    if idx - 1 != len(resnames):
        raise ValueError(
            f"Residue count mismatch in {in_pdb}: "
            f"got {idx - 1}, expected {len(resnames)}")

    with open(out_pdb, "w") as f:
        for line in lines:
            f.write(line)
        f.write("TER\nENDMDL\n")


def run_stage5(config: dict) -> list[str]:
    """Stage 5: Restore original sequence on minimized GLY structures.

    Returns list of paths to final backbone PDB files.
    """
    protein = config["protein"]
    name = protein["name"]
    out_dir = config["output_dir"]

    final_dir = os.path.join(out_dir, name, "final_backbones")
    existing = sorted(glob.glob(os.path.join(final_dir, "final_*.pdb")))
    if existing:
        log.info(
            f"Stage 5: {len(existing)} final PDBs already exist, skipping")
        return existing

    fasta = os.path.join(out_dir, name, "input", "sequence.fasta")
    if not os.path.exists(fasta):
        raise FileNotFoundError(
            f"FASTA not found: {fasta}. Run Stage 0 first.")

    record = next(SeqIO.parse(fasta, "fasta"))
    aa_seq = str(record.seq)
    resnames = [protein_letters_1to3[aa].upper() for aa in aa_seq]

    gly_dir = os.path.join(out_dir, name, "gly")
    pdbs = sorted(
        glob.glob(os.path.join(gly_dir, "minimized_backbone_*_gly.pdb")))
    if not pdbs:
        raise FileNotFoundError(
            f"No minimized GLY PDBs in {gly_dir}. Run Stage 4 first.")

    os.makedirs(final_dir, exist_ok=True)

    out_paths = []
    for pdb in pdbs:
        base = os.path.basename(pdb)
        base = base.replace("minimized_backbone_", "final_").replace(
            "_gly", "")
        out = os.path.join(final_dir, base)
        log.info(f"Restoring sequence: {os.path.basename(pdb)} -> {base}")
        try:
            restore_sequence(pdb, out, resnames)
            out_paths.append(out)
        except ValueError as e:
            log.error(f"Failed to restore {pdb}: {e}")

    log.info(
        f"Stage 5 complete: {len(out_paths)}/{len(pdbs)} structures restored")
    return out_paths
