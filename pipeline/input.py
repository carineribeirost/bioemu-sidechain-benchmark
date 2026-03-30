"""Stage 0: Input preparation — fetch, clean, and validate protein sequence."""

import os
import requests
from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from Bio.Align import PairwiseAligner
from io import StringIO

from pipeline.utils import log

STANDARD_AAS = set("ACDEFGHIKLMNPQRSTVWY")


def fetch_pdb_seqres(pdb_id: str, chain: str) -> str:
    """Download PDB and extract SEQRES for the specified chain."""
    url = f"https://files.rcsb.org/download/{pdb_id.upper()}.pdb"
    log.info(f"Downloading PDB {pdb_id} from RCSB...")
    resp = requests.get(url, timeout=30)
    resp.raise_for_status()

    pdb_io = StringIO(resp.text)
    for record in SeqIO.parse(pdb_io, "pdb-seqres"):
        rec_chain = (record.id.split(":")[-1] if ":" in record.id
                     else record.annotations.get("chain", ""))
        if rec_chain == chain:
            seq = str(record.seq)
            log.info(f"SEQRES chain {chain}: {len(seq)} residues")
            return seq

    raise ValueError(f"Chain {chain} not found in PDB {pdb_id}")


def fetch_uniprot_mapping(pdb_id: str, chain: str) -> str | None:
    """Query RCSB API for the UniProt accession mapped to a PDB chain."""
    url = (f"https://data.rcsb.org/rest/v1/core/"
           f"polymer_entity_instance/{pdb_id.upper()}/{chain}")
    log.info(f"Querying RCSB for UniProt mapping of {pdb_id}:{chain}...")
    resp = requests.get(url, timeout=30)
    if not resp.ok:
        return None

    data = resp.json()
    ids_key = "rcsb_polymer_entity_instance_container_identifiers"
    entity_id = data.get(ids_key, {}).get("entity_id")
    if not entity_id:
        return None

    entity_url = (f"https://data.rcsb.org/rest/v1/core/"
                  f"uniprot/{pdb_id.upper()}/{entity_id}")
    entity_resp = requests.get(entity_url, timeout=30)
    if not entity_resp.ok:
        return None

    up_data = entity_resp.json()
    if isinstance(up_data, list) and len(up_data) > 0:
        accession = (up_data[0]
                     .get("rcsb_uniprot_container_identifiers", {})
                     .get("uniprot_id"))
        if accession:
            log.info(f"UniProt accession: {accession}")
            return accession

    return None


def fetch_uniprot_seq(accession: str) -> str:
    """Fetch canonical sequence from UniProt."""
    url = f"https://rest.uniprot.org/uniprotkb/{accession}.fasta"
    log.info(f"Fetching UniProt canonical sequence for {accession}...")
    resp = requests.get(url, timeout=30)
    resp.raise_for_status()

    record = next(SeqIO.parse(StringIO(resp.text), "fasta"))
    seq = str(record.seq)
    log.info(f"UniProt canonical sequence: {len(seq)} residues")
    return seq


def align_and_trim(seqres: str, uniprot_seq: str) -> str:
    """Align SEQRES against UniProt canonical and extract matching region.

    Removes expression tags (His-tags, etc.) present in PDB
    but absent from the canonical UniProt sequence.
    """
    aligner = PairwiseAligner()
    aligner.mode = "local"
    aligner.match_score = 2
    aligner.mismatch_score = -1
    aligner.open_gap_score = -5
    aligner.extend_gap_score = -0.5

    best = aligner.align(seqres, uniprot_seq)[0]

    seqres_start = best.aligned[0][0][0]
    seqres_end = best.aligned[0][-1][1]
    up_start = best.aligned[1][0][0]
    up_end = best.aligned[1][-1][1]

    trimmed = uniprot_seq[up_start:up_end]
    n_trim = seqres_start
    c_trim = len(seqres) - seqres_end

    if n_trim > 0 or c_trim > 0:
        log.info(f"Trimmed {n_trim} from N-terminus, "
                 f"{c_trim} from C-terminus (expression artifacts)")
    else:
        log.info("No trimming needed — SEQRES matches UniProt canonical")

    log.info(f"Clean sequence: {len(trimmed)} residues")
    return trimmed


def validate_sequence(seq: str) -> None:
    """Validate sequence length and alphabet."""
    non_std = set(seq) - STANDARD_AAS
    if non_std:
        raise ValueError(
            f"Sequence contains non-standard amino acids: {non_std}. "
            f"BioEmu and FlowPacker only support the 20 standard AAs.")

    if len(seq) < 40:
        raise ValueError(
            f"Sequence too short: {len(seq)} residues. "
            f"FlowPacker requires at least 40 residues.")

    if len(seq) > 512:
        raise ValueError(
            f"Sequence too long: {len(seq)} residues. "
            f"FlowPacker supports at most 512 residues.")

    log.info(f"Validation passed: {len(seq)} residues, all standard AAs")


def save_fasta(seq: str, name: str, out_path: str) -> str:
    """Save sequence as FASTA file."""
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    record = SeqRecord(Seq(seq), id=name,
                       description=f"cleaned sequence for {name}")
    with open(out_path, "w") as f:
        SeqIO.write(record, f, "fasta")
    log.info(f"Saved FASTA: {out_path}")
    return out_path


def run_stage0(config: dict) -> str:
    """Execute Stage 0: input preparation.

    Returns the path to the clean FASTA file.
    """
    protein = config["protein"]
    name = protein["name"]
    out_dir = config["output_dir"]
    fasta_path = os.path.join(out_dir, name, "input", "sequence.fasta")

    if os.path.exists(fasta_path):
        log.info(f"Stage 0: output already exists at {fasta_path}, skipping")
        return fasta_path

    source = protein["source"]

    if source == "fasta":
        local = protein["fasta_path"]
        log.info(f"Reading local FASTA: {local}")
        record = next(SeqIO.parse(local, "fasta"))
        seq = str(record.seq)
        validate_sequence(seq)
        return save_fasta(seq, name, fasta_path)

    elif source == "pdb":
        pdb_id = protein["pdb_id"]
        chain = protein["chain"]
        seqres = fetch_pdb_seqres(pdb_id, chain)

        accession = fetch_uniprot_mapping(pdb_id, chain)
        if accession:
            up_seq = fetch_uniprot_seq(accession)
            seq = align_and_trim(seqres, up_seq)
        else:
            log.warning("No UniProt mapping found — using SEQRES directly")
            seq = seqres

        validate_sequence(seq)
        return save_fasta(seq, name, fasta_path)

    else:
        raise ValueError(
            f"Unknown source: {source}. Must be 'pdb' or 'fasta'.")
