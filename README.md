# BioEmu Sidechain Benchmark

An alternative sidechain reconstruction and MD refinement pipeline for [BioEmu](https://github.com/microsoft/bioemu) backbone ensembles. BioEmu natively supports sidechain packing via [HPacker](https://github.com/gvisani/hpacker) and relaxation via [OpenMM](https://openmm.org/). This project replaces those steps with [FlowPacker](https://gitlab.com/mjslee0921/flowpacker) and [GROMACS](https://www.gromacs.org/), and provides a pipeline and example workflow for comparing sidechain packing and MD refinement approaches.

## Motivation

As demonstrated by Lewis et al. [(bioRxiv, 2024)](https://doi.org/10.1101/2024.12.05.626885), BioEmu samples functionally relevant conformational changes including large-scale domain motions, local unfolding, and formation of cryptic binding pockets not visible in static crystal structures. The authors highlight potential applications in the identification of binding pockets and allosteric mechanisms in drug discovery, and generation of ensembles for dynamical protein design.

## How It Works

The pipeline chains together several computational tools in a fully automated workflow:

```
Input (PDB ID or FASTA)
        |
  [Stage 0]  Fetch, clean, validate sequence (UniProt alignment)
        |
  [Stage 1]  Sample backbone conformations (BioEmu)
        |
  [Stage 2]  Extract individual frames from trajectory
        |
  [Stage 3]  Mutate all residues to glycine (backbone-only)
        |
  [Stage 4]  Energy minimization in vacuum (GROMACS)
        |
  [Stage 5]  Restore original amino acid sequence
        |
  [Stage 6]  Sidechain repacking (FlowPacker)
        |
  [Stage 7]  NVT MD refinement in explicit solvent (GROMACS)
        |
  [Stage 8]  Structure quality scoring (reduce + probe + Biopython)
```

**Why this order:**

- Vacuum EM (Stage 4) fixes backbone geometry before sidechain placement. FlowPacker performs better on geometrically sound backbones.
- FlowPacker (Stage 6) adds sidechains before MD, since NVT in explicit solvent requires full-atom structures.
- NVT (Stage 7) relaxes the complete system in an aqueous environment, resolving steric clashes introduced by sidechain packing.

## Benchmark: FlowPacker + GROMACS vs HPacker + GROMACS

The central question of this project: does the choice of sidechain packer matter after MD refinement? To answer this, both [FlowPacker](https://gitlab.com/mjslee0921/flowpacker) and [HPacker](https://github.com/gvisani/hpacker) were run on ABL1 kinase (PDB 2HYY, 273 residues). BioEmu generated 10 backbone samples, of which 8 passed quality filtering. Both packers were applied to the same 8 structures, followed by the same GROMACS NVT refinement (CHARMM36, TIP3P, dodecahedron box, 50 ps, 300 K). This isolates the sidechain packer as the only variable.

| Stage | N | Clashscore | Rama Favored | Rama Outlier |
|---|---|---|---|---|
| FlowPacker (pre-MD) | 8 | 44.5 | 79.3% | 5.5% |
| HPacker (pre-MD) | 8 | 44.5 | 79.3% | 5.5% |
| FlowPacker + NVT | 7 | 0.8 | 80.0% | 6.7% |
| HPacker + NVT | 6 | 0.6 | 80.6% | 6.0% |

**Key findings:**

- Both packers produce nearly identical pre-MD quality (clashscore 44.5, 79.3% Ramachandran favored).
- After GROMACS NVT refinement, clashscores drop to < 1.0 for both — the MD refinement step dominates the final structure quality.
- HPacker+NVT is marginally better post-MD (0.6 vs 0.8 clashscore, 6.0% vs 6.7% outliers), but the difference is within noise for this sample size.
- **GROMACS success rate differs**: FlowPacker structures had 7/8 (87.5%) NVT success rate, while HPacker had 6/8 (75%). HPacker produced one additional structure with steric clashes too severe for energy minimization to resolve.

> **Note:** This benchmark used a short NVT simulation (50 ps) as a proof of concept. Results may differ with longer simulation times, different force fields, or NPT equilibration. This project provides a starting point and framework for more thorough comparisons.

FlowPacker is the default in this pipeline. HPacker is available as an optional alternative for comparison (see `run_comparison.py`).

## Limitations

| Tool | Min residues | Max residues | Notes |
|---|---|---|---|
| BioEmu | — | — | Memory scales as L²; long sequences need more VRAM |
| FlowPacker | 40 | 512 | Hardcoded limit in the model |
| GROMACS | — | — | Compute-bound, no sequence limit |

**Effective pipeline limit: 40–512 residues**, bottlenecked by FlowPacker.

Additional constraints:

- Only the 20 standard amino acids are supported (no selenomethionine, modified residues, etc.)
- BioEmu requires internet access on first run to download model weights and generate MSA embeddings via ColabFold
- GPU with compute capability >= 7.5 (sm_75) is recommended; the pipeline falls back to CPU automatically when no compatible GPU is available, but inference is significantly slower

## Requirements

### System

- Linux
- [uv](https://docs.astral.sh/uv/) (Python package manager)
- GROMACS >= 2021 (`sudo apt install gromacs`)
- A force field installed in the GROMACS data directory (only if the desired force field is not bundled with GROMACS — common ones like `amber99sb-ildn` and `oplsaa` are included by default; `charmm36-jul2020` used in the example requires a separate download)
- `reduce` and `probe` from MolProbity (for structure quality scoring)
- `git-lfs` (for FlowPacker checkpoint files)

### External Repositories (option to conda)

The pipeline calls BioEmu and FlowPacker as Python packages/scripts, so their
source code must be available locally. MolProbity only needs the `reduce` and
`probe` binaries.

```bash
mkdir protein_ensemble && cd protein_ensemble

# BioEmu — installed as a local Python package by the bioemu-env
git clone https://github.com/microsoft/bioemu.git

# FlowPacker — the pipeline runs sampler_pdb.py from this repo
# git-lfs is required for the checkpoint file (~276 MB)
sudo apt install -y git-lfs
git clone https://gitlab.com/mjslee0921/flowpacker
cd flowpacker && git lfs pull && cd ..

# This pipeline
git clone <this_repo> ensemble_pipeline
```

> **Note:** [HPacker](https://github.com/gvisani/hpacker) (optional, for comparison) is installed automatically from GitHub when setting up `hpacker-env` — no manual clone required.

Expected directory layout:

```
protein_ensemble/
    bioemu/
    flowpacker/
    ensemble_pipeline/
```

#### MolProbity (reduce + probe)

The pipeline only needs the `reduce` and `probe` command-line tools, not the
full MolProbity suite. You can install them in whichever way is most convenient:

**Option A — conda**:

```bash
conda install -c conda-forge reduce probe
```

**Option B — build from source**:

```bash
git clone https://github.com/rlabduke/MolProbity
cd MolProbity
./install_via_bootstrap.sh 4
echo 'export PATH="$HOME/protein_ensemble/MolProbity/bin/linux:$PATH"' >> ~/.bashrc
source ~/.bashrc
```

Verify the installation:

```bash
reduce -version
probe -version
```

## Installation

### 1. GROMACS and force field

```bash
sudo apt install -y gromacs

# Find the GROMACS data directory:
gmx --version | grep "Data prefix"
# Typically /usr, so the top dir is /usr/share/gromacs/top/

# Install a force field (example: CHARMM36)
# Download from http://mackerell.umaryland.edu/charmm_ff.shtml
sudo cp -r charmm36-jul2020.ff /usr/share/gromacs/top/

# Verify:
ls /usr/share/gromacs/top/ | grep charmm
```

### 2. Python environments

The pipeline uses two separate uv environments because BioEmu and FlowPacker have conflicting numpy requirements (>= 1.23 vs == 1.22.4).

```bash
cd ../ensemble_pipeline/envs

# BioEmu environment
cd bioemu-env
uv sync
cd ..

# FlowPacker environment
cd flowpacker-env
uv sync
cd ../..
```

> If the environments are not yet initialized, create them manually:
>
> ```bash
> # bioemu-env
> cd bioemu-env
> uv add ../../bioemu requests pyyaml \
>     --default-index https://download.pytorch.org/whl/cu124 \
>     --index https://pypi.org/simple
> cd ..
>
> # flowpacker-env (install deps matching FlowPacker's requirements)
> cd flowpacker-env
> uv add torch "numpy==1.22.4" \
>     --default-index https://download.pytorch.org/whl/cu124 \
>     --index https://pypi.org/simple
> uv add tqdm tensorboard pyyaml easydict biotite dm-tree \
>     biopython modelcif torch_geometric torch_cluster pandas e3nn \
>     --no-build-isolation \
>     --default-index https://download.pytorch.org/whl/cu124 \
>     --index https://pypi.org/simple
> cd ../..
> ```

## Usage

All commands are run from the `ensemble_pipeline/` directory.

### Full pipeline

```bash
uv run --project envs/bioemu-env python run_pipeline.py config.yaml
```

### Resume from a specific stage

```bash
# Skip stages 0–3, start from vacuum EM
uv run --project envs/bioemu-env python run_pipeline.py config.yaml --stage 4
```

### Run a single stage

```bash
# Run only FlowPacker sidechain repacking
uv run --project envs/bioemu-env python run_pipeline.py config.yaml --stage 6 --only
```

Each stage checks for existing output before running. If the output already exists, it skips automatically. This makes the pipeline fully **resumable** — if it fails at Stage 7, just re-run and it picks up where it left off.

## Configuration

All parameters are set in `config.yaml`. Here is a fully annotated example:

```yaml
protein:
  source: pdb               # "pdb" or "fasta"
  pdb_id: 2HYY              # PDB ID (used when source=pdb)
  chain: A                  # Chain ID (used when source=pdb)
  name: abl1                # Name for output directory
  # fasta_path: ./seq.fasta # Path to FASTA (used when source=fasta)

bioemu:
  num_samples: 10           # Number of backbone conformations to generate
  model_version: v1.2       # BioEmu model (v1.0, v1.1, v1.2)
  filter_samples: true      # Filter low-quality samples

gromacs:
  forcefield: charmm36-jul2020  # Must match a .ff directory in GROMACS top/
  water_model: tip3p            # Must match the force field (see table below)
  em_steps: 5000                # Max steps for vacuum energy minimization
  threads: 8                    # OpenMP threads for mdrun
  nvt:
    temperature: 300            # Kelvin
    dt: 0.002                   # Timestep in ps (2 fs)
    nsteps: 25000               # Total MD steps (50 ps at default dt)
    restraint_fc: 5             # CA position restraint force constant
    tcoupl: V-rescale           # Thermostat algorithm
    tau_t: 0.1                  # Temperature coupling time constant (ps)

flowpacker:
  repo_path: ../flowpacker              # Path to FlowPacker repo
  checkpoint: checkpoints/cluster.pth   # Checkpoint relative to repo_path
  num_steps: 10                         # ODE integration steps
  n_samples: 1                          # Samples per structure

molprobity:
  enabled: true             # Set to false to skip quality scoring

output_dir: ./output
```

### Water model compatibility

| Force field | Water model |
|---|---|
| charmm36-jul2020 | `tip3p` |
| amber99sb-ildn | `tip3p` |
| oplsaa | `tip4p` or `spc` |

Check which models are available in your force field directory:

```bash
ls /usr/share/gromacs/top/<your_ff>.ff/*.itp | grep -i tip
```

## Output Structure

```
output/<protein_name>/
    input/
        sequence.fasta          # Cleaned sequence
    bioemu/
        samples.xtc             # BioEmu trajectory
        topology.pdb            # Topology for trajectory
        backbone_*.pdb          # Extracted frames
    gly/
        backbone_*_gly.pdb      # Glycine-mutated backbones
        minimized_*_gly.pdb     # After vacuum EM
    final_backbones/
        final_*.pdb             # Backbones with restored sequence
    flowpacker/
        final_*.pdb             # Full-atom structures (with sidechains)
    nvt/
        receptor_*.pdb          # MD-refined final structures
    molprobity/
        scores.tsv              # Quality scores (clashscore + Ramachandran)
```

## Example: ABL1 Kinase Domain

The included `config.yaml` runs the pipeline on the ABL1 kinase domain (PDB [2HYY](https://www.rcsb.org/structure/2HYY), chain A, 273 residues).

```bash
uv run --project envs/bioemu-env python run_pipeline.py config.yaml
```

Expected results with default settings (10 samples):

| Metric | FlowPacker (pre-NVT) | NVT-refined (post-NVT) |
|---|---|---|
| Clashscore | ~44 | ~0.8 |
| Ramachandran favored | ~79% | ~80% |

The NVT refinement reduces clashes by roughly 50x while maintaining backbone geometry.

## Error Handling

- **Per-structure failures**: If GROMACS EM or NVT fails for one structure, it is logged and skipped. The pipeline continues with the remaining structures and reports a summary at the end.
- **Stage-level failures**: If BioEmu or FlowPacker fails entirely, the pipeline stops with an error message.
- **Validation failures**: If the input sequence is outside 40–512 residues or contains non-standard amino acids, the pipeline stops at Stage 0.

## Sequence Cleaning

When using `source: pdb`, the pipeline automatically removes expression artifacts (His-tags, purification tags) by:

1. Extracting SEQRES from the PDB file
2. Querying RCSB for the corresponding UniProt accession
3. Fetching the canonical UniProt sequence
4. Aligning SEQRES against UniProt and extracting only the matching region

When using `source: fasta`, the sequence is used as-is (only validation is performed).

## Project Structure

```
ensemble_pipeline/
    run_pipeline.py         # Entry point / orchestrator
    config.yaml             # User configuration
    README.md
    PLAN.md                 # Detailed design document
    pipeline/
        input.py            # Stage 0: input preparation
        bioemu.py           # Stages 1-2: sampling + extraction
        prepare.py          # Stages 3, 5: GLY mutation + sequence restore
        gromacs.py          # Stages 4, 7: vacuum EM + NVT refinement
        flowpacker.py       # Stage 6: sidechain repacking
        molprobity.py       # Stage 8: quality scoring
        utils.py            # Logger
    envs/
        bioemu-env/         # uv environment for BioEmu
        flowpacker-env/     # uv environment for FlowPacker
    mdp/
        em_vacuum.mdp       # GROMACS MDP for vacuum EM (Stage 4)
        em_solvated.mdp     # GROMACS MDP for solvated EM (Stage 7)
```

## License

This pipeline integrates several external tools, each with its own license:

- [BioEmu](https://github.com/microsoft/bioemu) — MIT License
- [FlowPacker](https://gitlab.com/mjslee0921/flowpacker) — MIT License
- [MolProbity](https://github.com/rlabduke/MolProbity) — BSD-style License
- [GROMACS](https://www.gromacs.org/) — LGPL 2.1
