# ARPES Spectroscopic Simulation and Visualization Suite

This project provides a Python command-line interface to process, interpolate, and visualize Electronic Band Structures from Density Functional Theory (DFT) outputs. It simulates Angle-Resolved Photoemission Spectroscopy (ARPES) profiles. It parses VASP `vaspout.h5` and `vasprun.xml` files.

---

## Capabilities

* **Single-Plane Analysis:** Project 3D bulk states onto reciprocal planes. Simulate constant-energy contours and dispersion cuts.
* **Multi-Plane Automation:** Process Miller-index reciprocal planes in sequence. Generate organized output directories.
* **Surface Brillouin Zone Correlation:** Map 3D reciprocal volumes to 2D slab representations. Analyze band-folding and surface states.
* **Synthetic Prototyping:** Generate tight-binding datasets to test execution without VASP files.

---

## Requirements

Install the dependencies using standard Python package managers.

* `numpy`
* `scipy`
* `matplotlib`
* `pymatgen`
* `h5py`
* `scikit-learn`
* `sumo` (Optional requirement for publication-format plots)

---

## Usage

Execute the main script via the command line. 

```bash
python arpes.py [OPTIONS]
```

### Execution Modes

Set the execution mode using the `--mode` flag.

* `single`: Project states onto a specific vector plane.
* `multi`: Process a predefined set of Miller-index planes sequentially.
* `surface_bz`: Correlate bulk and surface Brillouin zones.
* `surface_bands`: Project band structures along surface symmetry paths.

### Common CLI Arguments

| Argument | Description | Default |
| :--- | :--- | :--- |
| `--mode` | Set execution mode (`single`, `multi`, `surface_bz`, `surface_bands`). | `single` |
| `--input` | Path to VASP output file (`vaspout.h5` or `vasprun.xml`). | None |
| `--mock` | Force generation of synthetic tight-binding dataset. | `False` |
| `--outdir` | Directory path for generated plots. | `arpes_outputs` |
| `--normal` | Fractional normal vector of the projection plane (3 values). | `0.0 0.0 1.0` |
| `--energy` | Energy relative to Fermi Level in eV. | `0.0` |
| `--efermi_shift` | Rigid shift (eV) of the Fermi level (`E_F -> E_F + shift`); positive emulates electron doping and moves the simulated Fermi surface. | `0.0` |
| `--matrix_elements` | Weight each state's intensity by its orbital/site projection (requires `LORBIT=11/12`). | `False` |
| `--orbital_weights` | Orbital or shell weights, e.g. `s:1,p:0.5` or `dz2:2`; unlisted orbitals get 0 (or `default:x`). Implies `--matrix_elements`. | None |
| `--ion_weights` | Per-ion weights in POSCAR order, e.g. `1,1,0,0,0.5,0.5`. Implies `--matrix_elements`. | None |
| `--elimits` | Binding energy limits for dispersion slices (2 values). | `-3.0 1.0` |
| `--miller_surf` | Miller index for surface slab generation (3 values). | `0 0 1` |

### Examples

**Run a single plane projection using a synthetic dataset:**
```bash
python arpes.py --mode single --mock --normal 0.0 0.0 1.0 --energy -0.5
```

**Shift the Fermi level by +0.3 eV (rigid-band / doping emulation):**
```bash
python arpes.py --mode single --mock --normal 0.0 0.0 1.0 --efermi_shift 0.3
```

**Weight intensities by d-orbital character (matrix-element proxy):**
```bash
python arpes.py --mode single --input vasprun.xml --orbital_weights "d:1"
```

Large `vasprun.xml` files (>100 MB) are parsed with a constant-memory streaming
parser, and parse results are cached to `<input>.arpes_cache.npz` for instant
re-runs. Orbital projections are reduced to per-state weights during the parse,
so the full projection tensor is never held in memory.

**Generate multi-plane projections from a VASP HDF5 file:**
```bash
python arpes.py --mode multi --input vaspout.h5 --outdir results_dir
```

**Analyze Surface Brillouin Zone for the (111) surface:**
```bash
python arpes.py --mode surface_bz --input vasprun.xml --miller_surf 1 1 1
```

## Disclaimer
Google Gemini was used to assist in the development of this project.

