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
| `--elimits` | Binding energy limits for dispersion slices (2 values). | `-3.0 1.0` |
| `--miller_surf` | Miller index for the surface (3 values). | `0 0 1` |
| `--slab_min` / `--vac_min` | Minimum slab and vacuum thickness (A) for the real-space slab. | `15.0` / `20.0` |
| `--ubounds` / `--vbounds` | Plot window along u / v (A^-1). | k-point cloud footprint |
| `--temperature` | Sample temperature (K). Applies the Fermi-Dirac cutoff `f(E-Ef,T)` to the intensity. | None (no cutoff) |
| `--matrix_elements` | Weight intensity by orbital/site projections instead of treating every band as equally bright. Needs `LORBIT=11` or `12`. | `False` |
| `--orbital_weights` | Orbital or shell weights, e.g. `"s:1,p:0.5,dz2:2"` or `"d:1"`. Unlisted orbitals get 0 (override with `default:x`). Implies `--matrix_elements`. | None |
| `--ion_weights` | Per-ion weights in POSCAR order, e.g. `"1,1,0,0,0.5,0.5"`. Implies `--matrix_elements`. | None |

### Matrix elements

By default every band contributes equally to the simulated intensity. A real
photoemission intensity is modulated by the matrix element, which varies by
orders of magnitude with orbital character, so the default is a density of
states in disguise rather than a spectrum. Passing orbital or site weights
reduces the VASP projections to one weight per state and scales each band's
Lorentzian by it:

```bash
python arpes.py --mode single --input vasprun.xml --orbital_weights "d:1"
```

Projections are contracted as they stream, so the full
`(nspin, nkpt, nband, nion, norb)` tensor is never held in memory: a 573 MB
`vasprun.xml` reduces in about 3.6 s using 0.21 GB, and a 2.44 GB `vaspout.h5`
projections dataset in about 1.0 s using 0.34 GB.

For a **noncollinear** run (`LNONCOLLINEAR = T`) the projection block carries
four sets per k-point which are *(total, m_x, m_y, m_z)* - not four spin
channels. Only the first is the charge projection; the others are magnetisation
components and may be negative. The number of sets read follows the eigenvalue
spin-channel count, so noncollinear runs use the charge projection alone and a
collinear `ISPIN=2` run uses both channels.

### Large `vasprun.xml` files

`vasprun.xml` above 100 MB is read by a streaming parser instead of pymatgen's
DOM parser, which builds the whole document in memory. The `<projected>` block -
usually well over 95% of the file - is elided from the byte stream before the
tokenizer sees it.

Measured on a 9.33 GiB noncollinear `vasprun.xml` (136 bands, 19683 k-points):
**14.7 s at 0.29 GB peak RSS** - about 3% of the file size.

That measurement is also what found the remaining leak. Eliding `<projected>`
from the byte stream was not enough: `<eigenvalues>` still holds one `<r>`
element per band per k-point, and an ElementTree node costs far more than the
two floats it carries. On this file that subtree was 2.7M nodes and 1.2 GB -
the entire peak. Draining it k-point by k-point instead cut peak RSS from
1.29 GB to 0.29 GB with byte-identical output.

On a 600 MB file the streaming parser reproduces pymatgen's k-points,
eigenvalues and Fermi energy bit-for-bit (the reciprocal lattice to the ~8
digits the XML prints) and is about 9x faster; there is a regression check for
that equivalence.

Results are cached in `<input>.arpes_cache.npz` next to the input, so repeated
runs skip the parse entirely. The cache is used only when it is strictly newer
than the input file - an equal timestamp counts as stale - so editing or
re-running VASP invalidates it automatically.

### Fermi-Dirac cutoff

Without `--temperature` the simulated intensity is the bare spectral function,
so states above the Fermi level are drawn just as brightly as occupied ones -
useful for inspecting the band structure, but not what a photoemission
experiment measures. Giving a temperature applies the occupation factor:

```bash
python arpes.py --mode single --input vasprun.xml --temperature 20
```

The factor multiplies the spectral function at the *probed* energy, following
`I(k,w) ~ f(w,T) A(k,w)`, not at each band's own energy. That distinction
matters: a band sitting just above `E_F` still has Lorentzian weight below it,
and a real measurement sees that weight. Weighting per band would wrongly
delete it.

`--temperature 0` gives the hard step (half-occupied exactly at `E_F`). The
occupation is evaluated with a logistic form, so `|E| >> kT` returns 0 or 1
rather than overflowing to `NaN`. Runs that pass a temperature write their own
filenames (`..._T300K.png`), so they do not overwrite runs that do not.

**The flag is opt-in and output without it is bit-identical to before it
existed** - there is a regression check for exactly that.

### Behaviour worth knowing

* **Output filenames encode the parameters that change the figure** - energy or
  energy limits, broadening, colour scale, temperature and the orbital/ion
  weighting - so re-running with different settings no longer overwrites the
  previous plot. `--mode surface_bands` names also carry the surface Miller
  index, which previously collided: an (001) and a (111) run wrote the same
  files. A weighting that could not be applied (a `--mock` run has no
  projections) is *not* named, so a filename never claims a weighting its
  figure does not carry.
* **`--mode surface_bz` writes a PNG** into `--outdir`; it previously called
  `plt.show()` unconditionally and produced nothing at all when run headless.
* **`--slab_min` / `--vac_min` size the real-space slab**, not the surface
  Brillouin zone. The projected zone of an (hkl) surface is fixed by the 2D
  surface lattice, which the Miller index alone determines.
* **Degenerate inputs are rejected** rather than yielding a convincing blank
  figure: `--broadening` must be positive (zero makes the Lorentzian numerator
  zero and blanks the spectrum, negative gives negative intensity),
  `--resolution` / `--smooth` / `--n_energy` must be at least 1, and a
  zero-length `--normal` does not define a plane.
* **The plot window defaults to the k-point cloud's own footprint** rather than
  a fixed +/-2 A^-1. That fixed window is unrelated to any given calculation:
  for a cloud reaching +/-0.33 A^-1 along v it left 94-95% of every figure
  outside the convex hull, drawn as zero intensity. Deriving the window from the
  cloud takes the local 600 MB example from 94-95% empty on all 12 planes to
  53-74% on 7 of them, and the `--mock` demo from 73% empty to none. The bounds
  are printed, never applied silently, and an explicit `--ubounds` / `--vbounds`
  always wins - each axis independently, so you can pin one and derive the
  other. Because the window projects the whole cloud, it is an upper bound on
  the slice's own extent and can never crop real data.
* **Coverage of the k-point cloud is reported.** Points outside its convex hull
  interpolate to NaN and are drawn as zero intensity, which is indistinguishable
  from a genuine absence of spectral weight, so a warning is printed when a
  large fraction of the window carries no data.

### Examples

**Run a single plane projection using a synthetic dataset:**
```bash
python arpes.py --mode single --mock --normal 0.0 0.0 1.0 --energy -0.5
```

**Generate multi-plane projections from a VASP HDF5 file:**
```bash
python arpes.py --mode multi --input vaspout.h5 --outdir results_dir
```

**Analyze Surface Brillouin Zone for the (111) surface:**
```bash
python arpes.py --mode surface_bz --input vasprun.xml --miller_surf 1 1 1 --outdir results_dir
```

**Weight intensities by d-orbital character:**
```bash
python arpes.py --mode single --input vasprun.xml --orbital_weights "d:1"
```

---

## Tests

```bash
python3 tests/test_matrix_elements.py
```

Covers the weight algebra, the orbital/shell spec, both projection readers
against values read from the raw files, and regressions for the interpolation,
surface-BZ, input-validation and normalisation fixes. Checks needing the local
VASP outputs skip cleanly when those are absent. Exit code is non-zero on
failure.

## Disclaimer
Google Gemini was used to assist in the development of this project.

