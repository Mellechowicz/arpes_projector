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


## Disclaimer
Google Gemini was used to assist in the development of this project.

