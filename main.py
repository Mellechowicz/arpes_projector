"""
This file contains the execution loop of the program.
It coordinates file detection, invokes band interpolation,
generates energy and dispersion slices, instantiates the surface analyzer,
executes zone correlation, and triggers plot displays.

Inputs:
 - VASP calculation files (vaspout.h5 or vasprun.xml) in the execution directory, or synthetic tight-binding data parameters.

Outputs:
 - Interpolated energy plane grids.
 - Image files of Fermi surfaces and dispersion slices.
 - Console print of point correlation maps.
 - Display of visual plot windows.

Approach and Modules:
 - Surface Brillouin zone analysis: surface_bz.SurfaceBZAnalyzer.
 - Data interpolation and spectral plotting: arpes_projector modules.
 - Numerical arrays: numpy.
 - Display: matplotlib.pyplot.
"""

import os
import numpy as np
from arpes_projector.parser import VaspDataParser
from arpes_projector.geometry import KSpaceProjector
from arpes_projector.plotter import ARPESPlotter
from arpes_projector.surface_bz import SurfaceBZAnalyzer

def generate_mock_electronic_structure() -> dict:
    """
    Generates synthetic, physically intuitive eigenvalues for a cubic lattice
    to ensure the demo runs out-of-the-box without requiring large VASP output files.
    """
    # Reciprocal lattice vectors for simple cubic lattice (a = 3.14 Angstrom)
    a_param = 3.14
    rec_lattice = np.eye(3) * (2.0 * np.pi / a_param)

    # Establish uniform 3D k-mesh
    kx_lin = np.linspace(-0.5, 0.5, 14)
    ky_lin = np.linspace(-0.5, 0.5, 14)
    kz_lin = np.linspace(-0.5, 0.5, 14)
    kx, ky, kz = np.meshgrid(kx_lin, ky_lin, kz_lin, indexing='ij')
    kpoints = np.stack([kx.flatten(), ky.flatten(), kz.flatten()], axis=1)

    # Generate tight-binding dispersion surfaces representing electronic states
    nkpts = len(kpoints)
    eigenvalues = np.zeros((1, 3, nkpts))  # (nspin=1, nbands=3, nkpts)

    # Band 1: Deep valence state
    eigenvalues = 3.0 - 1.8 * (
            np.cos(2 * np.pi * kpoints[:, 0]) +
            np.cos(2 * np.pi * kpoints[:, 1]) +
            np.cos(2 * np.pi * kpoints[:, 2])
            )
    # Band 2: Intermediate state
    eigenvalues = 5.2 - 1.1 * (
            2.0 * np.cos(2 * np.pi * kpoints[:, 0]) +
            np.cos(2 * np.pi * kpoints[:, 1]) +
            np.cos(2 * np.pi * kpoints[:, 2])
            )
    # Band 3: State crossing the Fermi level
    eigenvalues = 6.0 - 0.9 * (
            np.cos(2 * np.pi * kpoints[:, 0]) +
            2.0 * np.cos(2 * np.pi * kpoints[:, 1]) +
            np.cos(2 * np.pi * kpoints[:, 2])
            )

    return {
            "kpoints": kpoints,
            "eigenvalues": (1,3,eigenvalues),
            "efermi": 5.2,  # Set Fermi level relative to mock bands
            "rec_lattice": rec_lattice,
            "is_spin_polarized": False
            }

def main():
    print("=" * 80)
    print("  SIMULATED SPECTROSCOPIC PROJECTOR FOR DFT-ARPES CORRELATIONS")
    print("=" * 80)

    output_dir = "arpes_plots"
    os.makedirs(output_dir, exist_ok=True)

    # Attempt to locate real local VASP output files, fallback to mock database if none resolved
    xml_path = "vasprun.xml"
    h5_path = "vaspout.h5"

    if os.path.exists(h5_path):
        print(f"Resolving binary HDF5 calculation database: {h5_path}")
        parser = VaspDataParser(h5_path)
        data = parser.parse()
    elif os.path.exists(xml_path):
        print(f"Resolving structural markup calculation database: {xml_path}")
        parser = VaspDataParser(xml_path)
        data = parser.parse()
    else:
        print("VASP calculation files not found in execution directory.")
        print("Initializing synthetic simple-cubic tight-binding dataset...")
        data = generate_mock_electronic_structure()

    # Extract structured reciprocal arrays
    kpoints = data["kpoints"]
    eigenvalues = data["eigenvalues"]
    rec_lattice = data["rec_lattice"]
    efermi = data["efermi"]

    # Instantiating reciprocal geometry handlers
    projector = KSpaceProjector(kpoints, eigenvalues, rec_lattice)

    # User input defined parameters: Select projection plane in reciprocal space
    # Target plane defined at Gamma (0,0,0) with a normal along  (equivalent to kz = 0 slice)
    normal_frac = np.array([0.0, 0.0, 1.0])
    point_frac = np.array([0.0, 0.0, 0.0])

    # Establish plane limits in reciprocal Cartesian Angstrom units
    u_bounds = (-2.0, 2.0)
    v_bounds = (-2.0, 2.0)

    print("\n[Geometry] Interpolating 3D bands onto targeted reciprocal plane slice...")
    u_grid, v_grid, interp_spectra = projector.interpolate_plane(
            normal_frac=normal_frac,
            point_frac=point_frac,
            u_range=u_bounds,
            v_range=v_bounds,
            grid_resolution=200,
            interpolate_factor=2  # Sumo-style interpolation smoothing multiplier
            )

    # Instantiating photoemission plotters
    plotter = ARPESPlotter(u_grid, v_grid, interp_spectra, efermi)

    # 1. Generate Constant Energy Slices (Fermi Surface Map)
    print("[Plotter] Simulating constant energy slice at E - Ef = 0.0 eV...")
    fermi_surface_file = os.path.join(output_dir, "fermi_surface_kz0.png")
    plotter.plot_constant_energy_cut(
            energy=0.0,
            broadening=0.06,
            cmap="magma",
            filename=fermi_surface_file
            )
    print(f"Saved Fermi surface slice: {fermi_surface_file}")

    # 2. Generate Band Dispersion Slices (E vs k_parallel)
    print("[Plotter] Simulating band dispersion cut at constant kv = 0.0 A^-1...")
    dispersion_file = os.path.join(output_dir, "dispersion_slice_kv0.png")
    plotter.plot_dispersion_slice(
            slice_coordinate=0.0,
            along_v=False,
            energy_limits=(-2.5, 1.5),
            broadening=0.06,
            cmap="magma",
            filename=dispersion_file
            )
    print(f"Saved dispersion slice: {dispersion_file}")

    # 3. Generate Momentum Distribution Curves (MDCs) at fixed energy
    # 1. Initialize the analyzer with the VASP output file
    # It dynamically supports parsing 'vaspout.h5' and 'vasprun.xml' structures
    analyzer = SurfaceBZAnalyzer("vaspout.h5")

    # 2. Generate the slab (e.g., (0,0,1) surface, 15 Angstrom slab, 20 Angstrom vacuum)
    analyzer.generate_slab(miller_index=(0, 0, 1), min_slab=15.0, min_vac=20.0)

    # 3. Calculate 3D to 2D band folding correlation
    correlation_data = analyzer.correlate_zones()
    print("Correlation Mapping Computed:", correlation_data["labels"])

    # 4. Generate visual outputs
    analyzer.visualize()


    print("\nPost-processing execution complete. Plots generated successfully.")
    print("=" * 80)

if __name__ == "__main__":
    main()

