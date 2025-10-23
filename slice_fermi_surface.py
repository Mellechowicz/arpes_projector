"""
This file contains an execution script. It parses HDF5 calculation databases,
projects band structures onto a constant energy slice, and generates a simulated Fermi surface map.

Inputs:
 - vaspout.h5 file located in the execution directory.
 - Internal definitions for target plane vectors and coordinate projection limits.

Outputs:
 - A single saved PNG image file of the simulated Fermi surface constant energy contour.

Approach and Modules:
 - File operations: Path validation via os.
 - Array definitions: Vector assignments via numpy.
 - Data pipeline: Extraction, interpolation, and plotting via arpes_projector modules.
"""

import os
import numpy as np
from arpes_projector.parser import VaspDataParser
from arpes_projector.geometry import KSpaceProjector
from arpes_projector.plotter import ARPESPlotter

def main():
    # 1. Define the path to your VASP HDF5 database
    h5_file = "vaspout.h5"

    if not os.path.exists(h5_file):
        print(f"Error: Could not find '{h5_file}' in the current directory.")
        print("Please place a valid VASP HDF5 output file here to run the example.")
        return

    # 2. Parse the VASP output file using the dynamic parser
    print(f"Parsing electron eigenvalues and k-points from {h5_file}...")
    parser = VaspDataParser(h5_file)
    data = parser.parse()

    kpoints = data["kpoints"]
    eigenvalues = data["eigenvalues"]
    rec_lattice = data["rec_lattice"]
    efermi = data["efermi"]

    print(f"Parsed {eigenvalues.shape[1]} bands across {len(kpoints)} k-points.")
    print(f"Fermi Energy: {efermi:.4f} eV")

    # 3. Initialize the reciprocal space projector
    projector = KSpaceProjector(kpoints, eigenvalues, rec_lattice)

    # 4. Set up the target plane in fractional coordinates
    # We slice at kz = 0. We define this by a normal vector  
    # and an origin point  in fractional reciprocal lattice units.
    normal_frac = np.array([0.0, 0.0, 1.0])
    point_frac = np.array([0.0, 0.0, 0.0])

    # Define coordinate limits of the projection plane (in inverse Angstroms)
    # This represents a square region from -1.5 to 1.5 A^-1
    u_bounds = (-1.5, 1.5)
    v_bounds = (-1.5, 1.5)

    print("Interpolating electronic structure onto the kz = 0 plane...")
    u_grid, v_grid, interpolated_spectra = projector.interpolate_plane(
            normal_frac=normal_frac,
            point_frac=point_frac,
            u_range=u_bounds,
            v_range=v_bounds,
            grid_resolution=300,   # Higher resolution for publication-quality grids
            interpolate_factor=2   # Apply sumo-style smoothing multiplier
            )

    # 5. Initialize the ARPES intensity simulator
    plotter = ARPESPlotter(u_grid, v_grid, interpolated_spectra, efermi)

    # 6. Generate and save the simulated Fermi Surface (energy slice at E - Ef = 0.0 eV)
    # We use a standard Lorentzian broadening of 0.05 eV to simulate lifetime effects.
    output_image = "simulated_fermi_surface_kz0.png"
    print(f"Simulating photoemission intensity at E - Ef = 0.0 eV and saving to {output_image}...")

    plotter.plot_constant_energy_cut(
            energy=0.0,            # Slice at the Fermi Level
            broadening=0.05,       # Lorentzian HWHM in eV
            cmap="inferno",        # High-contrast color map typical of ARPES plots
            filename=output_image
            )
    print("Execution complete. Visualized Fermi surface successfully.")

if __name__ == "__main__":
    main()

