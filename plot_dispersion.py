"""
This file contains an execution script. It parses XML calculation databases,
projects band structures onto a specified plane,
and generates a simulated energy versus momentum dispersion cut.

Inputs:
 - vasprun.xml file located in the execution directory.
 - Internal definitions for plane normal, plane origin, and coordinate boundaries.

Outputs:
 - A single saved PNG image file of the simulated band dispersion slice.

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
    # 1. Define path to your XML database
    xml_file = "vasprun.xml"

    if not os.path.exists(xml_file):
        print(f"Error: Could not find '{xml_file}' in the current directory.")
        print("Please place a valid vasprun.xml file here to run the example.")
        return

    # 2. Parse the VASP run using the dynamic parser
    print(f"Parsing VASP xml database: {xml_file}...")
    parser = VaspDataParser(xml_file)
    data = parser.parse()

    kpoints = data["kpoints"]
    eigenvalues = data["eigenvalues"]
    rec_lattice = data["rec_lattice"]
    efermi = data["efermi"]

    # 3. Instantiate reciprocal geometry handers
    projector = KSpaceProjector(kpoints, eigenvalues, rec_lattice)

    # Define a custom plane slanted in reciprocal space
    # e.g., Plane normal along  passing through Gamma
    normal_frac = np.array([0.0, 1.0, 0.0])
    point_frac = np.array([0.0, 0.0, 0.0])

    # Limits of the plane coordinate axes in inverse Angstroms
    u_bounds = (-0.8, 0.8)
    v_bounds = (-0.8, 0.8)

    print("Projecting 3D eigenvalues onto the  plane...")
    u_grid, v_grid, interpolated_spectra = projector.interpolate_plane(
            normal_frac=normal_frac,
            point_frac=point_frac,
            u_range=u_bounds,
            v_range=v_bounds,
            grid_resolution=250,
            interpolate_factor=1
            )

    # 4. Instantiate the ARPES visualizer
    plotter = ARPESPlotter(u_grid, v_grid, interpolated_spectra, efermi)

    # 5. Extract a dispersion slice along the k_u axis (at fixed k_v = 0.0 A^-1)
    # We restrict the binding energy axis to a window from -3.0 eV (occupied) to 1.0 eV (unoccupied).
    output_image = "simulated_dispersion_cut.png"
    print(f"Simulating dispersion slice and saving to {output_image}...")

    plotter.plot_dispersion_slice(
            slice_coordinate=0.0,       # Slice coordinate (k_v = 0.0 A^-1)
            along_v=False,              # Vary k_u (along_v=False means vary along the u axis)
            energy_limits=(-1.0, 0.5),  # Binding energy limits (E - Ef) in eV
            n_energy_points=600,        # Energy grid resolution
            broadening=0.03,            # Simulated experimental broadening in eV
            cmap="magma",
            filename=output_image
            )
    print("Execution complete. Visualized band dispersion successfully.")

if __name__ == "__main__":
    main()

