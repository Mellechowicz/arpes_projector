"""
This file contains an automated execution script.
It parses VASP output and projects three-dimensional band structures onto multiple predefined Miller index planes.

Inputs:
 - vaspout.h5 file located in the execution directory.
 - Internal list of fractional reciprocal directions.

Outputs:
 - Directory folder containing saved Fermi surface and dispersion PNG image files for each defined plane.

Approach and Modules:
 - File operations: Path validation and directory creation via os.
 - Array definitions: Vector assignments via numpy.
 - Data pipeline: Extraction, interpolation, and plotting via arpes_projector modules.
"""

import os
import numpy as np
from arpes_projector.parser import VaspDataParser
from arpes_projector.geometry import KSpaceProjector
from arpes_projector.plotter import ARPESPlotter

def main():
    # 1. Look for the VASP output file
    h5_file = "vaspout.h5"

    if not os.path.exists(h5_file):
        print(f"Error: Could not find '{h5_file}'.")
        print("Remember: You must run a VASP calculation using your POSCAR first!")
        return

    print(f"Parsing VASP HDF5 output: {h5_file}...")
    parser = VaspDataParser(h5_file)
    data = parser.parse()

    projector = KSpaceProjector(data["kpoints"], data["eigenvalues"], data["rec_lattice"])
    efermi = data["efermi"]

    # 2. Define the requested Miller indices (fractional reciprocal directions)
    miller_indices = [
            (1.0, 0.0, 0.0),  # 100
            (0.0, 1.0, 0.0),  # 010
            (0.0, 0.0, 1.0),  # 001
            (1.0, 1.0, 0.0),  # 110
            (1.0, 0.0, 1.0),  # 101
            (0.0, 1.0, 1.0),  # 011
            (1.0, 1.0, 1.0),  # 111
            (2.0, 1.0, 0.0),  # 210
            (1.0, 2.0, 0.0),  # 120
            (1.0, 1.0, 2.0),  # 112
            (2.0, 0.0, 1.0),  # 201
            (1.0, 2.0, 1.0),  # 121
            (1.0, 0.0, 2.0),  # 102
            (2.0, 1.0, 1.0),  # 211
            (0.0, 2.0, 1.0),  # 021
            (0.0, 1.0, 2.0),  # 012
            ]

    # Bounds of the projected image in inverse Angstroms
    u_bounds = (-3.0, 3.0)
    v_bounds = (-3.0, 3.0)

    # Create an output directory to avoid cluttering your root folder
    output_dir = "arpes_projections"
    os.makedirs(output_dir, exist_ok=True)

    # 3. Loop through each plane and generate the ARPES spectra
    for h, k, l in miller_indices:
        plane_name = f"{int(h)}{int(k)}{int(l)}"
        print(f"\n" + "="*50)
        print(f"Projecting electronic structure onto the {plane_name} plane...")

        normal_frac = np.array([h, k, l])
        point_frac = np.array([0.0, 0.0, 0.0])  # Plane intersecting Gamma

        # Interpolate the 3D data onto the 2D plane
        u_grid, v_grid, interpolated_spectra = projector.interpolate_plane(
                normal_frac=normal_frac,
                point_frac=point_frac,
                u_range=u_bounds,
                v_range=v_bounds,
                grid_resolution=600,
                interpolate_factor=2 
                )

        plotter = ARPESPlotter(u_grid, v_grid, interpolated_spectra, efermi)

        # Plot 1: Constant Energy Cut (Fermi Surface at 0.0 eV)
        fs_filename = os.path.join(output_dir, f"fermi_surface_{plane_name}.png")
        print(f"  -> Saving Fermi Surface: {fs_filename}")
        plotter.plot_constant_energy_cut(
                energy=0.0, 
                broadening=0.05, 
                cmap="inferno", 
                filename=fs_filename
                )

        # Plot 2: Dispersion Slice (E vs k_parallel across the center)
        disp_filename = os.path.join(output_dir, f"dispersion_{plane_name}.png")
        print(f"  -> Saving Dispersion: {disp_filename}")
        plotter.plot_dispersion_slice(
                slice_coordinate=0.0, 
                along_v=False, 
                energy_limits=(-3.0, 1.0), 
                broadening=0.05,
                cmap="magma",
                filename=disp_filename
                )

    print("\n" + "="*50)
    print("Execution complete! All requested planes have been processed.")

if __name__ == "__main__":
    main()

