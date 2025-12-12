"""
Command-line interface module for the ARPES projector suite.
Handles argument parsing to expose geometry, plotting, and file settings.
"""

import argparse

def build_parser() -> argparse.ArgumentParser:
    """
    Constructs and returns the argument parser for ARPES projections.
    """
    parser = argparse.ArgumentParser(
            description="General ARPES Simulation and Projection CLI tool.",
            formatter_class=argparse.ArgumentDefaultsHelpFormatter
            )

    # Mode selection
    parser.add_argument(
            "--mode", type=str, choices=["single", "multi", "surface_bz", "surface_bands"], default="single",
            help="Execution mode: 'single' plane, 'multi' automated, 'surface_bz' analysis, or 'surface_bands' path projections."
            )

    # I/O arguments
    io_group = parser.add_argument_group("Input/Output Options")
    io_group.add_argument("--input", type=str, default=None, help="Path to specific VASP output file (e.g., vaspout.h5 or vasprun.xml).")
    io_group.add_argument("--mock", action="store_true", help="Force generation of synthetic simple-cubic tight-binding dataset.")
    io_group.add_argument("--outdir", type=str, default="arpes_outputs", help="Directory where generated plots will be saved.")

    # Reciprocal Geometry arguments
    geom_group = parser.add_argument_group("Reciprocal Geometry Options")
    geom_group.add_argument("--normal", nargs=3, type=float, default=[0.0, 0.0, 1.0], help="Fractional normal vector of the projection plane.")
    geom_group.add_argument("--origin", nargs=3, type=float, default=[0.0, 0.0, 0.0], help="Fractional point the projection plane passes through.")
    geom_group.add_argument("--ubounds", nargs=2, type=float, default=[-2.0, 2.0], help="Coordinate limits of the projection plane u-axis (A^-1).")
    geom_group.add_argument("--vbounds", nargs=2, type=float, default=[-2.0, 2.0], help="Coordinate limits of the projection plane v-axis (A^-1).")
    geom_group.add_argument("--resolution", type=int, default=250, help="Grid resolution for the 2D projection plane.")
    geom_group.add_argument("--smooth", type=int, default=2, help="Sumo-style interpolation smoothing multiplier.")

    # Plotting & Physics arguments
    plot_group = parser.add_argument_group("Plotting & Physical Parameters")
    plot_group.add_argument("--energy", type=float, default=0.0, help="Target energy relative to Fermi Level (eV) for constant energy cuts.")
    plot_group.add_argument("--broadening", type=float, default=0.05, help="Lorentzian broadening (eV) to simulate lifetime effects.")
    plot_group.add_argument("--cmap", type=str, default="magma", help="Matplotlib colormap to use for simulated intensity.")

    # Dispersion specific
    disp_group = parser.add_argument_group("Dispersion Slice Options")
    disp_group.add_argument("--elimits", nargs=2, type=float, default=[-3.0, 1.0], help="Binding energy limits (E - Ef) in eV for dispersion slices.")
    disp_group.add_argument("--slice_coord", type=float, default=0.0, help="Constant coordinate value (A^-1) at which to take the dispersion slice.")
    disp_group.add_argument("--along_v", action="store_true", help="Vary along v-axis instead of u-axis for dispersion slice.")
    disp_group.add_argument("--n_energy", type=int, default=600, help="Energy grid resolution for dispersion map.")

    # Surface BZ specific
    bz_group = parser.add_argument_group("Surface BZ Options")
    bz_group.add_argument("--miller_surf", nargs=3, type=int, default=[0, 0, 1], help="Miller index for surface slab generation (Surface BZ mode).")
    bz_group.add_argument("--slab_min", type=float, default=15.0, help="Minimum slab thickness (A) for Surface BZ generation.")
    bz_group.add_argument("--vac_min", type=float, default=20.0, help="Minimum vacuum thickness (A) for Surface BZ generation.")

    return parser

