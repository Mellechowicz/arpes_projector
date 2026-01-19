"""
ARPES Spectroscopic Simulation and Visualization Suite.

This module provides a unified command-line interface for the processing,
interpolation, and visualization of Electronic Band Structures derived from
Density Functional Theory (DFT) outputs (e.g., VASP HDF5 or XML databases).

Capabilities include:
    1. Single-Plane Analysis: Projects 3D bulk states onto user-defined arbitrary
       reciprocal planes to simulate constant-energy contours (Fermi surfaces)
       and high-symmetry dispersion cuts (E vs. k).
    2. Multi-Plane Automation: Batch processes multiple Miller-index reciprocal
       planes sequentially, generating organized spectroscopic plot directories.
    3. Surface Brillouin Zone Correlation: Correlates 3D reciprocal volumes
       onto 2D slab representations, mapping band-folding and surface states.
    4. Synthetic Prototyping: Generates mock tight-binding cubic arrays to
       facilitate pipeline testing without heavy ab-initio dependencies.

Execution:
    Designed to be run via the command line with options defined in `arpes_projector.cli`.
    Use `python arpes.py --help` to view all configurable flags.
"""

import os
import sys
import numpy as np

# Adjust imports according to your package structure
from arpes_projector.parser import VaspDataParser
from arpes_projector.geometry import KSpaceProjector
from arpes_projector.plotter import ARPESPlotter
from arpes_projector.surface_bz import SurfaceBZAnalyzer
from arpes_projector.cli import build_parser

def generate_mock_electronic_structure() -> dict:
    """
    Generates synthetic, physically intuitive eigenvalues for a simple cubic
    lattice to ensure the demo runs out-of-the-box without requiring large
    VASP output files.
    """
    a_param = 3.14
    rec_lattice = np.eye(3) * (2.0 * np.pi / a_param)

    kx_lin = np.linspace(-0.5, 0.5, 14)
    ky_lin = np.linspace(-0.5, 0.5, 14)
    kz_lin = np.linspace(-0.5, 0.5, 14)
    kx, ky, kz = np.meshgrid(kx_lin, ky_lin, kz_lin, indexing='ij')
    kpoints = np.stack([kx.flatten(), ky.flatten(), kz.flatten()], axis=1)

    nkpts = len(kpoints)
    eigenvalues = np.zeros((1, 3, nkpts))  # (nspin=1, nbands=3, nkpts)

    # Band 1: Deep valence state
    eigenvalues[0, 0, :] = 3.0 - 1.8 * (
            np.cos(2 * np.pi * kpoints[:, 0]) +
            np.cos(2 * np.pi * kpoints[:, 1]) +
            np.cos(2 * np.pi * kpoints[:, 2])
            )
    # Band 2: Intermediate state
    eigenvalues[0, 1, :] = 5.2 - 1.1 * (
            2.0 * np.cos(2 * np.pi * kpoints[:, 0]) +
            np.cos(2 * np.pi * kpoints[:, 1]) +
            np.cos(2 * np.pi * kpoints[:, 2])
            )
    # Band 3: State crossing the Fermi level
    eigenvalues[0, 2, :] = 6.0 - 0.9 * (
            np.cos(2 * np.pi * kpoints[:, 0]) +
            2.0 * np.cos(2 * np.pi * kpoints[:, 1]) +
            np.cos(2 * np.pi * kpoints[:, 2])
            )

    return {
            "kpoints": kpoints,
            "eigenvalues": eigenvalues,
            "efermi": 5.2,
            "rec_lattice": rec_lattice,
            "is_spin_polarized": False
            }

def execute_projection(projector, efermi, args, normal_frac, plane_label):
    """Helper method to interpolate and plot projection slices."""
    print(f"\n[Geometry] Interpolating onto plane (Normal: {normal_frac})...")
    u_grid, v_grid, interp_spectra = projector.interpolate_plane(
            normal_frac=normal_frac,
            point_frac=np.array(args.origin),
            u_range=tuple(args.ubounds),
            v_range=tuple(args.vbounds),
            grid_resolution=args.resolution,
            interpolate_factor=args.smooth
            )

    plotter = ARPESPlotter(u_grid, v_grid, interp_spectra, efermi)

    # 1. Constant Energy Slice
    fs_file = os.path.join(args.outdir, f"fermi_surface_{plane_label}.png")
    fs_title = f"Constant Energy Contour ($E - E_F = {args.energy:.2f}$ eV)\nMiller/Label: {plane_label} | Vector: {np.round(normal_frac, 3)}"
    plotter.plot_constant_energy_cut(
            energy=args.energy, broadening=args.broadening, cmap=args.cmap,
            filename=fs_file, cscale=args.cscale, custom_title=fs_title
            )
    print(f" -> Saved Fermi surface slice: {fs_file}")

    # 2. Band Dispersion Slice
    disp_file = os.path.join(args.outdir, f"dispersion_{plane_label}.png")
    disp_title = f"Dispersion Slice\nMiller/Label: {plane_label} | Vector: {np.round(normal_frac, 3)}"
    plotter.plot_dispersion_slice(
            slice_coordinate=args.slice_coord, along_v=args.along_v,
            energy_limits=tuple(args.elimits), n_energy_points=args.n_energy,
            broadening=args.broadening, cmap=args.cmap,
            filename=disp_file, cscale=args.cscale, custom_title=disp_title
            )
    print(f" -> Saved dispersion slice: {disp_file}")

def make_bar_label(lbl: str) -> str:
    """Formats standard labels into LaTeX overbar notation for Surface BZ."""
    l = lbl.replace('$', '').replace('\\', '')
    if l.upper() == 'GAMMA': return r"$\bar{\Gamma}$"
    if '_' in l: return rf"$\bar{{{l.split('_')[0]}}}_{{{l.split('_')[1]}}}$"
    return rf"$\bar{{{l}}}$"

def main():
    parser = build_parser()
    args = parser.parse_args()

    print("=" * 80)
    print("  SIMULATED SPECTROSCOPIC PROJECTOR FOR DFT-ARPES CORRELATIONS")
    print("=" * 80)

    os.makedirs(args.outdir, exist_ok=True)

    # ---------------------------------------------------------
    # 1. Resolve Data Pipeline
    # ---------------------------------------------------------
    data = None
    input_resolved = None

    if args.mock:
        print("[I/O] Initializing synthetic simple-cubic tight-binding dataset...")
        data = generate_mock_electronic_structure()
    else:
        # Fallback resolution mechanism
        candidates = [args.input] if args.input else ["vaspout.h5", "vasprun.xml"]
        for candidate in candidates:
            if candidate and os.path.exists(candidate):
                print(f"[I/O] Resolving calculation database: {candidate}")
                parser_inst = VaspDataParser(candidate)
                data = parser_inst.parse()
                input_resolved = candidate
                break

        if data is None:
            print("[Warning] No VASP files found. Falling back to synthetic dataset.")
            data = generate_mock_electronic_structure()

    # ---------------------------------------------------------
    # 2. Execute Selected Mode
    # ---------------------------------------------------------
    if args.mode in ["single", "multi"]:
        projector = KSpaceProjector(data["kpoints"], data["eigenvalues"], data["rec_lattice"])

        if args.mode == "single":
            # Single Plane Execution
            normal_frac = np.array(args.normal)
            label = f"nx{normal_frac[0]}_ny{normal_frac[1]}_nz{normal_frac[2]}"
            execute_projection(projector, data["efermi"], args, normal_frac, label)

        elif args.mode == "multi":
            # Standard generalized multi-plane array
            miller_indices = [
                    (1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0),
                    (1.0, 1.0, 0.0), (1.0, 0.0, 1.0), (0.0, 1.0, 1.0),
                    (1.0, 1.0, 1.0), (2.0, 1.0, 0.0), (1.0, 2.0, 0.0),
                    (1.0, 1.0, 2.0), (2.0, 0.0, 1.0), (1.0, 2.0, 1.0)
                    ]
            for h, k, l in miller_indices:
                normal_frac = np.array([h, k, l])
                plane_name = f"{int(h)}{int(k)}{int(l)}"
                execute_projection(projector, data["efermi"], args, normal_frac, plane_name)

    elif args.mode == "surface_bz":
        # Surface Brillouin Zone Correlation Mode
        if input_resolved is None:
            print("[Error] Surface BZ analyzer requires a valid VASP file. Mock data is unsupported.")
            sys.exit(1)

        print("\n[Surface BZ] Initializing volumetric correlation analyzer...")
        analyzer = SurfaceBZAnalyzer(input_resolved)

        print(f"[Surface BZ] Generating slab {tuple(args.miller_surf)}...")
        analyzer.generate_slab(
                miller_index=tuple(args.miller_surf),
                min_slab=args.slab_min,
                min_vac=args.vac_min
                )

        correlation_data = analyzer.correlate_zones()
        print(" -> Correlation Mapping Computed:", correlation_data.get("labels", "N/A"))
        analyzer.visualize()

    elif args.mode == "surface_bands":
        if input_resolved is None:
            sys.exit("[Error] Surface bands mode requires a valid VASP file.")

        print(f"\n[Surface Bands] Initializing bulk band projection for {tuple(args.miller_surf)}...")
        analyzer = SurfaceBZAnalyzer(input_resolved)
        analyzer.generate_slab(tuple(args.miller_surf), args.slab_min, args.vac_min)
        corr = analyzer.correlate_zones()
        projector = KSpaceProjector(data["kpoints"], data["eigenvalues"], data["rec_lattice"])

        # Filter out unique points using dictionary keys
        unique_pts = {tuple(np.round(kpt, 4)): {'coord': kpt, 'label': make_bar_label(lbl), 'raw': lbl}
                      for kpt, lbl in zip(corr["projected_kpts"], corr["labels"])}

        gamma_key = min(unique_pts.keys(), key=np.linalg.norm)
        gamma_pt = unique_pts.pop(gamma_key)

        for pt_info in unique_pts.values():
#            if 'Z' in pt_info['label'][1:-1] or ('X' in pt_info['label'][1:-1] and '1' not in pt_info['label'][1:-1]):
            if True:
                print(f" -> Running high-symmetry point {pt_info['label']} for band projection.")
            else:
                print(f" -> Skipping high-symmetry point {pt_info['label']} for band projection.")
                continue
            p_vec = pt_info['coord']
            dist = np.linalg.norm(p_vec)
            if dist < 1e-4: continue

            # Create strictly aligned slicing vectors
            u_dir_aligned = np.array([p_vec[0], p_vec[1], 0.0]) / np.linalg.norm([p_vec[0], p_vec[1], 0.0])
            n_aligned = np.cross(u_dir_aligned, np.array([0.0, 0.0, 1.0]))

            # Map back to bulk Cartesian
            u_dir_cart = u_dir_aligned @ analyzer.R_align
            n_cart = n_aligned @ analyzer.R_align
            normal_frac = n_cart @ np.linalg.inv(projector.rec_lattice)

            print(f" -> Path: -{pt_info['raw']} -> Gamma -> +{pt_info['raw']}")

            u_grid, v_grid, interp_spectra = projector.interpolate_plane(
                normal_frac=normal_frac, point_frac=np.array([0.0, 0.0, 0.0]),
                u_range=(-dist, dist), v_range=(-3.0, 3.0),
                grid_resolution=args.resolution, interpolate_factor=args.smooth, u_dir_cart=u_dir_cart
            )

            plotter = ARPESPlotter(u_grid, v_grid, interp_spectra, data["efermi"])
            clean = pt_info['raw'].replace('$', '').replace('\\', '').replace('{', '').replace('}', '')

            bands_title = f"Surface Bands {tuple(args.miller_surf)}"
            plotter.plot_dispersion_slice(
                slice_coordinate=0.0, along_v=False, energy_limits=tuple(args.elimits),
                n_energy_points=args.n_energy, broadening=args.broadening, cmap=args.cmap,
                filename=os.path.join(args.outdir, f"sbz_bands_{clean}_G_{clean}.png"),
                integrate_v=True, custom_xticks=[-dist, 0.0, dist],
                custom_xticklabels=[f"$-{pt_info['label'][1:-1]}$", r'$\bar{\Gamma}$', f"$+{pt_info['label'][1:-1]}$"],
                cscale=args.cscale, custom_title=bands_title
            )

    print("\nPost-processing execution complete.")
    print("=" * 80)

if __name__ == "__main__":
    main()

