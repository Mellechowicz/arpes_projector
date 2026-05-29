"""
This file contains the VaspDataParser class.
It extracts electronic eigenvalues, k-points,
and reciprocal lattice parameters from VASP structural and binary outputs.

Inputs:
 - filepath: String specifying the path to vasprun.xml or vaspout.h5 files.

Outputs:
 - Dictionary containing parsed k-points arrays, eigenvalues arrays, Fermi energy float, reciprocal lattice matrix, and spin polarization boolean.

Approach and Modules:
 - XML parsing: Structural markup extraction via pymatgen.io.vasp.outputs (Vasprun, BSVasprun).
 - HDF5 parsing: Binary dataset extraction and dynamic shape matching via h5py.
 - Linear algebra: Reciprocal basis computation via numpy.
"""

import os
import numpy as np
from typing import Dict, Any
from pymatgen.io.vasp.outputs import Vasprun, BSVasprun
from pymatgen.electronic_structure.core import Spin

class VaspDataParser:
    """Parses and structures VASP electronic structure data for spectroscopic analysis."""

    def __init__(self, filepath: str):
        """
        Initialize the parser with the path to the VASP output file.

        Args:
            filepath (str): Path to vasprun.xml or vaspout.h5.
        """
        self.filepath = filepath
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"VASP output file not found at: {filepath}")

    def parse(self) -> Dict[str, Any]:
        """
        Dynamically dispatches parsing depending on the file extension.

        Returns:
            Dict[str, Any]: Structured data containing kpoints, eigenvalues, efermi,
                            and reciprocal lattice vectors.
        """
        _, ext = os.path.splitext(self.filepath)
        if ext.lower() == ".h5":
            return self._parse_h5()
        else:
            return self._parse_xml()

    def _parse_xml(self) -> Dict[str, Any]:
        """
        Parses vasprun.xml using pymatgen routines.

        Returns:
            Dict[str, Any]: Dictionary containing parsed arrays and floats.
        """
        try:
            # Attempt to parse as band structure mode
            run = BSVasprun(self.filepath, parse_projected_eigen=False)
            bs = run.get_band_structure()
        except Exception:
            # Fall back to standard ground-state parser
            run = Vasprun(self.filepath, parse_eigen=True)
            bs = run.get_band_structure()

        rec_lattice = bs.lattice_rec
        efermi = bs.efermi
        kpoints = np.array([kp.frac_coords for kp in bs.kpoints])

        spins = list(bs.bands.keys())
        nbands = bs.nb_bands
        nkpts = len(kpoints)
        nspins = len(spins)

        # Re-structure eigenvalues to shape: (nspins, nbands, nkpts)
        eigenvalues = np.zeros((nspins, nbands, nkpts))
        for i, spin in enumerate(spins):
            eigenvalues[i] = bs.bands[spin]

        return {
                "kpoints": kpoints,
                "eigenvalues": eigenvalues,
                "efermi": efermi,
                "rec_lattice": rec_lattice.matrix,
                "is_spin_polarized": bs.is_spin_polarized
                }

    def _parse_h5(self) -> Dict[str, Any]:
        """
        Directly parses vaspout.h5 using h5py.
        Uses a robust dynamic dataset matching strategy to find eigenvalues and k-points
        compatible with different VASP versions and calculation setups.

        Returns:
            Dict[str, Any]: Dictionary containing parsed arrays and floats.
        """
        import h5py
        data = {}
        with h5py.File(self.filepath, "r") as f:
            # 1. Gather all candidate k-points datasets
            kpoints_candidates = {}
            def find_kpoints(name, obj):
                if isinstance(obj, h5py.Dataset) and "kpoints" in name.lower():
                    shape = obj.shape
                    # Valid kpoints dataset should be 2D with shape (N, 3)
                    if len(shape) == 2 and shape[1] == 3:
                        kpoints_candidates[name] = obj[:]
            f.visititems(find_kpoints)

            # 2. Gather all candidate eigenvalues datasets
            eigenvalues_candidates = {}
            def find_eigenvalues(name, obj):
                if isinstance(obj, h5py.Dataset) and "eigenvalues" in name.lower():
                    shape = obj.shape
                    # Valid eigenvalues dataset should be 3D or 4D
                    if len(shape) in (3, 4):
                        eigenvalues_candidates[name] = obj

            f.visititems(find_eigenvalues)

            # 3. Perform matching based on number of k-points
            matched_pair = None
            # Prioritized order of paths to search for matching
            for eig_path, eig_ds in eigenvalues_candidates.items():
                eig_shape = eig_ds.shape
                # If 4D: (nstep, nspin, nkpoint, nband)
                # If 3D: (nspin, nkpoint, nband)
                nk = eig_shape[-2]

                # Find a k-points dataset of the same size
                for kp_path, kp_arr in kpoints_candidates.items():
                    if len(kp_arr) == nk:
                        # Score the match based on typical path names
                        score = 0
                        if "kpoints_opt" in eig_path:
                            score += 10
                        if "electron_eigenvalues" in eig_path:
                            score += 5
                        if "results" in eig_path:
                            score += 2
                        if "wan" in eig_path:
                            score -= 10  # Deprioritize Wannier grids

                        if matched_pair is None or score > matched_pair["score"]:
                            matched_pair = {
                                    "eig_path": eig_path,
                                    "eig_ds": eig_ds,
                                    "kp_path": kp_path,
                                    "kp_arr": kp_arr,
                                    "score": score
                                    }

            if matched_pair is None:
                print("[Parser] Warning: Dynamic HDF5 dataset matching failed. Using fallback paths.")
                potential_paths = [
                        "results/electron_eigenvalues_kpoints/eigenvalues",
                        "results/electron_eigenvalues/eigenvalues",
                        "results/electron_eigenvalues_kpoints_opt/eigenvalues",
                        "results/eigenvalues/eigenvalues"
                        ]
                for p in potential_paths:
                    if p in f:
                        eig_ds = f[p]
                        kp_path = p.replace("eigenvalues", "kpoints")
                        if kp_path in f:
                            matched_pair = {
                                    "eig_path": p,
                                    "eig_ds": eig_ds,
                                    "kp_path": kp_path,
                                    "kp_arr": f[kp_path][:],
                                    "score": 0
                                    }
                            break

            if matched_pair is None:
                # If all else fails, print the available datasets to aid user debugging
                available_datasets = []
                def list_all(name, obj):
                    if isinstance(obj, h5py.Dataset):
                        available_datasets.append(f"  {name}: {obj.shape}")
                f.visititems(list_all)
                datasets_str = "\n".join(available_datasets)
                raise KeyError(
                        f"Could not find any matching eigenvalues and k-points datasets in vaspout.h5.\n"
                        f"Available datasets in your file:\n{datasets_str}"
                        )

            # Extract data from matched pair
            eig_path = matched_pair["eig_path"]
            kp_path = matched_pair["kp_path"]
            eig_ds = matched_pair["eig_ds"]

            print(f"[Parser] Successfully resolved matching datasets in vaspout.h5:")
            print(f"  - Eigenvalues: '{eig_path}' {eig_ds.shape}")
            print(f"  - K-points:    '{kp_path}' {matched_pair['kp_arr'].shape}")

            evals = eig_ds[:]
            # VASP 4D: (nstep, nspin, nkpoint, nband) -> extract last ionic step
            if evals.ndim == 4:
                evals = evals[-1]
            # Transpose 3D shape (nspin, nkpoint, nband) to (nspin, nband, nkpoint)
            evals = np.transpose(evals, (0, 2, 1))

            data["eigenvalues"] = evals
            data["kpoints"] = matched_pair["kp_arr"]

            # Parse Fermi energy
            efermi_found = False
            parent_group = eig_path.rsplit("/", 1)
            if f"{parent_group}/efermi" in f:
                data["efermi"] = f[f"{parent_group}/efermi"][()]
                efermi_found = True

            if not efermi_found:
                for path in ["results/electron_dos/efermi", "results/eigenvalues/efermi", "results/electron_dos_kpoints_opt/efermi"]:
                    if path in f:
                        data["efermi"] = f[path][()]
                        efermi_found = True
                        break

            if not efermi_found:
                data["efermi"] = 0.0

            # Compute reciprocal lattice vectors from real-space basis
            if "results/positions/basis" in f:
                basis = f["results/positions/basis"][-1]
                # Mathematically correct triple scalar product for volume
                vol = np.dot(basis, np.cross(basis[1], basis[2]))
                rec_basis = np.zeros((3, 3))
                rec_basis = 2 * np.pi * np.cross(basis[1], basis[2]) / vol
                rec_basis[1] = 2 * np.pi * np.cross(basis[2], basis) / vol
                rec_basis[2] = 2 * np.pi * np.cross(basis, basis[1]) / vol
                data["rec_lattice"] = rec_basis
            else:
                data["rec_lattice"] = np.eye(3) * 2 * np.pi

            data["is_spin_polarized"] = evals.shape[0] > 1

        return data

