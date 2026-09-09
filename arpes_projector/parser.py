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


class _ProjectedSkippingReader:
    """
    File-like wrapper that elides every <projected>...</projected> block from
    the byte stream at raw-buffer speed.

    The projections block routinely accounts for >95% of a vasprun.xml
    produced with LORBIT set (99% for multi-GB files), yet standard XML
    parsers tokenize all of it even when the caller discards the result.
    Skipping it with bytes.find() keeps the streaming parse I/O-bound.
    """

    _OPEN = b"<projected>"
    _CLOSE = b"</projected>"

    def __init__(self, fh, chunk_size: int = 1 << 20):
        self._fh = fh
        self._chunk = chunk_size
        self._buf = b""
        self._pending = b""
        self._skipping = False
        self._eof = False
        # Retain this many trailing bytes when a tag is not found, in case it
        # straddles a chunk boundary.
        self._margin = len(self._CLOSE) - 1

    def _fill(self) -> None:
        data = self._fh.read(self._chunk)
        if not data:
            self._eof = True
        else:
            self._buf += data

    def read(self, size: int = -1) -> bytes:
        if size is None or size < 0:
            size = 1 << 62
        while len(self._pending) < size:
            if self._skipping:
                idx = self._buf.find(self._CLOSE)
                if idx >= 0:
                    self._buf = self._buf[idx + len(self._CLOSE):]
                    self._skipping = False
                    continue
                if self._eof:
                    self._buf = b""
                    break
                self._buf = self._buf[-self._margin:] if len(self._buf) > self._margin else self._buf
                self._fill()
                continue
            idx = self._buf.find(self._OPEN)
            if idx >= 0:
                self._pending += self._buf[:idx]
                self._buf = self._buf[idx + len(self._OPEN):]
                self._skipping = True
                continue
            if self._eof:
                self._pending += self._buf
                self._buf = b""
                break
            if len(self._buf) > self._margin:
                self._pending += self._buf[:-self._margin]
                self._buf = self._buf[-self._margin:]
            self._fill()
        out, self._pending = self._pending[:size], self._pending[size:]
        return out


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

    def parse(self, use_cache: bool = True) -> Dict[str, Any]:
        """
        Dynamically dispatches parsing depending on the file extension.

        Args:
            use_cache (bool): When True (default), reuse/write a small .npz
                cache next to the input file so repeated invocations skip
                re-parsing multi-GB outputs entirely.

        Returns:
            Dict[str, Any]: Structured data containing kpoints, eigenvalues, efermi,
                            and reciprocal lattice vectors.
        """
        cache_path = self.filepath + ".arpes_cache.npz"
        if use_cache:
            cached = self._load_cache(cache_path)
            if cached is not None:
                return cached

        _, ext = os.path.splitext(self.filepath)
        if ext.lower() == ".h5":
            data = self._parse_h5()
        else:
            data = self._parse_xml()

        if use_cache:
            self._write_cache(cache_path, data)
        return data

    def _load_cache(self, cache_path: str) -> Dict[str, Any]:
        """Returns cached parse results if present and newer than the input, else None."""
        try:
            # Strictly newer: equal timestamps (same clock tick) count as stale,
            # so ambiguity resolves toward re-parsing rather than stale data.
            if not (os.path.exists(cache_path)
                    and os.path.getmtime(cache_path) > os.path.getmtime(self.filepath)):
                return None
            with np.load(cache_path) as z:
                data = {
                        "kpoints": z["kpoints"],
                        "eigenvalues": z["eigenvalues"],
                        "efermi": float(z["efermi"]),
                        "rec_lattice": z["rec_lattice"],
                        "is_spin_polarized": bool(z["is_spin_polarized"])
                        }
            print(f"[Parser] Loaded cached parse results: {cache_path}")
            return data
        except Exception as exc:
            print(f"[Parser] Ignoring unreadable cache {cache_path}: {exc}")
            return None

    @staticmethod
    def _write_cache(cache_path: str, data: Dict[str, Any]) -> None:
        """Persists parse results next to the input file; failures are non-fatal."""
        try:
            np.savez(cache_path,
                     kpoints=data["kpoints"],
                     eigenvalues=data["eigenvalues"],
                     efermi=data["efermi"],
                     rec_lattice=data["rec_lattice"],
                     is_spin_polarized=data["is_spin_polarized"])
            print(f"[Parser] Cached parse results to {cache_path}")
        except OSError as exc:
            print(f"[Parser] Could not write cache {cache_path}: {exc}")

    # Above this file size, pymatgen's DOM-building parser is replaced by the
    # streaming parser. Measured on a 9.33 GiB noncollinear vasprun.xml
    # (136 bands, 19683 k-points): 14.7 s at 0.29 GB peak RSS, about 3% of the
    # file. The earlier "under ~100 MB" note here was an unmeasured estimate.
    STREAM_THRESHOLD_BYTES = 100 * 1024 * 1024

    def _parse_xml(self) -> Dict[str, Any]:
        """
        Parses vasprun.xml, dispatching between pymatgen (small files) and the
        constant-memory streaming parser (large files, or pymatgen missing).

        Returns:
            Dict[str, Any]: Dictionary containing parsed arrays and floats.
        """
        if os.path.getsize(self.filepath) > self.STREAM_THRESHOLD_BYTES:
            print(f"[Parser] Large vasprun.xml detected "
                  f"({os.path.getsize(self.filepath) / 1e6:.0f} MB); using streaming parser.")
            return self._parse_xml_stream()
        try:
            return self._parse_xml_pymatgen()
        except ImportError:
            print("[Parser] pymatgen not available; falling back to streaming XML parser.")
            return self._parse_xml_stream()

    def _parse_xml_stream(self) -> Dict[str, Any]:
        """
        Constant-memory vasprun.xml parser built on xml.etree.iterparse.

        Only the sections this suite needs are materialized (k-point list,
        final-step eigenvalues, Fermi energy, reciprocal basis); the
        <projected> block - typically >95% of the file - is elided from the
        byte stream before it ever reaches the XML tokenizer, and processed
        elements are cleared as parsing advances.

        Returns:
            Dict[str, Any]: Dictionary containing parsed arrays and floats.
        """
        import xml.etree.ElementTree as ET

        kpoints = None
        eigenvalues = None      # last <eigenvalues> block wins (final ionic step)
        efermi = None
        rec_lattice = None      # last <crystal> rec_basis wins (final structure)
        in_dos = False

        # Eigenvalues are drained k-point by k-point rather than left to
        # accumulate. The <projected> block is elided from the byte stream, but
        # <eigenvalues> still holds one <r> element per band per k-point, and an
        # ElementTree node costs far more than the two floats it carries: on a
        # 9.33 GiB file that subtree alone was 2.7M nodes and 1.2 GB, which was
        # the entire measured peak. Clearing each k-point <set> as its end tag
        # passes keeps only the finished float rows.
        in_eig = False
        eig_depth = 0           # <set> nesting inside <eigenvalues>: 1 outer, 2 spin, 3 k-point
        eig_spins = []          # one list of per-k-point band-energy arrays per spin

        # Elements whose subtrees are bulky and irrelevant once their end tag passes
        clear_on_end = {"scstep", "incar", "parameters", "atominfo",
                        "generation", "total", "partial", "varray", "structure"}

        with open(self.filepath, "rb") as raw:
            source = _ProjectedSkippingReader(raw)
            for event, elem in ET.iterparse(source, events=("start", "end")):
                if event == "start":
                    if elem.tag == "dos":
                        in_dos = True
                    elif elem.tag == "eigenvalues":
                        in_eig, eig_depth, eig_spins = True, 0, []
                    elif in_eig and elem.tag == "set":
                        eig_depth += 1
                        if eig_depth == 2:
                            eig_spins.append([])
                    continue

                tag = elem.tag
                if in_eig and tag == "set":
                    if eig_depth == 3 and eig_spins:
                        # <r> rows are "energy occupation"; keep the energies.
                        rows = np.array(
                                " ".join(r.text for r in elem if r.tag == "r").split(),
                                dtype=np.float64).reshape(-1, 2)
                        eig_spins[-1].append(rows[:, 0])
                        elem.clear()
                    eig_depth -= 1
                if tag == "varray":
                    name = elem.get("name")
                    if name == "kpointlist":
                        kpoints = np.array(
                                " ".join(v.text for v in elem).split(),
                                dtype=np.float64).reshape(-1, 3)
                    elif name == "rec_basis":
                        # vasprun.xml stores the crystallographic reciprocal
                        # basis (no 2*pi); include the physics convention here
                        rec_lattice = 2.0 * np.pi * np.array(
                                " ".join(v.text for v in elem).split(),
                                dtype=np.float64).reshape(3, 3)
                elif tag == "eigenvalues":
                    in_eig = False
                    if eig_spins and all(eig_spins):
                        # (nspins, nkpts, nbands) -> (nspins, nbands, nkpts)
                        eigenvalues = np.array(
                                [np.stack(ks) for ks in eig_spins]).transpose(0, 2, 1)
                    eig_spins = []
                    elem.clear()
                elif tag == "i" and in_dos and elem.get("name") == "efermi":
                    efermi = float(elem.text)
                elif tag == "dos":
                    in_dos = False
                    elem.clear()
                elif tag == "calculation":
                    elem.clear()

                if tag in clear_on_end:
                    elem.clear()

        if eigenvalues is None or kpoints is None:
            raise ValueError(
                    f"Streaming parse of {self.filepath} did not find "
                    f"{'eigenvalues' if eigenvalues is None else 'a k-point list'}; "
                    "the file may be truncated or from an unsupported calculation type.")
        if eigenvalues.shape[2] != len(kpoints):
            raise ValueError(
                    f"Streaming parse mismatch: {eigenvalues.shape[2]} eigenvalue "
                    f"k-points vs {len(kpoints)} k-points in kpointlist.")
        if efermi is None:
            print("[Parser] Warning: no Fermi energy found in vasprun.xml; defaulting to 0.0 eV.")
            efermi = 0.0
        if rec_lattice is None:
            raise ValueError(f"Streaming parse of {self.filepath} found no reciprocal basis.")

        print(f"[Parser] Streaming parse complete: {eigenvalues.shape[0]} spin(s), "
              f"{eigenvalues.shape[1]} bands, {len(kpoints)} k-points, E_F = {efermi:.4f} eV.")

        return {
                "kpoints": kpoints,
                "eigenvalues": eigenvalues,
                "efermi": efermi,
                "rec_lattice": rec_lattice,
                "is_spin_polarized": eigenvalues.shape[0] > 1
                }

    @staticmethod
    def _eigenvalues_from_element(elem) -> np.ndarray:
        """
        Converts an <eigenvalues> element into an array of shape (nspins, nbands, nkpts).

        Rows hold "energy occupation" pairs; only energies are kept.
        Returns None if the element does not contain the expected set structure.
        """
        outer = elem.find("array/set")
        if outer is None:
            return None
        spin_sets = [c for c in outer if c.tag == "set"]
        if not spin_sets:
            return None

        per_spin = []
        for spin_set in spin_sets:
            ksets = [c for c in spin_set if c.tag == "set"]
            if not ksets:
                return None
            rows = np.array(
                    " ".join(r.text for kset in ksets for r in kset).split(),
                    dtype=np.float64)
            nk = len(ksets)
            # rows = nk * nbands * 2 values (energy, occupation)
            bands_k = rows.reshape(nk, -1, 2)[:, :, 0]   # (nk, nbands)
            per_spin.append(bands_k.T)                   # (nbands, nk)
        return np.stack(per_spin)

    def _parse_xml_pymatgen(self) -> Dict[str, Any]:
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
                if isinstance(obj, h5py.Dataset) and "kpoint" in name.lower():
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
                        group = p.rsplit("/", 1)[0]
                        kp_path = next((c for c in (f"{group}/kpoint_coords", f"{group}/kpoints")
                                        if c in f), f"{group}/kpoints")
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

            # VASP 4D: (nstep, nspin, nkpoint, nband) -> slice only the last ionic
            # step; h5py reads lazily, so this avoids loading every step from disk
            evals = eig_ds[-1] if eig_ds.ndim == 4 else eig_ds[:]
            # Transpose 3D shape (nspin, nkpoint, nband) to (nspin, nband, nkpoint)
            evals = np.transpose(evals, (0, 2, 1))

            data["eigenvalues"] = evals
            data["kpoints"] = matched_pair["kp_arr"]

            # Parse Fermi energy
            efermi_found = False
            parent_group = eig_path.rsplit("/", 1)[0]
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

            # Compute reciprocal lattice vectors from the real-space basis.
            # VASP writes the cell as results/positions/lattice_vectors; older/other
            # layouts use results/positions/basis, and the POSCAR copy needs its scale.
            basis = None
            for path in ("results/positions/lattice_vectors", "results/positions/basis"):
                if path in f:
                    basis = np.asarray(f[path][()], dtype=float)
                    break
            if basis is None and "input/poscar/lattice_vectors" in f:
                basis = np.asarray(f["input/poscar/lattice_vectors"][()], dtype=float)
                if "input/poscar/scale" in f:
                    basis = basis * float(f["input/poscar/scale"][()])
            if basis is None:
                raise KeyError(
                        "No lattice vectors found in vaspout.h5 (looked for "
                        "results/positions/lattice_vectors, results/positions/basis and "
                        "input/poscar/lattice_vectors). Refusing to guess a reciprocal "
                        "lattice: every k-space coordinate would be wrong.")
            # Trajectory layouts store (nstep, 3, 3); take the final step.
            if basis.ndim == 3:
                basis = basis[-1]
            if basis.shape != (3, 3):
                raise ValueError(f"lattice vectors have shape {basis.shape}, expected (3, 3)")

            # b_i = 2*pi (a_j x a_k) / V   with   V = a_0 . (a_1 x a_2)
            vol = float(np.dot(basis[0], np.cross(basis[1], basis[2])))
            if abs(vol) < 1e-12:
                raise ValueError("degenerate lattice vectors: cell volume is zero")
            rec_basis = np.empty((3, 3))
            rec_basis[0] = 2 * np.pi * np.cross(basis[1], basis[2]) / vol
            rec_basis[1] = 2 * np.pi * np.cross(basis[2], basis[0]) / vol
            rec_basis[2] = 2 * np.pi * np.cross(basis[0], basis[1]) / vol
            data["rec_lattice"] = rec_basis

            data["is_spin_polarized"] = evals.shape[0] > 1

        return data

