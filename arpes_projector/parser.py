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

# VASP's canonical lm-decomposed orbital order (LORBIT=11/12); sliced to the
# actual column count when a file provides no field names (e.g. vaspout.h5).
CANONICAL_ORBITALS = ["s", "py", "pz", "px", "dxy", "dyz", "dz2", "dxz", "x2-y2",
                      "fy3x2", "fxyz", "fyz2", "fz3", "fxz2", "fzx2", "fx3"]


class _StopParse(Exception):
    """Raised internally to abort streaming once all needed data has been read."""


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

    def parse(self, use_cache: bool = True, weights_spec: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Dynamically dispatches parsing depending on the file extension.

        Args:
            use_cache (bool): When True (default), reuse/write a small .npz
                cache next to the input file so repeated invocations skip
                re-parsing multi-GB outputs entirely.
            weights_spec (Dict, optional): When given, per-state spectral
                weights are reduced from the orbital/site projections during
                parsing. Keys:
                  "orbital_weights": mapping of orbital or shell name to weight
                      (e.g. {"s": 1.0, "p": 0.5, "dz2": 2.0}); unlisted
                      orbitals default to 0.0 if any weight is given for their
                      shell's siblings, otherwise shells default from "default".
                  "ion_weights": sequence of per-ion weights (defaults to 1.0).

        Returns:
            Dict[str, Any]: Structured data containing kpoints, eigenvalues, efermi,
                            reciprocal lattice vectors and, when requested,
                            "weights" of shape (nspins, nbands, nkpts).
        """
        spec_key = self._weights_spec_key(weights_spec)
        cache_path = self.filepath + ".arpes_cache.npz"
        if use_cache:
            cached = self._load_cache(cache_path, spec_key)
            if cached is not None:
                return cached

        _, ext = os.path.splitext(self.filepath)
        if ext.lower() == ".h5":
            data = self._parse_h5(weights_spec)
        else:
            data = self._parse_xml(weights_spec)

        if use_cache:
            self._write_cache(cache_path, data, spec_key)
        return data

    @staticmethod
    def _weights_spec_key(weights_spec) -> str:
        """Canonical string form of a weights request, used for cache validation."""
        if weights_spec is None:
            return ""
        orb = sorted((weights_spec.get("orbital_weights") or {}).items())
        ion = list(weights_spec.get("ion_weights") or [])
        return f"orb={orb};ion={ion}"

    def _load_cache(self, cache_path: str, spec_key: str) -> Dict[str, Any]:
        """Returns cached parse results if present, newer than the input, and
        holding weights that match the requested spec (when one is given)."""
        try:
            # Strictly newer: equal timestamps (same clock tick) count as stale,
            # so ambiguity resolves toward re-parsing rather than stale data.
            if not (os.path.exists(cache_path)
                    and os.path.getmtime(cache_path) > os.path.getmtime(self.filepath)):
                return None
            with np.load(cache_path) as z:
                if spec_key and (("weights" not in z) or str(z["weights_spec"]) != spec_key):
                    return None      # cache lacks the requested weights
                data = {
                        "kpoints": z["kpoints"],
                        "eigenvalues": z["eigenvalues"],
                        "efermi": float(z["efermi"]),
                        "rec_lattice": z["rec_lattice"],
                        "is_spin_polarized": bool(z["is_spin_polarized"])
                        }
                if spec_key:
                    data["weights"] = z["weights"]
            print(f"[Parser] Loaded cached parse results: {cache_path}")
            return data
        except Exception as exc:
            print(f"[Parser] Ignoring unreadable cache {cache_path}: {exc}")
            return None

    @staticmethod
    def _write_cache(cache_path: str, data: Dict[str, Any], spec_key: str) -> None:
        """Persists parse results next to the input file; failures are non-fatal."""
        try:
            payload = {
                    "kpoints": data["kpoints"],
                    "eigenvalues": data["eigenvalues"],
                    "efermi": data["efermi"],
                    "rec_lattice": data["rec_lattice"],
                    "is_spin_polarized": data["is_spin_polarized"]
                    }
            if data.get("weights") is not None:
                payload["weights"] = data["weights"]
                payload["weights_spec"] = spec_key
            np.savez(cache_path, **payload)
            print(f"[Parser] Cached parse results to {cache_path}")
        except OSError as exc:
            print(f"[Parser] Could not write cache {cache_path}: {exc}")

    # Above this file size, pymatgen's DOM-building parser is replaced by the
    # constant-memory streaming parser (peak RSS for a 9 GB vasprun.xml drops
    # from tens of GB to under ~100 MB).
    STREAM_THRESHOLD_BYTES = 100 * 1024 * 1024

    def _parse_xml(self, weights_spec: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Parses vasprun.xml, dispatching between pymatgen (small files) and the
        constant-memory streaming parser (large files, pymatgen missing, or
        whenever projection weights are requested - only the streaming path
        can reduce them without materializing the full tensor).

        Returns:
            Dict[str, Any]: Dictionary containing parsed arrays and floats.
        """
        if weights_spec is not None:
            return self._parse_xml_stream(weights_spec)
        if os.path.getsize(self.filepath) > self.STREAM_THRESHOLD_BYTES:
            print(f"[Parser] Large vasprun.xml detected "
                  f"({os.path.getsize(self.filepath) / 1e6:.0f} MB); using streaming parser.")
            return self._parse_xml_stream()
        try:
            return self._parse_xml_pymatgen()
        except ImportError:
            print("[Parser] pymatgen not available; falling back to streaming XML parser.")
            return self._parse_xml_stream()

    def _parse_xml_stream(self, weights_spec: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Constant-memory vasprun.xml parser built on xml.etree.iterparse.

        Only the sections this suite needs are materialized (k-point list,
        final-step eigenvalues, Fermi energy, reciprocal basis); processed
        elements are cleared as parsing advances.

        Without a weights_spec, the <projected> block - typically >95% of the
        file - is elided from the byte stream before it ever reaches the XML
        tokenizer. With one, each band's (nion x norb) projection row-block is
        reduced to a single scalar weight the moment it streams past, so the
        projection tensor is never materialized; parsing stops early once as
        many projection spin sets as eigenvalue spin channels have been read
        (noncollinear files carry 4 sets - total + 3 components - of which
        only the first is needed).

        Returns:
            Dict[str, Any]: Dictionary containing parsed arrays and floats,
                plus "weights" (nspins, nbands, nkpts) when requested.
        """
        import xml.etree.ElementTree as ET

        want_weights = weights_spec is not None
        kpoints = None
        eigenvalues = None      # last <eigenvalues> block wins (final ionic step)
        efermi = None
        rec_lattice = None      # last <crystal> rec_basis wins (final structure)
        in_dos = False

        # --- projection-reduction state (only used when want_weights) ---
        in_projected = False
        in_proj_eig = False     # inside the redundant <eigenvalues> copy within <projected>
        set_depth = 0           # <set> nesting: 1 outer, 2 spin, 3 kpoint, 4 band
        last_fields = []        # <field> names of the most recent <array> header
        w_flat = []             # one reduced scalar per (spin, kpoint, band), document order
        w_orb = w_ion = None
        nion = norb = None
        bands_in_kpt = kpts_in_spin = spins_done = 0
        nb_proj = nk_proj = None

        # Elements whose subtrees are bulky and irrelevant once their end tag passes
        clear_on_end = {"scstep", "incar", "parameters", "atominfo",
                        "generation", "total", "partial", "varray", "structure"}

        with open(self.filepath, "rb") as raw:
            source = raw if want_weights else _ProjectedSkippingReader(raw)
            try:
                for event, elem in ET.iterparse(source, events=("start", "end")):
                    tag = elem.tag
                    if event == "start":
                        if tag == "dos":
                            in_dos = True
                        elif tag == "projected":
                            in_projected = True
                        elif in_projected and not in_proj_eig:
                            if tag == "eigenvalues":
                                in_proj_eig = True
                            elif tag == "array":
                                last_fields = []
                            elif tag == "set":
                                set_depth += 1
                        continue

                    # ---- end events inside <projected> (weights reduction) ----
                    if in_projected:
                        if tag == "projected":
                            in_projected = False
                            elem.clear()
                        elif in_proj_eig:
                            if tag == "eigenvalues":
                                in_proj_eig = False
                                elem.clear()    # discard the redundant eigenvalue copy
                        elif tag == "field":
                            last_fields.append((elem.text or "").strip())
                        elif tag == "set":
                            level = set_depth
                            set_depth -= 1
                            if level == 4:      # band set: nion rows of norb projections
                                if w_orb is None:
                                    rows = [r.text for r in elem]
                                    nion = len(rows)
                                    vals = np.array(" ".join(rows).split(),
                                                    dtype=np.float64).reshape(nion, -1)
                                    norb = vals.shape[1]
                                    w_orb = self._build_orbital_vector(
                                            last_fields, norb, weights_spec.get("orbital_weights"))
                                    w_ion = self._build_ion_vector(
                                            nion, weights_spec.get("ion_weights"))
                                else:
                                    vals = np.array(" ".join(r.text for r in elem).split(),
                                                    dtype=np.float64).reshape(nion, norb)
                                w_flat.append(float(w_ion @ vals @ w_orb))
                                bands_in_kpt += 1
                            elif level == 3:    # kpoint set complete
                                if nb_proj is None:
                                    nb_proj = bands_in_kpt
                                elif bands_in_kpt != nb_proj:
                                    raise ValueError("Inconsistent band count in <projected> block.")
                                bands_in_kpt = 0
                                kpts_in_spin += 1
                            elif level == 2:    # spin set complete
                                if nk_proj is None:
                                    nk_proj = kpts_in_spin
                                elif kpts_in_spin != nk_proj:
                                    raise ValueError("Inconsistent k-point count in <projected> block.")
                                kpts_in_spin = 0
                                spins_done += 1
                                if eigenvalues is not None and spins_done >= eigenvalues.shape[0]:
                                    raise _StopParse    # all needed spin channels read
                            elem.clear()
                        continue

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
                        parsed = self._eigenvalues_from_element(elem)
                        if parsed is not None:
                            eigenvalues = parsed
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
            except _StopParse:
                pass

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

        weights = None
        if want_weights:
            if not w_flat:
                raise ValueError(
                        f"No <projected> block found in {self.filepath}; orbital-resolved "
                        "weights need a calculation run with LORBIT=11 or 12.")
            nspins = eigenvalues.shape[0]
            expected = spins_done * nk_proj * nb_proj
            if len(w_flat) != expected or spins_done < nspins:
                raise ValueError(
                        f"Projection block inconsistent: {len(w_flat)} reduced weights for "
                        f"{spins_done} spin set(s) x {nk_proj} k-points x {nb_proj} bands.")
            if (nb_proj, nk_proj) != eigenvalues.shape[1:]:
                raise ValueError(
                        f"Projection dimensions ({nb_proj} bands, {nk_proj} k-points) do not "
                        f"match eigenvalues {eigenvalues.shape[1:]}.")
            # (nspins, nk, nb) in document order -> (nspins, nbands, nkpts)
            weights = np.asarray(w_flat, dtype=np.float32).reshape(
                    spins_done, nk_proj, nb_proj).transpose(0, 2, 1)[:nspins]
            print(f"[Parser] Reduced orbital projections to per-state weights "
                  f"({nion} ions x {norb} orbitals; range [{weights.min():.3f}, {weights.max():.3f}]).")

        print(f"[Parser] Streaming parse complete: {eigenvalues.shape[0]} spin(s), "
              f"{eigenvalues.shape[1]} bands, {len(kpoints)} k-points, E_F = {efermi:.4f} eV.")

        return {
                "kpoints": kpoints,
                "eigenvalues": eigenvalues,
                "efermi": efermi,
                "rec_lattice": rec_lattice,
                "is_spin_polarized": eigenvalues.shape[0] > 1,
                "weights": weights
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
            if rows.size == 0:
                return None
            bands_k = rows.reshape(nk, -1, 2)[:, :, 0]   # (nk, nbands)
            per_spin.append(bands_k.T)                   # (nbands, nk)
        return np.stack(per_spin)

    @staticmethod
    def _build_orbital_vector(field_names, norb: int, orbital_weights) -> np.ndarray:
        """
        Maps a user orbital-weight spec onto the file's orbital columns.

        Spec keys may be exact orbital names ("dz2", "x2-y2"), shell letters
        ("s", "p", "d", "f"), or "default" for unlisted orbitals (0.0 if
        omitted). No spec at all weights every orbital 1.0. Exact names win
        over shell letters.
        """
        names = [n.strip().lower() for n in field_names]
        if len(names) != norb:
            names = CANONICAL_ORBITALS[:norb]
        if not orbital_weights:
            return np.ones(norb)
        spec = {str(k).strip().lower(): float(v) for k, v in orbital_weights.items()}
        default = spec.get("default", 0.0)
        vec = np.full(norb, default)
        for i, name in enumerate(names):
            shell = "d" if name.startswith("x2") else name[0]
            if name in spec:
                vec[i] = spec[name]
            elif shell in spec:
                vec[i] = spec[shell]
        return vec

    @staticmethod
    def _build_ion_vector(nion: int, ion_weights) -> np.ndarray:
        """Per-ion weight vector; defaults to 1.0 for every ion."""
        if not ion_weights:
            return np.ones(nion)
        vec = np.asarray(list(ion_weights), dtype=np.float64)
        if len(vec) != nion:
            raise ValueError(
                    f"ion_weights has {len(vec)} entries but the calculation has {nion} ions.")
        return vec

    def _parse_xml_pymatgen(self) -> Dict[str, Any]:
        """
        Parses vasprun.xml using pymatgen routines.

        Returns:
            Dict[str, Any]: Dictionary containing parsed arrays and floats.
        """
        # Deferred import: keeps --mock and HDF5 workflows usable without pymatgen
        from pymatgen.io.vasp.outputs import Vasprun, BSVasprun

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
                "is_spin_polarized": bs.is_spin_polarized,
                "weights": None
                }

    def _parse_h5(self, weights_spec: Dict[str, Any] = None) -> Dict[str, Any]:
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

            # Compute reciprocal lattice vectors from real-space basis
            if "results/positions/basis" in f:
                basis = f["results/positions/basis"][:]
                # Handle both (3, 3) final basis and (nstep, 3, 3) trajectory layouts
                if basis.ndim == 3:
                    basis = basis[-1]
                # Mathematically correct triple scalar product for volume
                vol = np.dot(basis[0], np.cross(basis[1], basis[2]))
                rec_basis = np.zeros((3, 3))
                rec_basis[0] = 2 * np.pi * np.cross(basis[1], basis[2]) / vol
                rec_basis[1] = 2 * np.pi * np.cross(basis[2], basis[0]) / vol
                rec_basis[2] = 2 * np.pi * np.cross(basis[0], basis[1]) / vol
                data["rec_lattice"] = rec_basis
            else:
                data["rec_lattice"] = np.eye(3) * 2 * np.pi

            data["is_spin_polarized"] = evals.shape[0] > 1

            data["weights"] = None
            if weights_spec is not None:
                data["weights"] = self._reduce_h5_projections(
                        f, weights_spec, data["eigenvalues"].shape)

        return data

    def _reduce_h5_projections(self, f, weights_spec: Dict[str, Any],
                               eig_shape: tuple) -> np.ndarray:
        """
        Reduces the orbital projections in a vaspout.h5 to per-state weights.

        The 5D projections dataset (nspin, nkpts, nbands, nion, norb) is read
        in k-chunks and contracted immediately, so peak memory stays bounded
        by the chunk size regardless of the dataset's full extent.
        """
        import h5py
        nspins, nbands, nkpts = eig_shape

        candidates = []
        def find_projections(name, obj):
            if isinstance(obj, h5py.Dataset) and obj.ndim == 5:
                low = name.lower()
                if "projector" in low or low.rsplit("/", 1)[-1] == "par":
                    candidates.append(obj)
        f.visititems(find_projections)

        ds = next((c for c in candidates
                   if c.shape[1] == nkpts and c.shape[2] == nbands), None)
        if ds is None:
            raise KeyError(
                    "No orbital projections dataset matching the eigenvalues was found in "
                    "vaspout.h5; orbital-resolved weights need LORBIT=11 or 12.")

        nion, norb = ds.shape[3], ds.shape[4]
        w_orb = self._build_orbital_vector([], norb, weights_spec.get("orbital_weights"))
        w_ion = self._build_ion_vector(nion, weights_spec.get("ion_weights"))

        weights = np.empty((nspins, nbands, nkpts), dtype=np.float32)
        k_chunk = max(1, int(8e6) // max(1, nbands * nion * norb))
        for s in range(nspins):    # noncollinear files carry extra spin sets; take the first nspins
            for k0 in range(0, nkpts, k_chunk):
                block = ds[s, k0:k0 + k_chunk]
                weights[s, :, k0:k0 + k_chunk] = np.einsum(
                        "kbio,i,o->bk", block, w_ion, w_orb, optimize=True)
        print(f"[Parser] Reduced orbital projections to per-state weights "
              f"({nion} ions x {norb} orbitals; range [{weights.min():.3f}, {weights.max():.3f}]).")
        return weights

