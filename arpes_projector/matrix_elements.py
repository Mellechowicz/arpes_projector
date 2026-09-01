"""
Photoemission matrix-element weighting for simulated ARPES intensities.

Baseline behaviour of this suite treats every band at every k-point as equally
bright. A real photoemission intensity is modulated by the matrix element
|<f|A.p|i>|^2, which at the DFT level is approximated well by the site- and
orbital-resolved projections VASP already writes when LORBIT=11 or 12:

    I(E,k)  ~  sum_b  w_{skb} * L(E - eps_{skb}; gamma)

    w_{skb} = sum_{alpha,lm}  c_alpha  sigma_lm  P^{alpha,lm}_{skb}

with c_alpha a per-ion weight (e.g. surface sensitivity) and sigma_lm a per-
orbital weight (e.g. a photon-energy-dependent cross section).

The governing constraint is memory. The projection tensor has shape
(nspin, nkpt, nband, nion, norb) and reaches gigabytes for a dense mesh, while
its reduction w is a few megabytes. Every routine here therefore contracts each
band's (nion x norb) block to a single scalar the moment it streams past the XML
tokenizer; the full tensor is never held.

Noncollinear runs (LNONCOLLINEAR = T) are a trap worth naming: their <projected>
block carries FOUR sets per k-point, which are (total, m_x, m_y, m_z) - NOT four
spin channels. Only the first is the charge projection; the remaining three are
magnetisation components and may be negative. Summing them would be physically
meaningless. This module reads only as many sets as there are eigenvalue spin
channels, which selects the charge projection for noncollinear runs and both
channels for a genuine collinear ISPIN=2 run.
"""

import numpy as np

# VASP's lm-decomposed order for LORBIT=11/12, used when a source supplies no
# field names of its own (e.g. vaspout.h5).
CANONICAL_ORBITALS = ["s", "py", "pz", "px", "dxy", "dyz", "dz2", "dxz", "x2-y2",
                      "fy3x2", "fxyz", "fyz2", "fz3", "fxz2", "fzx2", "fx3"]


class _StopParse(Exception):
    """Raised internally once every needed projection set has been read."""


def parse_orbital_spec(text):
    """
    Parses a CLI orbital-weight string into a dict.

    Accepts comma-separated "name:value" pairs where name is an exact orbital
    ("dz2", "x2-y2"), a shell letter ("s", "p", "d", "f"), or "default".
    Example: "s:1,p:0.5,dz2:2" or "d:1".
    """
    if not text:
        return None
    spec = {}
    for part in text.split(","):
        part = part.strip()
        if not part:
            continue
        name, sep, value = part.partition(":")
        if not sep:
            raise ValueError(f"orbital weight '{part}' must be written name:value")
        spec[name.strip().lower()] = float(value)
    return spec


def parse_ion_spec(text):
    """Parses a comma-separated per-ion weight list in POSCAR order."""
    if not text:
        return None
    return [float(x) for x in text.split(",") if x.strip()]


def build_orbital_vector(field_names, norb, spec):
    """
    Maps an orbital-weight spec onto the projection columns of a specific file.

    Exact orbital names win over shell letters; anything unlisted takes
    spec["default"] (0.0 when absent). With no spec at all every orbital
    weights 1.0, which reproduces the total projection.
    """
    names = [str(n).strip().lower() for n in (field_names or [])]
    if len(names) != norb:
        names = CANONICAL_ORBITALS[:norb]
    if not spec:
        return np.ones(norb)
    default = spec.get("default", 0.0)
    vec = np.full(norb, float(default))
    for i, name in enumerate(names):
        # "x2-y2" belongs to the d shell but starts with 'x'
        shell = "d" if name.startswith("x2") else (name[:1] or "")
        if name in spec:
            vec[i] = spec[name]
        elif shell in spec:
            vec[i] = spec[shell]
    return vec


def build_ion_vector(nion, spec):
    """Per-ion weight vector in POSCAR order; defaults to 1.0 for every ion."""
    if not spec:
        return np.ones(nion)
    vec = np.asarray(list(spec), dtype=np.float64)
    if len(vec) != nion:
        raise ValueError(f"ion weights list has {len(vec)} entries "
                         f"but the calculation contains {nion} ions")
    return vec


def surface_ion_weights(structure_z, escape_depth, surface_z=None):
    """
    Exponential surface-sensitivity weights exp(-(z_surf - z)/lambda).

    Provides the depth attenuation a real photoemission experiment has and this
    simulation otherwise lacks. structure_z is a sequence of Cartesian z
    coordinates in Angstrom; escape_depth is the inelastic mean free path.
    """
    z = np.asarray(structure_z, dtype=np.float64)
    if escape_depth <= 0:
        raise ValueError("escape depth must be positive")
    top = float(np.max(z)) if surface_z is None else float(surface_z)
    return np.exp(-np.abs(top - z) / escape_depth)


def reduce_projections_xml(filepath, orbital_spec=None, ion_spec=None,
                           n_sets=1, verbose=True):
    """
    Streams a vasprun.xml <projected> block and reduces it to per-state weights.

    Args:
        filepath (str): path to vasprun.xml.
        orbital_spec (dict): orbital/shell weights, see build_orbital_vector.
        ion_spec (sequence): per-ion weights, see build_ion_vector.
        n_sets (int): how many leading projection sets to read. Pass the number
            of eigenvalue spin channels: 1 for a noncollinear run (selecting the
            charge projection and skipping m_x, m_y, m_z), 2 for collinear
            ISPIN=2.
        verbose (bool): print a one-line summary when finished.

    Returns:
        (weights, meta): weights of shape (n_sets, nbands, nkpts) as float32,
        and a dict with nion, norb, orbital field names and the applied vectors.

    Memory is O(one band block); the projection tensor is never materialized.
    """
    import xml.etree.ElementTree as ET

    in_projected = False
    in_proj_eig = False
    set_depth = 0
    fields = []
    flat = []
    w_orb = w_ion = None
    nion = norb = None
    bands_in_kpt = kpts_in_set = sets_done = 0
    nb = nk = None

    with open(filepath, "rb") as fh:
        try:
            for event, elem in ET.iterparse(fh, events=("start", "end")):
                tag = elem.tag
                if event == "start":
                    if tag == "projected":
                        in_projected = True
                    elif in_projected and not in_proj_eig:
                        if tag == "eigenvalues":
                            in_proj_eig = True          # redundant copy, skip it
                        elif tag == "array":
                            fields = []
                        elif tag == "set":
                            set_depth += 1
                    continue

                if not in_projected:
                    elem.clear()
                    continue

                if tag == "projected":
                    in_projected = False
                    elem.clear()
                elif in_proj_eig:
                    if tag == "eigenvalues":
                        in_proj_eig = False
                    elem.clear()
                elif tag == "field":
                    fields.append((elem.text or "").strip())
                elif tag == "set":
                    level = set_depth
                    set_depth -= 1
                    if level == 4:                      # one band: nion rows x norb
                        if w_orb is None:
                            rows = [r.text for r in elem]
                            nion = len(rows)
                            vals = np.array(" ".join(rows).split(),
                                            dtype=np.float64).reshape(nion, -1)
                            norb = vals.shape[1]
                            w_orb = build_orbital_vector(fields, norb, orbital_spec)
                            w_ion = build_ion_vector(nion, ion_spec)
                        else:
                            vals = np.array(" ".join(r.text for r in elem).split(),
                                            dtype=np.float64).reshape(nion, norb)
                        flat.append(float(w_ion @ vals @ w_orb))
                        bands_in_kpt += 1
                    elif level == 3:                    # k-point complete
                        if nb is None:
                            nb = bands_in_kpt
                        elif bands_in_kpt != nb:
                            raise ValueError("inconsistent band count in <projected>")
                        bands_in_kpt = 0
                        kpts_in_set += 1
                    elif level == 2:                    # projection set complete
                        if nk is None:
                            nk = kpts_in_set
                        elif kpts_in_set != nk:
                            raise ValueError("inconsistent k-point count in <projected>")
                        kpts_in_set = 0
                        sets_done += 1
                        if sets_done >= n_sets:
                            raise _StopParse
                    elem.clear()
        except _StopParse:
            pass

    if not flat:
        raise ValueError(
                f"no <projected> block found in {filepath}; matrix-element weighting "
                "requires a calculation run with LORBIT=11 or 12")
    if sets_done < n_sets or len(flat) != sets_done * nk * nb:
        raise ValueError(
                f"projection block inconsistent: {len(flat)} weights for {sets_done} "
                f"set(s) x {nk} k-points x {nb} bands (wanted {n_sets} set(s))")

    # document order is (set, kpoint, band) -> transpose to (set, band, kpoint)
    weights = np.asarray(flat, dtype=np.float32).reshape(sets_done, nk, nb).transpose(0, 2, 1)
    meta = {"nion": nion, "norb": norb, "fields": list(fields),
            "orbital_vector": w_orb, "ion_vector": w_ion,
            "nbands": nb, "nkpts": nk, "n_sets": sets_done}
    if verbose:
        print(f"[MatrixElements] Reduced {nion} ions x {norb} orbitals over "
              f"{nb} bands x {nk} k-points; weight range "
              f"[{weights.min():.4f}, {weights.max():.4f}]")
    return weights, meta


def reduce_projections_h5(filepath, orbital_spec=None, ion_spec=None,
                          n_sets=1, k_chunk_bytes=64 << 20, verbose=True):
    """
    Reduces the orbital projections in a vaspout.h5 to per-state weights.

    VASP stores them at results/projectors/par with axis order
    (nset, nion, norb, nkpt, nband) - note this differs from the vasprun.xml
    ordering - and the array reaches gigabytes (2.44 GB for a 5832-point mesh
    with 136 bands). It is therefore read in k-chunks and contracted
    immediately, so peak memory is bounded by one chunk rather than the file.

    The leading axis carries the same noncollinear trap as the XML path: for a
    noncollinear run its four entries are (total, m_x, m_y, m_z), so n_sets
    should be the number of eigenvalue spin channels.

    Returns:
        (weights, meta) with weights of shape (n_sets, nbands, nkpts), float32.
    """
    import h5py

    with h5py.File(filepath, "r") as f:
        ds = None
        for path in ("results/projectors/par", "results/projections/par"):
            if path in f:
                ds = f[path]
                break
        if ds is None:
            found = []
            f.visititems(lambda n, o: found.append(n)
                         if isinstance(o, h5py.Dataset) and o.ndim == 5 else None)
            raise KeyError(
                    "No orbital projections found in vaspout.h5 (expected "
                    "results/projectors/par); matrix-element weighting requires "
                    f"LORBIT=11 or 12. 5-D datasets present: {found}")

        nset, nion, norb, nk, nb = ds.shape
        if n_sets > nset:
            raise ValueError(f"requested {n_sets} projection sets but the file has {nset}")

        # Orbital names live alongside the array as fixed-width bytes.
        fields = []
        lchar = ds.parent.get("lchar")
        if lchar is not None:
            fields = [x.decode() if isinstance(x, bytes) else str(x) for x in lchar[()]]

        w_orb = build_orbital_vector(fields, norb, orbital_spec)
        w_ion = build_ion_vector(nion, ion_spec)

        # Bound the working set: one chunk is nion*norb*chunk*nb float64.
        per_k = nion * norb * nb * 8
        chunk = max(1, int(k_chunk_bytes // max(1, per_k)))

        weights = np.empty((n_sets, nb, nk), dtype=np.float32)
        for s in range(n_sets):
            for k0 in range(0, nk, chunk):
                block = ds[s, :, :, k0:k0 + chunk, :]          # (nion, norb, dk, nb)
                weights[s, :, k0:k0 + chunk] = np.einsum(
                        "iokb,i,o->bk", block, w_ion, w_orb, optimize=True)

    meta = {"nion": nion, "norb": norb, "fields": fields, "orbital_vector": w_orb,
            "ion_vector": w_ion, "nbands": nb, "nkpts": nk, "n_sets": n_sets,
            "k_chunk": chunk}
    if verbose:
        print(f"[MatrixElements] Reduced {nion} ions x {norb} orbitals over "
              f"{nb} bands x {nk} k-points (k-chunk {chunk}); weight range "
              f"[{weights.min():.4f}, {weights.max():.4f}]")
    return weights, meta
