"""
Checks for matrix-element intensity weighting.

Run:  python3 tests/test_matrix_elements.py
Real-data checks are skipped when 040/vasprun.xml is absent.
Exit code is non-zero if any check fails.
"""
import os, re, sys, numpy as np, matplotlib
matplotlib.use("Agg")
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
from arpes_projector.geometry import KSpaceProjector
from arpes_projector.plotter import ARPESPlotter
from arpes_projector.matrix_elements import (reduce_projections_xml, parse_orbital_spec,
                                             build_orbital_vector, surface_ion_weights)
FAIL = []
def check(name, cond, detail=""):
    print(("[PASS] " if cond else "[FAIL] ") + name + (("  " + detail) if detail else ""))
    if not cond:
        FAIL.append(name)

def toy():
    rec = np.eye(3) * 2.0
    lin = np.linspace(-.45, .45, 9)
    X, Y, Z = np.meshgrid(lin, lin, lin, indexing="ij")
    kp = np.stack([X.ravel(), Y.ravel(), Z.ravel()], 1)
    eig = np.zeros((1, 3, len(kp)))
    for b in range(3):
        eig[0, b] = b * 2.0 - 1.0 + np.cos(2*np.pi*kp[:, 0]) + 0.5*np.cos(2*np.pi*kp[:, 1])
    return kp, eig, rec

def plane(pr):
    return pr.interpolate_plane(np.array([0., 0, 1]), np.zeros(3), (-1, 1), (-1, 1), 20, 1)

kp, eig, rec = toy()
E = np.linspace(-2, 2, 25)
u, v, s0, w0 = plane(KSpaceProjector(kp, eig, rec))
_, _, s1, w1 = plane(KSpaceProjector(kp, eig, rec, weights=np.ones_like(eig)))
I0 = ARPESPlotter(u, v, s0, 0.0).calculate_spectral_density(E)
I1 = ARPESPlotter(u, v, s1, 0.0, weights=w1).calculate_spectral_density(E)

# Unit weights must be a no-op: guards against the weighting silently rescaling output.
check("unit weights reproduce the unweighted intensity", np.allclose(I0, I1, atol=1e-12),
      f"max dev {np.abs(I0-I1).max():.1e}")
# Linearity: zeroing a band removes exactly that band's contribution.
w = np.ones_like(eig); w[0, 1] = 0.0
_, _, s2, w2 = plane(KSpaceProjector(kp, eig, rec, weights=w))
I2 = ARPESPlotter(u, v, s2, 0.0, weights=w2).calculate_spectral_density(E)
only1 = ARPESPlotter(u, v, s0[:, 1:2], 0.0).calculate_spectral_density(E)
check("zeroing a band subtracts exactly that band", np.allclose(I0-I2, only1, atol=1e-10))
# Homogeneity.
_, _, s3, w3 = plane(KSpaceProjector(kp, eig, rec, weights=np.full_like(eig, 2.5)))
I3 = ARPESPlotter(u, v, s3, 0.0, weights=w3).calculate_spectral_density(E)
check("scaling weights scales intensity", np.allclose(I3, 2.5*I0, atol=1e-10))
# Weights must share the energies' convex hull, else the two are misaligned.
check("weight and energy NaN masks coincide", np.array_equal(np.isnan(s1), np.isnan(w1)))
# Shape mismatch must be rejected rather than broadcast.
try:
    KSpaceProjector(kp, eig, rec, weights=np.ones((1, 2, len(kp)))); ok = False
except ValueError:
    ok = True
check("mismatched weight shape is rejected", ok)

# Orbital spec mapping.
fields = ["s", "py", "pz", "px", "dxy", "dyz", "dz2", "dxz", "x2-y2"]
vec = build_orbital_vector(fields, 9, parse_orbital_spec("d:1"))
check("shell 'd' selects all five d orbitals incl. x2-y2",
      vec.tolist() == [0, 0, 0, 0, 1, 1, 1, 1, 1])
vec = build_orbital_vector(fields, 9, parse_orbital_spec("d:1,dz2:3"))
check("exact orbital name overrides its shell", vec[6] == 3.0 and vec[4] == 1.0)
check("no spec means uniform weights",
      build_orbital_vector(fields, 9, None).tolist() == [1]*9)
# Escape-depth weighting decays away from the surface.
sw = surface_ion_weights([0.0, 2.0, 4.0], escape_depth=2.0)
check("surface weights decay with depth", sw[2] > sw[1] > sw[0] and abs(sw[2]-1.0) < 1e-12)

XML = os.path.join(REPO, "040", "vasprun.xml")
if not os.path.exists(XML):
    print("[SKIP] real-data checks (040/vasprun.xml absent)")
else:
    w_s, meta = reduce_projections_xml(XML, parse_orbital_spec("s:1"), n_sets=1, verbose=False)
    w_t, _ = reduce_projections_xml(XML, None, n_sets=1, verbose=False)
    w_d, _ = reduce_projections_xml(XML, parse_orbital_spec("d:1"), n_sets=1, verbose=False)
    check("shape is (nsets, nbands, nkpts)", w_s.shape == (1, 160, 1000), str(w_s.shape))
    # Ground truth read from the raw file text for band 1 / k-point 1.
    check("s weight matches raw-file ground truth", abs(w_s[0, 0, 0] - 0.9671) < 1e-3,
          f"got {w_s[0,0,0]:.4f}")
    check("total weight matches raw-file ground truth", abs(w_t[0, 0, 0] - 0.9671) < 1e-3)
    check("d weight is zero for the pure-s band 1", abs(w_d[0, 0, 0]) < 1e-3)
    check("orbital subset never exceeds the total", bool((w_s <= w_t + 1e-4).all() and (w_d <= w_t + 1e-4).all()))
    check("all weights finite and non-negative", bool(np.isfinite(w_t).all() and w_t.min() >= -1e-3))
    check("d character has real structure", (w_d > 0.01).mean() > 0.5,
          f"{100*(w_d>0.01).mean():.1f}% of states")
    # Noncollinear guard: sets 2-4 are m_x,m_y,m_z and admit negative values.
    w4, _ = reduce_projections_xml(XML, None, n_sets=4, verbose=False)
    check("noncollinear magnetisation sets are excluded by n_sets=1",
          w4.min() < 0 and w_t.min() >= -1e-3,
          f"4 sets span [{w4.min():.3f},{w4.max():.3f}], 1 set [{w_t.min():.3f},{w_t.max():.3f}]")

H5 = os.path.join(REPO, "080", "vaspout.h5")
if not os.path.exists(H5):
    print("[SKIP] HDF5 checks (080/vaspout.h5 absent)")
else:
    import h5py
    from arpes_projector.matrix_elements import reduce_projections_h5
    h_t, hmeta = reduce_projections_h5(H5, None, n_sets=1, verbose=False)
    h_d, _ = reduce_projections_h5(H5, parse_orbital_spec("d:1"), n_sets=1, verbose=False)
    h_s, _ = reduce_projections_h5(H5, parse_orbital_spec("s:1"), n_sets=1, verbose=False)
    check("h5 shape is (nsets, nbands, nkpts)", h_t.shape == (1, 136, 5832), str(h_t.shape))
    check("h5 orbital names read from lchar",
          [n.strip() for n in hmeta["fields"][:4]] == ["s", "py", "pz", "px"])
    # Reference computed directly from the raw 5-D dataset for a slice of k-points.
    with h5py.File(H5, "r") as fh:
        par = fh["results/projectors/par"]
        ref_t = par[0, :, :, :40, :].sum(axis=(0, 1)).T
        lch = [x.decode().strip() for x in fh["results/projectors/lchar"][()]]
        dcols = [i for i, n in enumerate(lch) if n.startswith("d") or n.startswith("x2")]
        ref_d = par[0, :, dcols, :40, :].sum(axis=(0, 1)).T
    check("h5 total matches a direct sum over the raw dataset",
          np.allclose(h_t[0, :, :40], ref_t, atol=1e-5),
          f"max dev {np.abs(h_t[0,:,:40]-ref_t).max():.1e}")
    check("h5 d-shell matches a direct sum (x2-y2 included)",
          np.allclose(h_d[0, :, :40], ref_d, atol=1e-5) and len(dcols) == 5)
    check("h5 orbital subset never exceeds the total",
          bool((h_s <= h_t + 1e-4).all() and (h_d <= h_t + 1e-4).all()))
    h4, _ = reduce_projections_h5(H5, None, n_sets=4, verbose=False)
    check("h5 noncollinear magnetisation sets excluded by n_sets=1",
          h4.min() < 0 and h_t.min() >= -1e-3,
          f"4 sets [{h4.min():.3f},{h4.max():.3f}], 1 set [{h_t.min():.3f},{h_t.max():.3f}]")

# ---------------------------------------------------------------- regressions
# Guards for defects fixed after the initial matrix-element work. Each asserts
# the property the fix restored, not merely that the code runs.

# 1. One shared triangulation must give exactly what per-band interpolators gave.
from scipy.interpolate import LinearNDInterpolator
_rs = np.random.RandomState(0)
_rec = np.array([[1.3607, -0.3693, 0.], [1.3607, 0.3693, 0.], [0., 0., 1.4235]])
_lin = np.linspace(-.45, .45, 7)
_X, _Y, _Z = np.meshgrid(_lin, _lin, _lin, indexing="ij")
_kp = np.stack([_X.ravel(), _Y.ravel(), _Z.ravel()], 1)
_nb = 4
_eig = _rs.rand(1, _nb, len(_kp)) * 4 - 2
_wts = _rs.rand(1, _nb, len(_kp)) + 0.1
for _tag, _W in (("without weights", None), ("with weights", _wts)):
    _pr = KSpaceProjector(_kp, _eig, _rec, weights=_W)
    _u, _v, _s_new, _w_new = _pr.interpolate_plane(
            np.array([0., 0, 1]), np.zeros(3), (-0.8, 0.8), (-0.8, 0.8), 21, 1)
    _n, _p, _uh, _vh = _pr.define_plane_basis(np.array([0., 0, 1]), np.zeros(3))
    _uu, _vv = np.meshgrid(_u, _v)
    _pts = (_p + _uu[..., None] * _uh + _vv[..., None] * _vh).reshape(-1, 3)
    _kc = _kp @ _rec
    _s_ref = np.zeros_like(_s_new)
    for _b in range(_nb):
        _s_ref[0, _b] = LinearNDInterpolator(_kc, _eig[0, _b])(_pts).reshape(21, 21)
    check(f"shared triangulation matches per-band interpolation ({_tag})",
          np.array_equal(np.isnan(_s_new), np.isnan(_s_ref))
          and np.nanmax(np.abs(_s_new - _s_ref)) == 0.0,
          f"max|diff| {np.nanmax(np.abs(_s_new - _s_ref)):.1e}")

# 2. align_vector_to_z must return a proper rotation; -I would mirror the BZ.
try:
    from arpes_projector.surface_bz import SurfaceBZAnalyzer
    _an = SurfaceBZAnalyzer.__new__(SurfaceBZAnalyzer)
    _z = np.array([0., 0, 1])
    _R = _an.align_vector_to_z(-_z)
    _x, _y = np.array([1., 0, 0]), np.array([0, 1., 0])
    check("align_vector_to_z(-z) is a rotation, not a reflection",
          abs(np.linalg.det(_R) - 1.0) < 1e-12, f"det = {np.linalg.det(_R):+.1f}")
    check("align_vector_to_z(-z) maps -z to +z and preserves handedness",
          np.allclose(_R @ (-_z), _z)
          and np.allclose(_R @ np.cross(_x, _y), np.cross(_R @ _x, _R @ _y)))
    check("align_vector_to_z stays proper for general directions",
          all(abs(np.linalg.det(_an.align_vector_to_z(np.array(_v, float))) - 1) < 1e-9
              for _v in ([0, 0, 1], [1, 1, 1], [0, 0, -1], [1e-7, 0, -1])))
except ImportError as _exc:
    print(f"[SKIP] surface_bz checks ({_exc})")

# 3. Degenerate inputs must fail loudly instead of writing a convincing blank figure.
from arpes_projector.cli import build_parser
_parser = build_parser()
for _args, _label in ((["--broadening", "0"], "--broadening 0"),
                      (["--broadening", "-0.1"], "--broadening negative"),
                      (["--resolution", "0"], "--resolution 0"),
                      (["--smooth", "0"], "--smooth 0"),
                      (["--n_energy", "0"], "--n_energy 0")):
    try:
        _parser.parse_args(_args)
        _rejected = False
    except SystemExit:
        _rejected = True
    check(f"{_label} is rejected at parse time", _rejected)
try:
    KSpaceProjector(_kp, _eig, _rec).define_plane_basis(np.zeros(3), np.zeros(3))
    _raised = False
except ValueError:
    _raised = True
check("--normal 0 0 0 raises instead of producing a NaN plane", _raised)

# 4. integrate_v must divide by contributing samples, not by grid height. A flat
#    band viewed along (111) has varying hull coverage, so the old normalisation
#    imprinted a purely geometric gradient.
import matplotlib.pyplot as _plt
_flat = np.zeros((1, 1, len(_kp)))
_pr = KSpaceProjector(_kp, _flat, np.eye(3) * 2.0)
_u, _v, _s, _ = _pr.interpolate_plane(np.array([1., 1, 1]), np.zeros(3),
                                      (-1.6, 1.6), (-1.6, 1.6), 41, 1)
_captured = {}
_orig = _plt.Axes.pcolormesh
def _spy(self, *a, **k):
    # The colorbar draws its own pcolormesh afterwards; keep only the first.
    if "C" not in _captured and len(a) >= 3:
        _captured["C"] = np.asarray(a[2])
    return _orig(self, *a, **k)
_plt.Axes.pcolormesh = _spy
try:
    ARPESPlotter(_u, _v, _s, 0.0).plot_dispersion_slice(
            0.0, integrate_v=True, energy_limits=(-0.3, 0.3), n_energy_points=7,
            filename=os.path.join(REPO, "tests", "_tmp_integrate_v.png"))
finally:
    _plt.Axes.pcolormesh = _orig
    _tmp = os.path.join(REPO, "tests", "_tmp_integrate_v.png")
    if os.path.exists(_tmp):
        os.remove(_tmp)
_cov = np.isfinite(_s[0]).any(axis=0).sum(axis=0)
_row = _captured["C"][3][_cov > 0]
_spread = (_row.max() - _row.min()) / max(_row.mean(), 1e-30)
check("integrate_v is flat for a flat band despite varying hull coverage",
      _spread < 1e-9,
      f"coverage {_cov[_cov>0].min()}..{_cov.max()} of {len(_v)}, relative spread {_spread:.1e}")


# --- Fermi-Dirac occupation behind --temperature ------------------------------
from arpes_projector.plotter import K_BOLTZMANN_EV
from arpes_projector.cli import build_parser as _bp

# The whole point of the flag is that it is opt-in: without it, nothing moves.
_I_default = ARPESPlotter(u, v, s0, 0.0).calculate_spectral_density(E)
check("no --temperature reproduces the previous intensity bit-identically",
      np.array_equal(_I_default, I0))
check("temperature defaults to None on the parser",
      _bp().parse_args([]).temperature is None)

_T = 300.0
_pT = ARPESPlotter(u, v, s0, 0.0, temperature=_T)
_I_T = _pT.calculate_spectral_density(E)
_f = 1.0 / (1.0 + np.exp(E / (K_BOLTZMANN_EV * _T)))
# f multiplies the spectral function at the probed energy, so the whole
# constant-energy plane is scaled by one scalar per energy.
check("intensity is exactly f(E,T) times the unoccupied intensity",
      np.allclose(_I_T, I0 * _f[:, None, None], rtol=0, atol=1e-12),
      f"max dev {np.abs(_I_T - I0*_f[:,None,None]).max():.1e}")
check("occupation is 1/2 exactly at the Fermi level",
      abs(float(_pT._occupation(np.array([0.0]))[0]) - 0.5) < 1e-15)
_o = _pT._occupation(np.array([-1.0, -0.01, 0.0, 0.01, 1.0]))
check("occupation decreases monotonically through E_F", bool(np.all(np.diff(_o) < 0)))
check("occupation saturates at 1 below and 0 above",
      _o[0] > 1 - 1e-12 and _o[-1] < 1e-12)

# 50 eV at 1 K is E/kT ~ 6e5: a naive exp() overflows to inf and the ratio to NaN.
_cold = ARPESPlotter(u, v, s0, 0.0, temperature=1.0)._occupation(np.array([-50.0, 50.0]))
check("no overflow for |E| >> kT", bool(np.all(np.isfinite(_cold))) and _cold[1] == 0.0,
      f"f(+50 eV, 1 K) = {_cold[1]:g}")

_zero = ARPESPlotter(u, v, s0, 0.0, temperature=0.0)._occupation(np.array([-0.1, 0.0, 0.1]))
check("T = 0 K is a step with 1/2 at E_F", np.array_equal(_zero, np.array([1.0, 0.5, 0.0])))

try:
    ARPESPlotter(u, v, s0, 0.0, temperature=-1.0); _ok = False
except ValueError:
    _ok = True
check("a negative temperature is rejected by the plotter", _ok)
try:
    _bp().parse_args(["--temperature", "-5"]); _ok = False
except SystemExit:
    _ok = True
check("--temperature -5 is rejected at parse time", _ok)

# The dispersion slice must pick up the same cutoff; capture what is drawn.
def _draw(**kw):
    cap = {}
    orig = _plt.Axes.pcolormesh
    def spy(self, *a, **k):
        if "C" not in cap and len(a) >= 3:
            cap["C"] = np.asarray(a[2])
        return orig(self, *a, **k)
    _plt.Axes.pcolormesh = spy
    out = os.path.join(REPO, "tests", "_tmp_temperature.png")
    try:
        ARPESPlotter(u, v, s0, 0.0, **kw).plot_dispersion_slice(
                0.0, energy_limits=(-1.0, 1.0), n_energy_points=9, filename=out)
    finally:
        _plt.Axes.pcolormesh = orig
        if os.path.exists(out):
            os.remove(out)
    return cap["C"]

_Ed = np.linspace(-1.0, 1.0, 9)
_fd = 1.0 / (1.0 + np.exp(_Ed / (K_BOLTZMANN_EV * _T)))
_d0, _dT = _draw(), _draw(temperature=_T)
check("dispersion slice applies the same f(E,T)",
      np.allclose(_dT, _d0 * _fd[:, None], rtol=0, atol=1e-12),
      f"max dev {np.abs(_dT - _d0*_fd[:,None]).max():.1e}")
check("dispersion slice is unchanged without --temperature",
      np.array_equal(_d0, _draw()))


# --- streaming vasprun.xml parser -------------------------------------------
_XML = os.path.join(REPO, "040", "vasprun.xml")
if os.path.exists(_XML):
    from arpes_projector.parser import VaspDataParser
    _p = VaspDataParser(_XML)
    try:
        _ref = _p._parse_xml_pymatgen()
    except ImportError:
        _ref = None
    _str = _p._parse_xml_stream()
    if _ref is not None:
        # The streaming parser exists to avoid holding a multi-GB DOM; it is only
        # useful if it reproduces the DOM parser exactly.
        check("streaming parser reproduces pymatgen eigenvalues exactly",
              np.array_equal(np.asarray(_ref["eigenvalues"]), np.asarray(_str["eigenvalues"])))
        check("streaming parser reproduces pymatgen k-points exactly",
              np.array_equal(np.asarray(_ref["kpoints"]), np.asarray(_str["kpoints"])))
        check("streaming parser agrees on E_F and spin polarisation",
              _ref["efermi"] == _str["efermi"]
              and bool(_ref["is_spin_polarized"]) == bool(_str["is_spin_polarized"]))
        _dev = np.abs(np.asarray(_ref["rec_lattice"]) - np.asarray(_str["rec_lattice"])).max()
        # The XML prints the reciprocal basis to ~8 digits; that is the only
        # discrepancy allowed here.
        check("streaming parser agrees on the reciprocal lattice", _dev < 1e-6,
              f"max dev {_dev:.1e}")
    # The cache must never outlive the file it describes.
    _cp = os.path.join(REPO, "tests", "_tmp_cache.npz")
    VaspDataParser._write_cache(_cp, _str)
    _sp = VaspDataParser(_XML)
    _fresh = _sp._load_cache(_cp)
    check("a cache newer than the input is reused and round-trips",
          _fresh is not None
          and np.array_equal(_fresh["eigenvalues"], np.asarray(_str["eigenvalues"]))
          and _fresh["efermi"] == _str["efermi"])
    # Back-date it to one second before the input: the parse must be redone.
    _in_mtime = os.path.getmtime(_XML)
    os.utime(_cp, (_in_mtime - 1.0, _in_mtime - 1.0))
    check("a cache older than the input is refused", _sp._load_cache(_cp) is None)
    # Equal timestamps are ambiguous, so they must resolve toward re-parsing.
    os.utime(_cp, (_in_mtime, _in_mtime))
    check("a cache with the input's own timestamp is refused",
          _sp._load_cache(_cp) is None)
    os.remove(_cp)
else:
    print("[SKIP] streaming-parser checks (040/vasprun.xml absent)")


# --- filename tags ------------------------------------------------------------
import importlib.util as _ilu
_spec = _ilu.spec_from_file_location("_arpes_main", os.path.join(REPO, "arpes.py"))
_am = _ilu.module_from_spec(_spec); _spec.loader.exec_module(_am)
from arpes_projector.cli import build_parser as _bp2

def _tag(argv):
    return _am.figure_tag(_bp2().parse_args(argv), "E+0.00")

_plain = _tag([])
# Runs using none of the optional knobs must keep the names they already have.
check("the plain tag is unchanged by the shared helper", _plain == "E+0.00_g0.05_linear")
# Every knob that changes the figure must change the name.
_variants = {
        "plain": _plain,
        "temperature": _tag(["--temperature", "300"]),
        "orbital": _tag(["--orbital_weights", "d:1"]),
        "other orbital": _tag(["--orbital_weights", "p:1"]),
        "ion": _tag(["--ion_weights", "1,0,0"]),
        "bare --matrix_elements": _tag(["--matrix_elements"]),
        "broadening": _tag(["--broadening", "0.1"]),
        "cscale": _tag(["--cscale", "log"]),
        }
check("every figure-changing flag yields a distinct filename tag",
      len(set(_variants.values())) == len(_variants),
      f"{len(set(_variants.values()))} distinct of {len(_variants)}")
# This is the collision the multi-mode run actually hit: a d-weighted run wrote
# the same names as an unweighted one.
check("a weighted run no longer collides with an unweighted one",
      _variants["orbital"] != _plain, _variants["orbital"])
check("tags contain no path or shell-hostile characters",
      all(re.fullmatch(r"[0-9A-Za-z.+_-]+", t) for t in _variants.values()))
# --mock cannot supply projections, so a requested weighting is dropped with a
# warning; the name must not then claim a weighting the figure does not carry.
check("a dropped weighting is not advertised in the filename",
      _am.figure_tag(_bp2().parse_args(["--orbital_weights", "d:1"]), "E+0.00", False) == _plain)
check("an applied weighting is still advertised",
      _am.figure_tag(_bp2().parse_args(["--orbital_weights", "d:1"]), "E+0.00", True) != _plain)
# A long ion list must stay short without silently colliding.
_long_a = _tag(["--ion_weights", ",".join(["0.5"] * 40)])
_long_b = _tag(["--ion_weights", ",".join(["0.25"] * 40)])
check("a long weight spec is shortened but stays unique",
      _long_a != _long_b and len(_long_a) < 60, f"len {len(_long_a)}")
# The surface index defines the surface bands, so it must be in their name.
_ns = _bp2().parse_args(["--miller_surf", "1", "1", "1"])
_no = _bp2().parse_args(["--miller_surf", "0", "0", "1"])
_sbz = lambda a: "m" + "".join(f"{int(m):+d}" for m in a.miller_surf)
check("surface_bands names distinguish the Miller index", _sbz(_ns) != _sbz(_no),
      f"{_sbz(_ns)} vs {_sbz(_no)}")
check("negative Miller indices stay unambiguous",
      _sbz(_bp2().parse_args(["--miller_surf", "-1", "1", "1"]))
      != _sbz(_bp2().parse_args(["--miller_surf", "1", "-1", "1"])))


# --- cloud-derived plane bounds ----------------------------------------------
_prj = KSpaceProjector(kp, eig, rec)
_n = np.array([0., 0, 1])
_ur, _vr = _prj.suggest_plane_bounds(_n, np.zeros(3))
_nh, _pc, _uh, _vh = _prj.define_plane_basis(_n, np.zeros(3))
_sec = _prj._hull_plane_section(_nh, _pc)
check("the plane's hull cross-section is found", _sec is not None and len(_sec) >= 3,
      f"{0 if _sec is None else len(_sec)} crossings")
_su, _sv = (_sec - _pc) @ _uh, (_sec - _pc) @ _vh
check("derived bounds contain the whole hull cross-section",
      _ur[0] <= _su.min() and _su.max() <= _ur[1]
      and _vr[0] <= _sv.min() and _sv.max() <= _vr[1])
check("derived bounds are padded, not exactly touching",
      _ur[0] < _su.min() and _su.max() < _ur[1])
# The real guarantee is that nothing carrying data falls outside. Interpolate on
# a window twice as wide and confirm every finite pixel sits inside the bounds.
_wide_u = (_ur[0] * 2, _ur[1] * 2)
_wide_v = (_vr[0] * 2, _vr[1] * 2)
_res = 61
_wide = _prj.interpolate_plane(_n, np.zeros(3), _wide_u, _wide_v, _res, 1)[2][0, 0]
_gu = np.linspace(_wide_u[0], _wide_u[1], _res)
_gv = np.linspace(_wide_v[0], _wide_v[1], _res)
_fv, _fu = np.nonzero(np.isfinite(_wide))
check("no pixel carrying data falls outside the derived bounds",
      _gu[_fu].min() >= _ur[0] and _gu[_fu].max() <= _ur[1]
      and _gv[_fv].min() >= _vr[0] and _gv[_fv].max() <= _vr[1],
      f"data spans u [{_gu[_fu].min():.3f},{_gu[_fu].max():.3f}] "
      f"within [{_ur[0]:.3f},{_ur[1]:.3f}]")
# An oblique plane cuts a much smaller polygon than the cloud's full shadow;
# that gap is the whole reason for sectioning the hull rather than projecting.
_no = np.array([1., 1, 1])
_ou, _ov = _prj.suggest_plane_bounds(_no, np.zeros(3))
_onh, _opc, _ouh, _ovh = _prj.define_plane_basis(_no, np.zeros(3))
_proj_u = _prj.kpoints_cart @ _ouh
_area = (_ou[1] - _ou[0]) * (_ov[1] - _ov[0])
_shadow = (_proj_u.max() - _proj_u.min()) * (
        (_prj.kpoints_cart @ _ovh).max() - (_prj.kpoints_cart @ _ovh).min())
check("an oblique plane gets a tighter window than the cloud's shadow",
      _area < 0.95 * _shadow, f"{_area / _shadow:.2f} of the projected area")
# The whole point: a window tracking the cloud beats a fixed +/-2.
_cov = lambda ur, vr: 1.0 - float(np.isnan(
        _prj.interpolate_plane(_n, np.zeros(3), ur, vr, 30, 1)[2][0, 0]).mean())
_c_auto, _c_fixed = _cov(_ur, _vr), _cov((-2., 2.), (-2., 2.))
check("derived bounds cover more of the cloud than the fixed +/-2 window",
      _c_auto > 3 * _c_fixed, f"{_c_auto:.0%} vs {_c_fixed:.0%} of the window carries data")
# A cloud with no extent must not produce a zero-width or NaN window.
_deg = KSpaceProjector(np.zeros((1, 3)), np.zeros((1, 1, 1)), rec)
_du, _dv = _deg.suggest_plane_bounds(_n, np.zeros(3))
check("a degenerate cloud falls back to a finite window",
      all(np.isfinite(b) for b in _du + _dv) and _du[1] > _du[0] and _dv[1] > _dv[0],
      f"u {_du}, v {_dv}")

# Explicit bounds must win, and each axis is resolved independently.
_ns_auto = _bp2().parse_args([])
_ns_u = _bp2().parse_args(["--ubounds", "-1", "1"])
check("--ubounds and --vbounds default to None so 'unset' is distinguishable",
      _ns_auto.ubounds is None and _ns_auto.vbounds is None)
_ru, _rv = _am.resolve_bounds(_prj, _ns_u, _n)
check("an explicit --ubounds wins while v is still derived",
      _ru == (-1.0, 1.0) and _rv == _vr, f"u {_ru}, v {_rv}")
_ru2, _rv2 = _am.resolve_bounds(_prj, _bp2().parse_args(
        ["--ubounds", "-1", "1", "--vbounds", "-2", "2"]), _n)
check("both explicit bounds are passed through untouched",
      _ru2 == (-1.0, 1.0) and _rv2 == (-2.0, 2.0))
# Derived bounds are a deterministic function of the input, so they need no tag;
# explicitly chosen windows can differ between runs and must be named.
check("derived bounds add nothing to the filename tag",
      _am.figure_tag(_ns_auto, "E+0.00") == _plain)
# A leading minus is meaningful and must survive: "-1 2" and "1 2" are different
# windows, and an earlier version stripped the sign and named them alike.
check("a negative bound keeps its sign in the tag",
      _am.figure_tag(_bp2().parse_args(["--ubounds", "-1", "2"]), "E+0.00")
      != _am.figure_tag(_bp2().parse_args(["--ubounds", "1", "2"]), "E+0.00"),
      _am.figure_tag(_bp2().parse_args(["--ubounds", "-1", "2"]), "E+0.00"))
# argparse reads a bare "-1,1" as an option, so a negative spec needs the "="
# form; the slug must still keep the sign either way.
check("a negative ion weight keeps its sign in the tag",
      _am.figure_tag(_bp2().parse_args(["--ion_weights=-1,1"]), "E+0.00")
      != _am.figure_tag(_bp2().parse_args(["--ion_weights=1,1"]), "E+0.00"))
check("explicit bounds appear in the filename tag",
      _am.figure_tag(_ns_u, "E+0.00") != _plain
      and _am.figure_tag(_ns_u, "E+0.00") != _am.figure_tag(
              _bp2().parse_args(["--ubounds", "-3", "3"]), "E+0.00"))


print("\n" + ("ALL CHECKS PASSED" if not FAIL else f"{len(FAIL)} FAILED: {FAIL}"))
sys.exit(1 if FAIL else 0)
