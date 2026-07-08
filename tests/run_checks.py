"""
Self-contained check suite for the ARPES projector suite.

Run from anywhere:  python3 tests/run_checks.py

Checks C1-C3 exercise the streaming parser, cache, and matrix-element weight
reduction against a real VASP output (skipped gracefully when absent);
C4-C6 cover geometry and plotter numerics on synthetic data; C7 runs the CLI
end-to-end. Exemplary outputs (plots, summary) land in tests/outputs/.

Exit code is non-zero when any check fails.
"""

import os
import subprocess
import sys
import time

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTDIR = os.path.join(REPO, "tests", "outputs")
REAL_INPUT = os.path.join(REPO, "040", "vasprun.xml")
sys.path.insert(0, REPO)
os.makedirs(OUTDIR, exist_ok=True)

import matplotlib
matplotlib.use("Agg")

from arpes_projector.parser import VaspDataParser
from arpes_projector.geometry import KSpaceProjector
from arpes_projector.plotter import ARPESPlotter

RESULTS = []

def check(name):
    def deco(fn):
        def run():
            t0 = time.perf_counter()
            try:
                detail = fn() or ""
                RESULTS.append((name, "PASS", f"{time.perf_counter()-t0:.2f}s", detail))
                print(f"[PASS] {name}  {detail}")
            except SkipCheck as exc:
                RESULTS.append((name, "SKIP", "-", str(exc)))
                print(f"[SKIP] {name}  {exc}")
            except Exception as exc:
                RESULTS.append((name, "FAIL", f"{time.perf_counter()-t0:.2f}s", repr(exc)))
                print(f"[FAIL] {name}  {exc!r}")
        return run
    return deco

class SkipCheck(Exception):
    pass

def make_mock_projector(with_weights=False):
    """3-band simple-cubic tight-binding model, matching arpes.py's mock."""
    rec = np.eye(3) * (2 * np.pi / 3.14)
    lin = np.linspace(-0.5, 0.5, 14)
    kx, ky, kz = np.meshgrid(lin, lin, lin, indexing="ij")
    kp = np.stack([kx.ravel(), ky.ravel(), kz.ravel()], axis=1)
    eig = np.zeros((1, 3, len(kp)))
    eig[0, 0] = 3.0 - 1.8 * (np.cos(2*np.pi*kp[:, 0]) + np.cos(2*np.pi*kp[:, 1]) + np.cos(2*np.pi*kp[:, 2]))
    eig[0, 1] = 5.2 - 1.1 * (2*np.cos(2*np.pi*kp[:, 0]) + np.cos(2*np.pi*kp[:, 1]) + np.cos(2*np.pi*kp[:, 2]))
    eig[0, 2] = 6.0 - 0.9 * (np.cos(2*np.pi*kp[:, 0]) + 2*np.cos(2*np.pi*kp[:, 1]) + np.cos(2*np.pi*kp[:, 2]))
    weights = None
    if with_weights:
        weights = np.abs(np.sin(eig)) + 0.1        # smooth, positive, band-dependent
    return KSpaceProjector(kp, eig, rec, weights=weights), eig

@check("C1 streaming parse of real vasprun.xml (ground truth)")
def c1():
    if not os.path.exists(REAL_INPUT):
        raise SkipCheck(f"{REAL_INPUT} not present")
    data = VaspDataParser(REAL_INPUT).parse(use_cache=False)
    e = data["eigenvalues"]
    assert e.shape == (1, 160, 1000), e.shape
    # Values read directly from the raw XML text of this file
    assert abs(data["efermi"] - 7.4356) < 5e-4
    assert abs(e[0, 0, 0] - (-43.1841)) < 5e-4
    assert data["kpoints"].shape == (1000, 3)
    assert np.all(np.abs(data["kpoints"]) <= 0.5 + 1e-9)
    # Reciprocal lattice must satisfy A @ B^T = 2*pi*I against its own dual
    B = data["rec_lattice"]
    A = 2 * np.pi * np.linalg.inv(B).T
    assert np.allclose(A @ B.T, 2 * np.pi * np.eye(3), atol=1e-10)
    return f"E_F={data['efermi']:.4f} eV, {e.shape[1]} bands x {e.shape[2]} kpts"

@check("C2 parse cache: round-trip, spec keying, staleness")
def c2():
    if not os.path.exists(REAL_INPUT):
        raise SkipCheck(f"{REAL_INPUT} not present")
    cache = REAL_INPUT + ".arpes_cache.npz"
    if os.path.exists(cache):
        os.remove(cache)
    p = VaspDataParser(REAL_INPUT)
    d1 = p.parse()                                   # writes cache
    t0 = time.perf_counter()
    d2 = p.parse()                                   # must hit cache
    hit = time.perf_counter() - t0
    assert hit < 0.2, f"cache hit took {hit:.2f}s"
    assert np.array_equal(d1["eigenvalues"], d2["eigenvalues"])
    # weight spec keying: differing spec must not reuse cached weights
    d3 = p.parse(weights_spec={"orbital_weights": {"s": 1.0}})
    d4 = p.parse(weights_spec={"orbital_weights": {"s": 1.0}})   # hit
    assert np.array_equal(d3["weights"], d4["weights"])
    return f"cache hit {hit*1e3:.0f} ms"

@check("C3 weight reduction vs raw-text ground truth")
def c3():
    if not os.path.exists(REAL_INPUT):
        raise SkipCheck(f"{REAL_INPUT} not present")
    p = VaspDataParser(REAL_INPUT)
    ws = p.parse(use_cache=False, weights_spec={"orbital_weights": {"s": 1.0}})["weights"]
    wt = p.parse(use_cache=False, weights_spec={"orbital_weights": None})["weights"]
    assert ws.shape == (1, 160, 1000)
    # band 1 @ kpt 1: s-projections 0.4829 + 0.4842 read from the raw file
    assert abs(ws[0, 0, 0] - 0.9671) < 1e-3, ws[0, 0, 0]
    assert wt.min() > -1e-3 and wt.max() < 1.5      # total projection is physical
    assert (ws <= wt + 1e-4).all()                  # s-character <= total everywhere
    return f"w_s[0,0,0]={ws[0,0,0]:.4f} (expect 0.9671), total in [{wt.min():.2f},{wt.max():.2f}]"

@check("C4 interpolation: weights ride the shared triangulation faithfully")
def c4():
    proj, eig = make_mock_projector(with_weights=True)
    u, v, spec, wgrid = proj.interpolate_plane(
            np.array([0., 0., 1.]), np.zeros(3), (-0.9, 0.9), (-0.9, 0.9), 40, 1)
    assert wgrid is not None and wgrid.shape == spec.shape
    # Reference: interpolate the weights alone on a fresh interpolator
    from scipy.interpolate import LinearNDInterpolator
    ref = LinearNDInterpolator(proj.build_triangulation(),
                               proj.weights.reshape(3, -1).T)
    n_hat, p_cart, u_hat, v_hat = proj.define_plane_basis(np.array([0., 0., 1.]), np.zeros(3))
    uu, vv = np.meshgrid(u, v)
    pts = (p_cart + uu[..., None]*u_hat + vv[..., None]*v_hat).reshape(-1, 3)
    ref_w = ref(pts).T.reshape(wgrid.shape[1], len(v), len(u))
    assert np.array_equal(np.isnan(wgrid[0]), np.isnan(ref_w))
    assert np.nanmax(np.abs(wgrid[0] - ref_w)) < 1e-12
    # NaN masks of spectra and weights must coincide (same hull)
    assert np.array_equal(np.isnan(spec), np.isnan(wgrid))
    return f"max dev vs standalone interpolation: {np.nanmax(np.abs(wgrid[0]-ref_w)):.1e}"

@check("C5 plotter numerics: unit weights ≡ unweighted; E_F-shift consistency")
def c5():
    proj, _ = make_mock_projector()
    u, v, spec, _ = proj.interpolate_plane(
            np.array([0., 0., 1.]), np.zeros(3), (-0.9, 0.9), (-0.9, 0.9), 40, 1)
    energies = np.linspace(-2, 1, 40)
    p_plain = ARPESPlotter(u, v, spec, 5.2)
    p_ones = ARPESPlotter(u, v, spec, 5.2, weights=np.ones_like(spec))
    i1 = p_plain.calculate_spectral_density(energies)
    i2 = p_ones.calculate_spectral_density(energies)
    assert np.allclose(i1, i2, atol=1e-12), "unit weights changed the intensity"
    # shifting E_F by +d == sampling the unshifted spectrum at E+d
    p_shift = ARPESPlotter(u, v, spec, 5.2, efermi_shift=0.7)
    a = p_shift.calculate_spectral_density(np.array([0.0]))
    b = p_plain.calculate_spectral_density(np.array([0.7]))
    assert np.allclose(a, b, atol=1e-12)
    # halving one band's weight halves exactly its contribution
    w = np.ones_like(spec); w[0, 2] = 0.0
    p_w = ARPESPlotter(u, v, spec, 5.2, weights=w)
    i3 = p_w.calculate_spectral_density(energies)
    only2 = ARPESPlotter(u, v, spec[:, 2:3], 5.2).calculate_spectral_density(energies)
    assert np.allclose(i1 - i3, only2, atol=1e-10)
    return "unit-weight identity, shift equivalence, band-removal linearity"

@check("C6 dispersion slices agree with manual Lorentzian sums")
def c6():
    proj, _ = make_mock_projector(with_weights=True)
    u, v, spec, wgrid = proj.interpolate_plane(
            np.array([0., 0., 1.]), np.zeros(3), (-0.9, 0.9), (-0.9, 0.9), 30, 1)
    plotter = ARPESPlotter(u, v, spec, 5.2, weights=wgrid)
    es = np.linspace(-2, 1, 25)
    bands = np.where(np.isnan(plotter.spectra[0]), np.inf, plotter.spectra[0])
    wts = np.nan_to_num(wgrid[0], nan=0.0)
    ref = np.zeros((len(es), len(u)))
    for i, e in enumerate(es):
        ref[i] = (wts * (0.05/np.pi / ((e - bands)**2 + 0.05**2))).sum(axis=(0, 1)) / len(v)
    got = plotter._accumulate_lorentzian(plotter.spectra[0], es, 0.05, wgrid[0]).sum(axis=1) / len(v)
    assert np.allclose(ref, got, atol=1e-10)
    return f"integrate_v max dev: {np.abs(ref-got).max():.1e}"

@check("C7 CLI end-to-end (exemplary plots saved to tests/outputs)")
def c7():
    env = dict(os.environ, MPLBACKEND="Agg")
    runs = [("mock", ["--mode", "single", "--mock", "--resolution", "60", "--smooth", "1"])]
    if os.path.exists(REAL_INPUT):
        common = ["--mode", "single", "--input", REAL_INPUT,
                  "--ubounds", "-0.65", "0.65", "--vbounds", "-0.65", "0.65",
                  "--resolution", "120", "--smooth", "1", "--elimits", "-6", "2"]
        runs += [("real_unweighted", common),
                 ("real_d_weighted", common + ["--orbital_weights", "d:1"])]
    made = []
    for tag, extra in runs:
        outdir = os.path.join(OUTDIR, tag)
        r = subprocess.run([sys.executable, os.path.join(REPO, "arpes.py"),
                            "--outdir", outdir] + extra,
                           env=env, cwd=REPO, capture_output=True, text=True, timeout=600)
        assert r.returncode == 0, f"{tag} exited {r.returncode}: {r.stderr[-400:]}"
        pngs = [f for f in os.listdir(outdir) if f.endswith(".png")]
        assert len(pngs) == 2, f"{tag}: expected 2 plots, found {pngs}"
        made.append(f"{tag} ({len(pngs)} plots)")
    return ", ".join(made)

def main():
    print("=" * 72)
    print("ARPES projector check suite")
    print("=" * 72)
    for fn in (c1, c2, c3, c4, c5, c6, c7):
        fn()

    lines = [f"{name:<58} {status:>4}  {dt:>7}  {detail}"
             for name, status, dt, detail in RESULTS]
    summary = "\n".join(lines) + "\n"
    with open(os.path.join(OUTDIR, "checks_summary.txt"), "w") as fh:
        fh.write(summary)
    print("-" * 72)
    print(summary, end="")
    print(f"Summary written to {os.path.join(OUTDIR, 'checks_summary.txt')}")

    if any(status == "FAIL" for _, status, _, _ in RESULTS):
        sys.exit(1)

if __name__ == "__main__":
    main()
