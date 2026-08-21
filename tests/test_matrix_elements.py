"""
Checks for matrix-element intensity weighting.

Run:  python3 tests/test_matrix_elements.py
Real-data checks are skipped when 040/vasprun.xml is absent.
Exit code is non-zero if any check fails.
"""
import os, sys, numpy as np, matplotlib
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

print("\n" + ("ALL CHECKS PASSED" if not FAIL else f"{len(FAIL)} FAILED: {FAIL}"))
sys.exit(1 if FAIL else 0)
