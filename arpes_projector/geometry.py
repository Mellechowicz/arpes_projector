"""
This file contains the KSpaceProjector class.
It transforms coordinates, defines projection planes,
and executes multidimensional interpolation of electronic band structures.

Inputs:
 - kpoints: Array of fractional k-points coordinates.
 - eigenvalues: Array of electronic eigenvalues.
 - rec_lattice: Reciprocal lattice matrix.
 - normal_frac: Fractional normal vector defining the projection plane.
 - point_frac: Fractional vector representing a shift point on the plane.
 - u_range: Coordinate bounds for the in-plane u axis.
 - v_range: Coordinate bounds for the in-plane v axis.
 - grid_resolution: Integer specifying grid point count.
 - interpolate_factor: Integer specifying the scaling factor for smoothing.

Outputs:
 - Orthonormal basis vectors (n_hat, p_cart, u_hat, v_hat).
 - Two-dimensional interpolation grids (u_grid, v_grid).
 - Interpolated eigenvalue spectra arrays.

Approach and Modules:
 - Orthogonalization: Gram-Schmidt process via numpy.
 - Coordinate transformation: Matrix multiplication via numpy.
 - Interpolation: Linear multidimensional triangulation via scipy.interpolate.LinearNDInterpolator.
"""

import numpy as np
from scipy.interpolate import LinearNDInterpolator
from scipy.spatial import ConvexHull, Delaunay
from typing import Tuple

class KSpaceProjector:
    """Performs coordinates transformation, plane projection, and multidimensional interpolation."""

    def __init__(self, kpoints: np.ndarray, eigenvalues: np.ndarray, rec_lattice: np.ndarray,
                 weights: np.ndarray = None):
        """
        Initialize the projector.

        Args:
            kpoints (np.ndarray): Fractional k-points coordinates, shape (nkpts, 3).
            eigenvalues (np.ndarray): Eigenvalues array, shape (nspins, nbands, nkpts).
            rec_lattice (np.ndarray): Reciprocal lattice matrix, shape (3, 3).
        """
        if weights is not None and weights.shape != eigenvalues.shape:
            raise ValueError(f"weights shape {weights.shape} does not match "
                             f"eigenvalues shape {eigenvalues.shape}")
        self.kpoints_frac = kpoints
        self.eigenvalues = eigenvalues
        self.weights = weights
        self.rec_lattice = rec_lattice
        # Transform fractional k-points to Cartesian coordinates (A^-1)
        self.kpoints_cart = np.dot(kpoints, rec_lattice)
        # Lazily-built triangulation shared by every band, spin and weight column
        self._triangulation = None
        self._hull = None

    def build_triangulation(self) -> Delaunay:
        """
        Builds (once) and returns the Delaunay triangulation of the k-point cloud.

        Holding it explicitly lets every interpolated quantity reuse one
        triangulation - and, more importantly, one qhull point-location
        structure - instead of rebuilding both per band.
        """
        if self._triangulation is None:
            self._triangulation = Delaunay(self.kpoints_cart)
        return self._triangulation

    def convex_hull(self) -> ConvexHull:
        """
        Builds (once) and returns the convex hull of the k-point cloud.

        Separate from the Delaunay triangulation: the hull has a handful of
        facets where the triangulation has one simplex per mesh cell, so
        intersecting a plane with the hull is cheap.
        """
        if self._hull is None:
            self._hull = ConvexHull(self.kpoints_cart)
        return self._hull

    def _hull_plane_section(self, n_hat: np.ndarray, p_cart: np.ndarray) -> np.ndarray:
        """
        Returns the points where the plane crosses the convex hull's edges.

        Those crossings are the vertices of the hull's cross-section, which is
        exactly the region that can carry interpolated data. Returns None when
        the plane misses the hull or only grazes it.

        Args:
            n_hat (np.ndarray): Unit plane normal, Cartesian.
            p_cart (np.ndarray): A point on the plane, Cartesian.

        Returns:
            np.ndarray: Crossing points, shape (n, 3), or None.
        """
        try:
            hull = self.convex_hull()
        except Exception:
            # A degenerate cloud (coplanar or collinear points) has no 3D hull.
            return None

        signed = (self.kpoints_cart - p_cart) @ n_hat
        edges = set()
        for simplex in hull.simplices:
            for i in range(len(simplex)):
                a, b = simplex[i], simplex[(i + 1) % len(simplex)]
                edges.add((a, b) if a < b else (b, a))

        crossings = []
        for a, b in edges:
            da, db = signed[a], signed[b]
            if (da <= 0.0 <= db) or (db <= 0.0 <= da):
                if da == db:
                    # Edge lies in the plane; both endpoints are crossings.
                    crossings.extend((self.kpoints_cart[a], self.kpoints_cart[b]))
                    continue
                t = da / (da - db)
                crossings.append(self.kpoints_cart[a]
                                 + t * (self.kpoints_cart[b] - self.kpoints_cart[a]))
        return np.array(crossings) if len(crossings) >= 3 else None

    def define_plane_basis(self, normal_frac: np.ndarray, point_frac: np.ndarray, u_dir_cart: np.ndarray = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Constructs an orthonormal basis set for the specified projection plane.

        Args:
            normal_frac (np.ndarray): Normal vector in fractional coordinates.
            point_frac (np.ndarray): Shift point on the plane in fractional coordinates.
            u_dir_cart (np.ndarray, optional): Optional Cartesian vector to guide the u-axis direction. If None, an arbitrary orthogonal vector is generated.

        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]: n_hat, p_cart, u_hat, v_hat.
        """
        n_cart = np.dot(normal_frac, self.rec_lattice)
        p_cart = np.dot(point_frac, self.rec_lattice)

        # A zero (or unnormalisable) normal would make n_hat NaN, propagate NaN
        # through the whole grid and yield a uniform blank figure with exit 0.
        n_norm = np.linalg.norm(n_cart)
        if not np.isfinite(n_norm) or n_norm < 1e-12:
            raise ValueError(
                    f"the plane normal {np.asarray(normal_frac).tolist()} maps to a "
                    f"zero-length reciprocal vector; it does not define a plane")
        n_hat = n_cart / n_norm

        # Generate orthogonal vectors on the plane via Gram-Schmidt
        # Use a non-collinear starting vector
        if u_dir_cart is not None:
            u_cart = u_dir_cart - np.dot(u_dir_cart, n_hat) * n_hat
        else:
            aux_vec = np.array([1.0, 0.0, 0.0]) if np.abs(n_hat[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
            u_cart = aux_vec - np.dot(aux_vec, n_hat) * n_hat

        u_norm = np.linalg.norm(u_cart)
        if not np.isfinite(u_norm) or u_norm < 1e-12:
            raise ValueError("the requested in-plane direction is collinear with the "
                             "plane normal, so no in-plane axis can be built")
        u_hat = u_cart / u_norm
        v_hat = np.cross(n_hat, u_hat)

        return n_hat, p_cart, u_hat, v_hat

    def suggest_plane_bounds(self, normal_frac: np.ndarray, point_frac: np.ndarray,
                             u_dir_cart: np.ndarray = None, margin: float = 0.05
                             ) -> Tuple[Tuple[float, float], Tuple[float, float]]:
        """
        Proposes in-plane bounds covering the k-point cloud's footprint.

        A fixed +/-2 A^-1 window is unrelated to any particular calculation. For
        a cloud reaching only +/-0.33 A^-1 along v it leaves ~95% of the figure
        outside the convex hull, drawn as zero intensity and indistinguishable
        from a genuine absence of spectral weight.

        The window is the bounding box of the convex hull's cross-section by
        this plane - the exact region that can carry interpolated data - so it
        never crops real data. When the plane misses the hull, or the cloud is
        too degenerate to have one, it falls back to projecting every k-point,
        which bounds the same region from above.

        Args:
            normal_frac (np.ndarray): Fractional normal vector defining the plane.
            point_frac (np.ndarray): Fractional point the plane passes through.
            u_dir_cart (np.ndarray, optional): Cartesian vector guiding the u-axis.
            margin (float): Fraction of each span added as padding on both sides.

        Returns:
            Tuple[Tuple[float, float], Tuple[float, float]]: (u_range, v_range).
        """
        n_hat, p_cart, u_hat, v_hat = self.define_plane_basis(normal_frac, point_frac, u_dir_cart)

        # Prefer the hull's actual cross-section. Projecting the whole cloud
        # bounds the footprint from above and can overshoot badly for an oblique
        # plane, which is a parallelepiped's long diagonal rather than the much
        # smaller polygon the plane really cuts.
        section = self._hull_plane_section(n_hat, p_cart)
        rel = (section if section is not None else self.kpoints_cart) - p_cart
        ranges = []
        for axis in (u_hat, v_hat):
            proj = rel @ axis
            lo, hi = float(proj.min()), float(proj.max())
            span = hi - lo
            if not np.isfinite(span) or span <= 0.0:
                # A cloud with no extent along this axis - a single k-point, or a
                # plane containing a degenerate line - gives nothing to scale to.
                lo, hi, span = -1.0, 1.0, 2.0
            pad = margin * span
            ranges.append((lo - pad, hi + pad))
        return ranges[0], ranges[1]

    def interpolate_plane(self, normal_frac: np.ndarray, point_frac: np.ndarray,
                          u_range: Tuple[float, float], v_range: Tuple[float, float],
                          grid_resolution: int = 150, interpolate_factor: int = 1,
                          u_dir_cart: np.ndarray = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Interpolates discrete 3D eigenvalues onto a regular 2D plane grid using Scipy.

        Args:
            normal_frac (np.ndarray): Fractional normal vector defining the plane.
            point_frac (np.ndarray): Fractional coordinate vector representing a point on the plane.
            u_range (Tuple[float, float]): Range of in-plane coordinate u (min, max) in A^-1.
            v_range (Tuple[float, float]): Range of in-plane coordinate v (min, max) in A^-1.
            grid_resolution (int): Base number of grid points along each in-plane dimension.
            interpolate_factor (int): Scaling factor matching sumo smoothing defaults.

        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray]: u_grid, v_grid, interpolated_spectra.
        """
        n_hat, p_cart, u_hat, v_hat = self.define_plane_basis(normal_frac, point_frac, u_dir_cart)

        # Scale resolution based on Sumo's interpolation paradigms to enhance output quality
        total_resolution = int(grid_resolution * interpolate_factor)

        u_grid = np.linspace(u_range[0], u_range[1], total_resolution)
        v_grid = np.linspace(v_range[0], v_range[1], total_resolution)
        uu, vv = np.meshgrid(u_grid, v_grid)

        # Map 2D grid coordinates back to 3D Cartesian reciprocal coordinates
        grid_cart = (p_cart[None, None, :]
                     + uu[:, :, None] * u_hat[None, None, :]
                     + vv[:, :, None] * v_hat[None, None, :])
        grid_cart_flat = grid_cart.reshape(-1, 3)

        nspins, nbands, nkpts = self.eigenvalues.shape
        n_eig = nspins * nbands

        # One triangulation, one interpolator, one pass. Building a fresh
        # LinearNDInterpolator per band re-ran qhull nspins*nbands times per
        # plane; on a real VASP k-mesh - a regular grid, whose degenerate sliver
        # simplices make point-location pathologically slow - that dominated the
        # entire runtime. Bands and matrix-element weights become value columns
        # so each query point's simplex is located exactly once.
        values = self.eigenvalues.reshape(n_eig, nkpts).T
        if self.weights is not None:
            values = np.hstack([values, self.weights.reshape(n_eig, nkpts).T])
        interp = LinearNDInterpolator(self.build_triangulation(), values)

        # Chunk over grid points so the temporary stays bounded regardless of
        # resolution and band count.
        n_pixels = grid_cart_flat.shape[0]
        n_cols = values.shape[1]
        chunk = max(1, int(4e7) // max(1, n_cols))
        flat = np.empty((n_pixels, n_cols))
        for i0 in range(0, n_pixels, chunk):
            flat[i0:i0 + chunk] = interp(grid_cart_flat[i0:i0 + chunk])

        interpolated_spectra = np.ascontiguousarray(
                flat[:, :n_eig].T.reshape(nspins, nbands, total_resolution, total_resolution))
        interpolated_weights = None
        if self.weights is not None:
            interpolated_weights = np.ascontiguousarray(
                    flat[:, n_eig:].T.reshape(nspins, nbands, total_resolution, total_resolution))

        # Points outside the convex hull of the k-point cloud interpolate to NaN
        # and are later drawn as intensity 0 - indistinguishable from a genuine
        # absence of spectral weight. Report the coverage rather than let a
        # mostly-fabricated figure pass as data.
        nan_fraction = float(np.isnan(interpolated_spectra[0, 0]).mean())
        if nan_fraction >= 1.0:
            print("[Geometry] WARNING: the requested plane lies entirely outside the "
                  "k-point convex hull; the figure will be uniformly blank. Check "
                  "--normal, --origin, --ubounds and --vbounds.")
        elif nan_fraction > 0.25:
            print(f"[Geometry] WARNING: {nan_fraction:.0%} of the plot window lies outside "
                  f"the k-point convex hull and carries no data; it will render as zero "
                  f"intensity, which looks identical to zero spectral weight.")

        return u_grid, v_grid, interpolated_spectra, interpolated_weights

