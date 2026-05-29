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
from typing import Tuple

class KSpaceProjector:
    """Performs coordinates transformation, plane projection, and multidimensional interpolation."""

    def __init__(self, kpoints: np.ndarray, eigenvalues: np.ndarray, rec_lattice: np.ndarray):
        """
        Initialize the projector.

        Args:
            kpoints (np.ndarray): Fractional k-points coordinates, shape (nkpts, 3).
            eigenvalues (np.ndarray): Eigenvalues array, shape (nspins, nbands, nkpts).
            rec_lattice (np.ndarray): Reciprocal lattice matrix, shape (3, 3).
        """
        self.kpoints_frac = kpoints
        self.eigenvalues = eigenvalues
        self.rec_lattice = rec_lattice
        # Transform fractional k-points to Cartesian coordinates (A^-1)
        self.kpoints_cart = np.dot(kpoints, rec_lattice)

    def define_plane_basis(self, normal_frac: np.ndarray, point_frac: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Constructs an orthonormal basis set for the specified projection plane.

        Args:
            normal_frac (np.ndarray): Normal vector in fractional coordinates.
            point_frac (np.ndarray): Shift point on the plane in fractional coordinates.

        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]: n_hat, p_cart, u_hat, v_hat.
        """
        n_cart = np.dot(normal_frac, self.rec_lattice)
        p_cart = np.dot(point_frac, self.rec_lattice)

        n_hat = n_cart / np.linalg.norm(n_cart)

        # Generate orthogonal vectors on the plane via Gram-Schmidt
        # Use a non-collinear starting vector
        aux_vec = np.array([1.0, 0.0, 0.0]) if np.abs(n_hat[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
        u_cart = aux_vec - np.dot(aux_vec, n_hat) * n_hat
        u_hat = u_cart / np.linalg.norm(u_cart)
        v_hat = np.cross(n_hat, u_hat)

        return n_hat, p_cart, u_hat, v_hat

    def interpolate_plane(self, normal_frac: np.ndarray, point_frac: np.ndarray,
                          u_range: Tuple[float, float], v_range: Tuple[float, float],
                          grid_resolution: int = 150, interpolate_factor: int = 1) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
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
        n_hat, p_cart, u_hat, v_hat = self.define_plane_basis(normal_frac, point_frac)

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

        nspins, nbands, _ = self.eigenvalues.shape
        interpolated_spectra = np.zeros((nspins, nbands, total_resolution, total_resolution))

        # Perform Linear Triangulation-based 3D interpolation for each band and spin channel
        for s in range(nspins):
            for b in range(nbands):
                interp = LinearNDInterpolator(self.kpoints_cart, self.eigenvalues[s, b, :])
                flat_interp = interp(grid_cart_flat)
                interpolated_spectra[s, b] = flat_interp.reshape(total_resolution, total_resolution)

        return u_grid, v_grid, interpolated_spectra

