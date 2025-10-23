"""
This file contains the ARPESPlotter class.
It simulates photoemission intensities and generates
constant-energy contours and band dispersion slices.

Inputs:
 - u_grid: Array representing the local in-plane coordinate axis u.
 - v_grid: Array representing the local in-plane coordinate axis v.
 - interpolated_spectra: Array of interpolated energy bands.
 - efermi: Float specifying the Fermi energy.
 - energy_array: Array of energies for spectral density evaluation.
 - broadening: Float specifying Lorentzian half-width at half-maximum.
 - spin_channel: Integer specifying the target spin channel.

Outputs:
 - Calculated spectral density arrays.
 - Saved PNG image files or display windows for Fermi surface maps.
 - Saved PNG image files or display windows for band dispersion slices.

Approach and Modules:
 - Spectral evaluation: Lorentzian line-shape accumulation via numpy.
 - Graphics rendering: Plotting and colormesh generation via matplotlib.pyplot.
 - Styling: Publication formatting via sumo.plotting.formatting.
"""

import matplotlib.pyplot as plt
import numpy as np
from typing import Tuple, Optional

# Attempt to import sumo styling for publication-ready figures
try:
    from sumo.plotting.formatting import sumo_style
    HAS_SUMO = True
except ImportError:
    HAS_SUMO = False

class ARPESPlotter:
    """Simulates physical photoemission intensities and generates publication-ready plots."""

    def __init__(self, u_grid: np.ndarray, v_grid: np.ndarray, interpolated_spectra: np.ndarray, efermi: float):
        """
        Initialize the plotter.

        Args:
            u_grid (np.ndarray): Local in-plane coordinate axis u, shape (grid_res,).
            v_grid (np.ndarray): Local in-plane coordinate axis v, shape (grid_res,).
            interpolated_spectra (np.ndarray): Interpolated energies, shape (nspin, nband, grid_res, grid_res).
            efermi (float): Fermi energy in eV.
        """
        self.u_grid = u_grid
        self.v_grid = v_grid
        self.spectra = interpolated_spectra - efermi  # Shift Fermi level to 0.0 eV
        self.efermi = 0.0
        self._apply_styles()

    def _apply_styles(self):
        """Applies Sumo or default plotting parameters for publication-ready outputs."""
        if HAS_SUMO:
            plt.style.use(sumo_style)
        else:
            plt.rcParams.update({
                'font.family': 'sans-serif',
                'font.sans-serif': ['Tahoma', 'DejaVu Sans',
                                    'Lucida Grande', 'Verdana'],
                'axes.linewidth': 1.5,
                'xtick.major.size': 6,
                'xtick.major.width': 1.5,
                'ytick.major.size': 6,
                'ytick.major.width': 1.5,
                'font.size': 12,
                'axes.labelsize': 14,
                'axes.titlesize': 14
                })

    def calculate_spectral_density(self, energy_array: np.ndarray, broadening: float = 0.05, spin_channel: int = 0) -> np.ndarray:
        """
        Evaluates the Lorentzian spectral function representing intrinsic lifetime broadening.

        Args:
            energy_array (np.ndarray): Range of energies (relative to Ef) at which to calculate intensity.
            broadening (float): Lorentzian half-width at half-maximum (HWHM) in eV.
            spin_channel (int): Index of the target spin channel.

        Returns:
            np.ndarray: Calculated spectral density array, shape (n_energies, grid_res_v, grid_res_u).
        """
        grid_res_v = len(self.v_grid)
        grid_res_u = len(self.u_grid)
        n_energies = len(energy_array)

        intensity = np.zeros((n_energies, grid_res_v, grid_res_u))
        nbands = self.spectra.shape[1]  # Extracted correct dimension count for bands

        # Accumulate Lorentzian line-shapes for each band
        for b in range(nbands):
            band_energies = self.spectra[spin_channel, b]
            if np.isnan(band_energies).all():
                continue
            for idx, e in enumerate(energy_array):
                lorentzian = (1.0 / np.pi) * (broadening / ((e - band_energies) ** 2 + broadening ** 2))
                intensity[idx] += np.nan_to_num(lorentzian, nan=0.0)

        return intensity

    def plot_constant_energy_cut(self, energy: float, broadening: float = 0.05,
                                 spin_channel: int = 0, cmap: str = "inferno",
                                 filename: Optional[str] = None):
        """
        Generates and saves/displays a constant energy map (e.g., Fermi surface map).

        Args:
            energy (float): Energy slice coordinate (E - Ef) in eV.
            broadening (float): Broadening width in eV.
            spin_channel (int): Selected spin channel index.
            cmap (str): Matplotlib colormap.
            filename (Optional[str]): Target output file name for saving the plot.
        """
        energy_slice = np.array([energy])
        intensity = self.calculate_spectral_density(energy_slice, broadening, spin_channel)

        fig, ax = plt.subplots(figsize=(6, 5))
        im = ax.pcolormesh(self.u_grid, self.v_grid, intensity[0], cmap=cmap, shading='auto')
        fig.colorbar(im, ax=ax, label="Simulated ARPES Intensity (arb. u.)")

        ax.set_xlabel(r"$k_u$ ($\mathrm{\AA}^{-1}$)")
        ax.set_ylabel(r"$k_v$ ($\mathrm{\AA}^{-1}$)")
        ax.set_title(f"Constant Energy Contour ($E - E_F = {energy:.2f}$ eV)")
        ax.set_aspect('equal', 'box')

        plt.tight_layout()
        if filename:
            plt.savefig(filename, dpi=300)
            plt.close()
        else:
            plt.show()

    def plot_dispersion_slice(self, slice_coordinate: float, along_v: bool = False,
                              energy_limits: Tuple[float, float] = (-2.0, 1.0),
                              n_energy_points: int = 250, broadening: float = 0.05,
                              spin_channel: int = 0, cmap: str = "inferno",
                              filename: Optional[str] = None):
        """
        Plots an E vs k_parallel dispersion cut along a selected axis.

        Args:
            slice_coordinate (float): Fixed coordinate value on the secondary axis (A^-1).
            along_v (bool): If True, plots E vs k_v at a constant u. If False, plots E vs k_u at constant v.
            energy_limits (Tuple[float, float]): Min and max limits for the energy axis (eV).
            n_energy_points (int): Energy grid resolution.
            broadening (float): Broadening width in eV.
            spin_channel (int): Target spin channel.
            cmap (str): Target colormap.
            filename (Optional[str]): Output filename.
        """
        energy_axis = np.linspace(energy_limits[0], energy_limits[1], n_energy_points)
        nbands = self.spectra.shape[1]  # Extracted correct dimension count for bands

        if along_v:
            # Slicing along constant u coordinate
            idx = np.argmin(np.abs(self.u_grid - slice_coordinate))
            intensity_slice = np.zeros((n_energy_points, len(self.v_grid)))
            for b in range(nbands):
                band_v = self.spectra[spin_channel, b, :, idx]
                for i, e in enumerate(energy_axis):
                    lorentzian = (1.0 / np.pi) * (broadening / ((e - band_v) ** 2 + broadening ** 2))
                    intensity_slice[i] += np.nan_to_num(lorentzian, nan=0.0)
            k_axis = self.v_grid
            xlabel = r"$k_v$ ($\mathrm{\AA}^{-1}$)"
            title = f"Dispersion Slice at $k_u = {slice_coordinate:.2f}$ $\mathrm{{\AA}}^{{-1}}$"
        else:
            # Slicing along constant v coordinate
            idx = np.argmin(np.abs(self.v_grid - slice_coordinate))
            intensity_slice = np.zeros((n_energy_points, len(self.u_grid)))
            for b in range(nbands):
                band_u = self.spectra[spin_channel, b, idx, :]
                for i, e in enumerate(energy_axis):
                    lorentzian = (1.0 / np.pi) * (broadening / ((e - band_u) ** 2 + broadening ** 2))
                    intensity_slice[i] += np.nan_to_num(lorentzian, nan=0.0)
            k_axis = self.u_grid
            xlabel = r"$k_u$ ($\mathrm{\AA}^{-1}$)"
            title = f"Dispersion Slice at $k_v = {slice_coordinate:.2f}$ $\mathrm{{\AA}}^{{-1}}$"

        fig, ax = plt.subplots(figsize=(6, 5))
        im = ax.pcolormesh(k_axis, energy_axis, intensity_slice, cmap=cmap, shading='auto')
        fig.colorbar(im, ax=ax, label="Simulated ARPES Intensity (arb. u.)")

        ax.axhline(0.0, color="w", linestyle="--", alpha=0.6, label="Fermi Level")
        ax.set_xlabel(xlabel)
        ax.set_ylabel(r"$E - E_F$ (eV)")
        ax.set_title(title)

        plt.tight_layout()
        if filename:
            plt.savefig(filename, dpi=300)
            plt.close()
        else:
            plt.show()

