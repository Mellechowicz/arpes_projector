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

import os
from concurrent.futures import ThreadPoolExecutor

import matplotlib.pyplot as plt
import numpy as np
from typing import Tuple, Optional
from matplotlib.colors import LogNorm, PowerNorm

# Attempt to import sumo styling for publication-ready figures
try:
    from sumo.plotting.formatting import sumo_style
    HAS_SUMO = True
except ImportError:
    HAS_SUMO = False

class ARPESPlotter:
    """Simulates physical photoemission intensities and generates publication-ready plots."""

    def __init__(self, u_grid: np.ndarray, v_grid: np.ndarray, interpolated_spectra: np.ndarray, efermi: float,
                 efermi_shift: float = 0.0, weights: np.ndarray = None):
        """
        Initialize the plotter.

        Args:
            u_grid (np.ndarray): Local in-plane coordinate axis u, shape (grid_res,).
            v_grid (np.ndarray): Local in-plane coordinate axis v, shape (grid_res,).
            interpolated_spectra (np.ndarray): Interpolated energies, shape (nspin, nband, grid_res, grid_res).
            efermi (float): Fermi energy in eV.
            efermi_shift (float): Rigid shift (eV) added to the Fermi level. A positive value
                raises the chemical potential (electron doping), moving the simulated Fermi
                surface. All energies are referenced to the shifted level (E - (E_F + shift)).
            weights (np.ndarray, optional): Per-state spectral weights (matrix-element
                proxies), same shape as interpolated_spectra. Each band's Lorentzian is
                scaled by its weight; None weights every state equally.
        """
        if efermi is None:
            raise ValueError("Fermi energy is None; cannot reference band energies. "
                             "Provide a valid VASP file or set the Fermi level explicitly.")
        if weights is not None and weights.shape != interpolated_spectra.shape:
            raise ValueError(f"weights shape {weights.shape} does not match "
                             f"spectra shape {interpolated_spectra.shape}.")
        self.u_grid = u_grid
        self.v_grid = v_grid
        self.efermi_shift = efermi_shift
        # Reference all energies to the (optionally shifted) Fermi level, mapping E_F + shift -> 0.0 eV
        self.spectra = interpolated_spectra - efermi - efermi_shift
        self.weights = weights
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

    @staticmethod
    def _accumulate_lorentzian(band_values: np.ndarray, energy_array: np.ndarray, broadening: float,
                               weights: np.ndarray = None) -> np.ndarray:
        """
        Vectorized Lorentzian accumulation over bands, threaded over energies.

        Args:
            band_values (np.ndarray): Band energies with the band axis first, shape (nbands, ...).
            energy_array (np.ndarray): Energies (relative to Ef) at which to evaluate intensity.
            broadening (float): Lorentzian HWHM in eV.
            weights (np.ndarray, optional): Per-state weights, same shape as band_values;
                each state's Lorentzian is scaled by its weight.

        Returns:
            np.ndarray: Accumulated intensity, shape (n_energies, ...).
        """
        # NaN band values (grid points outside the k-point hull) are mapped to +inf,
        # whose Lorentzian weight is exactly 0 - equivalent to the previous
        # nan_to_num(...) of each Lorentzian, without per-energy NaN scans.
        bands = np.where(np.isnan(band_values), np.inf, band_values)
        if weights is not None:
            weights = np.nan_to_num(weights, nan=0.0)
        nbands = bands.shape[0]
        cell = bands[0].size if nbands else 0
        prefactor = broadening / np.pi

        n_energies = len(energy_array)
        intensity = np.zeros((n_energies,) + bands.shape[1:])
        if nbands == 0 or cell == 0:
            return intensity

        # Chunk the band axis so the largest temporary stays around ~40 MB
        band_chunk = max(1, int(5e6) // cell)

        def _one_energy(idx: int):
            acc = intensity[idx]
            e = energy_array[idx]
            for b0 in range(0, nbands, band_chunk):
                block = bands[b0:b0 + band_chunk]
                lor = prefactor / ((e - block) ** 2 + broadening ** 2)
                if weights is not None:
                    lor *= weights[b0:b0 + band_chunk]
                acc += lor.sum(axis=0)

        # numpy releases the GIL on large ufuncs, so threads parallelize well here
        n_workers = min(n_energies, os.cpu_count() or 1)
        if n_workers > 1 and n_energies * nbands * cell > 1_000_000:
            with ThreadPoolExecutor(max_workers=n_workers) as pool:
                list(pool.map(_one_energy, range(n_energies)))
        else:
            for idx in range(n_energies):
                _one_energy(idx)

        return intensity

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
        w = self.weights[spin_channel] if self.weights is not None else None
        return self._accumulate_lorentzian(self.spectra[spin_channel], np.asarray(energy_array), broadening, w)

    def plot_constant_energy_cut(self, energy: float, broadening: float = 0.05,
                                 spin_channel: int = 0, cmap: str = "inferno",
                                 filename: Optional[str] = None, cscale: str = "linear",
                                 custom_title: Optional[str] = None):
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
        norm = LogNorm(vmin=max(intensity[0].min(), 1e-5), vmax=intensity[0].max()) if cscale == "log" else (PowerNorm(0.5) if cscale == "sqrt" else None)

        fig, ax = plt.subplots(figsize=(6, 5))
        im = ax.pcolormesh(self.u_grid, self.v_grid, intensity[0], cmap=cmap, shading='auto', norm=norm)
        fig.colorbar(im, ax=ax, label="Simulated Intensity (a.u.)")

        ax.set_xlabel(r"$k_u$ ($\mathrm{\AA}^{-1}$)")
        ax.set_ylabel(r"$k_v$ ($\mathrm{\AA}^{-1}$)")
        ax.set_title(custom_title if custom_title else f"Constant Energy Contour ($E - E_F = {energy:.2f}$ eV)")
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
                              filename: Optional[str] = None, integrate_v: bool = False,
                              custom_xticks: Optional[list] = None, custom_xticklabels: Optional[list] = None,
                              cscale: str = "linear", custom_title: Optional[str] = None):
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
        w = self.weights[spin_channel] if self.weights is not None else None

        if integrate_v and not along_v:
            # Accumulate over the full plane, then average out the v axis
            intensity_slice = self._accumulate_lorentzian(
                    self.spectra[spin_channel], energy_axis, broadening, w).sum(axis=1)
            intensity_slice /= len(self.v_grid)
            k_axis = self.u_grid
            xlabel = r"$k_\parallel$ ($\mathrm{\AA}^{-1}$)"
            title = "Projected Surface Band Structure"

        elif along_v:
            # Slicing along constant u coordinate
            idx = np.argmin(np.abs(self.u_grid - slice_coordinate))
            intensity_slice = self._accumulate_lorentzian(
                    self.spectra[spin_channel, :, :, idx], energy_axis, broadening,
                    w[:, :, idx] if w is not None else None)
            k_axis = self.v_grid
            xlabel = r"$k_v$ ($\mathrm{\AA}^{-1}$)"
            title = rf"Dispersion Slice at $k_u = {slice_coordinate:.2f}$ $\mathrm{{\AA}}^{{-1}}$"
        else:
            # Slicing along constant v coordinate
            idx = np.argmin(np.abs(self.v_grid - slice_coordinate))
            intensity_slice = self._accumulate_lorentzian(
                    self.spectra[spin_channel, :, idx, :], energy_axis, broadening,
                    w[:, idx, :] if w is not None else None)
            k_axis = self.u_grid
            xlabel = r"$k_u$ ($\mathrm{\AA}^{-1}$)"
            title = rf"Dispersion Slice at $k_v = {slice_coordinate:.2f}$ $\mathrm{{\AA}}^{{-1}}$"

        norm = LogNorm(vmin=max(intensity_slice.min(), 1e-5), vmax=intensity_slice.max()) if cscale == "log" else (PowerNorm(0.5) if cscale == "sqrt" else None)
        fig, ax = plt.subplots(figsize=(6, 5))
        im = ax.pcolormesh(k_axis, energy_axis, intensity_slice, cmap=cmap, shading='auto', norm=norm)
        fig.colorbar(im, ax=ax, label="Simulated Intensity (a.u.)")

        ax.axhline(0.0, color="w", linestyle="--", alpha=0.6, label="Fermi Level")
        ax.set_xlabel(xlabel)
        ax.set_ylabel(r"$E - E_F$ (eV)")

        if custom_xticks is not None:
            ax.set_xticks(custom_xticks)
            if custom_xticklabels is not None:
                ax.set_xticklabels(custom_xticklabels, fontsize=16)
            ax.set_xlabel("High Symmetry Momentum Path")
        else:
            ax.set_xlabel(xlabel)

        ax.set_title(custom_title if custom_title else title)

        plt.tight_layout()
        if filename:
            plt.savefig(filename, dpi=300)
            plt.close()
        else:
            plt.show()

