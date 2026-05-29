"""
This file contains the SurfaceBZAnalyzer class.
It parses crystal structures, cleaves slabs, computes reciprocal lattices,
projects three-dimensional Brillouin zones to two-dimensional planes, identifies k-paths,
clusters projected points, and generates visualizations.

Inputs:
 - filepath: Path to a structure file (CONTCAR, POSCAR, vasprun.xml, or vaspout.h5).
 - miller_index: Three-integer tuple specifying the surface plane.
 - min_slab: Float specifying the minimum slab thickness in Angstroms.
 - min_vac: Float specifying the minimum vacuum thickness in Angstroms.

Outputs:
 - A symmetrized slab structure.
 - Three-dimensional and two-dimensional reciprocal lattice matrices.
 - A dictionary containing projected k-points, labels, and clustering assignments.
 - Visual plots of the bulk and projected Brillouin zones.

Approach and Modules:
 - Slab generation: pymatgen.core.surface.SlabGenerator.
 - High-symmetry path detection: pymatgen.symmetry.bandstructure.HighSymmKpath.
 - Wigner-Seitz cell calculation: Voronoi tessellation via scipy.spatial.Voronoi.
 - Vertex sorting: Convex hull boundary calculation via scipy.spatial.ConvexHull.
 - Density clustering: DBSCAN point grouping via sklearn.cluster.DBSCAN.
 - Spatial matching: Nearest neighbors mapping via sklearn.neighbors.NearestNeighbors.
 - Linear algebra: Vector projections via numpy.
 - Graphics: Plot rendering via matplotlib.pyplot and Poly3DCollection.
"""
import os
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from scipy.spatial import Voronoi, ConvexHull
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors
from pymatgen.io.vasp.outputs import Vasprun, Vaspout
from pymatgen.core.surface import SlabGenerator
from pymatgen.symmetry.bandstructure import HighSymmKpath

class SurfaceBZAnalyzer:
    def __init__(self, filepath):
        """Initializes the bulk structure from a VASP output file."""
        self.filepath = filepath
        self.bulk_structure = self._parse_structure()

    def _parse_structure(self):
        """Parses vaspout.h5 (modern) or vasprun.xml (legacy)."""
        if self.filepath.endswith(".h5"):
            parser = Vaspout(self.filepath)
        else:
            parser = Vasprun(self.filepath)
        return parser.final_structure

    def generate_slab(self, miller_index, min_slab, min_vac):
        """Generates a slab and derives 3D/2D reciprocal lattices."""
        # Generate symmetrical slab
        slabgen = SlabGenerator(self.bulk_structure, miller_index, min_slab, min_vac, center_slab=True)
        # Extract the first unique, symmetrically balanced termination
        self.slab = slabgen.get_slabs(symmetrize=True)

        # Physics convention (includes 2 * pi)
        self.bulk_recip = self.bulk_structure.lattice.reciprocal_lattice.matrix
        self.slab_recip = self.slab.lattice.reciprocal_lattice.matrix

        # 2D in-plane projection of the slab reciprocal lattice
        self.slab_recip_2d = self.slab_recip[:2, :2]

    def get_wigner_seitz(self, recip_matrix, dim=3):
        """Calculates Wigner-Seitz cell via Voronoi tessellation."""
        # Generate local grid of reciprocal lattice nodes
        grid_range = np.mgrid[-1:2, -1:2, -1:2] if dim == 3 else np.mgrid[-1:2, -1:2]
        pts = np.tensordot(recip_matrix, grid_range, axes=2).reshape(dim, -1).T

        vor = Voronoi(pts)

        # Find the region associated with the origin
        origin_idx = np.argmin(np.linalg.norm(pts, axis=1))
        region_idx = vor.point_region[origin_idx]
        region_vertices = vor.regions[region_idx]

        return vor.vertices[region_vertices]

    def correlate_zones(self):
        """Finds high-symmetry paths and maps 3D bulk states to 2D slab plane."""
        # 3D Bulk High-Symmetry Points
        bulk_kpath = HighSymmKpath(self.bulk_structure)
        bulk_kpts = bulk_kpath.kpath["kpoints"]

        # Project 3D fractional coordinates to Cartesian, then drop the z-component
        projected_kpts = []
        labels = []
        for label, frac_coord in bulk_kpts.items():
            cart_coord = np.dot(frac_coord, self.bulk_recip)
            projected_kpts.append([cart_coord[0], cart_coord[1]])
            labels.append(label)

        projected_kpts = np.array(projected_kpts)

        # Cluster overlapped points using DBSCAN to identify band folding
        clustering = DBSCAN(eps=1e-4, min_samples=1).fit(projected_kpts)

        # 2D Surface High-Symmetry Points
        slab_kpath = HighSymmKpath(self.slab)

        # Map clusters using NearestNeighbors
        nn = NearestNeighbors(n_neighbors=1).fit(projected_kpts)

        self.correlation_data = {
                "projected_kpts": projected_kpts,
                "labels": labels,
                "clusters": clustering.labels_
                }
        return self.correlation_data

    def visualize(self):
        """Renders the 3D and 2D Brillouin Zones."""
        fig = plt.figure(figsize=(12, 5))

        # Plot 3D BZ
        ax1 = fig.add_subplot(121, projection='3d')
        bz_3d_verts = self.get_wigner_seitz(self.bulk_recip, dim=3)
        hull = ConvexHull(bz_3d_verts)
        for simplex in hull.simplices:
            poly = Poly3DCollection([bz_3d_verts[simplex]], alpha=0.3, facecolor='cyan', edgecolor='k')
            ax1.add_collection3d(poly)
        ax1.set_title("Bulk 3D Brillouin Zone")

        # Plot 2D BZ with Projections
        ax2 = fig.add_subplot(122)
        bz_2d_verts = self.get_wigner_seitz(self.slab_recip_2d, dim=2)
        hull_2d = ConvexHull(bz_2d_verts)
        for simplex in hull_2d.simplices:
            ax2.plot(bz_2d_verts[simplex, 0], bz_2d_verts[simplex, 1], 'k-')

        # Scatter projected clusters
        proj = self.correlation_data["projected_kpts"]
        ax2.scatter(proj[:, 0], proj[:, 1], c=self.correlation_data["clusters"], cmap='viridis', s=50, zorder=5)

        for i, label in enumerate(self.correlation_data["labels"]):
            ax2.text(proj[i, 0], proj[i, 1], f"${label}$", fontsize=12)

        ax2.set_title("Projected 2D Surface Brillouin Zone")
        ax2.set_aspect('equal')
        plt.tight_layout()
        plt.show()

