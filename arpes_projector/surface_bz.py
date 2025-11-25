"""
This file contains the SurfaceBZAnalyzer class.
It implements a mathematically rigorous projection of 3D bulk states onto 
arbitrary 2D surface planes, visualizing the First Brillouin Zones and band foldings.
"""
import itertools
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from scipy.spatial import Voronoi
from sklearn.cluster import DBSCAN
from pymatgen.io.vasp.outputs import Vasprun, Vaspout
from pymatgen.core.surface import SlabGenerator
from pymatgen.symmetry.bandstructure import HighSymmKpath

class SurfaceBZAnalyzer:
    def __init__(self, filepath):
        """Initializes the bulk structure from a VASP output file."""
        self.filepath = filepath
        self.bulk_structure = self._parse_structure()

    def _parse_structure(self):
        if self.filepath.endswith(".h5"):
            return Vaspout(self.filepath).final_structure
        return Vasprun(self.filepath).final_structure

    def align_vector_to_z(self, v):
        """Calculates the Cartesian rotation matrix to align a vector to the Z-axis."""
        v = np.array(v, dtype=float)
        v = v / np.linalg.norm(v)
        z = np.array([0.0, 0.0, 1.0])

        if np.allclose(v, z): return np.eye(3)
        if np.allclose(v, -z): return -np.eye(3)

        axis = np.cross(v, z)
        axis = axis / np.linalg.norm(axis)
        angle = np.arccos(np.clip(np.dot(v, z), -1.0, 1.0))

        K = np.array([[0, -axis[2], axis[1]],
                      [axis[2], 0, -axis[0]],
                      [-axis[1], axis[0], 0]])
        return np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * (K @ K)

    def generate_slab(self, miller_index, min_slab, min_vac):
        """Prepares the aligned Cartesian coordinate frames based on the Miller surface."""
        self.miller_index = miller_index

        # 1. Base bulk reciprocal lattice
        self.bulk_recip = self.bulk_structure.lattice.reciprocal_lattice.matrix
        h, k, l = miller_index

        # 2. Find the reciprocal vector strictly normal to the (hkl) planes
        n_vec = h * self.bulk_recip[0] + k * self.bulk_recip[1] + l * self.bulk_recip[2]

        # 3. Create a rotation matrix that stands the bulk BZ perfectly upright
        self.R_align = self.align_vector_to_z(n_vec)
        self.bulk_recip_aligned = self.bulk_recip @ self.R_align.T

        # 4. Generate the 2D surface reciprocal lattice nodes (for Umklapp folding)
        # By projecting a 7x7x7 grid of 3D nodes onto the XY plane and filtering unique states
        translations = np.array(list(itertools.product(range(-3, 4), repeat=3)))
        pts_3d = translations @ self.bulk_recip_aligned
        pts_2d = pts_3d[:, :2]
        self.surface_nodes_2d = np.unique(np.round(pts_2d, 5), axis=0)

    def get_wigner_seitz_3d(self):
        """Calculates the upright 3D Wigner-Seitz cell faces."""
        translations = np.array(list(itertools.product([-1, 0, 1], repeat=3)))
        pts = translations @ self.bulk_recip_aligned
        vor = Voronoi(pts)
        origin_idx = np.argmin(np.linalg.norm(pts, axis=1))

        faces = []
        for point_pair, ridge_verts in zip(vor.ridge_points, vor.ridge_vertices):
            if origin_idx in point_pair and -1 not in ridge_verts:
                faces.append(vor.vertices[ridge_verts])
        return faces

    def get_wigner_seitz_2d(self):
        """Calculates the 2D Surface Wigner-Seitz cell boundary."""
        vor = Voronoi(self.surface_nodes_2d)
        origin_idx = np.argmin(np.linalg.norm(self.surface_nodes_2d, axis=1))

        region_idx = vor.point_region[origin_idx]
        region_vertices_indices = vor.regions[region_idx]
        vertices = vor.vertices[region_vertices_indices]

        # Sort counter-clockwise to form a closed polygon
        center = np.mean(vertices, axis=0)
        angles = np.arctan2(vertices[:, 1] - center[1], vertices[:, 0] - center[0])
        return vertices[np.argsort(angles)]

    def fold_to_first_bz(self, pt_2d):
        """Mathematically translates an extended state into the First Brillouin Zone."""
        distances = np.linalg.norm(self.surface_nodes_2d - pt_2d, axis=1)
        closest_node = self.surface_nodes_2d[np.argmin(distances)]
        return pt_2d - closest_node

    def correlate_zones(self):
        """Maps 3D paths to 2D projections and identifies band foldings."""
        bulk_kpath = HighSymmKpath(self.bulk_structure)
        bulk_kpts = bulk_kpath.kpath["kpoints"]

        bulk_3d_kpts, projected_kpts, extended_kpts, labels = [], [], [], []

        for label, frac_coord in bulk_kpts.items():
            # 1. Map to upright Cartesian frame
            cart_3d = frac_coord @ self.bulk_recip_aligned

            # 2. Perfect projection: simply drop the Z axis!
            cart_2d_ext = cart_3d[:2] 

            # 3. Fold into First Brillouin Zone
            cart_2d_folded = self.fold_to_first_bz(cart_2d_ext)

            bulk_3d_kpts.append(cart_3d)
            extended_kpts.append(cart_2d_ext)
            projected_kpts.append(cart_2d_folded)
            labels.append(label)

        self.correlation_data = {
                "bulk_3d_kpts": np.array(bulk_3d_kpts),
                "extended_kpts": np.array(extended_kpts),
                "projected_kpts": np.array(projected_kpts),
                "labels": labels
                }

        # Cluster overlapped points using DBSCAN on the folded coordinates
        clustering = DBSCAN(eps=1e-4, min_samples=1).fit(self.correlation_data["projected_kpts"])
        self.correlation_data["clusters"] = clustering.labels_
        return self.correlation_data

    def visualize(self):
        """Renders the Brillouin Zones with physically accurate vertical projection lines."""
        fig = plt.figure(figsize=(14, 6))

        # ==========================================
        # Subplot 1: 3D BZ
        # ==========================================
        ax1 = fig.add_subplot(121, projection='3d')
        ax1.view_init(elev=20, azim=-45) # Set a good default viewing angle

        # Plot 3D BZ Faces
        faces_3d = self.get_wigner_seitz_3d()
        for face in faces_3d:
            poly = Poly3DCollection([face], alpha=0.15, facecolor='cyan', edgecolor='k', linewidth=1)
            ax1.add_collection3d(poly)

        # Draw 2D Surface W-S cell embedded at kz=0
        faces_2d = self.get_wigner_seitz_2d()
        faces_2d_in_3d = np.column_stack((faces_2d, np.zeros(len(faces_2d))))
        poly_2d = Poly3DCollection([faces_2d_in_3d], alpha=0.15, facecolor='red', edgecolor='red', linewidth=2)
        ax1.add_collection3d(poly_2d)

        bulk_3d = self.correlation_data["bulk_3d_kpts"]
        proj_2d = self.correlation_data["projected_kpts"]
        ext_2d = self.correlation_data["extended_kpts"]

        # Plot Data and Projection Lines
        for b_pt, p_pt, e_pt in zip(bulk_3d, proj_2d, ext_2d):
            # 1. Straight vertical drop from bulk state to the extended zone (kz=0)
            ax1.plot([b_pt[0], b_pt[0]], [b_pt[1], b_pt[1]], [b_pt[2], 0], 'k--', alpha=0.3, linewidth=1.2)

            # 2. If the state folded, draw a red translation line back into the 1st BZ W-S cell
            if np.linalg.norm(p_pt - e_pt) > 1e-4:
                ax1.plot([b_pt[0], p_pt[0]], [b_pt[1], p_pt[1]], [0, 0], 'r:', alpha=0.7, linewidth=1.5)

        # Scatter actual points
        ax1.scatter(bulk_3d[:, 0], bulk_3d[:, 1], bulk_3d[:, 2], c='blue', s=30, zorder=10)
        ax1.scatter(proj_2d[:, 0], proj_2d[:, 1], np.zeros(len(proj_2d)), c=self.correlation_data["clusters"], cmap='viridis', s=40, zorder=11)

        # Text labels on 3D points
        for b_pt, label in zip(bulk_3d, self.correlation_data["labels"]):
            ax1.text(b_pt[0], b_pt[1], b_pt[2] + 0.02, f"${label}$", fontsize=10, zorder=12)

        max_val = np.max(np.abs(np.vstack(faces_3d))) * 1.1
        ax1.set_xlim([-max_val, max_val])
        ax1.set_ylim([-max_val, max_val])
        ax1.set_zlim([-max_val, max_val])
        ax1.set_box_aspect((1, 1, 1))
        ax1.set_title(f"Bulk 3D BZ & {self.miller_index} Projection")

        # ==========================================
        # Subplot 2: 2D Surface Brillouin Zone Flat
        # ==========================================
        ax2 = fig.add_subplot(122)

        perim = np.vstack((faces_2d, faces_2d[0]))
        ax2.plot(perim[:, 0], perim[:, 1], 'k-', linewidth=2)
        ax2.scatter(proj_2d[:, 0], proj_2d[:, 1], c=self.correlation_data["clusters"], cmap='viridis', s=70, zorder=5)

        for p_pt, label in zip(proj_2d, self.correlation_data["labels"]):
            ax2.text(p_pt[0] + 0.02, p_pt[1] + 0.02, f"${label}$", fontsize=12)

        ax2.set_title(f"Projected 2D Surface Brillouin Zone {self.miller_index}")
        ax2.set_aspect('equal')

        plt.tight_layout()
        plt.show()

