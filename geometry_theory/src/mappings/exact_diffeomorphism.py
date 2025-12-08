"""
Exact diffeomorphism Ψ: U → H between unit cylinder and HSV manifold.
"""
import sys 
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '../../..'))

import numpy as np
from typing import Tuple, Optional
import plotly.graph_objects as go 
from geometry_theory.src.spaces.unit_cylinder import UnitCylinder
from geometry_theory.src.spaces.hsv_manifold import HSVManifold

class ExactDiffeomorphism:
    """
    Exact diffeomorphism Ψ: U → H.
    
    This is an isometric diffeomorphism between the unit cylinder U
    and the HSV manifold H.
    """
    def __init__(self, hsv_radius: float = 1/(2*np.pi)):
        """
        Parameters
        ----------
        hsv_radius : float
            Radius of HSV cylinder. For isometry, set to 1/(2π).
        """
        self.unit_cylinder = UnitCylinder()
        self.hsv_manifold = HSVManifold(radius=hsv_radius)
        
        # For isometry: scaling factor from u1 to hue
        self.scale_factor = 2 * np.pi
    
    def forward(self, point_u: np.ndarray) -> np.ndarray:
        """
        Apply Ψ: U → H.
        
        Parameters
        ----------
        point_u : np.ndarray
            Point in U (u1, u2, u3).
            
        Returns
        -------
        np.ndarray
            Point in H (hue, saturation, value).
        """
        assert self.unit_cylinder.contains(point_u), "Point must be in U"
        
        u1, u2, u3 = point_u
        
        # Map u1 ∈ [0,1) to hue ∈ [0, 2π)
        hue = self.scale_factor * u1
        
        # Saturation and value map directly
        saturation = u2
        value = u3
        
        return np.array([hue, saturation, value])
    
    def inverse(self, point_h: np.ndarray) -> np.ndarray:
        """
        Apply Ψ⁻¹: H → U.
        
        Parameters
        ----------
        point_h : np.ndarray
            Point in H (hue, saturation, value).
            
        Returns
        -------
        np.ndarray
            Point in U (u1, u2, u3).
        """
        assert self.hsv_manifold.contains(point_h), "Point must be in H"
        
        hue, saturation, value = point_h
        
        # Map hue ∈ [0, 2π) to u1 ∈ [0,1)
        u1 = (hue % self.scale_factor) / self.scale_factor
        
        # Saturation and value map directly
        u2 = saturation
        u3 = value
        
        return np.array([u1, u2, u3])
    
    def forward_trajectory(self, trajectory_u: np.ndarray) -> np.ndarray:
        """
        Apply Ψ to a trajectory in U.
        
        Parameters
        ----------
        trajectory_u : np.ndarray
            Trajectory in U, shape (n_steps, 3).
            
        Returns
        -------
        np.ndarray
            Trajectory in H, shape (n_steps, 3).
        """
        return np.array([self.forward(p) for p in trajectory_u])
    
    def inverse_trajectory(self, trajectory_h: np.ndarray) -> np.ndarray:
        """
        Apply Ψ⁻¹ to a trajectory in H.
        
        Parameters
        ----------
        trajectory_h : np.ndarray
            Trajectory in H, shape (n_steps, 3).
            
        Returns
        -------
        np.ndarray
            Trajectory in U, shape (n_steps, 3).
        """
        return np.array([self.inverse(p) for p in trajectory_h])
    
    def jacobian(self, point_u: np.ndarray) -> np.ndarray:
        """
        Compute Jacobian matrix J_Ψ at a point.
        
        Parameters
        ----------
        point_u : np.ndarray
            Point in U.
            
        Returns
        -------
        np.ndarray
            Jacobian matrix (3×3).
        """
        u1, u2, u3 = point_u
        
        # Jacobian of Ψ: (hue, saturation, value) = (2π·u1, u2, u3)
        J = np.zeros((3, 3))
        J[0, 0] = self.scale_factor  # ∂hue/∂u1
        J[1, 1] = 1.0                # ∂saturation/∂u2
        J[2, 2] = 1.0                # ∂value/∂u3
        
        return J
    
    def inverse_jacobian(self, point_h: np.ndarray) -> np.ndarray:
        """
        Compute Jacobian matrix J_Ψ⁻¹ at a point.
        
        Parameters
        ----------
        point_h : np.ndarray
            Point in H.
            
        Returns
        -------
        np.ndarray
            Jacobian matrix (3×3).
        """
        # Jacobian of Ψ⁻¹: (u1, u2, u3) = (hue/2π, saturation, value)
        J_inv = np.zeros((3, 3))
        J_inv[0, 0] = 1.0 / self.scale_factor  # ∂u1/∂hue
        J_inv[1, 1] = 1.0                      # ∂u2/∂saturation
        J_inv[2, 2] = 1.0                      # ∂u3/∂value
        
        return J_inv
    
    def test_diffeomorphism(self, n_points: int = 100) -> dict:
        """
        Test that Ψ is indeed a diffeomorphism.
        
        Parameters
        ----------
        n_points : int
            Number of test points.
            
        Returns
        -------
        dict
            Test results.
        """
        results = {
            "forward_inverse_error": [],
            "inverse_forward_error": [],
            "jacobian_product_error": []
        }
        
        # Test random points
        test_points_u = self.unit_cylinder.random_point(n_points)
        
        for point_u in test_points_u:
            # Test forward then inverse
            point_h = self.forward(point_u)
            point_u_recon = self.inverse(point_h)
            error1 = np.linalg.norm(point_u - point_u_recon)
            results["forward_inverse_error"].append(error1)
            
            # Test inverse then forward
            point_h = self.hsv_manifold.random_point(1)[0]
            point_u = self.inverse(point_h)
            point_h_recon = self.forward(point_u)
            error2 = np.linalg.norm(point_h - point_h_recon)
            results["inverse_forward_error"].append(error2)
            
            # Test Jacobian: J_Ψ * J_Ψ⁻¹ should be identity
            J = self.jacobian(point_u)
            J_inv = self.inverse_jacobian(point_h)
            J_product = J @ J_inv
            identity_error = np.linalg.norm(J_product - np.eye(3))
            results["jacobian_product_error"].append(identity_error)
        
        # Compute statistics
        stats = {}
        for key, errors in results.items():
            stats[f"{key}_mean"] = np.mean(errors)
            stats[f"{key}_std"] = np.std(errors)
            stats[f"{key}_max"] = np.max(errors)
        
        return stats
    
    def visualize_mapping(self, n_points: int = 100) -> Tuple[go.Figure, go.Figure]:
        """
        Visualize the diffeomorphism.
        
        Parameters
        ----------
        n_points : int
            Number of points to visualize.
            
        Returns
        -------
        Tuple[go.Figure, go.Figure]
            Figures for U and H spaces.
        """
        # Generate points in U
        points_u = self.unit_cylinder.random_point(n_points)
        
        # Map to H
        points_h = self.forward_trajectory(points_u)
        
        # Visualize U
        fig_u = go.Figure()
        
        # Convert U points to cartesian-like for visualization
        angles_u, radii_u, heights_u = points_u.T
        x_u = radii_u * np.cos(2*np.pi*angles_u)
        y_u = radii_u * np.sin(2*np.pi*angles_u)
        z_u = heights_u
        
        fig_u.add_trace(go.Scatter3d(
            x=x_u, y=y_u, z=z_u,
            mode='markers',
            marker=dict(
                size=4,
                color=angles_u,  # Color by u1 (circular coordinate)
                colorscale='Rainbow',
                opacity=0.8
            ),
            name="Points in U"
        ))
        
        fig_u.update_layout(
            title="Unit Cylinder U = [0,1) × [0,1]²",
            scene=dict(
                xaxis_title='X',
                yaxis_title='Y',
                zaxis_title='u₃ (Value-like)'
            ),
            width=600,
            height=500
        )
        
        # Visualize H
        fig_h = self.hsv_manifold.visualize_3d(points=points_h, show_cylinder=True)
        fig_h.update_layout(title="HSV Manifold H = S¹ × [0,1]²")
        
        return fig_u, fig_h