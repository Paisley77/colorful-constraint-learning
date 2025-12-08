"""
Formal definition of the HSV semantic manifold H = S¹ × [0,1]².
"""
import numpy as np
from typing import Tuple, Optional
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import plotly.graph_objects as go

class HSVManifold:
    """
    HSV semantic manifold H = S¹ × [0,1]².
    
    A point h ∈ H is (hue, saturation, value) where:
    - hue ∈ S¹ (circle, [0, 2π))
    - saturation ∈ [0,1] (radial coordinate)
    - value ∈ [0,1] (vertical coordinate)
    """
    def __init__(self, radius: float = 1.0, height: float = 1.0):
        """
        Parameters
        ----------
        radius : float
            Radius of the hue circle.
        height : float
            Height of the value axis.
        """
        self.radius = radius
        self.height = height
        self.dimension = 3
        self.name = "HSVManifold"
    
    def contains(self, point: np.ndarray) -> bool:
        """
        Check if point is in H.
        
        Parameters
        ----------
        point : np.ndarray
            Point (hue, saturation, value).
            
        Returns
        -------
        bool
            True if point ∈ H.
        """
        if point.shape != (3,):
            return False
        
        hue, saturation, value = point
        
        # hue is circular, any real number is valid (interpreted mod 2π)
        # saturation, value ∈ [0,1]
        if not (0.0 <= saturation <= 1.0 and 0.0 <= value <= 1.0):
            return False
        
        return True
    
    def to_cartesian(self, point: np.ndarray) -> np.ndarray:
        """
        Convert HSV point to Cartesian coordinates for visualization.
        
        Parameters
        ----------
        point : np.ndarray
            Point (hue, saturation, value) in H.
            
        Returns
        -------
        np.ndarray
            Cartesian coordinates (x, y, z).
        """
        hue, saturation, value = point
        
        # Convert to cylindrical coordinates
        x = self.radius * saturation * np.cos(hue)
        y = self.radius * saturation * np.sin(hue)
        z = self.height * value
        
        return np.array([x, y, z])
    
    def from_cartesian(self, cartesian: np.ndarray) -> np.ndarray:
        """
        Convert Cartesian coordinates to HSV.
        
        Parameters
        ----------
        cartesian : np.ndarray
            Cartesian coordinates (x, y, z).
            
        Returns
        -------
        np.ndarray
            HSV point (hue, saturation, value).
        """
        x, y, z = cartesian
        
        # Convert to cylindrical
        hue = np.arctan2(y, x)  # ∈ (-π, π]
        if hue < 0:
            hue += 2 * np.pi  # ∈ [0, 2π)
        
        saturation = np.sqrt(x**2 + y**2) / self.radius
        saturation = np.clip(saturation, 0, 1)
        
        value = z / self.height
        value = np.clip(value, 0, 1)
        
        return np.array([hue, saturation, value])
    
    def distance(self, point1: np.ndarray, point2: np.ndarray, 
                 metric: str = "cylindrical") -> float:
        """
        Distance on the HSV manifold.
        
        Parameters
        ----------
        point1, point2 : np.ndarray
            Points in H.
        metric : str
            Type of distance: "cylindrical" or "cartesian".
            
        Returns
        -------
        float
            Distance between points.
        """
        assert self.contains(point1) and self.contains(point2), \
            "Points must be in HSV manifold"
        
        if metric == "cylindrical":
            hue1, sat1, val1 = point1
            hue2, sat2, val2 = point2
            
            # Circular distance for hue
            delta_hue = abs((hue1 % (2*np.pi)) - (hue2 % (2*np.pi)))
            circular_dist_hue = min(delta_hue, 2*np.pi - delta_hue)
            
            # Scale hue distance by radius
            hue_distance = self.radius * circular_dist_hue / (2 * np.pi)
            
            # Radial distance for saturation
            sat_distance = abs(sat1 - sat2)
            
            # Vertical distance for value
            val_distance = abs(val1 - val2)
            
            return np.sqrt(hue_distance**2 + sat_distance**2 + val_distance**2)
        
        elif metric == "cartesian":
            cart1 = self.to_cartesian(point1)
            cart2 = self.to_cartesian(point2)
            return np.linalg.norm(cart1 - cart2)
        
        else:
            raise ValueError(f"Unknown metric: {metric}")
    
    def random_point(self, n_samples: int = 1) -> np.ndarray:
        """
        Sample uniformly from H.
        
        Parameters
        ----------
        n_samples : int
            Number of samples.
            
        Returns
        -------
        np.ndarray
            Array of shape (n_samples, 3).
        """
        hue = np.random.uniform(0, 2*np.pi, n_samples)
        saturation = np.random.uniform(0, 1, n_samples)
        value = np.random.uniform(0, 1, n_samples)
        return np.column_stack([hue, saturation, value])
    
    def visualize_3d(self, points: Optional[np.ndarray] = None, 
                     trajectory: Optional[np.ndarray] = None,
                     show_cylinder: bool = True,
                     title: str = "HSV Semantic Manifold") -> go.Figure:
        """
        Create interactive 3D visualization using Plotly.
        
        Parameters
        ----------
        points : np.ndarray, optional
            Points to plot (shape: (n_points, 3)).
        trajectory : np.ndarray, optional
            Trajectory to plot as line (shape: (n_steps, 3)).
        show_cylinder : bool
            Whether to show the cylinder surface.
        title : str
            Plot title.
            
        Returns
        -------
        go.Figure
            Plotly figure.
        """
        fig = go.Figure()
        
        # Create cylinder surface
        if show_cylinder:
            # Generate cylinder points
            theta = np.linspace(0, 2*np.pi, 50)
            z = np.linspace(0, self.height, 20)
            theta_grid, z_grid = np.meshgrid(theta, z)
            
            # Cylinder coordinates
            x_cyl = self.radius * np.cos(theta_grid)
            y_cyl = self.radius * np.sin(theta_grid)
            z_cyl = z_grid
            
            # Add cylinder surface (semi-transparent)
            fig.add_trace(go.Surface(
                x=x_cyl, y=y_cyl, z=z_cyl,
                opacity=0.2,
                colorscale='Viridis',
                showscale=False,
                name="HSV Cylinder"
            ))
        
        # Plot points if provided
        if points is not None:
            # Convert to Cartesian for plotting
            cartesian_points = np.array([self.to_cartesian(p) for p in points])
            x, y, z = cartesian_points.T
            
            # Create color based on HSV values
            colors = []
            for point in points:
                hue, sat, val = point
                # Convert HSV to RGB for plotting
                from matplotlib.colors import hsv_to_rgb
                rgb = hsv_to_rgb([hue/(2*np.pi), sat, val])
                colors.append(f'rgb({int(rgb[0]*255)}, {int(rgb[1]*255)}, {int(rgb[2]*255)})')
            
            fig.add_trace(go.Scatter3d(
                x=x, y=y, z=z,
                mode='markers',
                marker=dict(
                    size=5,
                    color=colors,
                    opacity=0.8
                ),
                name="Semantic Points"
            ))
        
        # Plot trajectory if provided
        if trajectory is not None:
            cartesian_traj = np.array([self.to_cartesian(p) for p in trajectory])
            x_traj, y_traj, z_traj = cartesian_traj.T
            
            # Color trajectory by hue
            hues = trajectory[:, 0] / (2 * np.pi)  # Normalize to [0,1]
            
            fig.add_trace(go.Scatter3d(
                x=x_traj, y=y_traj, z=z_traj,
                mode='lines',
                line=dict(
                    width=4,
                    color=hues,
                    colorscale='Rainbow',
                    cmin=0,
                    cmax=1
                ),
                name="Semantic Trajectory"
            ))
        
        # Set layout
        fig.update_layout(
            title=title,
            scene=dict(
                xaxis_title='X',
                yaxis_title='Y',
                zaxis_title='Value',
                aspectmode='data'
            ),
            width=800,
            height=600
        )
        
        return fig
    
    def hue_to_color(self, hue: float) -> str:
        """
        Convert hue value to CSS color string.
        
        Parameters
        ----------
        hue : float
            Hue in [0, 2π).
            
        Returns
        -------
        str
            CSS color string.
        """
        # Convert hue to [0, 1]
        h = hue / (2 * np.pi)
        
        # Simple HSV to RGB conversion
        from matplotlib.colors import hsv_to_rgb
        rgb = hsv_to_rgb([h, 0.8, 0.9])  # Fixed saturation and value
        return f'rgb({int(rgb[0]*255)}, {int(rgb[1]*255)}, {int(rgb[2]*255)})'