"""
Formal definition of the unit cylinder U = [0,1) × [0,1]² with identification.
"""
import numpy as np
from typing import Tuple, Optional
from dataclasses import dataclass

@dataclass
class UnitCylinder:
    """
    The canonical unit cylinder U = [0,1) × [0,1]² with identification.
    
    This space is topologically equivalent to S¹ × [0,1]².
    """
    def __init__(self):
        self.dimension = 3
        self.name = "UnitCylinder"
        
        # Identification: (0, s, v) ~ (1, s, v)
        self.identification_boundary = 0.0  # u1 = 0 is identified with u1 = 1
    
    def contains(self, point: np.ndarray) -> bool:
        """
        Check if point is in U.
        
        Parameters
        ----------
        point : np.ndarray
            Point (u1, u2, u3).
            
        Returns
        -------
        bool
            True if point ∈ U.
        """
        if point.shape != (3,):
            return False
        
        u1, u2, u3 = point
        
        # u1 ∈ [0,1) with identification
        if not (0.0 <= u1 < 1.0):
            return False
        
        # u2, u3 ∈ [0,1]
        if not (0.0 <= u2 <= 1.0 and 0.0 <= u3 <= 1.0):
            return False
        
        return True
    
    def distance(self, point1: np.ndarray, point2: np.ndarray) -> float:
        """
        Cylindrical distance in U.
        
        Parameters
        ----------
        point1, point2 : np.ndarray
            Points in U.
            
        Returns
        -------
        float
            Distance respecting circular coordinate.
        """
        assert self.contains(point1) and self.contains(point2), \
            "Points must be in unit cylinder"
        
        u1_1, u2_1, u3_1 = point1
        u1_2, u2_2, u3_2 = point2
        
        # Circular distance for u1
        delta_u1 = abs(u1_1 - u1_2)
        circular_distance_u1 = min(delta_u1, 1.0 - delta_u1)
        
        # Euclidean distance for u2, u3
        euclidean_distance = np.sqrt((u2_1 - u2_2)**2 + (u3_1 - u3_2)**2)
        
        # Combined distance
        return np.sqrt(circular_distance_u1**2 + euclidean_distance**2)
    
    def to_circle_representation(self, point: np.ndarray) -> Tuple[float, float, float]:
        """
        Convert to angular representation on S¹.
        
        Parameters
        ----------
        point : np.ndarray
            Point (u1, u2, u3) in U.
            
        Returns
        -------
        Tuple[float, float, float]
            (angle, radius1, radius2) where angle ∈ [0, 2π).
        """
        u1, u2, u3 = point
        angle = 2 * np.pi * u1
        return angle, u2, u3
    
    def from_circle_representation(self, angle: float, radius1: float, radius2: float) -> np.ndarray:
        """
        Convert from angular representation.
        
        Parameters
        ----------
        angle : float
            Angle in [0, 2π).
        radius1, radius2 : float
            Coordinates in [0,1].
            
        Returns
        -------
        np.ndarray
            Point in U.
        """
        u1 = (angle % (2 * np.pi)) / (2 * np.pi)
        return np.array([u1, radius1, radius2])
    
    def random_point(self, n_samples: int = 1) -> np.ndarray:
        """
        Sample uniformly from U.
        
        Parameters
        ----------
        n_samples : int
            Number of samples.
            
        Returns
        -------
        np.ndarray
            Array of shape (n_samples, 3).
        """
        u1 = np.random.uniform(0, 1, n_samples)
        u2 = np.random.uniform(0, 1, n_samples)
        u3 = np.random.uniform(0, 1, n_samples)
        return np.column_stack([u1, u2, u3])
    
    def identify_boundary(self, points: np.ndarray) -> np.ndarray:
        """
        Apply boundary identification: map points near u1=1 to u1=0.
        
        Parameters
        ----------
        points : np.ndarray
            Points in [0,1] × [0,1]².
            
        Returns
        -------
        np.ndarray
            Points in U with proper identification.
        """
        points = points.copy()
        u1 = points[:, 0]
        
        # Map points with u1 ≈ 1 to u1 ≈ 0
        near_one = u1 > 0.95
        points[near_one, 0] = points[near_one, 0] - 1.0
        
        return points