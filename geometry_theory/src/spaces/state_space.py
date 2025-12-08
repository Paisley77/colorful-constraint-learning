"""
Formal definition of the state space S ⊆ ℝⁿ with Euclidean structure.
"""
import numpy as np
from typing import Union, List, Tuple, Optional
from dataclasses import dataclass

@dataclass
class StateSpace:
    """
    Represents the state space S ⊆ ℝⁿ with Euclidean topology.
    
    Parameters
    ----------
    dimension : int
        Dimension n of the state space.
    bounds : Optional[List[Tuple[float, float]]]
        Optional bounds for each dimension.
    name : str
        Name of the state space (e.g., "PendulumStateSpace").
    """
    dimension: int
    bounds: Optional[List[Tuple[float, float]]] = None
    name: str = "StateSpace"
    
    def __post_init__(self):
        if self.bounds is not None:
            assert len(self.bounds) == self.dimension, \
                f"Bounds must have length {self.dimension}, got {len(self.bounds)}"
    
    def contains(self, point: np.ndarray) -> bool:
        """
        Check if a point is in the state space.
        
        Parameters
        ----------
        point : np.ndarray
            Point in ℝⁿ.
            
        Returns
        -------
        bool
            True if point is in S.
        """
        if point.shape != (self.dimension,):
            return False
        
        if self.bounds is not None:
            for i, (lower, upper) in enumerate(self.bounds):
                if not (lower <= point[i] <= upper):
                    return False
        return True
    
    def distance(self, point1: np.ndarray, point2: np.ndarray) -> float:
        """
        Euclidean distance between two points in S.
        
        Parameters
        ----------
        point1, point2 : np.ndarray
            Points in ℝⁿ.
            
        Returns
        -------
        float
            Euclidean distance.
        """
        assert self.contains(point1) and self.contains(point2), \
            "Points must be in state space"
        return np.linalg.norm(point1 - point2)
    
    def random_point(self, n_samples: int = 1) -> np.ndarray:
        """
        Sample random points uniformly from the bounded space.
        
        Parameters
        ----------
        n_samples : int
            Number of samples.
            
        Returns
        -------
        np.ndarray
            Array of shape (n_samples, dimension).
        """
        if self.bounds is None:
            # Sample from unit ball if no bounds specified
            samples = np.random.randn(n_samples, self.dimension)
            samples /= np.linalg.norm(samples, axis=1, keepdims=True)
        else:
            samples = np.zeros((n_samples, self.dimension))
            for i, (lower, upper) in enumerate(self.bounds):
                samples[:, i] = np.random.uniform(lower, upper, n_samples)
        return samples
    
    def project_to_manifold(self, point: np.ndarray) -> np.ndarray:
        """
        Project a point onto the data manifold (if known).
        For now, identity projection.
        
        Parameters
        ----------
        point : np.ndarray
            Point in ℝⁿ.
            
        Returns
        -------
        np.ndarray
            Projected point on manifold.
        """
        return point.copy()


class PendulumStateSpace(StateSpace):
    """
    Specialized state space for inverted pendulum.
    State: [x, ẋ, θ, θ̇]ᵀ
    """
    def __init__(self):
        # Typical bounds for pendulum states
        bounds = [
            (-2.4, 2.4),      # x: cart position (m)
            (-5.0, 5.0),      # ẋ: cart velocity (m/s)
            (-np.pi, np.pi),  # θ: pole angle (rad)
            (-10.0, 10.0)     # θ̇: pole angular velocity (rad/s)
        ]
        super().__init__(dimension=4, bounds=bounds, name="PendulumStateSpace")
    
    def angle_normalize(self, theta: float) -> float:
        """
        Normalize angle to [-π, π).
        
        Parameters
        ----------
        theta : float
            Angle in radians.
            
        Returns
        -------
        float
            Normalized angle.
        """
        return ((theta + np.pi) % (2 * np.pi)) - np.pi
    
    def is_safe(self, state: np.ndarray, max_angle: float = 0.35) -> bool:
        """
        Check if state satisfies pendulum safety constraint.
        
        Parameters
        ----------
        state : np.ndarray
            State vector [x, ẋ, θ, θ̇].
        max_angle : float
            Maximum allowable pole angle (rad).
            
        Returns
        -------
        bool
            True if state is safe.
        """
        theta = state[2]
        return abs(self.angle_normalize(theta)) <= max_angle