import numpy as np
import torch
import torch.nn as nn
from sklearn.decomposition import PCA
from typing import Tuple, Optional
import warnings

class ColorManifoldEmbedding:
    """
    Implements the HSV color space embedding φ from the paper.
    Uses PCA to project concept vectors to structured HSV space.
    """
    
    def __init__(self, concept_dim: int = 8):
        self.concept_dim = concept_dim
        self.pca = PCA(n_components=concept_dim)
        self.is_fitted = False
        self.mu = None
        self.eigenvalues = None
        
    def fit(self, concept_vectors) -> None:
        """
        Fit PCA transformation on concept vectors.
        
        Args:
            concept_vectors: [n_samples, concept_dim] array of concept activations
        """
        self.mu = concept_vectors.mean(axis=0)
        self.pca.fit(concept_vectors)
        self.eigenvalues = self.pca.explained_variance_
        self.is_fitted = True
        
    def transform(self, concept_vectors):
        """
        Transform concept vectors to HSV color space.
        
        Args:
            concept_vectors: [n_samples, concept_dim] array of concept activations
            
        Returns:
            hsv_coordinates: [n_samples, 3] array of (H, S, V) coordinates
        """
        if not self.is_fitted:
            raise ValueError("ColorManifoldEmbedding must be fitted before transformation")
            
        # Project to principal components (Equation 10)
        p_scores = self.pca.transform(concept_vectors - self.mu)
        
        # Compute HSV coordinates (Equations 11-14)
        h = (np.arctan2(p_scores[:, 1], p_scores[:, 0]) + 2 * np.pi) / (2 * np.pi) % 1.0
        # Saturation: normalize by maximum possible value in training data
        raw_s = np.sqrt(np.sum(p_scores**2 / self.eigenvalues, axis=1))
        max_s = np.max(raw_s)
        s = np.clip(raw_s / max_s, 0, 1)  # Now bounded [0, 1]
        # Value: normalize
        raw_v = np.tanh(np.linalg.norm(concept_vectors, axis=1) / np.std(concept_vectors))
        max_v = np.max(raw_v)
        v = np.clip(raw_v / max_v, 0, 1)
        
        hsv_coordinates = np.column_stack([h, s, v])
        return hsv_coordinates
    
    def fit_transform(self, concept_vectors):
        """Fit and transform in one step."""
        self.fit(concept_vectors)
        return self.transform(concept_vectors)
    
    def smooth_trajectory(self, hsv_trajectory, alpha: float = 0.8):
        """
        Apply recurrent smoothing to HSV trajectory (Equation 15).
        
        Args:
            hsv_trajectory: [T, 3] array of HSV coordinates over time
            alpha: smoothing factor (0 < alpha < 1)
            
        Returns:
            smoothed_trajectory: [T, 3] array of smoothed HSV coordinates
        """
        smoothed = np.zeros_like(hsv_trajectory)
        smoothed[0] = hsv_trajectory[0]
        
        for t in range(1, hsv_trajectory.shape[0]):
            # Handle circular hue dimension
            current_hue = hsv_trajectory[t, 0]
            prev_hue = smoothed[t-1, 0]
            
            # Find shortest path for hue (circular continuity)
            hue_diff = current_hue - prev_hue
            if hue_diff > 0.5:
                current_hue -= 1.0
            elif hue_diff < -0.5:
                current_hue += 1.0
                
            smoothed_hue = alpha * prev_hue + (1 - alpha) * current_hue
            smoothed_hue = smoothed_hue % 1.0  # Wrap to [0, 1)
            
            # Smooth saturation and value normally
            smoothed_s = alpha * smoothed[t-1, 1] + (1 - alpha) * hsv_trajectory[t, 1]
            smoothed_v = alpha * smoothed[t-1, 2] + (1 - alpha) * hsv_trajectory[t, 2]
            
            smoothed[t] = [smoothed_hue, smoothed_s, smoothed_v]
            
        return smoothed