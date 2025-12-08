"""
Persistent homology computation for topological validation.
"""
import numpy as np
from typing import Tuple, Dict, Optional
import matplotlib.pyplot as plt
from giotto_tda.homology import VietorisRipsPersistence
from giotto_tda.diagrams import PersistenceEntropy, BettiCurve
from sklearn.preprocessing import MinMaxScaler


def compute_persistence_diagram(point_cloud: np.ndarray, 
                               homology_dimensions: Tuple[int, ...] = (0, 1),
                               max_edge_length: float = 1.0) -> np.ndarray:
    """
    Compute persistence diagram for a point cloud.
    
    Parameters
    ----------
    point_cloud : np.ndarray
        Point cloud of shape (n_points, n_dimensions).
    homology_dimensions : Tuple[int, ...]
        Homology dimensions to compute.
    max_edge_length : float
        Maximum edge length in Vietoris-Rips complex.
        
    Returns
    -------
    np.ndarray
        Persistence diagram as array of shape (n_points, 3)
        where each row is (birth, death, dimension).
    """
    # Normalize point cloud
    scaler = MinMaxScaler()
    point_cloud_normalized = scaler.fit_transform(point_cloud)
    
    # Reshape for giotto-tda
    point_cloud_reshaped = point_cloud_normalized.reshape(1, *point_cloud_normalized.shape)
    
    # Compute persistence diagram
    vr = VietorisRipsPersistence(
        homology_dimensions=homology_dimensions,
        max_edge_length=max_edge_length,
        n_jobs=-1
    )
    
    diagrams = vr.fit_transform(point_cloud_reshaped)
    
    return diagrams[0]


def has_prominent_loop(persistence_diagram: np.ndarray,
                      persistence_threshold: float = 0.1,
                      min_persistence: float = 0.05) -> Dict:
    """
    Check if persistence diagram has a prominent 1-dimensional loop.
    
    Parameters
    ----------
    persistence_diagram : np.ndarray
        Persistence diagram from compute_persistence_diagram.
    persistence_threshold : float
        Threshold for considering a feature prominent.
    min_persistence : float
        Minimum persistence to consider.
        
    Returns
    -------
    Dict
        Analysis results.
    """
    # Filter for dimension 1 (loops)
    dim1_features = persistence_diagram[persistence_diagram[:, 2] == 1]
    
    if len(dim1_features) == 0:
        return {
            'has_prominent_loop': False,
            'n_loops': 0,
            'max_persistence': 0.0,
            'prominent_loops': []
        }
    
    # Compute persistence (death - birth)
    persistences = dim1_features[:, 1] - dim1_features[:, 0]
    
    # Find prominent loops (persistence > threshold)
    prominent_mask = persistences > persistence_threshold
    prominent_loops = dim1_features[prominent_mask]
    
    # Also consider loops with persistence > min_persistence
    valid_mask = persistences > min_persistence
    valid_loops = dim1_features[valid_mask]
    
    return {
        'has_prominent_loop': len(prominent_loops) > 0,
        'n_loops': len(dim1_features),
        'n_valid_loops': len(valid_loops),
        'n_prominent_loops': len(prominent_loops),
        'max_persistence': np.max(persistences) if len(persistences) > 0 else 0.0,
        'mean_persistence': np.mean(persistences) if len(persistences) > 0 else 0.0,
        'prominent_loops': prominent_loops,
        'all_loops': dim1_features,
        'persistences': persistences
    }


def topological_similarity(diagram1: np.ndarray, 
                         diagram2: np.ndarray,
                         dimension: int = 1) -> Dict:
    """
    Compute topological similarity between two persistence diagrams.
    
    Parameters
    ----------
    diagram1, diagram2 : np.ndarray
        Persistence diagrams.
    dimension : int
        Homology dimension to compare.
        
    Returns
    -------
    Dict
        Similarity metrics.
    """
    # Filter for specified dimension
    dim1_features1 = diagram1[diagram1[:, 2] == dimension]
    dim1_features2 = diagram2[diagram2[:, 2] == dimension]
    
    # Compute persistences
    persistences1 = dim1_features1[:, 1] - dim1_features1[:, 0] if len(dim1_features1) > 0 else np.array([0.0])
    persistences2 = dim1_features2[:, 1] - dim1_features2[:, 0] if len(dim1_features2) > 0 else np.array([0.0])
    
    # Simple similarity metrics
    max_persistence_diff = abs(np.max(persistences1) - np.max(persistences2)) if len(persistences1) > 0 and len(persistences2) > 0 else 1.0
    n_features_diff = abs(len(dim1_features1) - len(dim1_features2))
    
    # Wasserstein-like distance (simplified)
    # Sort persistences and compute L1 distance
    sorted_p1 = np.sort(persistences1)[-5:]  # Top 5 most persistent features
    sorted_p2 = np.sort(persistences2)[-5:]
    
    # Pad with zeros if needed
    max_len = max(len(sorted_p1), len(sorted_p2))
    padded_p1 = np.pad(sorted_p1, (0, max_len - len(sorted_p1)), constant_values=0)
    padded_p2 = np.pad(sorted_p2, (0, max_len - len(sorted_p2)), constant_values=0)
    
    wasserstein_distance = np.sum(np.abs(padded_p1 - padded_p2))
    
    return {
        'max_persistence_diff': max_persistence_diff,
        'n_features_diff': n_features_diff,
        'wasserstein_distance': wasserstein_distance,
        'topology_preserved': wasserstein_distance < 0.2,  # Threshold
        'diagram1_features': len(dim1_features1),
        'diagram2_features': len(dim1_features2)
    }


def plot_persistence_diagram(diagram: np.ndarray,
                           title: str = "Persistence Diagram",
                           save_path: Optional[str] = None):
    """
    Plot persistence diagram.
    
    Parameters
    ----------
    diagram : np.ndarray
        Persistence diagram.
    title : str
        Plot title.
    save_path : Optional[str]
        Path to save figure.
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Plot diagonal
    max_death = np.max(diagram[:, 1])
    ax.plot([0, max_death], [0, max_death], 'k--', alpha=0.5, label='Diagonal')
    
    # Plot points by dimension
    colors = ['blue', 'red', 'green']  # for dim 0, 1, 2
    labels = ['H₀ (Components)', 'H₁ (Loops)', 'H₂ (Voids)']
    
    for dim in [0, 1, 2]:
        dim_points = diagram[diagram[:, 2] == dim]
        if len(dim_points) > 0:
            ax.scatter(dim_points[:, 0], dim_points[:, 1], 
                      color=colors[dim], s=50, alpha=0.7, label=labels[dim])
    
    ax.set_xlabel('Birth')
    ax.set_ylabel('Death')
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig, ax


def analyze_trajectory_topology(trajectory: np.ndarray,
                               homology_dimensions: Tuple[int, ...] = (0, 1),
                               persistence_threshold: float = 0.1) -> Dict:
    """
    Complete topological analysis of a trajectory.
    
    Parameters
    ----------
    trajectory : np.ndarray
        Trajectory points.
    homology_dimensions : Tuple[int, ...]
        Homology dimensions to compute.
    persistence_threshold : float
        Threshold for prominent features.
        
    Returns
    -------
    Dict
        Complete topological analysis.
    """
    # Compute persistence diagram
    diagram = compute_persistence_diagram(trajectory, homology_dimensions)
    
    # Analyze loops
    loop_analysis = has_prominent_loop(diagram, persistence_threshold)
    
    # Compute Betti numbers (simplified)
    betti_numbers = {}
    for dim in homology_dimensions:
        dim_features = diagram[diagram[:, 2] == dim]
        # Count features with persistence > 0.01
        persistent_features = dim_features[(dim_features[:, 1] - dim_features[:, 0]) > 0.01]
        betti_numbers[f'betti_{dim}'] = len(persistent_features)
    
    # Compute topological entropy (measure of complexity)
    entropy_calculator = PersistenceEntropy()
    diagram_reshaped = diagram.reshape(1, *diagram.shape)
    entropy = entropy_calculator.fit_transform(diagram_reshaped)[0]
    
    return {
        'persistence_diagram': diagram,
        'loop_analysis': loop_analysis,
        'betti_numbers': betti_numbers,
        'topological_entropy': entropy,
        'has_loop': loop_analysis['has_prominent_loop'],
        'max_persistence': loop_analysis['max_persistence']
    }