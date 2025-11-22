import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
from matplotlib.colors import hsv_to_rgb
import pickle
import matplotlib.pyplot as plt
from pathlib import Path
from src.embedding.color_manifold import ColorManifoldEmbedding
from src.perception.concept_network import ConceptNetwork
import torch

def load_demonstrations():
    """Load expert and violator demonstrations."""
    with open("data/expert/expert_demos.pkl", 'rb') as f:
        expert_trajs = pickle.load(f)
    with open("data/violator/violator_demos.pkl", 'rb') as f:
        violator_trajs = pickle.load(f)
    return expert_trajs, violator_trajs

def extract_concept_vectors(trajectories, concept_net):
    """Extract concept vectors from trajectories using the concept network."""
    all_vectors = []
    all_trajectory_colors = []
    
    for traj in trajectories:
        states = np.array(traj['states'])
        with torch.no_grad():
            concept_vectors = concept_net(torch.FloatTensor(states)).numpy()
        all_vectors.extend(concept_vectors)
        all_trajectory_colors.append(concept_vectors)
    
    return np.array(all_vectors), all_trajectory_colors

def visualize_manifold_2d():
    """Create 2D visualization of the HSV manifold."""
    print("Loading demonstrations...")
    expert_trajs, violator_trajs = load_demonstrations()
    
    # Initialize concept network
    model_path = os.path.join(os.path.dirname(__file__), '../', 'results/models/concept_net_stage1.pth')
    concept_net = ConceptNetwork(state_dim=4, concept_dim=8)
    concept_net.load_state_dict(torch.load(model_path))
    
    print("Extracting concept vectors...")
    expert_vectors, expert_traj_vectors = extract_concept_vectors(expert_trajs, concept_net)
    violator_vectors, violator_traj_vectors = extract_concept_vectors(violator_trajs, concept_net)
    
    # Combine all vectors for PCA fitting
    # all_vectors = np.vstack([expert_vectors, violator_vectors])
    
    print("Fitting color manifold embedding...")
    color_embedder = ColorManifoldEmbedding(concept_dim=8)

    expert_hsv = color_embedder.fit_transform(expert_vectors)
    violator_hsv = color_embedder.fit_transform(violator_vectors)
    # all_hsv = color_embedder.fit_transform(all_vectors)
    
    # Split back into expert and violator
    # expert_hsv = all_hsv[:len(expert_vectors)]
    # violator_hsv = all_hsv[len(expert_vectors):]
    
    print("Creating 2D visualization...")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Plot 1: Hue-Saturation space
    expert_rgb = hsv_to_rgb(expert_hsv)
    violator_rgb = hsv_to_rgb(violator_hsv)
    scatter1 = ax1.scatter(expert_hsv[:, 0], expert_hsv[:, 1], 
                          c=expert_rgb, alpha=0.6, 
                          s=20, label='Expert')
    ax1.scatter(violator_hsv[:, 0], violator_hsv[:, 1], marker='*',
               c=violator_rgb, alpha=0.6, s=20, label='Violator')
    ax1.set_xlabel('Hue (Semantic Category)')
    ax1.set_ylabel('Saturation (Confidence)')
    ax1.set_title('HSV Semantic Space: Hue vs Saturation')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Hue-Value space  
    scatter2 = ax2.scatter(expert_hsv[:, 0], expert_hsv[:, 2], 
                          c=expert_rgb, alpha=0.6, 
                          s=20, label='Expert')
    ax2.scatter(violator_hsv[:, 0], violator_hsv[:, 2], marker='*',
               c=violator_rgb, alpha=0.6, s=20, label='Violator')
    ax2.set_xlabel('Hue (Semantic Category)')
    ax2.set_ylabel('Value (Prominence)')
    ax2.set_title('HSV Semantic Space: Hue vs Value')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('results/plots/manifold_2d.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("2D visualization saved to results/plots/manifold_2d.png")

if __name__ == "__main__":
    # Create necessary directories
    Path("results/plots").mkdir(parents=True, exist_ok=True)
    visualize_manifold_2d()