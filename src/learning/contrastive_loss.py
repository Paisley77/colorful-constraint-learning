import torch
import torch.nn as nn
import numpy as np
from typing import Tuple, Dict

class ConceptContrastiveLoss(nn.Module):
    """
    Contrastive loss applied directly to concept vectors (before HSV projection).
    """
    
    def __init__(self, margin: float = 10.0, alpha: float = 3.0, beta: float = 0.3, gamma: float = 0.3):
        super().__init__()
        self.margin = margin
        self.alpha = alpha  # separation loss weight
        self.beta = beta    # expert clustering weight  
        self.gamma = gamma  # violator clustering weight
        
    def forward(self, 
                expert_concepts: torch.Tensor, 
                violator_concepts: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute contrastive loss between expert and violator concept trajectories.
        
        Args:
            expert_concepts: [batch_size, seq_len, concept_dim] expert concept trajectories
            violator_concepts: [batch_size, seq_len, concept_dim] violator concept trajectories
            
        Returns:
            loss: contrastive loss value
            loss_components: dictionary of individual loss components
        """
        batch_size_e = expert_concepts.shape[0]
        batch_size_v = violator_concepts.shape[0]
        
        # Compute trajectory centroids (mean over time dimension)
        expert_centroids = torch.mean(expert_concepts, dim=1)  # [batch_size_e, concept_dim]
        violator_centroids = torch.mean(violator_concepts, dim=1)  # [batch_size_v, concept_dim]
        
        # Separation loss: push expert and violator centroids apart
        separation_loss = 0.0
        for i in range(batch_size_e):
            for j in range(batch_size_v):
                distance = torch.norm(expert_centroids[i] - violator_centroids[j], p=2)
                separation_loss += torch.clamp(self.margin - distance, min=0) ** 2
        separation_loss /= (batch_size_e * batch_size_v)
        
        # Expert clustering loss: pull expert trajectories together
        expert_clustering_loss = 0.0
        count_e = 0
        for i in range(batch_size_e):
            for j in range(i + 1, batch_size_e):
                expert_clustering_loss += torch.norm(expert_centroids[i] - expert_centroids[j], p=2) ** 2
                count_e += 1
        if count_e > 0:
            expert_clustering_loss /= count_e
        
        # Violator clustering loss: pull violator trajectories together  
        violator_clustering_loss = 0.0
        count_v = 0
        for i in range(batch_size_v):
            for j in range(i + 1, batch_size_v):
                violator_clustering_loss += torch.norm(violator_centroids[i] - violator_centroids[j], p=2) ** 2
                count_v += 1
        if count_v > 0:
            violator_clustering_loss /= count_v
        
        # Total loss
        total_loss = (self.alpha * separation_loss + 
                     self.beta * expert_clustering_loss + 
                     self.gamma * violator_clustering_loss)
        
        loss_components = {
            'separation_loss': separation_loss.item(),
            'expert_clustering_loss': expert_clustering_loss.item() if count_e > 0 else 0.0,
            'violator_clustering_loss': violator_clustering_loss.item() if count_v > 0 else 0.0,
            'total_loss': total_loss.item()
        }
        
        return total_loss, loss_components

    def compute_distances(self, 
                         expert_centroids: torch.Tensor, 
                         violator_centroids: torch.Tensor) -> Dict[str, float]:
        """
        Compute distance metrics for analysis.
        """
        with torch.no_grad():
            # Inter-class distances (expert vs violator)
            inter_distances = []
            for i in range(expert_centroids.shape[0]):
                for j in range(violator_centroids.shape[0]):
                    dist = torch.norm(expert_centroids[i] - violator_centroids[j], p=2)
                    inter_distances.append(dist.item())
            
            # Intra-class distances (expert vs expert)
            intra_expert_distances = []
            for i in range(expert_centroids.shape[0]):
                for j in range(i + 1, expert_centroids.shape[0]):
                    dist = torch.norm(expert_centroids[i] - expert_centroids[j], p=2)
                    intra_expert_distances.append(dist.item())
            
            # Intra-class distances (violator vs violator)
            intra_violator_distances = []
            for i in range(violator_centroids.shape[0]):
                for j in range(i + 1, violator_centroids.shape[0]):
                    dist = torch.norm(violator_centroids[i] - violator_centroids[j], p=2)
                    intra_violator_distances.append(dist.item())
            
            return {
                'mean_inter_distance': np.mean(inter_distances) if inter_distances else 0.0,
                'mean_intra_expert_distance': np.mean(intra_expert_distances) if intra_expert_distances else 0.0,
                'mean_intra_violator_distance': np.mean(intra_violator_distances) if intra_violator_distances else 0.0,
                'separation_ratio': np.mean(inter_distances) / (np.mean(intra_expert_distances) + np.mean(intra_violator_distances) + 1e-8) 
                if intra_expert_distances and intra_violator_distances else 0.0
            }