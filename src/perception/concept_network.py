import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any

class ConceptNetwork(nn.Module):
    """
    Neural perception module that maps raw states to concept activation vectors.
    Implements the g_θ function from the paper.
    """
    
    def __init__(self, 
                 state_dim: int = 4,
                 concept_dim: int = 8,
                 hidden_dims: list = [64, 32],
                 activation: str = "relu"):
        super().__init__()
        
        self.state_dim = state_dim
        self.concept_dim = concept_dim
        
        # Build MLP layers
        layers = []
        prev_dim = state_dim
        
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            if activation == "relu":
                layers.append(nn.ReLU())
            elif activation == "tanh":
                layers.append(nn.Tanh())
            elif activation == "leaky_relu":
                layers.append(nn.LeakyReLU())
            layers.append(nn.Dropout(0.1))
            prev_dim = hidden_dim
        
        # Final layer to concept space
        layers.append(nn.Linear(prev_dim, concept_dim))
        layers.append(nn.Tanh())  # Constrain outputs to [-1, 1]
        
        self.network = nn.Sequential(*layers)
        
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """
        Map state to concept activation vector.
        
        Args:
            state: [batch_size, state_dim] or [state_dim]
            
        Returns:
            concept_vector: [batch_size, concept_dim] or [concept_dim]
        """
        return self.network(state)
    
    def get_activations(self, state: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Get intermediate activations for analysis."""
        activations = {}
        x = state
        
        for i, layer in enumerate(self.network):
            x = layer(x)
            if isinstance(layer, nn.Linear):
                activations[f'layer_{i}'] = x.clone()
                
        return activations