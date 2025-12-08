"""
Normalizing flow f_φ: ℝᵈ → ℝᵈ for learning diffeomorphism.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import normflows as nf
from typing import Tuple, List, Optional, Dict 

class DiffeomorphicFlow(nn.Module):
    """
    Normalizing flow that learns a diffeomorphism f_φ: ℝᵈ → ℝᵈ.
    
    By construction, normalizing flows are invertible and have
    computable Jacobian determinants.
    """
    def __init__(self, 
                 dim: int = 8,
                 n_layers: int = 10,
                 hidden_dim: int = 32,
                 flow_type: str = 'maf',
                 base_distribution: str = 'gaussian'):
        """
        Parameters
        ----------
        dim : int
            Dimension of latent space.
        n_layers : int
            Number of flow layers.
        hidden_dim : int
            Dimension of hidden layers in each flow.
        flow_type : str
            Type of flow: 'maf', 'realnvp', 'glow'.
        """
        super().__init__()
        
        self.dim = dim
        self.n_layers = n_layers
        
        # Define base distribution
        if base_distribution == 'uniform':
            # Uniform distribution on [0, 1]^dim
            self.base_dist = nf.distributions.base.Uniform(dim, 0, 1)
        elif base_distribution == 'gaussian':
            self.base_dist = nf.distributions.DiagGaussian(dim, trainable=False)
        else:
            raise ValueError(f"Unknown base distribution: {base_distribution}")
        
        # Create flow layers
        flows = []
        
        for i in range(n_layers):
            if flow_type == 'maf':
                # Masked Autoregressive Flow
                flows.append(nf.flows.MaskedAffineAutoregressive(dim, hidden_dim, num_blocks=2))
                flows.append(nf.flows.Permute(dim, mode='swap'))
            
            elif flow_type == 'realnvp':
                # RealNVP
                scale_map = nf.nets.MLP([2, hidden_dim, hidden_dim, dim])
                translation_map = nf.nets.MLP([2, hidden_dim, hidden_dim, dim])
                flows.append(nf.flows.AffineCouplingBlock(translation_map, scale_map))
                flows.append(nf.flows.Permute(dim, mode='swap'))
            
            elif flow_type == 'glow':
                # GLOW (simplified)
                param_map = nf.nets.MLP([dim, hidden_dim, hidden_dim, dim])
                flows.append(nf.flows.InvertibleAffine(param_map))
                flows.append(nf.flows.Permute(dim, mode='swap'))
            
            else:
                raise ValueError(f"Unknown flow type: {flow_type}")
        
        # Remove last permutation if present
        if isinstance(flows[-1], nf.flows.Permute):
            flows = flows[:-1]
        
        # Create normalizing flow
        self.flow = nf.NormalizingFlow(self.base_dist, flows)
    
    def forward(self, z: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Transform z through the flow: f_φ(z).
        
        Parameters
        ----------
        z : torch.Tensor
            Input tensor of shape (batch_size, dim).
            
        Returns
        -------
        Tuple[torch.Tensor, torch.Tensor]
            (transformed_z, log_det_jacobian)
        """
        return self.flow.forward(z)
    
    def inverse(self, u: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Inverse transform: f_φ⁻¹(u).
        
        Parameters
        ----------
        u : torch.Tensor
            Transformed tensor.
            
        Returns
        -------
        Tuple[torch.Tensor, torch.Tensor]
            (original_z, negative_log_det_jacobian)
        """
        return self.flow.inverse(u)
    
    def sample(self, n_samples: int = 1) -> torch.Tensor:
        """
        Sample from the flow.
        
        Parameters
        ----------
        n_samples : int
            Number of samples.
            
        Returns
        -------
        torch.Tensor
            Samples from flow.
        """
        return self.flow.sample(n_samples)[0]
    
    def log_prob(self, z: torch.Tensor) -> torch.Tensor:
        """
        Compute log probability of z under the flow.
        
        Parameters
        ----------
        z : torch.Tensor
            Input tensor.
            
        Returns
        -------
        torch.Tensor
            Log probability.
        """
        return self.flow.log_prob(z)
    
    def train_to_uniform(self,
                        data1: torch.Tensor,
                        data2: torch.Tensor,
                        epochs: int = 100,
                        batch_size: int = 128,
                        lr: float = 1e-3,
                        device: str = 'cpu') -> Dict[str, List[float]]:
        """
        Train flow to map both expert and violator data to uniform distribution.
        
        Parameters
        ----------
        data1, data2 : torch.Tensor
            Two datasets to map to uniform (e.g., expert and violator).
        epochs : int
            Number of training epochs.
        batch_size : int
            Batch size.
        lr : float
            Learning rate.
        device : str
            Device to use.
            
        Returns
        -------
        Dict[str, List[float]]
            Training metrics.
        """
        self.to(device)
        self.train()
        
        # Combine both datasets
        all_data = torch.cat([data1, data2], dim=0)
        
        # Create data loader
        dataset = torch.utils.data.TensorDataset(all_data)
        data_loader = torch.utils.data.DataLoader(
            dataset, batch_size=batch_size, shuffle=True
        )
        
        # Optimizer
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        
        metrics = {
            'loss': [],
            'expert_log_prob': [],
            'violator_log_prob': []
        }
        
        for epoch in range(epochs):
            epoch_loss = 0.0
            expert_log_prob_sum = 0.0
            violator_log_prob_sum = 0.0
            n_batches = 0
            
            for batch in data_loader:
                z_batch = batch[0].to(device)
                
                # Compute negative log likelihood
                optimizer.zero_grad()
                loss = -self.log_prob(z_batch).mean()
                
                # Backward pass
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
                optimizer.step()
                
                # Track probabilities for both datasets
                batch_size_half = z_batch.shape[0] // 2
                expert_batch = z_batch[:batch_size_half]
                violator_batch = z_batch[batch_size_half:]
                
                with torch.no_grad():
                    expert_log_prob = self.log_prob(expert_batch).mean().item()
                    violator_log_prob = self.log_prob(violator_batch).mean().item()
                
                epoch_loss += loss.item()
                expert_log_prob_sum += expert_log_prob
                violator_log_prob_sum += violator_log_prob
                n_batches += 1
            
            # Store metrics
            metrics['loss'].append(epoch_loss / max(n_batches, 1))
            metrics['expert_log_prob'].append(expert_log_prob_sum / max(n_batches, 1))
            metrics['violator_log_prob'].append(violator_log_prob_sum / max(n_batches, 1))
            
            if (epoch + 1) % 10 == 0:
                print(f"Flow Epoch {epoch+1}/{epochs}: "
                      f"Loss={metrics['loss'][-1]:.4f}, "
                      f"Expert logP={metrics['expert_log_prob'][-1]:.4f}, "
                      f"Violator logP={metrics['violator_log_prob'][-1]:.4f}")
        
        return metrics
    
    def test_invertibility(self, 
                          test_data: torch.Tensor,
                          rtol: float = 1e-5,
                          atol: float = 1e-8) -> dict:
        """
        Test that the flow is indeed invertible.
        
        Parameters
        ----------
        test_data : torch.Tensor
            Test data.
        rtol, atol : float
            Relative and absolute tolerances.
            
        Returns
        -------
        dict
            Test results.
        """
        self.eval()
        
        with torch.no_grad():
            # Forward then inverse
            u, _ = self.forward(test_data)
            z_recon, _ = self.inverse(u)
            
            # Inverse then forward
            z, _ = self.inverse(test_data)  # Using test_data as 'u' for inverse test
            u_recon, _ = self.forward(z)
            
            # Compute errors
            forward_inverse_error = torch.norm(test_data - z_recon, dim=1).mean().item()
            inverse_forward_error = torch.norm(test_data - u_recon, dim=1).mean().item()
            
            # Check Jacobian determinant (should be positive)
            _, log_det = self.forward(test_data)
            jac_det = torch.exp(log_det)
            min_det = jac_det.min().item()
            max_det = jac_det.max().item()
        
        return {
            'forward_inverse_error': forward_inverse_error,
            'inverse_forward_error': inverse_forward_error,
            'jacobian_det_min': min_det,
            'jacobian_det_max': max_det,
            'is_invertible': forward_inverse_error < rtol and inverse_forward_error < rtol
        }


class SquashingLayer(nn.Module):
    """
    Smooth squashing function σ: ℝᵈ → U.
    
    Maps to unit hypercube [0,1]ᵈ with careful handling of boundaries.
    """
    def __init__(self, dim: int = 3, eps: float = 1e-6):
        """
        Parameters
        ----------
        dim : int
            Dimension.
        eps : float
            Small epsilon to avoid exact 0 or 1.
        """
        super().__init__()
        self.dim = dim
        self.eps = eps
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply squashing: sigmoid with boundary handling.
        
        Parameters
        ----------
        x : torch.Tensor
            Input tensor.
            
        Returns
        -------
        torch.Tensor
            Squashed tensor in [eps, 1-eps]ᵈ.
        """
        # Sigmoid maps to (0,1), we clip to avoid exact boundaries
        squashed = torch.sigmoid(x)
        
        # Ensure we don't get exactly 0 or 1
        squashed = self.eps + (1 - 2 * self.eps) * squashed
        
        return squashed
    
    def inverse(self, y: torch.Tensor) -> torch.Tensor:
        """
        Inverse of squashing: logit function.
        
        Parameters
        ----------
        y : torch.Tensor
            Tensor in [eps, 1-eps]ᵈ.
            
        Returns
        -------
        torch.Tensor
            Inverse transformed tensor.
        """
        # Remove eps scaling
        y_scaled = (y - self.eps) / (1 - 2 * self.eps)
        
        # Apply logit
        # Add small epsilon to avoid log(0)
        y_scaled = torch.clamp(y_scaled, 1e-7, 1 - 1e-7)
        
        return torch.log(y_scaled / (1 - y_scaled))
    
    def log_det_jacobian(self, x: torch.Tensor) -> torch.Tensor:
        """
        Log determinant of Jacobian of forward transformation.
        
        Parameters
        ----------
        x : torch.Tensor
            Input tensor.
            
        Returns
        -------
        torch.Tensor
            Log determinant.
        """
        # For sigmoid: ∂σ/∂x = σ(x)(1-σ(x))
        sig = torch.sigmoid(x)
        jac_diag = sig * (1 - sig) * (1 - 2 * self.eps)
        
        # Jacobian is diagonal, so determinant is product
        log_det = torch.sum(torch.log(jac_diag + 1e-8), dim=-1)
        
        return log_det