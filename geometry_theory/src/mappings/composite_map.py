"""
Complete geometric-semantic interface Φ = Ψ ∘ σ ∘ f_φ ∘ E_θ.
"""
import sys 
import os 
sys.path.append(os.path.join(os.path.dirname(__file__), '../../..'))

import torch
import torch.nn as nn
import numpy as np
from typing import Tuple, Optional, Dict, List 

from geometry_theory.src.mappings.encoder import ConceptEncoder, Autoencoder
from geometry_theory.src.mappings.normalizing_flow import DiffeomorphicFlow, SquashingLayer
from geometry_theory.src.mappings.exact_diffeomorphism import ExactDiffeomorphism
from geometry_theory.src.spaces.hsv_manifold import HSVManifold

class GeometricSemanticInterface(nn.Module):
    """
    Complete pipeline: Φ = Ψ ∘ σ ∘ f_φ ∘ E_θ
    
    Maps states s ∈ S to semantic points h ∈ H in the HSV manifold.
    """
    def __init__(self, 
                 state_dim: int = 4,
                 concept_dim: int = 8,
                 flow_dim: int = 3,  # Must be 3 for HSV
                 encoder_hidden: Tuple[int, ...] = (64, 32),
                 flow_layers: int = 10,
                 hsv_radius: float = 1/(2*np.pi)):
        """
        Parameters
        ----------
        state_dim : int
            Dimension of state space.
        concept_dim : int
            Dimension of concept space.
        flow_dim : int
            Dimension for normalizing flow (must be 3 for HSV).
        encoder_hidden : Tuple[int, ...]
            Hidden dimensions for encoder.
        flow_layers : int
            Number of layers in normalizing flow.
        hsv_radius : float
            Radius of HSV cylinder.
        """
        super().__init__()
        
        assert flow_dim == 3, "Flow dimension must be 3 for HSV manifold"
        
        # Components
        self.encoder = ConceptEncoder(
            input_dim=state_dim,
            latent_dim=concept_dim,
            hidden_dims=encoder_hidden,
            activation='relu'
        )
        
        # Projection from concept_dim to flow_dim
        self.projection = nn.Linear(concept_dim, flow_dim)
        
        self.flow = DiffeomorphicFlow(
            dim=flow_dim,
            n_layers=flow_layers,
            hidden_dim=32,
            flow_type='maf'
        )
        
        self.squashing = SquashingLayer(dim=flow_dim)
        self.exact_diffeomorphism = ExactDiffeomorphism(hsv_radius=hsv_radius)
        self.hsv_manifold = HSVManifold(radius=hsv_radius)
        
        self.state_dim = state_dim
        self.concept_dim = concept_dim
        self.flow_dim = flow_dim
        
    def forward(self, states: torch.Tensor) -> Tuple[torch.Tensor, Dict]:
        """
        Full forward pass: s → h ∈ H.
        
        Parameters
        ----------
        states : torch.Tensor
            Batch of states, shape (batch_size, state_dim).
            
        Returns
        -------
        Tuple[torch.Tensor, Dict]
            (HSV_points, intermediate_results)
        """
        batch_size = states.shape[0]
        intermediate = {}
        
        # Step 1: Encode to concepts
        concepts = self.encoder(states)  # (batch, concept_dim)
        intermediate['concepts'] = concepts.detach().cpu().numpy()
        
        # Step 2: Project to flow dimension
        projected = self.projection(concepts)  # (batch, flow_dim=3)
        intermediate['projected'] = projected.detach().cpu().numpy()
        
        # Step 3: Apply normalizing flow
        # flow_output, log_det = self.flow.forward(projected)
        flow_output = self.flow.forward(projected)
        intermediate['flow_output'] = flow_output.detach().cpu().numpy()
        # intermediate['log_det_jacobian'] = log_det.detach().cpu().numpy()
        
        # Step 4: Squash to unit cube
        unit_cube = self.squashing(flow_output)  # ∈ [0,1]³
        intermediate['unit_cube'] = unit_cube.detach().cpu().numpy()
        
        # Step 5: Apply exact diffeomorphism to HSV
        # Convert to numpy for exact diffeomorphism (which is not learnable)
        unit_cube_np = unit_cube.detach().cpu().numpy()
        
        # Map to HSV
        hsv_points = np.zeros((batch_size, 3))
        for i in range(batch_size):
            try:
                hsv_points[i] = self.exact_diffeomorphism.forward(unit_cube_np[i])
            except:
                print("UNIT CUBE: ", unit_cube_np[i])
                exit(1)
        
        intermediate['hsv_points'] = hsv_points
        
        return torch.tensor(hsv_points, dtype=torch.float32), intermediate
    
    def inverse(self, hsv_points: torch.Tensor) -> Tuple[torch.Tensor, Dict]:
        """
        Inverse pass: h ∈ H → s (approximate).
        
        Parameters
        ----------
        hsv_points : torch.Tensor
            Points in HSV manifold.
            
        Returns
        -------
        Tuple[torch.Tensor, Dict]
            (reconstructed_states, intermediate_results)
        """
        batch_size = hsv_points.shape[0]
        intermediate = {}
        
        # Convert to numpy for exact diffeomorphism
        hsv_np = hsv_points.detach().cpu().numpy()
        
        # Step 1: Inverse of exact diffeomorphism
        unit_cube_np = np.zeros((batch_size, 3))
        for i in range(batch_size):
            unit_cube_np[i] = self.exact_diffeomorphism.inverse(hsv_np[i])
        
        unit_cube = torch.tensor(unit_cube_np, dtype=torch.float32, device=hsv_points.device)
        intermediate['unit_cube_inv'] = unit_cube_np
        
        # Step 2: Inverse of squashing
        flow_output = self.squashing.inverse(unit_cube)
        intermediate['flow_output_inv'] = flow_output.detach().cpu().numpy()
        
        # Step 3: Inverse of normalizing flow
        projected, _ = self.flow.inverse(flow_output)
        intermediate['projected_inv'] = projected.detach().cpu().numpy()
        
        # Step 4: Project back to concept space
        # Note: We need an inverse projection, which is not strictly defined
        # For simplicity, use a linear decoder (not ideal but works for demo)
        concepts_recon = torch.matmul(projected, self.projection.weight.T) + self.projection.bias
        intermediate['concepts_inv'] = concepts_recon.detach().cpu().numpy()
        
        # Step 5: Decode to states (need a decoder, not implemented here)
        # For demo, return zeros
        states_recon = torch.zeros(batch_size, self.state_dim, device=hsv_points.device)
        intermediate['states_recon'] = states_recon.detach().cpu().numpy()
        
        return states_recon, intermediate
    
    def train_autoencoder(self,
                         states: torch.Tensor,
                         epochs: int = 100,
                         batch_size: int = 32,
                         lr: float = 1e-3,
                         device: str = 'cpu') -> List[float]:
        """
        Train autoencoder (encoder + decoder) for reconstruction.
        
        Parameters
        ----------
        states : torch.Tensor
            Training states.
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
        List[float]
            Training losses.
        """
        # Create autoencoder
        autoencoder = Autoencoder(
            input_dim=self.state_dim,
            latent_dim=self.concept_dim,
            encoder_hidden=(64, 32),  # Match encoder architecture
            decoder_hidden=(32, 64)
        )
        
        # Train
        losses = autoencoder.train_autoencoder(
            states=states,
            epochs=epochs,
            batch_size=batch_size,
            lr=lr,
            device=device
        )
        
        # Copy encoder weights
        self.encoder.load_state_dict(autoencoder.encoder.state_dict())
        
        return losses
    
    def train_flow_to_uniform(self,
                             expert_states: torch.Tensor,
                             violator_states: torch.Tensor,
                             epochs: int = 100,
                             batch_size: int = 128,
                             lr: float = 1e-3,
                             device: str = 'cpu') -> Dict[str, List[float]]:
        """
        Train normalizing flow to map encoded concepts to uniform distribution.
        
        Parameters
        ----------
        expert_states, violator_states : torch.Tensor
            Expert and violator states.
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
        self.eval()  # Freeze encoder
        
        # Encode states
        with torch.no_grad():
            expert_concepts = self.encoder(expert_states.to(device))
            violator_concepts = self.encoder(violator_states.to(device))
            
            # Project to flow dimension
            expert_projected = self.projection(expert_concepts)
            violator_projected = self.projection(violator_concepts)
        
        # Train flow
        metrics = self.flow.train_to_uniform(
            data1=expert_projected,
            data2=violator_projected,
            epochs=epochs,
            batch_size=batch_size,
            lr=lr,
            device=device
        )
        
        return metrics
    
    def test_topological_preservation(self,
                                     trajectory: np.ndarray,
                                     device: str = 'cpu') -> Dict:
        """
        Test if a periodic trajectory preserves its loop structure.
        
        Parameters
        ----------
        trajectory : np.ndarray
            State trajectory (n_steps, state_dim).
        device : str
            Device to use.
            
        Returns
        -------
        Dict
            Topological preservation metrics.
        """
        self.eval()
        
        # Convert to tensor
        traj_tensor = torch.tensor(trajectory, dtype=torch.float32, device=device)
        
        with torch.no_grad():
            # Map to HSV
            hsv_points, _ = self.forward(traj_tensor)
            hsv_np = hsv_points.cpu().numpy()
            
            # Convert to Cartesian for analysis
            cartesian_points = np.array([self.hsv_manifold.to_cartesian(p) for p in hsv_np])
            
            # Compute loop closure error
            loop_error = np.linalg.norm(cartesian_points[0] - cartesian_points[-1])
            
            # Compute circular variance in hue (should be low for periodic)
            hues = hsv_np[:, 0]
            # Convert to complex numbers on unit circle
            complex_hues = np.exp(1j * hues)
            hue_circular_var = 1 - np.abs(np.mean(complex_hues))
            
            # Check if trajectory forms a loop in HSV space
            # Simple heuristic: start and end close in HSV space
            hsv_start = hsv_np[0]
            hsv_end = hsv_np[-1]
            
            # Account for circular hue
            hue_diff = min(abs(hsv_start[0] - hsv_end[0]),
                          2*np.pi - abs(hsv_start[0] - hsv_end[0]))
            
            hsv_distance = np.sqrt(
                (hue_diff / (2*np.pi))**2 +
                (hsv_start[1] - hsv_end[1])**2 +
                (hsv_start[2] - hsv_end[2])**2
            )
            
            # Periodic score (1 = perfect loop, 0 = no loop)
            periodic_score = max(0, 1 - hsv_distance / 0.5)  # Normalize
            
        return {
            'loop_closure_error': loop_error,
            'hue_circular_variance': hue_circular_var,
            'hsv_end_to_end_distance': hsv_distance,
            'periodic_score': periodic_score,
            'hsv_trajectory': hsv_np,
            'cartesian_trajectory': cartesian_points
        }