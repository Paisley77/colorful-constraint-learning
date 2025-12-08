"""
Neural encoder E_θ: S → ℝᵈ for dimensionality reduction.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, List 

class ConceptEncoder(nn.Module):
    """
    Neural network that maps states to concept vectors.
    
    This should ideally learn to be a diffeomorphism when restricted
    to the data manifold M_d ⊂ S.
    """
    def __init__(self, 
                 input_dim: int = 4,
                 latent_dim: int = 8,
                 hidden_dims: Tuple[int, ...] = (64, 32),
                 activation: str = 'relu'):
        """
        Parameters
        ----------
        input_dim : int
            Dimension of state space (n).
        latent_dim : int
            Dimension of concept space (d).
        hidden_dims : Tuple[int, ...]
            Dimensions of hidden layers.
        activation : str
            Activation function ('relu', 'tanh', 'elu').
        """
        super().__init__()
        
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        
        # Build network layers
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            
            if activation == 'relu':
                layers.append(nn.ReLU())
            elif activation == 'tanh':
                layers.append(nn.Tanh())
            elif activation == 'elu':
                layers.append(nn.ELU())
            elif activation == 'leaky_relu':
                layers.append(nn.LeakyReLU(0.2))
            else:
                raise ValueError(f"Unknown activation: {activation}")
            
            layers.append(nn.BatchNorm1d(hidden_dim))
            prev_dim = hidden_dim
        
        # Final layer to latent dimension
        layers.append(nn.Linear(prev_dim, latent_dim))
        
        self.network = nn.Sequential(*layers)
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize network weights for better conditioning."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight, gain=0.5)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Map state to concept vector.
        
        Parameters
        ----------
        x : torch.Tensor
            State tensor of shape (batch_size, input_dim).
            
        Returns
        -------
        torch.Tensor
            Concept vector of shape (batch_size, latent_dim).
        """
        return self.network(x)
    
    def jacobian(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute Jacobian matrix ∂E_θ/∂x at point x.
        
        Parameters
        ----------
        x : torch.Tensor
            State tensor of shape (batch_size, input_dim).
            
        Returns
        -------
        torch.Tensor
            Jacobian matrix of shape (batch_size, latent_dim, input_dim).
        """
        batch_size = x.shape[0]
        
        # Enable gradient computation
        x.requires_grad_(True)
        
        # Compute output
        z = self.forward(x)
        
        # Compute Jacobian row by row
        jacobian = torch.zeros(batch_size, self.latent_dim, self.input_dim, device=x.device)
        
        for i in range(self.latent_dim):
            # Compute gradient of z_i w.r.t. x
            grad_outputs = torch.zeros_like(z)
            grad_outputs[:, i] = 1.0
            
            if x.grad is not None:
                x.grad.zero_()
            
            z.backward(gradient=grad_outputs, retain_graph=True)
            jacobian[:, i, :] = x.grad.clone()
        
        # Clean up
        x.grad = None
        x.requires_grad_(False)
        
        return jacobian
    
    def rank_test(self, x: torch.Tensor) -> torch.Tensor:
        """
        Test if Jacobian has full rank (injective condition).
        
        Parameters
        ----------
        x : torch.Tensor
            State tensor.
            
        Returns
        -------
        torch.Tensor
            Singular values of Jacobian.
        """
        jac = self.jacobian(x)
        
        # Compute SVD for each batch element
        singular_values = []
        for i in range(x.shape[0]):
            U, S, V = torch.svd(jac[i])
            singular_values.append(S)
        
        return torch.stack(singular_values)

class ConceptDecoder(nn.Module):
    """
    Decoder that maps concept vectors back to state space.
    Mirror of ConceptEncoder for autoencoder training.
    """
    def __init__(self, 
                 input_dim: int = 8,
                 output_dim: int = 4,
                 hidden_dims: Tuple[int, ...] = (32, 64),
                 activation: str = 'relu'):
        super().__init__()
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        
        # Build network layers (reverse of encoder)
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            
            if activation == 'relu':
                layers.append(nn.ReLU())
            elif activation == 'tanh':
                layers.append(nn.Tanh())
            elif activation == 'elu':
                layers.append(nn.ELU())
            elif activation == 'leaky_relu':
                layers.append(nn.LeakyReLU(0.2))
            else:
                raise ValueError(f"Unknown activation: {activation}")
            
            layers.append(nn.BatchNorm1d(hidden_dim))
            prev_dim = hidden_dim
        
        # Final layer to output dimension
        layers.append(nn.Linear(prev_dim, output_dim))
        
        self.network = nn.Sequential(*layers)
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize network weights."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight, gain=0.5)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Map concept vector back to state space.
        
        Parameters
        ----------
        z : torch.Tensor
            Concept tensor of shape (batch_size, input_dim).
            
        Returns
        -------
        torch.Tensor
            Reconstructed state of shape (batch_size, output_dim).
        """
        return self.network(z)


class Autoencoder(nn.Module):
    """
    Autoencoder for training the encoder to be approximately invertible.
    """
    def __init__(self, 
                 input_dim: int = 4,
                 latent_dim: int = 8,
                 encoder_hidden: Tuple[int, ...] = (64, 32),
                 decoder_hidden: Tuple[int, ...] = (32, 64)):
        super().__init__()
        
        self.encoder = ConceptEncoder(
            input_dim=input_dim,
            latent_dim=latent_dim,
            hidden_dims=encoder_hidden,
            activation='relu'
        )
        
        self.decoder = ConceptDecoder(
            input_dim=latent_dim,
            output_dim=input_dim,
            hidden_dims=decoder_hidden,
            activation='relu'
        )
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Encode and decode.
        
        Parameters
        ----------
        x : torch.Tensor
            Input state.
            
        Returns
        -------
        Tuple[torch.Tensor, torch.Tensor]
            (latent_code, reconstructed_state)
        """
        z = self.encoder(x)
        x_recon = self.decoder(z)
        return z, x_recon
    
    def reconstruction_loss(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute reconstruction loss.
        
        Parameters
        ----------
        x : torch.Tensor
            Input state.
            
        Returns
        -------
        torch.Tensor
            Reconstruction loss.
        """
        _, x_recon = self.forward(x)
        return F.mse_loss(x_recon, x, reduction='mean')
    
    def train_autoencoder(self, 
                states: torch.Tensor,
                epochs: int = 100,
                batch_size: int = 32,
                lr: float = 1e-3,
                device: str = 'cpu') -> List[float]:
        """
        Train the autoencoder.
        
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
        self.to(device)
        self.train()
        
        # Create data loader
        dataset = torch.utils.data.TensorDataset(states)
        data_loader = torch.utils.data.DataLoader(
            dataset, batch_size=batch_size, shuffle=True
        )

        # Create optimizer 
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        
        losses = []
        
        for epoch in range(epochs):
            epoch_loss = 0.0
            n_batches = 0
            
            for batch in data_loader:
                x_batch = batch[0].to(device)
                
                # Forward pass
                optimizer.zero_grad()
                loss = self.reconstruction_loss(x_batch)
                
                # Backward pass
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item()
                n_batches += 1
            
            avg_loss = epoch_loss / max(n_batches, 1)
            losses.append(avg_loss)
            
            if (epoch + 1) % 10 == 0:
                print(f"Autoencoder Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.6f}")
        
        return losses