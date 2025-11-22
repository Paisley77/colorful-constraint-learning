import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import List, Dict, Any, Tuple
import matplotlib.pyplot as plt
from matplotlib.colors import hsv_to_rgb
from pathlib import Path
from tqdm import tqdm 

from src.perception.concept_network import ConceptNetwork
from src.embedding.color_manifold import ColorManifoldEmbedding
from src.learning.contrastive_loss import ConceptContrastiveLoss

class StagedTrainer:
    """
    Implements the three-stage training procedure from the paper.
    """
    
    def __init__(self, 
                 concept_net: ConceptNetwork,
                 color_embedder: ColorManifoldEmbedding,
                 config: Dict[str, Any]):
        self.concept_net = concept_net
        self.color_embedder = color_embedder
        self.config = config
        
        # Training components
        self.optimizer = optim.Adam(
            self.concept_net.parameters(), 
            lr=config['training']['learning_rate']
        )
        self.contrastive_loss = ConceptContrastiveLoss(
            margin=config['training']['contrastive_margin']
        )
        
        # Tracking
        self.loss_history = []
        self.embedding_history = []
        
    def prepare_trajectory_batch(self, 
                               expert_trajectories: np.ndarray,
                               violator_trajectories: np.ndarray,
                               batch_size: int = 32) -> Tuple[np.ndarray,np.ndarray]:
        """Prepare batches of trajectories for training."""
        B, T, _ = expert_trajectories.shape 
        if batch_size > B:
            raise ValueError("batch_size cannot be greater than the total number of trajectories (B)")
        indices = torch.randperm(B)[:batch_size]  
        return expert_trajectories[indices], violator_trajectories[indices] # [batch_size, T, 4]
    
    def get_concept_batches(self, 
                    expert_states: List[np.ndarray],
                    violator_states: List[np.ndarray]) -> Tuple[torch.Tensor, torch.Tensor]:
        """Project states to HSV space and sample batches."""
        # Let B = batch_size --> (B, T, 4)
        expert_batch, violator_batch = self.prepare_trajectory_batch(np.stack(expert_states), np.stack(violator_states))
        batch_size = expert_batch.shape[0]
        state_dim  = expert_batch.shape[2]
        expert_batch = expert_batch.reshape(-1, state_dim) # [BxT, 4]
        violator_batch = violator_batch.reshape(-1, state_dim) # [BxT, 4]
        expert_concepts = self.concept_net(torch.FloatTensor(expert_batch)) # [BxT, k]
        violator_concepts = self.concept_net(torch.FloatTensor(violator_batch)) # [BxT, k]
        k = expert_concepts.shape[-1]
        return expert_concepts.reshape(batch_size, -1, k), violator_concepts.reshape(batch_size, -1, k) # [B, T, k]


    def get_color_trajectories(self, expert_concepts, violator_concepts):
        # Expert projection 
        expert_hsv = self.color_embedder.fit_transform(expert_concepts) # [BxT, 3]
        expert_hsv_smooth = self.color_embedder.smooth_trajectory(expert_hsv) # [BxT, 3]

        # Violator projection
        violator_hsv = self.color_embedder.fit_transform(violator_concepts) # [BxT, 3]
        violator_hsv_smooth = self.color_embedder.smooth_trajectory(violator_hsv) # [BxT, 3]

        return expert_hsv_smooth, violator_hsv_smooth
    
    def stage1_foundation_learning(self,
                                 expert_states: List[np.ndarray],
                                 violator_states: List[np.ndarray],
                                 num_epochs: int = 100,
                                 save_interval: int = 100) -> None:
        """
        Implement Stage 1: Foundation Learning with contrastive loss.
        """
        print("Starting Stage 1: Foundation Learning...")
        
        for epoch in tqdm(range(num_epochs)):
            self.concept_net.train()
            # size: [B, T, k]
            expert_concepts, violator_concepts = self.get_concept_batches(expert_states, violator_states)
            # Compute contrastive loss
            loss, loss_components = self.contrastive_loss(
                expert_concepts, violator_concepts
            )
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            # Track progress
            self.loss_history.append(loss_components)
            
            if epoch % 50 == 0:
                print(f"Epoch {epoch}/{num_epochs}, Total Loss: {loss.item():.4f}")
                print(f"  Separation: {loss_components['separation_loss']:.4f}, "
                      f"Expert Cluster: {loss_components['expert_clustering_loss']:.4f}, "
                      f"Violator Cluster: {loss_components['violator_clustering_loss']:.4f}")
            
            # Save embedding state for animation
            if epoch % save_interval == 0:
                self._save_embedding_state(expert_states, violator_states, epoch)
        
        print("Stage 1 completed!")
    
    def _save_embedding_state(self, 
                            expert_states: List[np.ndarray], 
                            violator_states: List[np.ndarray],
                            epoch: int) -> None:
        """Save current embedding state for animation."""
        self.concept_net.eval()
        
        with torch.no_grad():
            # Get all concept vectors
            all_expert_concepts = []
            all_violator_concepts = []
            
            for states in expert_states:
                concepts = self.concept_net(torch.FloatTensor(states)).detach().numpy()
                all_expert_concepts.append(concepts) # B [T, k] arrays 
                
            for states in violator_states:
                concepts = self.concept_net(torch.FloatTensor(states)).detach().numpy()
                all_violator_concepts.append(concepts)
            
            # Flatten for visualization
            expert_flat = np.vstack(all_expert_concepts) # [BxT, k]
            violator_flat = np.vstack(all_violator_concepts) # [BxT, k]
            
            # Transform to HSV
            expert_hsv, violator_hsv = self.get_color_trajectories(expert_flat, violator_flat) # [BxT, 3]
            
            self.embedding_history.append({
                'epoch': epoch,
                'expert_hsv': expert_hsv, # [BxT,3]
                'violator_hsv': violator_hsv
            })
    
    def create_training_animation(self, save_path: str = "results/plots/training_evolution.gif") -> None:
        """Create animation showing the evolution of embeddings during training."""
        print("Creating training evolution animation...")
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 15))

        ax1 = axes[0,0]
        ax2 = axes[0,1]
        ax3 = axes[1,0]
        ax4 = axes[1,1]
        
        def update(frame):
            ax1.clear()
            ax2.clear()
            ax3.clear()
            ax4.clear()

            data = self.embedding_history[frame]
            epoch = data['epoch']
            expert_hsv = data['expert_hsv'] # [BxT,3]
            violator_hsv = data['violator_hsv']
            
            # Plot 1: Hue-Saturation space
            expert_rgb = hsv_to_rgb(expert_hsv)
            violator_rgb = hsv_to_rgb(violator_hsv)

            scatter1 = ax1.scatter(expert_hsv[:, 0], expert_hsv[:, 1], 
                                  c=expert_rgb, alpha=0.6, 
                                  s=10, label='Expert', marker='o')
            ax1.scatter(violator_hsv[:, 0], violator_hsv[:, 1], 
                       c=violator_rgb, alpha=0.6, s=10, label='Violator', marker='x')
            ax1.set_xlabel('Hue (Semantic Category)')
            ax1.set_ylabel('Saturation (Confidence)')
            ax1.set_title(f'Hue vs Saturation - Epoch {epoch}')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            ax1.set_xlim(-0.2, 1.2)
            ax1.set_ylim(-0.2, np.max([expert_hsv[:, 1].max(), violator_hsv[:, 1].max()]) * 1.1)
            
            # Plot 2: Hue-Value space
            scatter2 = ax2.scatter(expert_hsv[:, 0], expert_hsv[:, 2], 
                                  c=expert_rgb, alpha=0.6, 
                                  s=10, label='Expert', marker='o')
            ax2.scatter(violator_hsv[:, 0], violator_hsv[:, 2], 
                       c=violator_rgb, alpha=0.6, s=10, label='Violator', marker='x')
            ax2.set_xlabel('Hue (Semantic Category)')
            ax2.set_ylabel('Value (Prominence)')
            ax2.set_title(f'Hue vs Value - Epoch {epoch}')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
            ax2.set_xlim(-0.2, 1.2)
            ax2.set_ylim(-0.2, 1.2)

            # Plot 3: Hue-Sat, same color
            scatter3 = ax3.scatter(expert_hsv[:, 0], expert_hsv[:, 1], 
                                  c=expert_hsv[:,0], cmap='hsv', alpha=0.6, 
                                  s=10, label='Expert', marker='o')
            ax3.scatter(violator_hsv[:, 0], violator_hsv[:, 1], 
                       c='red', alpha=0.6, s=10, label='Violator', marker='x')
            ax3.set_xlabel('Hue (Semantic Category)')
            ax3.set_ylabel('Saturation (Confidence)')
            ax3.set_title(f'Hue vs Saturation - Epoch {epoch}')
            ax3.legend()
            ax3.grid(True, alpha=0.3)
            ax3.set_xlim(-0.2, 1.2)
            ax3.set_ylim(-0.2, np.max([expert_hsv[:, 1].max(), violator_hsv[:, 1].max()]) * 1.1)

            # Plot 4: Hue-Value, same color
            scatter4 = ax4.scatter(expert_hsv[:, 0], expert_hsv[:, 2], 
                                  c=expert_hsv[:,0], cmap='hsv', alpha=0.6, 
                                  s=10, label='Expert', marker='o')
            ax4.scatter(violator_hsv[:, 0], violator_hsv[:, 2], 
                       c='red', alpha=0.6, s=10, label='Violator', marker='x')
            ax4.set_xlabel('Hue (Semantic Category)')
            ax4.set_ylabel('Value (Prominence)')
            ax4.set_title(f'Hue vs Value - Epoch {epoch}')
            ax4.legend()
            ax4.grid(True, alpha=0.3)
            ax4.set_xlim(-0.2, 1.2)
            ax4.set_ylim(-0.2, np.max([expert_hsv[:, 2].max(), violator_hsv[:, 2].max()]) * 1.1)

            
            return scatter1, scatter2, scatter3, scatter4
        
        from matplotlib.animation import FuncAnimation
        
        anim = FuncAnimation(fig, update, frames=len(self.embedding_history), 
                           interval=500, blit=False, repeat=True)
        
        # Save animation
        # Path(save_path).mkdir(exist_ok=True)
        anim.save(save_path, writer='pillow', fps=2, dpi=100)
        
        plt.close()
        print(f"Training animation saved to {save_path}")
    
    def plot_loss_history(self, save_path: str = "results/plots/loss_history.png") -> None:
        """Plot the training loss history."""
        epochs = range(len(self.loss_history))
        
        separation_loss = [x['separation_loss'] for x in self.loss_history]
        expert_cluster_loss = [x['expert_clustering_loss'] for x in self.loss_history]
        violator_cluster_loss = [x['violator_clustering_loss'] for x in self.loss_history]
        total_loss = [x['total_loss'] for x in self.loss_history]
        
        plt.figure(figsize=(12, 8))
        
        plt.subplot(2, 1, 1)
        plt.plot(epochs, separation_loss, label='Separation Loss', linewidth=2)
        plt.plot(epochs, expert_cluster_loss, label='Expert Clustering Loss', linewidth=2)
        plt.plot(epochs, violator_cluster_loss, label='Violator Clustering Loss', linewidth=2)
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Component Losses During Stage 1 Training')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.subplot(2, 1, 2)
        plt.plot(epochs, total_loss, label='Total Loss', color='black', linewidth=3)
        plt.xlabel('Epoch')
        plt.ylabel('Total Loss')
        plt.title('Total Contrastive Loss During Stage 1 Training')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"Loss history plot saved to {save_path}")