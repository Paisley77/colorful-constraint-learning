import torch
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm 
import sys 
sys.path.append('../..')
from src.learning.simple_manifold import CylindricalManifold
from src.learning.smooth_temporal import temporal_constraint_loss
from src.learning.contrastive_loss import ConceptContrastiveLoss

class Stage2Trainer:
    def __init__(self, concept_net, color_embedder, config):
        self.concept_net = concept_net
        self.color_embedder = color_embedder
        self.config = config
        
        # Stage 2 components
        self.manifold = CylindricalManifold()
        self.manifold_optimizer = optim.Adam(self.manifold.parameters(), lr=1e-2)
        self.contrastive_loss = ConceptContrastiveLoss() 
        
        # Tracking
        self.history = []
        
    def stage2_alternating_optimization(self, expert_states, violator_states, num_alternations=10):
        """Simple alternating optimization between concept net and manifold"""
        
        for alt_step in tqdm(range(num_alternations)):
            
            # Phase A: Update manifold with fixed concepts
            self._update_manifold(expert_states, violator_states, num_epochs=5)
            
            # Phase B: Update concept network with fixed manifold  
            self._update_concept_network(expert_states, violator_states, num_epochs=5)
            
            # Save visualization state
            self._save_alternation_state(expert_states, violator_states, alt_step)
    
    def _update_manifold(self, expert_states, violator_states, num_epochs):
        """Update manifold to separate expert/violator in current embedding"""
        self.concept_net.eval()
        
        with torch.no_grad():
            # Get current HSV embeddings
            expert_hsv, violator_hsv = self._get_hsv_embeddings(expert_states, violator_states)
        
        # Convert to tensors
        expert_tensor = torch.FloatTensor(expert_hsv)
        violator_tensor = torch.FloatTensor(violator_hsv)
        
        for epoch in range(num_epochs):
            self.manifold_optimizer.zero_grad()
            
            loss = temporal_constraint_loss(expert_tensor, violator_tensor, self.manifold)
            
            loss.backward()
            self.manifold_optimizer.step()
    
    def _update_concept_network(self, expert_states, violator_states, num_epochs):
        """Update concept network with combined loss"""
        self.concept_net.train()
        concept_optimizer = optim.Adam(self.concept_net.parameters(), lr=1e-3)
        
        for epoch in range(num_epochs):
            # Sample batch
            expert_batch, violator_batch = self._sample_batch(expert_states, violator_states)
            
            # Forward pass
            expert_concepts = torch.stack([self.concept_net(traj) for traj in expert_batch])
            violator_concepts = torch.stack([self.concept_net(traj) for traj in violator_batch])
            
            # Get HSV embeddings
            expert_hsv = torch.stack([torch.FloatTensor(
                self.color_embedder.fit_transform(concepts.detach().numpy())) 
                for concepts in expert_concepts])
            violator_hsv = torch.stack([torch.FloatTensor(
                self.color_embedder.fit_transform(concepts.detach().numpy())) 
                for concepts in violator_concepts])
            
            # Combined loss
            contrastive_loss_comp, _ = self.contrastive_loss(expert_concepts, violator_concepts)
            temporal_loss = temporal_constraint_loss(expert_hsv, violator_hsv, self.manifold)
            
            total_loss = 0.5 * contrastive_loss_comp + temporal_loss
            
            # Update
            concept_optimizer.zero_grad()
            total_loss.backward()
            concept_optimizer.step()
    
    def _get_hsv_embeddings(self, expert_states, violator_states):
        """Get current HSV embeddings for all data"""
        self.concept_net.eval()
        
        with torch.no_grad():
            all_expert_concepts = []
            for states in expert_states:
                concepts = self.concept_net(torch.FloatTensor(states)).detach().numpy()
                all_expert_concepts.append(concepts)
            
            all_violator_concepts = []
            for states in violator_states:
                concepts = self.concept_net(torch.FloatTensor(states)).detach().numpy()
                all_violator_concepts.append(concepts)
            
            expert_flat = np.vstack(all_expert_concepts)
            violator_flat = np.vstack(all_violator_concepts)

            expert_hsv = self.color_embedder.fit_transform(expert_flat)
            violator_hsv = self.color_embedder.fit_transform(violator_flat)
            
            expert_hsv = self.color_embedder.smooth_trajectory(expert_hsv)
            violator_hsv = self.color_embedder.smooth_trajectory(violator_hsv)
            
            return expert_hsv, violator_hsv
    
    def _sample_batch(self, expert_states, violator_states, batch_size=8):
        """Sample batch of trajectories"""
        expert_idx = np.random.choice(len(expert_states), batch_size, replace=True)
        violator_idx = np.random.choice(len(violator_states), batch_size, replace=True)
        
        expert_batch = [torch.FloatTensor(expert_states[i]) for i in expert_idx]
        violator_batch = [torch.FloatTensor(violator_states[i]) for i in violator_idx]
        
        return expert_batch, violator_batch
    
    def _save_alternation_state(self, expert_states, violator_states, alt_step):
        """Save state for visualization"""
        expert_hsv, violator_hsv = self._get_hsv_embeddings(expert_states, violator_states)

        with torch.no_grad():
            expert_tensor = torch.FloatTensor(expert_hsv)
            violator_tensor = torch.FloatTensor(violator_hsv)
            
            expert_dists = self.manifold(expert_tensor).detach().numpy()
            violator_dists = self.manifold(violator_tensor).detach().numpy()
        
        # Get tube visualization data
        if hasattr(self.manifold, 'get_safe_boundary'):
            tube_points, tube_colors = self.manifold.get_safe_boundary()
            wireframe_points = self.manifold.get_tube_wireframe()
        else:
            tube_points, tube_colors, wireframe_points = None, None, None

        # Get manifold parameters
        if hasattr(self.manifold, 'get_constrained_params'):
            manifold_params = self.manifold.get_constrained_params()
        else:
            manifold_params = {
                'hue_center': self.manifold.hue_center.item(),
                'hue_width': self.manifold.hue_width.item(),
                'value_center': self.manifold.value_center.item(),
                'value_tolerance': self.manifold.value_tolerance.item(),
            }
        
        self.history.append({
            'alt_step': alt_step,
            'expert_hsv': expert_hsv,
            'violator_hsv': violator_hsv,
            'expert_dists': expert_dists,
            'violator_dists': violator_dists,
            'tube_points': tube_points,
            'tube_colors': tube_colors,
            'wireframe_points': wireframe_points,
            'manifold_params': manifold_params
        })