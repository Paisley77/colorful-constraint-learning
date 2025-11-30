import torch
import torch.optim as optim
import numpy as np
from tqdm import tqdm 
import sys
sys.path.append('../..')
from src.learning.temporal_tcn import TemporalPatternBank

class Stage3Trainer:
    def __init__(self, concept_net, color_embedder, manifold, config):
        self.concept_net = concept_net
        self.color_embedder = color_embedder  
        self.manifold = manifold
        self.config = config
        
        # Stage 3 components
        self.tcn_bank = TemporalPatternBank(num_tcns=4)
        self.tcn_optimizer = optim.Adam(self.tcn_bank.parameters(), lr=1e-3)
        
        # Tracking
        self.history = []
        
    def stage3_temporal_learning(self, expert_states, violator_states, num_epochs=100):
        """Learn temporal patterns that complement the geometric manifold"""
        
        # Freeze earlier components
        self.concept_net.eval()
        self.manifold.eval()
        
        for epoch in tqdm(range(num_epochs)):
            self.tcn_bank.train()
            
            # Sample batch
            expert_batch, violator_batch = self._sample_batch(expert_states, violator_states)
            
            # Get HSV trajectories
            expert_hsv = self._states_to_hsv_trajectories(expert_batch) # [batch_size, T, 3]
            violator_hsv = self._states_to_hsv_trajectories(violator_batch)
            
            # Get TCN scores
            expert_scores, expert_patterns = self.tcn_bank(expert_hsv) # [batch_size, ]
            violator_scores, violator_patterns = self.tcn_bank(violator_hsv)
            
            # Loss: TCNs should separate trajectories further
            margin = 1.0
            tcn_loss = torch.clamp(margin - (expert_scores.mean() - violator_scores.mean()), min=0)
            
            # Sparsity regularization
            sparsity_loss = torch.norm(torch.softmax(self.tcn_bank.weights, dim=0), p=1)
            
            total_loss = tcn_loss + 0.1 * sparsity_loss
            
            # Update
            self.tcn_optimizer.zero_grad()
            total_loss.backward()
            self.tcn_optimizer.step()
              
            # Save state for visualization
            if epoch % 10 == 0:
                self._save_temporal_state(expert_states, violator_states, epoch)

        # torch.save(self.tcn_bank.state_dict(), 'results/models/TCN_stage3.pth')
        # print("Trained TCNs saved to results/models/TCN_stage3.pth")
    
    def _states_to_hsv_trajectories(self, state_batch):
        """Convert state batch to HSV trajectories"""
        hsv_trajectories = []
        concept_batch = []
        
        with torch.no_grad():
            for states in state_batch:
                concepts = self.concept_net(torch.FloatTensor(states)) # [T, k]
                concept_batch.append(concepts.detach().numpy())

            concept_batch = np.vstack(concept_batch) # [batch_size x T, k]
            
            hsv = self.color_embedder.fit_transform(concept_batch) # [batch_size x T, 3]
            hsv = self.color_embedder.smooth_trajectory(hsv) # [batch_size x T, 3]
            batch_size = len(state_batch)
            hsv_trajectories = hsv.reshape(batch_size, -1, 3)
            return torch.FloatTensor(hsv_trajectories) # [batch_size, T, 3]
    
    def _sample_batch(self, expert_states, violator_states, batch_size=32):
        """Sample batch of trajectories"""
        expert_idx = np.random.choice(len(expert_states), batch_size, replace=True)
        violator_idx = np.random.choice(len(violator_states), batch_size, replace=True)
        
        expert_batch = [expert_states[i] for i in expert_idx]
        violator_batch = [violator_states[i] for i in violator_idx]
        
        return expert_batch, violator_batch
    
    def _save_temporal_state(self, expert_states, violator_states, epoch):
        """Save state for temporal pattern visualization"""
        self.tcn_bank.eval()
        
        with torch.no_grad():
            # Get all trajectories
            expert_hsv = self._states_to_hsv_trajectories(expert_states) # [batch_size, T, 3]
            violator_hsv = self._states_to_hsv_trajectories(violator_states)
            
            # Get TCN outputs
            expert_scores, expert_patterns = self.tcn_bank(expert_hsv) # [batch_size, 1], [batch_size, num_tcns] 
            violator_scores, violator_patterns = self.tcn_bank(violator_hsv)
            
            # Get manifold distances - compute mean distance per trajectory
            expert_manifold_dists = self.manifold(expert_hsv.reshape(-1, 3)) # [batch_size x T, ]
            violator_manifold_dists = self.manifold(violator_hsv.reshape(-1,3))
        
        self.history.append({
            'epoch': epoch,
            'expert_hsv': expert_hsv.view(-1, 3).detach().numpy(), # [batch_size x T, 3]
            'violator_hsv': violator_hsv.view(-1, 3).detach().numpy(),
            'expert_tcn_scores': expert_scores.detach().numpy(), # [batch_size, ]
            'violator_tcn_scores': violator_scores.detach().numpy(), 
            'expert_patterns': expert_patterns.detach().numpy(), # [batch_size, num_tcns] 
            'violator_patterns': violator_patterns.detach().numpy(), 
            'tcn_weights': torch.softmax(self.tcn_bank.weights, dim=0).detach().numpy(), # [num_tcns, ]
            'expert_manifold_dists': expert_manifold_dists.detach().numpy(), # [batch_size x T, ]
            'violator_manifold_dists': violator_manifold_dists.detach().numpy()
        })