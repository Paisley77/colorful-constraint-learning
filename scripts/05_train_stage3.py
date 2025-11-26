import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import pickle
import torch
import numpy as np 
from src.learning.stage3_training import Stage3Trainer
from src.learning.temporal_tcn import TemporalPatternBank
from src.perception.concept_network import ConceptNetwork
from src.embedding.color_manifold import ColorManifoldEmbedding
from src.learning.simple_manifold import CylindricalManifold

def main():
    # Load previous stages
    concept_net = ConceptNetwork()
    concept_net.load_state_dict(torch.load('results/models/concept_net_stage2.pth'))
    
    color_embedder = ColorManifoldEmbedding()
    manifold = CylindricalManifold()
    manifold.load_state_dict(torch.load('results/models/manifold_stage2.pth'))
    
    # Load data
    with open("data/expert/expert_demos.pkl", 'rb') as f:
        expert_data = pickle.load(f)
    with open("data/violator/violator_demos.pkl", 'rb') as f:
        violator_data = pickle.load(f)
    
    expert_states = [np.array(traj['states']) for traj in expert_data]
    violator_states = [np.array(traj['states']) for traj in violator_data]
    
    # Run Stage 3
    trainer = Stage3Trainer(concept_net, color_embedder, manifold, config={})
    trainer.stage3_temporal_learning(expert_states, violator_states, num_epochs=100)
    
    # Create visualization
    from src.perception.stage3_animation import create_temporal_animation
    create_temporal_animation(trainer.history)
    
    print("Stage 3 completed! Check results/plots/temporal_evolution.gif")

if __name__ == "__main__":
    main()