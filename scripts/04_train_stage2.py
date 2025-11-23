import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import pickle
from src.learning.stage2_training import Stage2Trainer
from src.perception.concept_network import ConceptNetwork
from src.embedding.color_manifold import ColorManifoldEmbedding
import torch
import numpy as np

def main():
    # Load Stage 1 trained model
    concept_net = ConceptNetwork()
    concept_net.load_state_dict(torch.load('results/models/concept_net_stage1.pth'))
    
    color_embedder = ColorManifoldEmbedding()
    
    # Load data
    with open("data/expert/expert_demos.pkl", 'rb') as f:
        expert_data = pickle.load(f)
    with open("data/violator/violator_demos.pkl", 'rb') as f:
        violator_data = pickle.load(f)
    
    expert_states = [np.array(traj['states']) for traj in expert_data]
    violator_states = [np.array(traj['states']) for traj in violator_data]
    
    # Run Stage 2
    trainer = Stage2Trainer(concept_net, color_embedder, config={})
    trainer.stage2_alternating_optimization(expert_states, violator_states, num_alternations=8)
    
    # Create visualization
    from src.perception.stage2_animation import create_tube_animation, create_circle_animation
    create_tube_animation(trainer.history)
    create_circle_animation(trainer.history)
    
    print("Stage 2 completed!")

if __name__ == "__main__":
    main()