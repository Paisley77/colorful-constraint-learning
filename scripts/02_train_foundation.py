import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import pickle
import yaml
import torch
from pathlib import Path

from src.perception.concept_network import ConceptNetwork
from src.embedding.color_manifold import ColorManifoldEmbedding
from src.learning.staged_training import StagedTrainer

def load_config():
    """Load configuration file."""
    with open("configs/default.yaml", 'r') as f:
        return yaml.safe_load(f)

def load_training_data():
    """Load expert and violator demonstrations."""
    with open("data/expert/expert_demos.pkl", 'rb') as f:
        expert_data = pickle.load(f)
    with open("data/violator/violator_demos.pkl", 'rb') as f:
        violator_data = pickle.load(f)
    return expert_data, violator_data

def extract_states(trajectories):
    """Extract state sequences from trajectories."""
    return [np.array(traj['states']) for traj in trajectories]

def main():
    """Run Stage 1 foundation learning."""
    print("Starting Stage 1 Training...")
    
    # Load configuration and data
    config = load_config()
    expert_data, violator_data = load_training_data()
    
    print(f"Loaded {len(expert_data)} expert and {len(violator_data)} violator trajectories")
    
    # Extract states
    expert_states = extract_states(expert_data)
    violator_states = extract_states(violator_data)
    
    # Initialize models
    concept_net = ConceptNetwork(
        state_dim=config['concept_network']['state_dim'],
        concept_dim=config['concept_network']['concept_dim'],
        hidden_dims=config['concept_network']['hidden_dims']
    )
    
    color_embedder = ColorManifoldEmbedding(
        concept_dim=config['concept_network']['concept_dim']
    )
    
    # Initialize trainer
    trainer = StagedTrainer(concept_net, color_embedder, config)
    
    # Run Stage 1 training
    trainer.stage1_foundation_learning(
        expert_states=expert_states,
        violator_states=violator_states,
        num_epochs=config['training']['num_epochs_stage1'],
        save_interval=config['stage1']['save_interval']
    )
    
    # Create visualizations
    trainer.create_training_animation()
    trainer.plot_loss_history()
    
    # Save trained model
    torch.save(concept_net.state_dict(), 'results/models/concept_net_stage1.pth')
    print("Trained model saved to results/models/concept_net_stage1.pth")
    
    print("Stage 1 training completed!")

if __name__ == "__main__":
    # Create necessary directories
    Path("results/models").mkdir(parents=True, exist_ok=True)
    Path("results/plots").mkdir(parents=True, exist_ok=True)
    
    main()