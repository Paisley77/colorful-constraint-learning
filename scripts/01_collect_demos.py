import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import torch
import pickle
from pathlib import Path
from src.environments.cartpole import InvertedPendulumEnv
from src.controller.lqr_controller import LQRController

def collect_expert_demonstrations(num_trajectories: int = 50, 
                                 max_steps: int = 200,
                                 save_path: str = "data/expert") -> None:
    """Collect expert demonstrations that maintain stability."""
    env = InvertedPendulumEnv()
    lqr = LQRController(env.config)
    expert_trajectories = []
    
    print(f"Collecting {num_trajectories} expert demonstrations...")
    
    for episode in range(num_trajectories):
        state, _ = env.reset()
        trajectory = {
            'states': [],
            'actions': [], 
            'rewards': [],
            'costs': [],
            'done': False
        }
        
        # Simple stabilizing controller (simulates expert)
        for step in range(max_steps):
            # LQR Controller 
            force = lqr.compute_control(state)
            
            next_state, reward, done, truncated, info = env.step(np.array([force]))
            
            trajectory['states'].append(state)
            trajectory['actions'].append(force)
            trajectory['rewards'].append(reward)
            trajectory['costs'].append(info['cost'])
            
            state = next_state
            
            if done or truncated:
                trajectory['done'] = True
                state, _ = env.reset() 
        
        expert_trajectories.append(trajectory)
        
        if (episode + 1) % 10 == 0:
            print(f"Collected {episode + 1}/{num_trajectories} expert trajectories")
    
    # Save expert demonstrations
    Path(save_path).mkdir(parents=True, exist_ok=True)
    with open(f"{save_path}/expert_demos.pkl", 'wb') as f:
        pickle.dump(expert_trajectories, f)
    
    print(f"Expert demonstrations saved to {save_path}/expert_demos.pkl")
    env.close()

    return expert_trajectories

def collect_violator_demonstrations(num_trajectories: int = 50,
                                   max_steps: int = 200,
                                   save_path: str = "data/violator") -> None:
    """Collect trajectories that violate constraints (unstable behavior)."""
    env = InvertedPendulumEnv()
    violator_trajectories = []
    
    print(f"Collecting {num_trajectories} violator demonstrations...")
    
    for episode in range(num_trajectories):
        # Start from more challenging initial conditions
        theta0 = np.random.uniform(-2.75, 2.75)  # Larger initial angle
        state, _ = env.reset(options={"initial_state": np.array([0.0, 0.0, theta0, 0.0])})
        
        trajectory = {
            'states': [],
            'actions': [],
            'rewards': [], 
            'costs': [],
            'done': False
        }
        
        # Apply destabilizing or random control
        for step in range(max_steps):
            # Sometimes apply destabilizing force, sometimes random
            if np.random.random() < 0.7:
                # Destabilizing control
                force = 5.0 * np.sign(state[2])  # Push in direction of tilt
            else:
                # Random control
                force = np.random.uniform(-env.config['max_force'], env.config['max_force'])
            
            next_state, reward, done, truncated, info = env.step(np.array([force]))
            
            trajectory['states'].append(state)
            trajectory['actions'].append(force)
            trajectory['rewards'].append(reward)
            trajectory['costs'].append(info['cost'])
            
            state = next_state
            
            if done or truncated:
                trajectory['done'] = True
                state, _ = env.reset() 
        
        violator_trajectories.append(trajectory)
        
        if (episode + 1) % 10 == 0:
            print(f"Collected {episode + 1}/{num_trajectories} violator trajectories")
    
    # Save violator demonstrations
    Path(save_path).mkdir(parents=True, exist_ok=True)
    with open(f"{save_path}/violator_demos.pkl", 'wb') as f:
        pickle.dump(violator_trajectories, f)
    
    print(f"Violator demonstrations saved to {save_path}/violator_demos.pkl")
    env.close()

    return violator_trajectories

if __name__ == "__main__":
    # Collect both expert and violator demonstrations
    env = InvertedPendulumEnv()
    env.reset()
    expert = collect_expert_demonstrations(num_trajectories=100)
    for i in range(3):
        for j in range(len(expert[i]['states'])):
            env.set_state(expert[i]['states'][j])
            env.render() 
    violator = collect_violator_demonstrations(num_trajectories=100)
    for i in range(3):
        for j in range(len(violator[i]['states'])):
            env.set_state(violator[i]['states'][j])
            env.render() 
    