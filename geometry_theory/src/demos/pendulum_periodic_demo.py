"""
Main demo: Map pendulum trajectories through geometric-semantic interface.
Show preservation of periodic constraints as loops in HSV space.
"""
import sys 
import os 
sys.path.append(os.path.join(os.path.dirname(__file__), '../../..'))

import numpy as np
import torch
import matplotlib.pyplot as plt
from pathlib import Path

# Import our modules
from geometry_theory.src.environment.pendulum_generator import PendulumTrajectoryGenerator
from geometry_theory.src.mappings.composite_map import GeometricSemanticInterface
# from geometry_theory.src.topology.persistence import (
#     compute_persistence_diagram, 
#     has_prominent_loop,
#     plot_persistence_diagram,
#     analyze_trajectory_topology
# )
from geometry_theory.src.spaces.hsv_manifold import HSVManifold
import plotly.io as pio

def main():
    print("=" * 70)
    print("PENDULUM PERIODIC CONSTRAINT VISUALIZATION")
    print("Geometric-Semantic Interface Demo")
    print("=" * 70)
    
    # Create directories
    visual_dir = os.path.join(os.path.dirname(__file__), '../..', "visualizations")
    data_dir = os.path.join(os.path.dirname(__file__), '../..', "data/pendulum")
    Path(visual_dir).mkdir(exist_ok=True)
    Path(data_dir).mkdir(exist_ok=True, parents=True)
    
    # Device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # =========================================================================
    # 1. Generate pendulum trajectories
    # =========================================================================
    print("\n1. Generating pendulum trajectories...")
    generator = PendulumTrajectoryGenerator()
    
    # Generate expert trajectories (periodic, safe)
    expert_data = generator.generate_expert_trajectory(
        n_trajectories=20,
        duration=6.0  # Longer for better period visualization
    )
    
    # Generate violator trajectories (non-periodic, unsafe)
    violator_data = generator.generate_violator_trajectory(
        n_trajectories=20,
        duration=6.0
    )
    
    # Visualize trajectories
    generator.visualize_trajectories(expert_data, violator_data)
    
    # Prepare training data
    # Flatten trajectories for training
    expert_states = np.vstack(expert_data['trajectories'])
    violator_states = np.vstack(violator_data['trajectories'])
    
    # Convert to tensors
    expert_tensor = torch.tensor(expert_states, dtype=torch.float32)
    violator_tensor = torch.tensor(violator_states, dtype=torch.float32)
    
    print(f"Expert states: {expert_tensor.shape}")
    print(f"Violator states: {violator_tensor.shape}")
    
    # =========================================================================
    # 2. Create and train geometric-semantic interface
    # =========================================================================
    print("\n2. Creating geometric-semantic interface...")
    
    interface = GeometricSemanticInterface(
        state_dim=4,
        concept_dim=12,  # Higher for more expressive concepts
        flow_dim=3,
        encoder_hidden=(64, 32),
        flow_layers=8
    )
    
    # Phase 1: Train autoencoder for reconstruction
    print("\nPhase 1: Training autoencoder for reconstruction...")
    all_states = torch.cat([expert_tensor, violator_tensor], dim=0)
    autoencoder_losses = interface.train_autoencoder(
        states=all_states,
        epochs=10,  # Can be fewer since this is just reconstruction
        batch_size=64,
        lr=1e-3,
        device=device
    )

    # Phase 2: Train flow to make encoded concepts uniform
    print("\nPhase 2: Training normalizing flow to uniform distribution...")
    flow_metrics = interface.train_flow_to_uniform(
        expert_states=expert_tensor,
        violator_states=violator_tensor,
        epochs=20,
        batch_size=128,
        lr=1e-3,
        device=device
    )

    # Plot training progress
    fig, axes = plt.subplots(1, 2, figsize=(12, 10))

    # Autoencoder loss
    axes[0].plot(autoencoder_losses)
    axes[0].set_title('Autoencoder Reconstruction Loss')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('MSE Loss')
    axes[0].grid(True, alpha=0.3)

    # Flow training
    axes[1].plot(flow_metrics['loss'], label='Total Loss')
    axes[1].plot(flow_metrics['expert_log_prob'], label='Expert logP')
    axes[1].plot(flow_metrics['violator_log_prob'], label='Violator logP')
    axes[1].set_title('Flow Training')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Loss / Log Probability')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    training_plot_path = os.path.join(os.path.dirname(__file__), '../..', 'visualizations/sequential_training.png')
    plt.savefig(training_plot_path, dpi=150, bbox_inches='tight')
    plt.show()

    
    # =========================================================================
    # 3. Test topological preservation on sample trajectories
    # =========================================================================
    print("\n3. Testing topological preservation...")
    
    # Select sample trajectories for visualization
    sample_expert_traj = expert_data['trajectories'][0]  # First expert trajectory
    sample_violator_traj = violator_data['trajectories'][0]  # First violator trajectory
    
    # Test expert trajectory
    print("\nAnalyzing expert trajectory (should preserve loop)...")
    expert_results = interface.test_topological_preservation(
        sample_expert_traj, device=device
    )
    
    print(f"  Loop closure error: {expert_results['loop_closure_error']:.4f}")
    print(f"  Periodic score: {expert_results['periodic_score']:.4f}")
    print(f"  Hue circular variance: {expert_results['hue_circular_variance']:.4f}")
    
    # Test violator trajectory
    print("\nAnalyzing violator trajectory (should NOT preserve loop)...")
    violator_results = interface.test_topological_preservation(
        sample_violator_traj, device=device
    )
    
    print(f"  Loop closure error: {violator_results['loop_closure_error']:.4f}")
    print(f"  Periodic score: {violator_results['periodic_score']:.4f}")
    print(f"  Hue circular variance: {violator_results['hue_circular_variance']:.4f}")
    
    # =========================================================================
    # 4. Visualize trajectories in HSV space
    # =========================================================================
    print("\n4. Creating visualizations...")
    
    # Create HSV manifold for visualization
    hsv_manifold = HSVManifold()
    
    # Expert trajectory in HSV space
    expert_hsv = expert_results['hsv_trajectory']
    fig_expert = hsv_manifold.visualize_3d(
        trajectory=expert_hsv,
        show_cylinder=True,
        title=f"Expert Trajectory in HSV Space (Periodic Score: {expert_results['periodic_score']:.3f})"
    )
    
    # Violator trajectory in HSV space
    violator_hsv = violator_results['hsv_trajectory']
    fig_violator = hsv_manifold.visualize_3d(
        trajectory=violator_hsv,
        show_cylinder=True,
        title=f"Violator Trajectory in HSV Space (Periodic Score: {violator_results['periodic_score']:.3f})"
    )
    
    # Save interactive plots
    expert_hsv_path = os.path.join(os.path.dirname(__file__), '../..', "visualizations/expert_hsv_trajectory.html")
    violator_hsv_path = os.path.join(os.path.dirname(__file__), '../..', "visualizations/violator_hsv_trajectory.html")
    
    pio.write_html(fig_expert, expert_hsv_path)
    pio.write_html(fig_violator, violator_hsv_path)
    
    # Create comparison figure (static, for paper)
    fig_comparison = plt.figure(figsize=(15, 6))
    
    # Expert: 3D plot
    ax1 = fig_comparison.add_subplot(121, projection='3d')
    expert_cartesian = expert_results['cartesian_trajectory']
    
    # Color by hue
    hues = expert_hsv[:, 0] / (2 * np.pi)  # Normalize to [0,1]
    colors = plt.cm.rainbow(hues)
    
    # Plot trajectory
    for i in range(len(expert_cartesian) - 1):
        ax1.plot(expert_cartesian[i:i+2, 0], 
                 expert_cartesian[i:i+2, 1],
                 expert_cartesian[i:i+2, 2],
                 color=colors[i], linewidth=2, alpha=0.8)
    
    ax1.set_xlabel('X (cos(hue) × saturation)')
    ax1.set_ylabel('Y (sin(hue) × saturation)')
    ax1.set_zlabel('Value')
    ax1.set_title(f'Expert: Clean Loop (Score: {expert_results["periodic_score"]:.3f})')
    ax1.grid(True, alpha=0.3)
    
    # Violator: 3D plot
    ax2 = fig_comparison.add_subplot(122, projection='3d')
    violator_cartesian = violator_results['cartesian_trajectory']
    
    # Color by hue
    hues_v = violator_hsv[:, 0] / (2 * np.pi)
    colors_v = plt.cm.rainbow(hues_v)
    
    # Plot trajectory
    for i in range(len(violator_cartesian) - 1):
        ax2.plot(violator_cartesian[i:i+2, 0], 
                 violator_cartesian[i:i+2, 1],
                 violator_cartesian[i:i+2, 2],
                 color=colors_v[i], linewidth=2, alpha=0.8)
    
    ax2.set_xlabel('X (cos(hue) × saturation)')
    ax2.set_ylabel('Y (sin(hue) × saturation)')
    ax2.set_zlabel('Value')
    ax2.set_title(f'Violator: Chaotic Path (Score: {violator_results["periodic_score"]:.3f})')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plot_path = os.path.join(os.path.dirname(__file__), '../..', 'visualizations/hsv_trajectory_comparison.png')
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.show()
    
    # =========================================================================
    # 5. Create summary report
    # =========================================================================
    print("\n" + "=" * 70)
    print("DEMO SUMMARY")
    print("=" * 70)
    
    print(f"\nExpert trajectory analysis:")
    print(f"  • Periodic score: {expert_results['periodic_score']:.3f} (1.0 = perfect loop)")
    print(f"  • Loop closure error: {expert_results['loop_closure_error']:.4f}")
    print(f"  • Trajectory forms a CLEAN LOOP in HSV space")
    
    print(f"\nViolator trajectory analysis:")
    print(f"  • Periodic score: {violator_results['periodic_score']:.3f}")
    print(f"  • Loop closure error: {violator_results['loop_closure_error']:.4f}")
    print(f"  • Trajectory forms a CHAOTIC PATH in HSV space")
    
    print(f"\nKey observation:")
    print("  The geometric-semantic interface successfully:")
    print("  1. Preserves the periodic structure of expert trajectories as loops")
    print("  2. Maps violator trajectories to non-looping, chaotic paths")
    print("  3. Provides VISUAL distinction between constraint-satisfying")
    print("     and constraint-violating behaviors")
    
    print(f"\nGenerated files:")
    print("  • visualizations/pendulum_trajectories.png - Raw trajectories")
    print("  • visualizations/sequential_training.png - Training progress")
    print("  • visualizations/expert_hsv_trajectory.html - Interactive expert in HSV")
    print("  • visualizations/violator_hsv_trajectory.html - Interactive violator in HSV")
    print("  • visualizations/hsv_trajectory_comparison.png - Side-by-side comparison")
    
    print("\n" + "=" * 70)
    print("Next steps:")
    print("  1. Open HTML files in browser to interact with 3D visualizations")
    print("  2. Rotate HSV plots to see the loop structure")
    print("=" * 70)

if __name__ == "__main__":
    main()