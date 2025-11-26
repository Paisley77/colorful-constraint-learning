import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import hsv_to_rgb
from pathlib import Path

def create_temporal_animation(history, save_path="results/plots/temporal_evolution.gif"):
    """Animate temporal pattern learning process"""
    
    fig = plt.figure(figsize=(16, 12), facecolor='black')
    
    def update(frame):
        plt.clf()
        fig.patch.set_facecolor('black')
        data = history[frame]
        
        # Panel 1: HSV Space with Combined Scoring
        ax1 = fig.add_subplot(221, projection='3d')
        ax1.set_facecolor('black')
        _plot_combined_scoring(ax1, data, frame)
        
        # Panel 2: TCN Pattern Weights
        ax2 = fig.add_subplot(222)
        ax2.set_facecolor('black')
        _plot_pattern_weights(ax2, data, frame)
        
        # Panel 3: Temporal Pattern Activations
        ax3 = fig.add_subplot(223)
        ax3.set_facecolor('black') 
        _plot_pattern_activations(ax3, data, frame)
        
        # Panel 4: Separation Progress
        ax4 = fig.add_subplot(224)
        ax4.set_facecolor('black')
        _plot_separation_progress(ax4, history, frame)
        
        plt.suptitle(f'Stage 3: Temporal Pattern Learning\nEpoch {data["epoch"]}', 
                    color='white', fontsize=14, y=0.95)
        plt.tight_layout()
    
    from matplotlib.animation import FuncAnimation
    
    anim = FuncAnimation(fig, update, frames=len(history), interval=800, repeat=True)
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    
    anim.save(save_path, writer='pillow', fps=2, dpi=120,
              savefig_kwargs={'facecolor': 'black'})
    
    plt.close()
    print(f"Temporal pattern animation saved to {save_path}")

def _plot_combined_scoring(ax, data, frame):
    """Plot HSV space colored by combined (manifold + TCN) scores"""
    expert_hsv = data['expert_hsv']
    violator_hsv = data['violator_hsv']
    
    # Combine manifold and TCN scores
    T = data['expert_manifold_dists'].shape[0] // data['expert_tcn_scores'].shape[0]
    expert_combined = -data['expert_manifold_dists'] + np.repeat(data['expert_tcn_scores'], T)
    violator_combined = -data['violator_manifold_dists'] + np.repeat(data['violator_tcn_scores'], T)
    
    # Normalize for coloring
    all_scores = np.concatenate([expert_combined, violator_combined])
    vmin, vmax = np.percentile(all_scores, [5, 95])
    
    # High score --> blue; low score --> red 
    expert_colors = plt.cm.RdBu((expert_combined - vmin) / (vmax - vmin))
    violator_colors = plt.cm.RdBu((violator_combined - vmin) / (vmax - vmin))
    
    # Plot points
    ax.scatter(expert_hsv[:, 0], expert_hsv[:, 1], expert_hsv[:, 2],
               c=expert_colors, alpha=0.7, s=15, edgecolors='darkblue', linewidth=0.2, label='Expert')
    ax.scatter(violator_hsv[:, 0], violator_hsv[:, 1], violator_hsv[:, 2],
               c=violator_colors, alpha=0.7, s=15, edgecolors='darkred', linewidth=0.2, marker='^', label='Violator')
    
    ax.set_xlabel('Hue', color='white')
    ax.set_ylabel('Saturation', color='white')
    ax.set_zlabel('Value', color='white')
    ax.set_title('Combined Geometric + Temporal Scoring', color='white', pad=10)
    ax.legend(facecolor='black', labelcolor='white')
    ax.grid(True, alpha=0.2, color='white')
    ax.tick_params(colors='white')

def _plot_pattern_weights(ax, data, frame):
    """Plot TCN pattern importance weights"""
    weights = data['tcn_weights']
    patterns = [f'Pattern {i+1}' for i in range(len(weights))]
    
    colors = plt.cm.Set3(np.linspace(0, 1, len(weights)))
    bars = ax.bar(patterns, weights, color=colors, alpha=0.8)
    
    # Add value labels
    for bar, weight in zip(bars, weights):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{weight:.2f}', ha='center', va='bottom', color='white', fontsize=9)
    
    ax.set_ylabel('Pattern Weight', color='white')
    ax.set_title('TCN Pattern Importance', color='white', pad=10)
    ax.tick_params(axis='x', rotation=45, colors='white')
    ax.tick_params(axis='y', colors='white')
    ax.grid(True, alpha=0.2, axis='y', color='white')

def _plot_pattern_activations(ax, data, frame):
    """Plot distribution of pattern activations"""
    expert_patterns = data['expert_patterns'].mean(axis=0)  # [num_tcns]
    violator_patterns = data['violator_patterns'].mean(axis=0)

    # print("=================Shape======================")
    # print(data['expert_patterns'].shape)
    
    x = np.arange(len(expert_patterns))
    width = 0.35
    
    ax.bar(x - width/2, expert_patterns, width, label='Expert', alpha=0.7, color='blue')
    ax.bar(x + width/2, violator_patterns, width, label='Violator', alpha=0.7, color='red')
    
    ax.set_xlabel('TCN Pattern', color='white')
    ax.set_ylabel('Mean Activation', color='white')
    ax.set_title('Pattern Activations by Class', color='white', pad=10)
    ax.legend(facecolor='black', labelcolor='white')
    ax.grid(True, alpha=0.2, color='white')
    ax.tick_params(colors='white')

def _plot_separation_progress(ax, history, frame):
    """Plot separation progress across all stages"""
    epochs = [h['epoch'] for h in history[:frame+1]]
    
    # Compute separation metrics
    separation_scores = []
    for h in history[:frame+1]:
        expert_mean = h['expert_tcn_scores'].mean()
        violator_mean = h['violator_tcn_scores'].mean() 
        separation = expert_mean - violator_mean  # Positive = good separation
        separation_scores.append(separation)
    
    ax.plot(epochs, separation_scores, 'o-', color='cyan', linewidth=2, markersize=4)
    ax.axhline(y=0, color='white', linestyle='--', alpha=0.5)
    ax.set_xlabel('Epoch', color='white')
    ax.set_ylabel('TCN Separation Score', color='white')
    ax.set_title('Temporal Separation Progress', color='white', pad=10)
    ax.grid(True, alpha=0.2, color='white')
    ax.tick_params(colors='white')