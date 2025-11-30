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


def visualize_temporal_patterns(history, save_path="results/plots/stage3_temporal_patterns.png"):
    """Create comprehensive visualization of Stage 3 temporal pattern learning"""
    
    # Get key epochs for progression
    initial_data = history[0]
    mid_data = history[len(history)//2] 
    final_data = history[-1]
    
    fig = plt.figure(figsize=(20, 12))
    
    # Panel 1: TCN Weight Evolution
    ax1 = plt.subplot2grid((3, 4), (0, 0), colspan=2)
    _plot_weight_evolution(ax1, history)
    
    # Panel 2: Final Pattern Weights
    ax2 = plt.subplot2grid((3, 4), (0, 2))
    _plot_final_weights(ax2, final_data)
    
    # Panel 3: Pattern Activations
    ax3 = plt.subplot2grid((3, 4), (0, 3))
    _plot_pattern_activations(ax3, final_data)
    
    # Panel 4: Sample Trajectory with Pattern Detection
    ax4 = plt.subplot2grid((3, 4), (1, 0), colspan=2)
    _plot_trajectory_patterns(ax4, final_data)
    
    # Panel 5: Temporal Pattern Separation
    ax5 = plt.subplot2grid((3, 4), (1, 2), colspan=2)
    _plot_temporal_separation(ax5, final_data)
    
    # Panel 6: Combined Scoring
    ax6 = plt.subplot2grid((3, 4), (2, 0), colspan=2)
    _plot_combined_scoring(ax6, final_data)
    
    # Panel 7: Learning Progress
    ax7 = plt.subplot2grid((3, 4), (2, 2), colspan=2)
    _plot_learning_progress(ax7, history)
    
    plt.suptitle('Stage 3: Temporal Pattern Learning Analysis', fontsize=16, y=0.95)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

def _plot_weight_evolution(ax, history):
    """Plot how TCN weights evolve during training"""
    epochs = [h['epoch'] for h in history]
    weights_history = np.array([h['tcn_weights'] for h in history])
    
    num_patterns = weights_history.shape[1]
    colors = plt.cm.Set3(np.linspace(0, 1, num_patterns))
    
    for i in range(num_patterns):
        ax.plot(epochs, weights_history[:, i], linewidth=2.5, 
                color=colors[i], label=f'Pattern {i+1}', alpha=0.8)
    
    ax.set_xlabel('Training Epoch')
    ax.set_ylabel('TCN Pattern Weight')
    ax.set_title('TCN Weight Evolution')
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.grid(True, alpha=0.3)

def _plot_final_weights(ax, data):
    """Plot final TCN pattern weights as a bar chart"""
    weights = data['tcn_weights']
    patterns = [f'P{i+1}' for i in range(len(weights))]
    
    colors = plt.cm.viridis(np.linspace(0, 1, len(weights)))
    bars = ax.bar(patterns, weights, color=colors, alpha=0.7)
    
    # Add value labels on bars
    for bar, weight in zip(bars, weights):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{weight:.2f}', ha='center', va='bottom', fontsize=9)
    
    ax.set_ylabel('Final Weight')
    ax.set_title('Final Pattern Importance')
    ax.grid(True, alpha=0.3, axis='y')

def _plot_pattern_activations(ax, data):
    """Plot pattern activations for expert vs violator trajectories"""
    expert_patterns = data['expert_patterns']  # [batch_size, num_patterns]
    violator_patterns = data['violator_patterns']
    
    # Take mean across trajectories
    expert_mean = expert_patterns.mean(axis=0)
    violator_mean = violator_patterns.mean(axis=0)
    
    x = np.arange(len(expert_mean))
    width = 0.35
    
    ax.bar(x - width/2, expert_mean, width, label='Expert', 
           alpha=0.7, color='blue', edgecolor='darkblue')
    ax.bar(x + width/2, violator_mean, width, label='Violator',
           alpha=0.7, color='red', edgecolor='darkred')
    
    ax.set_xlabel('TCN Pattern')
    ax.set_ylabel('Mean Activation')
    ax.set_title('Pattern Activations by Class')
    ax.legend()
    ax.grid(True, alpha=0.3)

def _plot_trajectory_patterns(ax, data):
    """Plot a sample trajectory with pattern activations over time"""
    # Take first trajectory from each class
    expert_patterns = data['expert_patterns'][0]  # [seq_len, num_patterns]
    violator_patterns = data['violator_patterns'][0]
    
    time_steps = np.arange(len(expert_patterns))
    num_patterns = expert_patterns.shape[1]
    
    # Create subplots for each pattern
    for i in range(num_patterns):
        ax.plot(time_steps, expert_patterns[:, i], 
                label=f'Expert Pattern {i+1}', linewidth=2, alpha=0.8)
        ax.plot(time_steps, violator_patterns[:, i], '--',
                label=f'Violator Pattern {i+1}', linewidth=2, alpha=0.8)
    
    ax.set_xlabel('Time Step')
    ax.set_ylabel('Pattern Activation')
    ax.set_title('Temporal Pattern Activations\n(Sample Trajectories)')
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.grid(True, alpha=0.3)

def _plot_temporal_separation(ax, data):
    """Plot distribution of TCN scores for expert vs violator"""
    expert_scores = data['expert_tcn_scores']
    violator_scores = data['violator_tcn_scores']
    
    # Create violin plot
    positions = [1, 2]
    data_to_plot = [expert_scores, violator_scores]
    labels = ['Expert', 'Violator']
    
    parts = ax.violinplot(data_to_plot, positions=positions, 
                         showmeans=True, showmedians=True)
    
    # Color the violins
    for pc, color in zip(parts['bodies'], ['lightblue', 'lightcoral']):
        pc.set_facecolor(color)
        pc.set_alpha(0.7)
    
    ax.set_xticks(positions)
    ax.set_xticklabels(labels)
    ax.set_ylabel('TCN Score')
    ax.set_title('Temporal Pattern Separation')
    ax.grid(True, alpha=0.3)
    
    # Add some statistics
    expert_mean = np.mean(expert_scores)
    violator_mean = np.mean(violator_scores)
    ax.text(0.5, 0.95, f'Separation: {violator_mean - expert_mean:.3f}',
            transform=ax.transAxes, ha='center', 
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))

def _plot_combined_scoring(ax, data):
    """Plot combined geometric + temporal scoring"""
    expert_combined = -data['expert_manifold_dists'] + data['expert_tcn_scores']
    violator_combined = -data['violator_manifold_dists'] + data['violator_tcn_scores']
    
    # Create histogram
    bins = np.linspace(min(np.min(expert_combined), np.min(violator_combined)),
                      max(np.max(expert_combined), np.max(violator_combined)), 30)
    
    ax.hist(expert_combined, bins=bins, alpha=0.7, color='blue', 
            label='Expert', density=True)
    ax.hist(violator_combined, bins=bins, alpha=0.7, color='red',
            label='Violator', density=True)
    
    ax.set_xlabel('Combined Score (Geometric + Temporal)')
    ax.set_ylabel('Density')
    ax.set_title('Final Combined Scoring')
    ax.legend()
    ax.grid(True, alpha=0.3)

def _plot_learning_progress(ax, history):
    """Plot learning progress metrics"""
    epochs = [h['epoch'] for h in history]
    
    # Compute separation metrics
    separation_scores = []
    accuracy_scores = []
    
    for h in history:
        expert_mean = np.mean(h['expert_tcn_scores'])
        violator_mean = np.mean(h['violator_tcn_scores'])
        separation = violator_mean - expert_mean
        separation_scores.append(separation)
        
        # Simple accuracy estimate
        expert_correct = np.sum(h['expert_tcn_scores'] < 0)  # Experts should have low scores
        violator_correct = np.sum(h['violator_tcn_scores'] > 0)  # Violators should have high scores
        accuracy = (expert_correct + violator_correct) / (len(h['expert_tcn_scores']) + len(h['violator_tcn_scores']))
        accuracy_scores.append(accuracy)
    
    # Plot both metrics
    ax.plot(epochs, separation_scores, 'o-', linewidth=2.5, 
            label='Separation Score', color='green', markersize=4)
    ax.plot(epochs, accuracy_scores, 's-', linewidth=2.5,
            label='Classification Accuracy', color='purple', markersize=4)
    
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Metric Value')
    ax.set_title('Learning Progress')
    ax.legend()
    ax.grid(True, alpha=0.3)

def create_temporal_animation(history_path="results/stage3_history.pkl", 
                            save_path="figures/stage3_evolution.gif"):
    """Create animation showing temporal pattern learning evolution"""
    
    with open(history_path, 'rb') as f:
        history = pickle.load(f)
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    
    def update(frame):
        for ax in [ax1, ax2, ax3, ax4]:
            ax.clear()
        
        data = history[frame]
        epoch = data['epoch']
        
        # Panel 1: Weight evolution up to current epoch
        epochs_so_far = [h['epoch'] for h in history[:frame+1]]
        weights_so_far = np.array([h['tcn_weights'] for h in history[:frame+1]])
        
        num_patterns = weights_so_far.shape[1]
        colors = plt.cm.Set3(np.linspace(0, 1, num_patterns))
        
        for i in range(num_patterns):
            ax1.plot(epochs_so_far, weights_so_far[:, i], 
                    color=colors[i], label=f'Pattern {i+1}', linewidth=2)
        
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Pattern Weight')
        ax1.set_title(f'Weight Evolution (Epoch {epoch})')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Panel 2: Current pattern activations
        expert_mean = data['expert_patterns'].mean(axis=0)
        violator_mean = data['violator_patterns'].mean(axis=0)
        
        x = np.arange(len(expert_mean))
        width = 0.35
        
        ax2.bar(x - width/2, expert_mean, width, label='Expert', 
               alpha=0.7, color='blue')
        ax2.bar(x + width/2, violator_mean, width, label='Violator',
               alpha=0.7, color='red')
        
        ax2.set_xlabel('Pattern')
        ax2.set_ylabel('Activation')
        ax2.set_title('Current Pattern Activations')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Panel 3: Separation progress
        separation_so_far = []
        for h in history[:frame+1]:
            sep = np.mean(h['violator_tcn_scores']) - np.mean(h['expert_tcn_scores'])
            separation_so_far.append(sep)
        
        ax3.plot(epochs_so_far, separation_so_far, 'o-', 
                color='green', linewidth=2, markersize=4)
        ax3.set_xlabel('Epoch')
        ax3.set_ylabel('Separation Score')
        ax3.set_title('Temporal Separation Progress')
        ax3.grid(True, alpha=0.3)
        
        # Panel 4: Current weights
        weights = data['tcn_weights']
        patterns = [f'P{i+1}' for i in range(len(weights))]
        
        bars = ax4.bar(patterns, weights, alpha=0.7, 
                      color=plt.cm.viridis(np.linspace(0, 1, len(weights))))
        
        for bar, weight in zip(bars, weights):
            height = bar.get_height()
            ax4.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{weight:.2f}', ha='center', va='bottom', fontsize=9)
        
        ax4.set_ylabel('Weight')
        ax4.set_title('Current Pattern Weights')
        ax4.grid(True, alpha=0.3, axis='y')
    
    from matplotlib.animation import FuncAnimation
    
    anim = FuncAnimation(fig, update, frames=len(history), 
                        interval=500, repeat=True)
    
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    anim.save(save_path, writer='pillow', fps=2, dpi=120)
    
    plt.close()
    print(f"Temporal pattern animation saved to {save_path}")

# Usage
if __name__ == "__main__":
    # Create static analysis figure
    visualize_temporal_patterns()
    
    # Create evolution animation
    create_temporal_animation()