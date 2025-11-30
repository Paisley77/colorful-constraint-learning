import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import hsv_to_rgb
from pathlib import Path
import sys
sys.path.append('../..')


def _plot_standard_hsv_view(ax1, data, frame):
    expert_hsv = data['expert_hsv']
    violator_hsv = data['violator_hsv']
    
    # Color points by distance to manifold
    expert_colors = plt.cm.RdBu_r(0.5 + 0.5 * np.tanh(data['expert_dists'] * 2))
    violator_colors = plt.cm.RdBu_r(0.5 + 0.5 * np.tanh(data['violator_dists'] * 2))
    
    # Plot expert points
    scatter1 = ax1.scatter(expert_hsv[:, 0], expert_hsv[:, 1], expert_hsv[:, 2],
                            c=expert_colors, alpha=0.7, s=15, 
                            edgecolors='darkblue', linewidths=0.2, label='Expert',
                            depthshade=False)
    
    # Plot violator points  
    scatter2 = ax1.scatter(violator_hsv[:, 0], violator_hsv[:, 1], violator_hsv[:, 2],
                            c=violator_colors, alpha=0.7, s=15,
                            edgecolors='darkred', linewidths=0.2, marker='^', label='Violator',
                            depthshade=False)
    
    # Plot tube manifold if available
    if 'tube_points' in data and 'tube_colors' in data:
        tube_pts = data['tube_points']
        tube_colors = data['tube_colors']
        
        # Reshape for surface plot
        n_height = len(np.unique(tube_pts[:, 2]))
        n_angles = n_angles = int(len(tube_pts) / n_height)
        
        if n_angles * n_height == len(tube_pts):
            # Create surface plot of the tube
            H = tube_pts[:, 0].reshape(n_height, n_angles)
            S = tube_pts[:, 1].reshape(n_height, n_angles) 
            V = tube_pts[:, 2].reshape(n_height, n_angles)

            C = hsv_to_rgb(tube_colors).reshape(n_height, n_angles, 3) 

            # Close cylinder surface
            H_closed = np.column_stack([H, H[:, 0]])
            S_closed = np.column_stack([S, S[:, 0]]) 
            V_closed = np.column_stack([V, V[:, 0]])
            C_closed = np.concatenate([C, C[:, :1, :]], axis=1)
            
            surface = ax1.plot_surface(H_closed, S_closed, V_closed, facecolors=C_closed,
                                        alpha=0.3, rstride=1, cstride=1,
                                        linewidth=0.3, edgecolor='white', shade=False, antialiased=True)
            # ax1.plot_wireframe(H, S, V, color='white', alpha=0.2, linewidth=0.5, rstride=5, cstride=5)
    
    # Plot wireframe for better tube visualization
    if 'wireframe_points' in data:
        wireframe = data['wireframe_points']
        # Plot horizontal circles
        n_points = len(wireframe) // 3
        for i in range(3):
            start_idx = i * n_points
            end_idx = (i + 1) * n_points
            circle_pts = wireframe[start_idx:end_idx]
            ax1.plot(circle_pts[:, 0], circle_pts[:, 1], circle_pts[:, 2],
                    color='cyan', linewidth=2, alpha=0.8)
        
        # Plot vertical lines
        for i in range(0, n_points, n_points//8):
            for j in range(2):
                idx = j * n_points + i
                if idx + n_points < len(wireframe):
                    line_pts = [wireframe[idx], wireframe[idx + n_points]]
                    line_pts = np.array(line_pts)
                    ax1.plot(line_pts[:, 0], line_pts[:, 1], line_pts[:, 2],
                            color='cyan', linewidth=1, alpha=0.6)
    
    ax1.set_xlabel('Hue\n(Semantic Theme)', color='white', fontsize=10)
    ax1.set_ylabel('Saturation\n(Confidence)', color='white', fontsize=10)
    ax1.set_zlabel('Value\n(Prominence)', color='white', fontsize=10)
    ax1.set_title(f'Cylindrical Constraint Manifold\n'
                    f'Alternation {data["alt_step"]}', 
                    color='white', fontsize=12, pad=20)
    
    # Styling
    ax1.xaxis.pane.fill = False
    ax1.yaxis.pane.fill = False
    ax1.zaxis.pane.fill = False
    ax1.grid(True, alpha=0.2, color='white')
    ax1.tick_params(colors='white')
    
    # Set consistent view
    ax1.view_init(elev=15, azim=frame * 8)
    ax1.set_xlim(0, 1)
    ax1.set_ylim(0, 1)
    ax1.set_zlim(0, 1)


def _plot_cylindrical_view(ax, data, frame):
    """Plot in cylindrical coordinates to show true distances"""
    expert_hsv = data['expert_hsv'].copy()
    violator_hsv = data['violator_hsv'].copy()
    
    # Convert to cylindrical coordinates for plotting
    expert_cyl = _hsv_to_cylindrical(expert_hsv)
    violator_cyl = _hsv_to_cylindrical(violator_hsv)
    
    # Color points by distance to manifold (using correct circular distance)
    expert_colors = plt.cm.RdBu_r(0.5 + 0.5 * np.tanh(data['expert_dists'] * 2))
    violator_colors = plt.cm.RdBu_r(0.5 + 0.5 * np.tanh(data['violator_dists'] * 2))
    
    # Plot in cylindrical coordinates
    scatter1 = ax.scatter(expert_cyl[:, 0], expert_cyl[:, 1], expert_cyl[:, 2],
                         c=expert_colors, alpha=0.7, s=15, 
                         edgecolors='darkblue', linewidths=0.2, label='Expert')
    
    scatter2 = ax.scatter(violator_cyl[:, 0], violator_cyl[:, 1], violator_cyl[:, 2],
                         c=violator_colors, alpha=0.7, s=15, 
                         edgecolors='darkred', linewidths=0.2, marker='^', label='Violator')
    
    # Plot tube manifold in cylindrical coordinates
    if 'tube_points' in data:
        tube_pts = data['tube_points'].copy()
        tube_cyl = _hsv_to_cylindrical(tube_pts)
        
        # Reshape for surface plot
        n_height = len(np.unique(tube_pts[:, 2]))
        n_angles = int(len(tube_pts) / n_height)
        
        if n_angles * n_height == len(tube_pts):
            X_cyl = tube_cyl[:, 0].reshape(n_height, n_angles)
            Y_cyl = tube_cyl[:, 1].reshape(n_height, n_angles)
            Z_cyl = tube_cyl[:, 2].reshape(n_height, n_angles)
            
            # Convert colors
            colors_rgb = hsv_to_rgb(data['tube_colors']).reshape(n_height, n_angles, 3)
            
            surface = ax.plot_surface(X_cyl, Y_cyl, Z_cyl, facecolors=colors_rgb,
                                    alpha=0.3, rstride=1, cstride=1,
                                    linewidth=0.3, edgecolor='white', shade=False)
    
    ax.set_xlabel('X (cos(hue))', color='white', fontsize=9)
    ax.set_ylabel('Y (sin(hue))', color='white', fontsize=9) 
    ax.set_zlabel('Value', color='white', fontsize=9)
    ax.set_title('Cylindrical Projection\nTrue Semantic Distances', 
                 color='white', fontsize=10, pad=15)
    
    # Styling
    ax.grid(True, alpha=0.2, color='white')
    ax.tick_params(colors='white')


def _plot_circular_projection(ax, data, frame):
    """2D circular projection showing true hue relationships"""
    expert_hsv = data['expert_hsv']
    violator_hsv = data['violator_hsv']
    
    # Convert to polar for 2D plot
    expert_angles = 2 * np.pi * expert_hsv[:, 0]
    expert_radii = expert_hsv[:, 1]
    
    violator_angles = 2 * np.pi * violator_hsv[:, 0]  
    violator_radii = violator_hsv[:, 1]
    
    # Convert to Cartesian for plotting
    expert_x = expert_radii * np.cos(expert_angles)
    expert_y = expert_radii * np.sin(expert_angles)
    
    violator_x = violator_radii * np.cos(violator_angles)
    violator_y = violator_radii * np.sin(violator_angles)
    
    # Color by distance to manifold
    expert_colors = plt.cm.RdBu_r(0.5 + 0.5 * np.tanh(data['expert_dists'] * 2))
    violator_colors = plt.cm.RdBu_r(0.5 + 0.5 * np.tanh(data['violator_dists'] * 2))
    
    # Plot
    ax.scatter(expert_x, expert_y, c=expert_colors, alpha=0.7, s=15, edgecolors='darkblue', linewidths=0.2, label='Expert')
    ax.scatter(violator_x, violator_y, c=violator_colors, alpha=0.7, s=15, marker='^', edgecolors='darkred', linewidths=0.2, label='Violator')
    
    # Plot manifold boundary
    tube_pts = data['tube_points'].copy()
    # Extract points at the same level 
    height_idx = tube_pts[:, 2] == np.max(tube_pts[:, 2])
    tube_pts_level = tube_pts[height_idx]
    tube_colors_level = data['tube_colors'][height_idx]

    # Convert to cylindrical coordinates
    tube_cyl = _hsv_to_cylindrical(tube_pts_level)
    X_cyl = tube_cyl[:, 0]
    Y_cyl = tube_cyl[:, 1]

    # Use actual HS colors with constant Value for visibility
    colors_rgb = hsv_to_rgb(tube_colors_level)
    # colors_rgb = np.array([hsv_to_rgb([h, s, 0.8]) for h, s, _ in tube_colors_level])

    ax.scatter(X_cyl, Y_cyl, c=colors_rgb)
        
    ax.set_xlabel('X (s·cos(2πh))', color='white')
    ax.set_ylabel('Y (s·sin(2πh))', color='white')
    ax.set_title('Circular Hue-Saturation Plane\n'
                 f'Alternation {data["alt_step"]}', 
                 color='white', fontsize=10)
    ax.legend(facecolor='black', labelcolor='white', framealpha=0.3, fontsize=8)
    ax.grid(True, alpha=0.2, color='white')
    ax.tick_params(colors='white')
    ax.set_aspect('equal')


def _hsv_to_cylindrical(hsv_points):
    """Convert HSV points to cylindrical coordinates for plotting"""
    h, s, v = hsv_points[:, 0], hsv_points[:, 1], hsv_points[:, 2]
    
    # Convert hue to angle, saturation to radius
    angles = 2 * np.pi * h
    x = s * np.cos(angles)
    y = s * np.sin(angles) 
    z = v  # Value stays as height
    
    return np.column_stack([x, y, z])


def create_tube_animation(history, save_path="results/plots/tube_evolution.gif"):
    """Create beautiful tube manifold animation"""
    
    fig = plt.figure(figsize=(15, 6), facecolor='black')
    
    def update(frame):
        plt.clf()
        fig.patch.set_facecolor('black')
        data = history[frame]
        
        # Panel 1: 3D HSV Space with Tube Manifold
        ax1 = fig.add_subplot(121, projection='3d')
        ax1.set_facecolor('black')
        _plot_standard_hsv_view(ax1, data, frame)
        
        # Panel 2: Crylindrical Projections
        ax2 = fig.add_subplot(122, projection='3d')  
        ax2.set_facecolor('black')
        _plot_cylindrical_view(ax2, data, frame) 
        
        
        plt.tight_layout()
    
    from matplotlib.animation import FuncAnimation
    
    anim = FuncAnimation(fig, update, frames=len(history), interval=1000, repeat=True)
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    
    anim.save(save_path, writer='pillow', fps=1, dpi=150, 
              savefig_kwargs={'facecolor': 'black'})
    
    plt.close()
    print(f"Tube manifold animation saved to {save_path}")


def create_circle_animation(history, save_path="results/plots/circular_evolution.gif"):
    """Create animation on 2D circular plane."""
    fig = plt.figure(figsize=(15, 6), facecolor='black')

    def update(frame):
        plt.clf()
        fig.patch.set_facecolor('black')
        data = history[frame]
        ax = fig.add_subplot(111)
        ax.set_facecolor('black')
        _plot_circular_projection(ax, data, frame)

    from matplotlib.animation import FuncAnimation
    
    anim = FuncAnimation(fig, update, frames=len(history), interval=1000, repeat=True)
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    
    anim.save(save_path, writer='pillow', fps=1, dpi=150, 
              savefig_kwargs={'facecolor': 'black'})
    
    plt.close()
    print(f"Projected circular manifold animation saved to {save_path}")


import matplotlib.patches as patches
from matplotlib.colors import LinearSegmentedColormap

def create_stage2_polar_figure(history, save_path="results/plots/stage2_polar_evolution.png"):
    """Create the 3-panel Stage 2 manifold evolution figure in polar coordinates"""
    
    # Get key alternation steps
    initial_data = history[0]  # Alternation 0
    mid_data = history[len(history)//2]  # Middle alternation
    final_data = history[-1]  # Final alternation
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 6), subplot_kw=dict(projection='polar'))
    
    # Create colormap for distance-based coloring
    cmap = LinearSegmentedColormap.from_list('safe_violate', ['blue', 'green', 'red'])
    
    def hsv_to_polar(hsv_points):
        """Convert HSV points to polar coordinates for plotting"""
        # Hue becomes angle (0-2π), Saturation becomes radius (0-1)
        angles = 2 * np.pi * hsv_points[:, 0]  # Convert hue to radians
        radii = hsv_points[:, 1]  # Saturation as radius
        return angles, radii
    
    def plot_polar_manifold_panel(ax, data, title):
        """Plot one panel with manifold and colored points in polar coordinates"""
        expert_hsv = data['expert_hsv']
        violator_hsv = data['violator_hsv']
        manifold_params = data['manifold_params']
        
        # Extract manifold bounds
        h_center = manifold_params['hue_center']
        h_width = manifold_params['hue_width']
        s_center = manifold_params['sat_center'] 
        s_tolerance = manifold_params['sat_tolerance']

        # Convert to polar
        expert_angles, expert_radii = hsv_to_polar(expert_hsv)
        violator_angles, violator_radii = hsv_to_polar(violator_hsv)

        scatter = ax.scatter(expert_angles, expert_radii, c='blue', alpha=0.5, s=15)
        ax.scatter(violator_angles, violator_radii, c='red', marker='x', alpha=0.5, s=15)
        
        # Draw manifold as an annular sector (circular band)
        # Create points for the manifold boundary
        n_points = 100
        manifold_angles = np.linspace(h_center - h_width, h_center + h_width, n_points) * 2 * np.pi
        
        # Inner boundary (s_center - s_tolerance)
        inner_radii = np.ones(n_points) * max(0, s_center - s_tolerance)
        # Outer boundary (s_center + s_tolerance)  
        outer_radii = np.ones(n_points) * min(1, s_center + s_tolerance)
        
        # Fill between inner and outer boundaries
        ax.fill_between(manifold_angles, inner_radii, outer_radii, 
                       color='green', alpha=0.3, label='Safe Region')
        
        # Draw boundary lines
        ax.plot(manifold_angles, inner_radii, 'g-', linewidth=2, alpha=0.8)
        ax.plot(manifold_angles, outer_radii, 'g-', linewidth=2, alpha=0.8)
        
        # Draw radial boundary lines to close the sector
        ax.plot([manifold_angles[0], manifold_angles[0]], [inner_radii[0], outer_radii[0]], 
               'g-', linewidth=2, alpha=0.8)
        ax.plot([manifold_angles[-1], manifold_angles[-1]], [inner_radii[-1], outer_radii[-1]], 
               'g-', linewidth=2, alpha=0.8)
        
        # Polar plot styling
        ax.set_theta_zero_location('E')  # 0° on the right
        ax.set_theta_direction(1)  # Counterclockwise
        ax.set_rmax(1.0)  # Maximum radius
        ax.grid(True, alpha=0.3)
        
        # Set title
        ax.set_title(title, pad=20, fontsize=12)
        
        # # Add colorbar for the final panel only
        # if title.startswith('c)'):
        #     # Create a separate axis for colorbar to avoid distortion
        #     from mpl_toolkits.axes_grid1 import make_axes_locatable
        #     divider = make_axes_locatable(ax)
        #     cax = divider.append_axes("right", size="5%", pad=0.1)
        #     plt.colorbar(scatter, cax=cax, label='Manifold Distance\n(blue=safe, red=violating)')
    
    # Plot all three panels
    plot_polar_manifold_panel(axes[0], initial_data, 'a) Initial: Poorly Fitted Manifold')
    plot_polar_manifold_panel(axes[1], mid_data, 'b) Mid-Optimization: Adaptive Refinement') 
    plot_polar_manifold_panel(axes[2], final_data, 'c) Final: Optimized Safe Region')
    
    # Add a super title
    plt.suptitle('Stage 2: Geometric Manifold Learning in Polar Coordinates\n(Hue → Angle, Saturation → Radius)', 
                 fontsize=14, y=1.05)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    # plt.show()

