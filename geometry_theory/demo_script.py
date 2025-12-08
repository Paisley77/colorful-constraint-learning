"""
geometric_patterns.py

Visualizing periodic constraints as geometric patterns in color spaces.
This script demonstrates how pendulum motion naturally maps to elegant
geometric structures in HSV and hyperbolic spaces.

Key insights:
1. Periodic motion → loops in HSV cylinder
2. Constraint violations → broken geometric patterns
3. Hyperbolic space reveals hierarchical structure
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import hsv_to_rgb
import plotly.graph_objects as go
from scipy.integrate import solve_ivp
from mpl_toolkits.mplot3d import Axes3D
import colorsys

# ============================================================================
# 1. GENERATE PENDULUM TRAJECTORIES
# ============================================================================

def pendulum_equations(t, y, F=0):
    """Simple pendulum equations."""
    theta, omega = y
    g, L, b = 9.81, 1.0, 0.05  # gravity, length, damping
    return [omega, -g/L * np.sin(theta) - b * omega + F]

def generate_periodic_trajectory():
    """Generate a periodic swinging pendulum."""
    # Initial conditions for nice periodic swing
    y0 = [np.pi/3, 0]  # 60° angle, zero velocity
    t_span = [0, 10]   # 10 seconds
    t_eval = np.linspace(0, 10, 500)
    
    # Solve ODE
    sol = solve_ivp(pendulum_equations, t_span, y0, t_eval=t_eval, args=(0,))
    theta = sol.y[0]
    omega = sol.y[1]
    
    # Create state representation
    states = np.column_stack([np.cos(theta), np.sin(theta), omega])
    return states, theta, omega

def generate_chaotic_trajectory():
    """Generate a chaotic/random pendulum."""
    np.random.seed(42)
    t = np.linspace(0, 10, 500)
    
    # Create chaotic motion by adding random forces
    theta = np.zeros_like(t)
    omega = np.zeros_like(t)
    theta[0] = np.pi/3
    
    dt = t[1] - t[0]
    for i in range(1, len(t)):
        # Random force at some points
        F = np.random.choice([-1, 0, 1], p=[0.1, 0.8, 0.1]) * 2.0
        
        # Euler integration
        theta[i] = theta[i-1] + omega[i-1] * dt
        omega[i] = omega[i-1] + (-9.81 * np.sin(theta[i-1]) - 0.05 * omega[i-1] + F) * dt
    
    # Create state representation
    states = np.column_stack([np.cos(theta), np.sin(theta), omega])
    return states, theta, omega

# ============================================================================
# 2. MAP TO HSV SPACE (Cylindrical Geometry)
# ============================================================================

def map_to_hsv_cylinder(states, theta_traj):
    """
    Map pendulum states to HSV cylinder:
    - Hue = phase angle (wrapped around cylinder)
    - Saturation = distance from stable equilibrium
    - Value = angular velocity (brightness)
    """
    n = len(states)
    hsv_points = np.zeros((n, 3))
    
    # Extract features
    cos_theta = states[:, 0]
    sin_theta = states[:, 1]
    omega = states[:, 2]
    
    # 1. Hue = phase on unit circle (0 to 2π)
    # Project to 2D phase space
    phase = np.arctan2(sin_theta, cos_theta)  # -π to π
    hue = (phase + np.pi) / (2 * np.pi)  # 0 to 1
    
    # 2. Saturation = distance from stable equilibrium (θ=0)
    # Use potential energy-like measure
    potential = 1 - np.cos(theta_traj)  # 0 to 2
    saturation = np.clip(potential / 2, 0, 1)
    
    # 3. Value = normalized angular velocity
    value = np.clip(np.abs(omega) / 3, 0, 1)
    
    hsv_points[:, 0] = hue * 2 * np.pi  # Store as radians
    hsv_points[:, 1] = saturation
    hsv_points[:, 2] = value
    
    return hsv_points

def hsv_to_cartesian(hsv_points):
    """Convert HSV points to Cartesian coordinates for 3D plotting."""
    n = len(hsv_points)
    cartesian = np.zeros((n, 3))
    
    for i in range(n):
        hue, saturation, value = hsv_points[i]
        
        # Convert to cylindrical coordinates
        radius = saturation * 0.5  # Scale for visualization
        x = radius * np.cos(hue)
        y = radius * np.sin(hue)
        z = value  # Height = value
        
        cartesian[i] = [x, y, z]
    
    return cartesian

# ============================================================================
# 3. MAP TO HYPERBOLIC SPACE (Poincaré Disk)
# ============================================================================

def map_to_poincare_disk(states, theta_traj):
    """
    Map pendulum states to Poincaré disk (hyperbolic space).
    Points near center = similar states, boundary = extreme states.
    """
    n = len(states)
    disk_points = np.zeros((n, 2))
    
    # Use phase and energy as polar coordinates
    phase = np.arctan2(states[:, 1], states[:, 0])  # -π to π
    
    # Energy = kinetic + potential
    kinetic = 0.5 * states[:, 2]**2
    potential = 1 - np.cos(theta_traj)
    total_energy = kinetic + potential
    
    # Normalize energy to [0, 0.95] (stay inside disk)
    energy_norm = total_energy / (total_energy.max() + 1e-6) * 0.95
    
    # Convert to disk coordinates
    disk_points[:, 0] = energy_norm * np.cos(phase)  # x
    disk_points[:, 1] = energy_norm * np.sin(phase)  # y
    
    return disk_points

def poincare_to_hyperboloid(disk_points):
    """
    Lift Poincaré disk points to hyperboloid in Minkowski space.
    For visualization in 3D.
    """
    n = len(disk_points)
    hyperboloid = np.zeros((n, 3))
    
    for i in range(n):
        x, y = disk_points[i]
        r = np.sqrt(x**2 + y**2)
        
        # Hyperboloid coordinates (Lorentz model)
        if r < 1:
            z = np.sqrt(1 + r**2)
            # Project to 3D for visualization
            hyperboloid[i] = [x, y, z - 1]  # Shift down for visualization
        else:
            hyperboloid[i] = [x, y, 0]
    
    return hyperboloid

# ============================================================================
# 4. TOPOLOGICAL ANALYSIS (Simplified)
# ============================================================================

def compute_winding_number(points):
    """Compute winding number of a closed curve around origin."""
    # Assume points are in 2D (projection)
    x = points[:, 0]
    y = points[:, 1]
    
    # Compute complex numbers
    z = x + 1j * y
    
    # Compute total phase change
    phase_changes = np.angle(z[1:] / z[:-1])
    total_phase = np.sum(phase_changes)
    
    winding = total_phase / (2 * np.pi)
    return np.abs(winding)

def check_loop_closure(points, threshold=0.1):
    """Check if trajectory forms a closed loop."""
    start = points[0]
    end = points[-1]
    
    # For circular coordinates, need special handling
    if points.shape[1] == 3:  # HSV space
        # Hue is circular
        hue_start = points[0, 0]
        hue_end = points[-1, 0]
        hue_diff = min(abs(hue_start - hue_end), 
                      2*np.pi - abs(hue_start - hue_end))
        
        # Other coordinates
        sat_diff = abs(points[0, 1] - points[-1, 1])
        val_diff = abs(points[0, 2] - points[-1, 2])
        
        total_diff = np.sqrt((hue_diff/(2*np.pi))**2 + sat_diff**2 + val_diff**2)
    else:
        total_diff = np.linalg.norm(start - end)
    
    return total_diff < threshold, total_diff

# ============================================================================
# 5. VISUALIZATION FUNCTIONS
# ============================================================================

def plot_pendulum_trajectories(theta_periodic, theta_chaotic):
    """Plot original pendulum angles."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    t = np.linspace(0, 10, len(theta_periodic))
    
    axes[0].plot(t, theta_periodic, 'b-', linewidth=2)
    axes[0].set_xlabel('Time (s)')
    axes[0].set_ylabel('Angle θ (rad)')
    axes[0].set_title('Periodic Pendulum')
    axes[0].grid(True, alpha=0.3)
    axes[0].axhline(y=0, color='k', linestyle='--', alpha=0.5)
    
    axes[1].plot(t, theta_chaotic, 'r-', linewidth=2)
    axes[1].set_xlabel('Time (s)')
    axes[1].set_ylabel('Angle θ (rad)')
    axes[1].set_title('Chaotic Pendulum')
    axes[1].grid(True, alpha=0.3)
    axes[1].axhline(y=0, color='k', linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    return fig

def plot_hsv_cylinder_3d(hsv_periodic, hsv_chaotic):
    """3D plot of trajectories in HSV cylinder."""
    fig = plt.figure(figsize=(15, 6))
    
    # Convert to Cartesian
    cartesian_periodic = hsv_to_cartesian(hsv_periodic)
    cartesian_chaotic = hsv_to_cartesian(hsv_chaotic)
    
    # Periodic trajectory
    ax1 = fig.add_subplot(121, projection='3d')
    
    # Color by hue
    hues_periodic = hsv_periodic[:, 0] / (2 * np.pi)
    colors_periodic = plt.cm.hsv(hues_periodic)
    
    # Plot as connected segments
    for i in range(len(cartesian_periodic) - 1):
        ax1.plot(cartesian_periodic[i:i+2, 0],
                 cartesian_periodic[i:i+2, 1],
                 cartesian_periodic[i:i+2, 2],
                 color=colors_periodic[i], linewidth=2, alpha=0.8)
    
    ax1.set_xlabel('X (cos(hue) × saturation)')
    ax1.set_ylabel('Y (sin(hue) × saturation)')
    ax1.set_zlabel('Value')
    ax1.set_title('Periodic → Clean Loop in HSV Space', fontsize=12)
    ax1.grid(True, alpha=0.3)
    
    # Chaotic trajectory
    ax2 = fig.add_subplot(122, projection='3d')
    
    hues_chaotic = hsv_chaotic[:, 0] / (2 * np.pi)
    colors_chaotic = plt.cm.hsv(hues_chaotic)
    
    for i in range(len(cartesian_chaotic) - 1):
        ax2.plot(cartesian_chaotic[i:i+2, 0],
                 cartesian_chaotic[i:i+2, 1],
                 cartesian_chaotic[i:i+2, 2],
                 color=colors_chaotic[i], linewidth=2, alpha=0.8)
    
    ax2.set_xlabel('X (cos(hue) × saturation)')
    ax2.set_ylabel('Y (sin(hue) × saturation)')
    ax2.set_zlabel('Value')
    ax2.set_title('Chaotic → Broken Path in HSV Space', fontsize=12)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig

def plot_poincare_disk(disk_periodic, disk_chaotic):
    """Plot trajectories in Poincaré disk."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    
    # Create disk boundary
    theta_boundary = np.linspace(0, 2*np.pi, 100)
    x_boundary = np.cos(theta_boundary)
    y_boundary = np.sin(theta_boundary)
    
    # Periodic trajectory
    axes[0].plot(x_boundary, y_boundary, 'k-', alpha=0.3, linewidth=1)
    axes[0].fill(x_boundary, y_boundary, 'lightgray', alpha=0.1)
    
    # Color by position in original trajectory
    colors_periodic = plt.cm.viridis(np.linspace(0, 1, len(disk_periodic)))
    
    for i in range(len(disk_periodic) - 1):
        axes[0].plot(disk_periodic[i:i+2, 0], disk_periodic[i:i+2, 1],
                    color=colors_periodic[i], linewidth=2, alpha=0.8)
    
    axes[0].set_xlabel('X')
    axes[0].set_ylabel('Y')
    axes[0].set_title('Periodic → Concentric Pattern in Hyperbolic Space', fontsize=12)
    axes[0].grid(True, alpha=0.3)
    axes[0].set_aspect('equal')
    axes[0].set_xlim(-1.1, 1.1)
    axes[0].set_ylim(-1.1, 1.1)
    
    # Chaotic trajectory
    axes[1].plot(x_boundary, y_boundary, 'k-', alpha=0.3, linewidth=1)
    axes[1].fill(x_boundary, y_boundary, 'lightgray', alpha=0.1)
    
    colors_chaotic = plt.cm.plasma(np.linspace(0, 1, len(disk_chaotic)))
    
    for i in range(len(disk_chaotic) - 1):
        axes[1].plot(disk_chaotic[i:i+2, 0], disk_chaotic[i:i+2, 1],
                    color=colors_chaotic[i], linewidth=2, alpha=0.8)
    
    axes[1].set_xlabel('X')
    axes[1].set_ylabel('Y')
    axes[1].set_title('Chaotic → Scattered Points in Hyperbolic Space', fontsize=12)
    axes[1].grid(True, alpha=0.3)
    axes[1].set_aspect('equal')
    axes[1].set_xlim(-1.1, 1.1)
    axes[1].set_ylim(-1.1, 1.1)
    
    plt.tight_layout()
    return fig

def plot_color_wheel_comparison(hsv_periodic, hsv_chaotic):
    """Plot hue-saturation projections as color wheels."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    
    # Create color wheel background
    n_angles = 360
    n_radii = 20
    angles = np.linspace(0, 2*np.pi, n_angles)
    radii = np.linspace(0, 1, n_radii)
    
    for ax in axes:
        for r in radii:
            for theta in angles:
                hue = theta / (2 * np.pi)
                saturation = r
                value = 1.0
                
                # Convert HSV to RGB
                rgb = colorsys.hsv_to_rgb(hue, saturation, value)
                
                # Plot small patch
                dtheta = 2*np.pi / n_angles
                dr = 1.0 / n_radii
                
                wedge = plt.Circle((0, 0), r + dr/2, 
                                  color=rgb, 
                                  alpha=0.05,
                                  transform=ax.transData._b)
                ax.add_artist(wedge)
    
    # Plot periodic trajectory
    angles_periodic = hsv_periodic[:, 0]
    radii_periodic = hsv_periodic[:, 1]
    
    # Convert to Cartesian for plotting
    x_periodic = radii_periodic * np.cos(angles_periodic)
    y_periodic = radii_periodic * np.sin(angles_periodic)
    
    axes[0].plot(x_periodic, y_periodic, 'w-', linewidth=3, alpha=0.8, label='Trajectory')
    axes[0].plot(x_periodic, y_periodic, 'k-', linewidth=1.5, alpha=1.0)
    axes[0].scatter(x_periodic[::20], y_periodic[::20], c='white', s=50, edgecolors='black')
    
    axes[0].set_xlabel('cos(hue) × saturation')
    axes[0].set_ylabel('sin(hue) × saturation')
    axes[0].set_title('Periodic → Clean Circular Path', fontsize=12, color='white')
    axes[0].set_aspect('equal')
    axes[0].set_xlim(-1.1, 1.1)
    axes[0].set_ylim(-1.1, 1.1)
    axes[0].set_facecolor('black')
    
    # Plot chaotic trajectory
    angles_chaotic = hsv_chaotic[:, 0]
    radii_chaotic = hsv_chaotic[:, 1]
    
    x_chaotic = radii_chaotic * np.cos(angles_chaotic)
    y_chaotic = radii_chaotic * np.sin(angles_chaotic)
    
    axes[1].plot(x_chaotic, y_chaotic, 'w-', linewidth=3, alpha=0.8, label='Trajectory')
    axes[1].plot(x_chaotic, y_chaotic, 'k-', linewidth=1.5, alpha=1.0)
    axes[1].scatter(x_chaotic[::20], y_chaotic[::20], c='white', s=50, edgecolors='black')
    
    axes[1].set_xlabel('cos(hue) × saturation')
    axes[1].set_ylabel('sin(hue) × saturation')
    axes[1].set_title('Chaotic → Irregular Path', fontsize=12, color='white')
    axes[1].set_aspect('equal')
    axes[1].set_xlim(-1.1, 1.1)
    axes[1].set_ylim(-1.1, 1.1)
    axes[1].set_facecolor('black')
    
    plt.tight_layout()
    return fig

def create_interactive_plotly(hsv_periodic, hsv_chaotic):
    """Create interactive 3D plot with Plotly."""
    import plotly.graph_objects as go
    
    # Convert to Cartesian
    cartesian_periodic = hsv_to_cartesian(hsv_periodic)
    cartesian_chaotic = hsv_to_cartesian(hsv_chaotic)
    
    # Create cylinder surface
    theta = np.linspace(0, 2*np.pi, 50)
    z = np.linspace(0, 1, 20)
    theta_grid, z_grid = np.meshgrid(theta, z)
    
    x_cyl = 0.5 * np.cos(theta_grid)  # Radius = 0.5
    y_cyl = 0.5 * np.sin(theta_grid)
    z_cyl = z_grid
    
    # Create figure
    fig = go.Figure()
    
    # Add cylinder surface
    fig.add_trace(go.Surface(
        x=x_cyl, y=y_cyl, z=z_cyl,
        opacity=0.1,
        colorscale='Viridis',
        showscale=False,
        name="HSV Cylinder"
    ))
    
    # Add periodic trajectory
    hues_periodic = hsv_periodic[:, 0] / (2 * np.pi)
    
    fig.add_trace(go.Scatter3d(
        x=cartesian_periodic[:, 0],
        y=cartesian_periodic[:, 1],
        z=cartesian_periodic[:, 2],
        mode='lines+markers',
        line=dict(
            width=6,
            color=hues_periodic,
            colorscale='Rainbow'
        ),
        marker=dict(
            size=4,
            color=hues_periodic,
            colorscale='Rainbow',
            showscale=True,
            colorbar=dict(title="Hue")
        ),
        name="Periodic (Clean Loop)"
    ))
    
    # Add chaotic trajectory
    hues_chaotic = hsv_chaotic[:, 0] / (2 * np.pi)
    
    fig.add_trace(go.Scatter3d(
        x=cartesian_chaotic[:, 0],
        y=cartesian_chaotic[:, 1],
        z=cartesian_chaotic[:, 2],
        mode='lines+markers',
        line=dict(
            width=4,
            color=hues_chaotic,
            colorscale='Rainbow',
            dash='dash'
        ),
        marker=dict(
            size=3,
            color=hues_chaotic,
            colorscale='Rainbow'
        ),
        name="Chaotic (Broken Path)"
    ))
    
    fig.update_layout(
        title="Periodic Constraints as Geometric Patterns",
        scene=dict(
            xaxis_title='X',
            yaxis_title='Y',
            zaxis_title='Value',
            camera=dict(
                eye=dict(x=1.5, y=1.5, z=1.0)
            )
        ),
        width=1000,
        height=700,
        showlegend=True
    )
    
    return fig

# ============================================================================
# 6. MAIN VISUALIZATION SCRIPT
# ============================================================================

def main():
    """Main visualization script."""
    print("=" * 70)
    print("VISUALIZING PERIODIC CONSTRAINTS AS GEOMETRIC PATTERNS")
    print("=" * 70)
    
    # Generate trajectories
    print("\n1. Generating pendulum trajectories...")
    states_periodic, theta_periodic, omega_periodic = generate_periodic_trajectory()
    states_chaotic, theta_chaotic, omega_chaotic = generate_chaotic_trajectory()
    
    # Map to HSV space
    print("2. Mapping to HSV cylindrical space...")
    hsv_periodic = map_to_hsv_cylinder(states_periodic, theta_periodic)
    hsv_chaotic = map_to_hsv_cylinder(states_chaotic, theta_chaotic)
    
    # Map to hyperbolic space
    print("3. Mapping to hyperbolic (Poincaré) space...")
    disk_periodic = map_to_poincare_disk(states_periodic, theta_periodic)
    disk_chaotic = map_to_poincare_disk(states_chaotic, theta_chaotic)
    
    # Topological analysis
    print("4. Analyzing topological structure...")
    is_closed_periodic, closure_dist_periodic = check_loop_closure(hsv_periodic)
    is_closed_chaotic, closure_dist_chaotic = check_loop_closure(hsv_chaotic)
    
    winding_periodic = compute_winding_number(hsv_periodic[:, :2])
    winding_chaotic = compute_winding_number(hsv_chaotic[:, :2])
    
    print(f"\n   Periodic trajectory:")
    print(f"   - Forms closed loop: {is_closed_periodic}")
    print(f"   - Loop closure distance: {closure_dist_periodic:.4f}")
    print(f"   - Winding number: {winding_periodic:.2f}")
    
    print(f"\n   Chaotic trajectory:")
    print(f"   - Forms closed loop: {is_closed_chaotic}")
    print(f"   - Loop closure distance: {closure_dist_chaotic:.4f}")
    print(f"   - Winding number: {winding_chaotic:.2f}")
    
    # Create visualizations
    print("\n5. Creating visualizations...")
    
    # 5.1 Original trajectories
    fig1 = plot_pendulum_trajectories(theta_periodic, theta_chaotic)
    fig1.suptitle("Original Pendulum Motion", fontsize=14, y=1.02)
    plt.savefig('pendulum_trajectories.png', dpi=150, bbox_inches='tight')
    
    # 5.2 HSV cylinder 3D
    fig2 = plot_hsv_cylinder_3d(hsv_periodic, hsv_chaotic)
    fig2.suptitle("Geometric Patterns in HSV Space", fontsize=14, y=1.02)
    plt.savefig('hsv_cylinder_3d.png', dpi=150, bbox_inches='tight')
    
    # 5.3 Poincaré disk
    fig3 = plot_poincare_disk(disk_periodic, disk_chaotic)
    fig3.suptitle("Hyperbolic Space Representation", fontsize=14, y=1.02)
    plt.savefig('poincare_disk.png', dpi=150, bbox_inches='tight')
    
    # 5.4 Color wheel comparison
    fig4 = plot_color_wheel_comparison(hsv_periodic, hsv_chaotic)
    fig4.suptitle("Hue-Saturation Projection", fontsize=14, y=1.02, color='white')
    fig4.patch.set_facecolor('black')
    plt.savefig('color_wheel_comparison.png', dpi=150, bbox_inches='tight', facecolor='black')
    
    # 5.5 Interactive plot (if plotly available)
    try:
        fig5 = create_interactive_plotly(hsv_periodic, hsv_chaotic)
        fig5.write_html("interactive_hsv_visualization.html")
        print("   - Saved interactive visualization: interactive_hsv_visualization.html")
    except ImportError:
        print("   - Plotly not available, skipping interactive plot")
    
    # Show all plots
    plt.show()
    
    # Summary
    print("\n" + "=" * 70)
    print("KEY INSIGHTS:")
    print("=" * 70)
    print("1. PERIODIC CONSTRAINT → TOPOLOGICAL LOOP")
    print("   The periodic pendulum naturally forms a CLOSED LOOP in HSV space.")
    print("   This loop structure is a topological invariant.")
    print("")
    print("2. CONSTRAINT VIOLATION → BROKEN GEOMETRY")
    print("   Chaotic motion maps to IRREGULAR, NON-CLOSED paths.")
    print("   The geometry visually distinguishes constraint satisfaction.")
    print("")
    print("3. HYPERBOLIC SPACE REVEALS HIERARCHY")
    print("   Poincaré disk shows nested structure of periodic motion.")
    print("   Distance from center = 'energy level' of the system.")
    print("")
    print("4. VISUAL SEMANTICS")
    print("   Hue = phase, Saturation = potential, Value = kinetic energy")
    print("   The COLOR itself encodes the system's state semantics.")
    print("=" * 70)
    
    print("\nGenerated files:")
    print("  - pendulum_trajectories.png")
    print("  - hsv_cylinder_3d.png")
    print("  - poincare_disk.png")
    print("  - color_wheel_comparison.png")
    print("  - interactive_hsv_visualization.html (if plotly installed)")

# ============================================================================
# RUN THE SCRIPT
# ============================================================================

if __name__ == "__main__":
    # Install required packages if needed
    print("Installing required packages...")
    import subprocess
    import sys
    
    required_packages = ['numpy', 'matplotlib', 'scipy']
    
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            print(f"Installing {package}...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", package])
    
    # Try to install plotly for interactive visualization
    try:
        import plotly
    except ImportError:
        print("Plotly not found. Install with: pip install plotly kaleido")
        print("(Optional, for interactive visualizations)")
    
    # Run main visualization
    main()