"""
Basic demonstration of the mathematical spaces and exact diffeomorphism.
"""
import sys 
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '../../..'))

import numpy as np
import matplotlib.pyplot as plt
from geometry_theory.src.spaces.state_space import PendulumStateSpace
from geometry_theory.src.spaces.unit_cylinder import UnitCylinder
from geometry_theory.src.spaces.hsv_manifold import HSVManifold
from geometry_theory.src.mappings.exact_diffeomorphism import ExactDiffeomorphism
import plotly.io as pio
import plotly.graph_objects as go 

def demo_state_space():
    """Demonstrate state space functionality."""
    print("=== State Space Demo ===")
    
    # Create pendulum state space
    pendulum_space = PendulumStateSpace()
    print(f"Space: {pendulum_space.name}")
    print(f"Dimension: {pendulum_space.dimension}")
    print(f"Bounds: {pendulum_space.bounds}")
    
    # Generate random states
    random_states = pendulum_space.random_point(5)
    print(f"\nRandom states:\n{random_states}")
    
    # Check safety
    for i, state in enumerate(random_states):
        safe = pendulum_space.is_safe(state)
        print(f"State {i}: safe={safe}, θ={state[2]:.3f}")
    
    return pendulum_space

def demo_unit_cylinder():
    """Demonstrate unit cylinder functionality."""
    print("\n=== Unit Cylinder Demo ===")
    
    cylinder = UnitCylinder()
    print(f"Space: {cylinder.name}")
    print(f"Dimension: {cylinder.dimension}")
    
    # Test points
    test_points = np.array([
        [0.2, 0.5, 0.5],   # Valid
        [0.8, 0.3, 0.7],   # Valid
        [1.1, 0.5, 0.5],   # Invalid (u1 >= 1)
        [0.5, 1.2, 0.5],   # Invalid (u2 > 1)
    ])
    
    for i, point in enumerate(test_points):
        valid = cylinder.contains(point)
        print(f"Point {i}: {point} -> valid={valid}")
    
    # Test distance with circular coordinate
    p1 = np.array([0.1, 0.5, 0.5])
    p2 = np.array([0.9, 0.5, 0.5])  # Should be close due to identification
    dist = cylinder.distance(p1, p2)
    print(f"\nDistance between {p1} and {p2}: {dist:.4f}")
    print("(Should be small due to circular identification)")
    
    return cylinder

def demo_hsv_manifold():
    """Demonstrate HSV manifold functionality."""
    print("\n=== HSV Manifold Demo ===")
    
    hsv_space = HSVManifold()
    print(f"Space: {hsv_space.name}")
    print(f"Dimension: {hsv_space.dimension}")
    print(f"Radius: {hsv_space.radius}")
    print(f"Height: {hsv_space.height}")
    
    # Test points
    test_points = np.array([
        [0.0, 0.5, 0.5],        # hue = 0
        [np.pi, 0.8, 0.3],      # hue = π
        [2*np.pi - 0.1, 0.2, 0.9], # hue ≈ 2π
    ])
    
    for i, point in enumerate(test_points):
        valid = hsv_space.contains(point)
        cartesian = hsv_space.to_cartesian(point)
        print(f"\nPoint {i}: HSV={point}")
        print(f"  Valid: {valid}")
        print(f"  Cartesian: {cartesian}")
        print(f"  Color: {hsv_space.hue_to_color(point[0])}")
    
    # Test distance
    p1 = np.array([0.0, 0.5, 0.5])
    p2 = np.array([2*np.pi - 0.1, 0.5, 0.5])  # Should be close
    dist_cyl = hsv_space.distance(p1, p2, metric="cylindrical")
    dist_cart = hsv_space.distance(p1, p2, metric="cartesian")
    print(f"\nDistance between hue=0 and hue≈2π:")
    print(f"  Cylindrical: {dist_cyl:.4f}")
    print(f"  Cartesian: {dist_cart:.4f}")
    print("(Should be small due to circular topology)")
    
    return hsv_space

def demo_exact_diffeomorphism():
    """Demonstrate exact diffeomorphism Ψ: U → H."""
    print("\n=== Exact Diffeomorphism Demo ===")
    
    diffeo = ExactDiffeomorphism()
    
    # Test diffeomorphism properties
    print("Testing diffeomorphism properties...")
    test_results = diffeo.test_diffeomorphism(n_points=50)
    
    for key, value in test_results.items():
        print(f"{key}: {value:.6f}")
    
    print("\nAll errors should be near zero (machine precision).")
    
    # Visual example
    print("\nVisualizing mapping of a helix...")
    
    # Create a helix in U (simulating a periodic trajectory)
    t = np.linspace(0, 4*np.pi, 100)
    helix_u = np.column_stack([
        (t / (4*np.pi)) % 1.0,  # u1: goes from 0 to 1, wraps
        np.ones_like(t) * 0.7,   # u2: constant saturation
        0.3 + 0.4 * (t / (4*np.pi))  # u3: increasing value
    ])
    
    # Map to H
    helix_h = diffeo.forward_trajectory(helix_u)
    
    # Visualize
    fig_u, fig_h = diffeo.visualize_mapping(n_points=200)
    
    # Add helix to U visualization
    angles_u, radii_u, heights_u = helix_u.T
    x_u_helix = radii_u * np.cos(2*np.pi*angles_u)
    y_u_helix = radii_u * np.sin(2*np.pi*angles_u)
    z_u_helix = heights_u
    
    fig_u.add_trace(go.Scatter3d(
        x=x_u_helix, y=y_u_helix, z=z_u_helix,
        mode='lines',
        line=dict(width=4, color='red'),
        name="Helix Trajectory"
    ))
    
    # Add helix to H visualization
    fig_h = diffeo.hsv_manifold.visualize_3d(
        trajectory=helix_h,
        show_cylinder=True,
        title="Helix Mapped to HSV Manifold"
    )
    
    # Save figures
    root_dir = os.path.join(os.path.dirname(__file__), '../..')
    u_file = os.path.join(root_dir, "visualizations/unit_cylinder_with_helix.html")
    h_file = os.path.join(root_dir, "visualizations/hsv_manifold_with_helix.html")
    pio.write_html(fig_u, u_file)
    pio.write_html(fig_h, h_file)
    
    print("\nVisualizations saved to:")
    print("  visualizations/unit_cylinder_with_helix.html")
    print("  visualizations/hsv_manifold_with_helix.html")
    
    return diffeo, helix_u, helix_h

def main():
    """Run all demos."""
    print("=" * 60)
    print("Geometric-Semantic Interface: Mathematical Foundations Demo")
    print("=" * 60)
    
    # Demo 1: State Space
    pendulum_space = demo_state_space()
    
    # Demo 2: Unit Cylinder
    cylinder = demo_unit_cylinder()
    
    # Demo 3: HSV Manifold
    hsv_space = demo_hsv_manifold()
    
    # Demo 4: Exact Diffeomorphism
    diffeo, helix_u, helix_h = demo_exact_diffeomorphism()
    
    print("\n" + "=" * 60)
    print("Demo Complete!")
    print("Next steps:")
    print("1. Open the HTML files in your browser to see 3D visualizations")
    print("2. Examine how the helix preserves its loop structure")
    print("3. Compare with PCA mapping (next demo)")
    print("=" * 60)

if __name__ == "__main__":
    main()