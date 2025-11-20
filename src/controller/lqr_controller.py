import sys
import os
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))
sys.path.insert(0, project_root)

from src.environments.cartpole import InvertedPendulumEnv

import numpy as np
from scipy.linalg import solve_continuous_are
import matplotlib.pyplot as plt
from typing import Tuple, Dict, Any
import warnings

class LQRController:
    """
    Linear Quadratic Regulator for the inverted pendulum system.
    
    Implements the LQR solution for the linearized cart-pole system around the 
    unstable equilibrium point (upright position).
    """
    
    def __init__(self, env_config: Dict[str, Any], Q: np.ndarray = None, R: float = None):
        """
        Initialize LQR controller.
        
        Args:
            env_config : dict
                Environment configuration parameters (mass, length, friction, etc.)
            Q : np.ndarray, optional
                State cost matrix (4x4). If None, uses default values.
            R : float, optional  
                Control cost scalar. If None, uses default value.
        """
        self.env_config = env_config
        self.state_dim = 4
        self.control_dim = 1
        
        # Set default cost matrices if not provided
        if Q is None:
            # Default: prioritize angle stabilization, then position, then velocities
            self.Q = np.diag([1.0, 0.1, 10.0, 0.1])  # [x, x_dot, theta, theta_dot]
        else:
            self.Q = Q
            
        if R is None:
            self.R = np.array([[0.01]])  # Control effort cost
        else:
            self.R = np.array([[R]])
        
        # Compute linearized system matrices and optimal gain
        self.A, self.B = self._linearize_system()
        self.K = self._compute_optimal_gain()
    
    def _linearize_system(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Linearize the nonlinear dynamics around the unstable equilibrium.
        
        Returns:
            A : np.ndarray
                State matrix (4x4) of linearized system
            B : np.ndarray  
                Control matrix (4x1) of linearized system
        """
        # Extract parameters
        m_c = self.env_config['cart_mass']
        m_p = self.env_config['pole_mass'] 
        l = self.env_config['pole_length']
        g = self.env_config['gravity']
        mu_c = self.env_config['cart_friction']
        mu_p = self.env_config['pole_friction']
        
        # Linearize around equilibrium: [x=0, x_dot=0, theta=0, theta_dot=0, F=0]
        # Using small angle approximations: sin(theta) ≈ theta, cos(theta) ≈ 1
        
        # State matrix A (4x4)
        A = np.zeros((4, 4))
        
        # dx/dt = x_dot
        A[0, 1] = 1.0
        
        # d(x_dot)/dt = f2(x, x_dot, theta, theta_dot)
        # From Equation (1) linearization:
        A[1, 1] = -mu_c / m_c                    # Friction for cart velocity
        A[1, 2] = (m_p * g) / m_c                # Gravity effect on angle
        A[1, 3] = -mu_p / (m_c * l)              # Pole friction effect
        
        # d(theta)/dt = theta_dot  
        A[2, 3] = 1.0
        
        # d(theta_dot)/dt = f4(x, x_dot, theta, theta_dot)
        # From Equation (2) linearization:
        A[3, 1] = -mu_c / (l * m_c)              # Cart friction effect on angular acceleration
        A[3, 2] = (g * (m_c + m_p)) / (l * m_c)  # Gravity torque
        A[3, 3] = -((m_c + m_p) * mu_p) / (m_p * l**2 * m_c)  # Pole friction
        
        # Control matrix B (4x1)
        B = np.zeros((4, 1))
        B[1, 0] = 1.0 / m_c                      # Force effect on cart acceleration
        B[3, 0] = 1.0 / (l * m_c)                # Force effect on angular acceleration
        
        return A, B
    
    def _compute_optimal_gain(self) -> np.ndarray:
        """
        Compute the optimal LQR gain matrix by solving the continuous-time ARE.
        
        Returns:
            K : np.ndarray
                Optimal feedback gain matrix (1x4)
        """
        try:
            # Solve the continuous-time algebraic Riccati equation
            P = solve_continuous_are(self.A, self.B, self.Q, self.R)
            
            # Compute optimal gain: K = R^{-1} B^T P
            K = np.linalg.inv(self.R) @ self.B.T @ P
            
            # Ensure gain is properly shaped (1x4)
            K = K.reshape(1, -1)
            
            return K
            
        except Exception as e:
            warnings.warn(f"ARE solution failed: {e}. Using pole placement fallback.")
            return self._pole_placement_fallback()
    
    def _pole_placement_fallback(self) -> np.ndarray:
        """
        Fallback method using pole placement if ARE solution fails.
        
        Returns:
            K : np.ndarray
                Stabilizing feedback gain matrix (1x4)
        """
        # Simple heuristic gains that should stabilize the system
        # These are tuned manually and may need adjustment
        K_heuristic = np.array([[-1.0, -2.0, -20.0, -5.0]])  # [x, x_dot, theta, theta_dot]
        
        print("Using pole placement fallback with heuristic gains")
        return K_heuristic
    
    def compute_control(self, state: np.ndarray) -> float:
        """
        Compute LQR control action for given state.
        
        Args:
            state : np.ndarray
                Current state vector [x, x_dot, theta, theta_dot]
            
        Returns:
            force : float
                Optimal control force to apply
        """
        # LQR control law: u = -Kx
        force = -self.K @ state
        
        # Clip to environment force limits
        max_force = self.env_config['max_force']
        force_clipped = np.clip(force[0], -max_force, max_force)
        
        return force_clipped
    
    def analyze_performance(self) -> Dict[str, Any]:
        """
        Analyze LQR controller performance characteristics.
        
        Returns:
            analysis : dict
                Performance analysis results
        """
        # Closed-loop system matrix
        A_cl = self.A - self.B @ self.K
        
        # Eigenvalues indicate stability
        eigenvalues = np.linalg.eigvals(A_cl)
        
        analysis = {
            'open_loop_eigenvalues': np.linalg.eigvals(self.A),
            'closed_loop_eigenvalues': eigenvalues,
            'gain_matrix': self.K.flatten(),
            'is_stable': np.all(np.real(eigenvalues) < 0),
            'settling_time_estimate': self._estimate_settling_time(eigenvalues),
            'control_authority': np.linalg.norm(self.B @ self.K)
        }
        
        return analysis
    
    def _estimate_settling_time(self, eigenvalues: np.ndarray) -> float:
        """
        Estimate settling time from closed-loop eigenvalues.
        
        Args:
            eigenvalues : np.ndarray
                Eigenvalues of closed-loop system
            
        Returns:
            settling_time : float
                Estimated 2% settling time in seconds
        """
        # Find dominant pole (slowest real part)
        real_parts = np.real(eigenvalues)
        if len(real_parts) == 0 or np.all(real_parts >= 0):
            return float('inf')
        
        # Settling time to 2%: t_s ≈ 4 / |Re(dominant_pole)|
        dominant_pole = np.min(np.abs(real_parts[real_parts < 0]))
        settling_time = 4.0 / dominant_pole if dominant_pole > 0 else float('inf')
        
        return settling_time


def run_lqr_demonstration():
    """
    Demonstrate LQR controller performance on the inverted pendulum.
    """
    print("=== LQR Controller Demonstration ===")
    
    # Use the loaded config for both environment and LQR
    env = InvertedPendulumEnv()
    lqr = LQRController(env.config)
    
    # Performance analysis
    analysis = lqr.analyze_performance()
    print("\n--- LQR Performance Analysis ---")
    for key, value in analysis.items():
        if key != 'gain_matrix':
            print(f"{key}: {value}")
    
    # Test trajectories
    test_initial_conditions = [
        np.array([0.0, 0.0, np.deg2rad(30), 0.0]),    # stable
        np.array([1.0, 1.5, np.deg2rad(180), 5.5])
    ]
    
    results = []
    
    for i, initial_state in enumerate(test_initial_conditions):
        print(f"\n--- Test {i+1}: Initial state {initial_state} ---")
        
        state, _ = env.reset(options={"initial_state": initial_state})
        trajectory = [state.copy()]
        forces = []
        rewards = []
        
        env.render()
        
        for step in range(300):  # 6 seconds at dt=0.02
            # Compute LQR control
            force = lqr.compute_control(state)
            force = np.array([force])
            
            # Step environment
            state, reward, done, truncated, info = env.step(force)
            
            # Store data
            trajectory.append(state.copy())
            forces.append(force)
            rewards.append(reward)
            
            env.render()
            
            # if done:
            #     print(f"  Terminated at step {step}: {info}")
            #     break
        
        results.append({
            'trajectory': np.array(trajectory),
            'forces': np.array(forces),
            'rewards': np.array(rewards),
            'steps': len(trajectory) - 1
        })
        
        # Plot individual test results
        # plot_trajectory_results(results[-1], initial_state, i+1)
    
    env.close()
    
    # Comparative analysis
    print("\n=== Comparative Performance ===")
    for i, result in enumerate(results):
        total_reward = np.sum(result['rewards'])
        max_force = np.max(np.abs(result['forces']))
        print(f"Test {i+1}: {result['steps']} steps, "
              f"Total reward: {total_reward:.3f}, Max force: {max_force:.2f}N")
    
    return results, lqr


if __name__ == "__main__":
    # Run demonstration
    demonstration_results, lqr_controller = run_lqr_demonstration()
