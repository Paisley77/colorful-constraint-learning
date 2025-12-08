"""
Generate pendulum trajectories with periodic constraints.
"""
import numpy as np
from typing import Tuple, Dict, Optional
from scipy.integrate import solve_ivp
import os 

class PendulumDynamics:
    """Physics-accurate inverted pendulum dynamics."""
    
    def __init__(self, 
                 cart_mass: float = 1.0,
                 pole_mass: float = 0.1,
                 pole_length: float = 0.5,
                 gravity: float = 9.81,
                 cart_friction: float = 0.1,
                 pole_friction: float = 0.05):
        self.m_c = cart_mass
        self.m_p = pole_mass
        self.l = pole_length
        self.g = gravity
        self.mu_c = cart_friction
        self.mu_p = pole_friction
    
    def equations_of_motion(self, t: float, state: np.ndarray, F: float) -> np.ndarray:
        """
        Lagrangian equations of motion for cart-pole system.
        
        Parameters
        ----------
        t : float
            Time (unused, for ODE interface).
        state : np.ndarray
            [x, ẋ, θ, θ̇]
        F : float
            Applied force.
            
        Returns
        -------
        np.ndarray
            State derivatives [ẋ, ẍ, θ̇, θ̈].
        """
        x, x_dot, theta, theta_dot = state
        
        # Intermediate terms
        sin_theta = np.sin(theta)
        cos_theta = np.cos(theta)
        total_mass = self.m_c + self.m_p
        
        # ẍ (cart acceleration)
        numerator_x = (F - self.mu_c * np.sign(x_dot) 
                      - self.m_p * self.l * theta_dot**2 * sin_theta
                      + self.m_p * self.g * sin_theta * cos_theta
                      - (self.mu_p * theta_dot * cos_theta) / self.l)
        denominator_x = self.m_c + self.m_p * sin_theta**2
        x_ddot = numerator_x / denominator_x
        
        # θ̈ (pole angular acceleration)
        numerator_theta = (self.g * total_mass * sin_theta
                          - (total_mass * self.mu_p * theta_dot) / (self.m_p * self.l)
                          + cos_theta * (F - self.mu_c * np.sign(x_dot))
                          - self.m_p * self.l * theta_dot**2 * sin_theta * cos_theta)
        denominator_theta = self.l * (self.m_c + self.m_p * sin_theta**2)
        theta_ddot = numerator_theta / denominator_theta
        
        return np.array([x_dot, x_ddot, theta_dot, theta_ddot])
    
    def simulate(self, 
                 initial_state: np.ndarray,
                 controller,
                 duration: float = 4.0,
                 dt: float = 0.02) -> Tuple[np.ndarray, np.ndarray]:
        """
        Simulate pendulum with given controller.
        
        Parameters
        ----------
        initial_state : np.ndarray
            Initial state [x, ẋ, θ, θ̇].
        controller : callable
            Function taking (state, t) and returning force F.
        duration : float
            Simulation duration in seconds.
        dt : float
            Time step.
            
        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            (states, times)
        """
        n_steps = int(duration / dt)
        times = np.linspace(0, duration, n_steps)
        states = np.zeros((n_steps, 4))
        states[0] = initial_state
        
        for i in range(1, n_steps):
            t = times[i-1]
            state = states[i-1]
            
            # Get control force
            F = controller(state, t)
            
            # Solve one step of ODE
            derivs = self.equations_of_motion(t, state, F)
            states[i] = state + derivs * dt
            
            # Add small noise
            states[i] += np.random.normal(0, 0.01, 4)
        
        return states, times


class PendulumTrajectoryGenerator:
    """Generate expert and violator pendulum trajectories."""
    
    def __init__(self, dynamics: Optional[PendulumDynamics] = None):
        self.dynamics = dynamics or PendulumDynamics()
        
    def expert_controller(self, state: np.ndarray, t: float) -> float:
        """
        Expert controller: LQR-like that maintains periodic swinging
        while respecting angle constraint |θ| < 0.35 rad (~20°).
        
        Parameters
        ----------
        state : np.ndarray
            [x, ẋ, θ, θ̇]
        t : float
            Time.
            
        Returns
        -------
        float
            Control force.
        """
        x, x_dot, theta, theta_dot = state
        
        # Normalize angle
        theta = ((theta + np.pi) % (2 * np.pi)) - np.pi
        
        # Target: swing with period ~2π seconds
        target_theta = 0.2 * np.sin(2 * np.pi * t / 2.0)
        
        # PD control on angle
        Kp, Kd = 10.0, 2.0
        F_theta = Kp * (target_theta - theta) + Kd * (-theta_dot)
        
        # Keep cart near center
        F_x = -1.0 * x - 0.5 * x_dot
        
        # Combined force (clipped)
        F = F_theta + F_x
        F = np.clip(F, -10.0, 10.0)
        
        return F
    
    def violator_controller(self, state: np.ndarray, t: float) -> float:
        """
        Violator controller: Random or destabilizing control.
        
        Parameters
        ----------
        state : np.ndarray
            [x, ẋ, θ, θ̇]
        t : float
            Time.
            
        Returns
        -------
        float
            Control force.
        """
        # Option 1: Random control
        if np.random.random() < 0.7:
            return np.random.uniform(-15.0, 15.0)
        
        # Option 2: Destabilizing push when pole is upright
        x, x_dot, theta, theta_dot = state
        theta_norm = ((theta + np.pi) % (2 * np.pi)) - np.pi
        
        if abs(theta_norm) < 0.1:  # Pole nearly upright
            return 20.0 * np.sign(theta_dot)  # Push in direction of motion
        
        return 0.0
    
    def generate_expert_trajectory(self, 
                                  n_trajectories: int = 10,
                                  duration: float = 4.0) -> Dict:
        """
        Generate expert trajectories with periodic constraints.
        
        Parameters
        ----------
        n_trajectories : int
            Number of trajectories.
        duration : float
            Duration of each trajectory.
            
        Returns
        -------
        Dict
            Dictionary with trajectories and metadata.
        """
        trajectories = []
        metadata = []
        
        for i in range(n_trajectories):
            # Random initial state near stable swinging
            initial_state = np.array([
                np.random.uniform(-0.5, 0.5),      # x
                np.random.uniform(-0.5, 0.5),      # ẋ
                np.random.uniform(-0.3, 0.3),      # θ
                np.random.uniform(-1.0, 1.0)       # θ̇
            ])
            
            states, times = self.dynamics.simulate(
                initial_state=initial_state,
                controller=self.expert_controller,
                duration=duration,
                dt=0.02
            )
            
            trajectories.append(states)
            
            # Calculate trajectory metadata
            theta_traj = states[:, 2]
            theta_norm = ((theta_traj + np.pi) % (2 * np.pi)) - np.pi
            
            meta = {
                'id': f'expert_{i}',
                'initial_state': initial_state,
                'max_angle': np.max(np.abs(theta_norm)),
                'periodic_score': self._calculate_periodicity(states),
                'is_safe': np.all(np.abs(theta_norm) <= 0.35)  # |θ| ≤ 20°
            }
            metadata.append(meta)
        
        return {
            'trajectories': trajectories,
            'times': times,
            'metadata': metadata,
            'type': 'expert'
        }
    
    def generate_violator_trajectory(self,
                                    n_trajectories: int = 10,
                                    duration: float = 4.0) -> Dict:
        """
        Generate violator trajectories that break constraints.
        
        Parameters
        ----------
        n_trajectories : int
            Number of trajectories.
        duration : float
            Duration of each trajectory.
            
        Returns
        -------
        Dict
            Dictionary with trajectories and metadata.
        """
        trajectories = []
        metadata = []
        
        for i in range(n_trajectories):
            # Start from similar initial conditions as experts
            initial_state = np.array([
                np.random.uniform(-0.5, 0.5),      # x
                np.random.uniform(-0.5, 0.5),      # ẋ
                np.random.uniform(-0.3, 0.3),      # θ
                np.random.uniform(-1.0, 1.0)       # θ̇
            ])
            
            states, times = self.dynamics.simulate(
                initial_state=initial_state,
                controller=self.violator_controller,
                duration=duration,
                dt=0.02
            )
            
            trajectories.append(states)
            
            # Calculate trajectory metadata
            theta_traj = states[:, 2]
            theta_norm = ((theta_traj + np.pi) % (2 * np.pi)) - np.pi
            
            meta = {
                'id': f'violator_{i}',
                'initial_state': initial_state,
                'max_angle': np.max(np.abs(theta_norm)),
                'periodic_score': self._calculate_periodicity(states),
                'is_safe': np.all(np.abs(theta_norm) <= 0.35)
            }
            metadata.append(meta)
        
        return {
            'trajectories': trajectories,
            'times': times,
            'metadata': metadata,
            'type': 'violator'
        }
    
    def _calculate_periodicity(self, states: np.ndarray) -> float:
        """
        Calculate how periodic the trajectory is.
        
        Parameters
        ----------
        states : np.ndarray
            State trajectory.
            
        Returns
        -------
        float
            Periodicity score (0=non-periodic, 1=perfectly periodic).
        """
        # Use pole angle for periodicity detection
        theta = states[:, 2]
        
        # Auto-correlation of detrended signal
        theta_detrended = theta - np.mean(theta)
        autocorr = np.correlate(theta_detrended, theta_detrended, mode='full')
        autocorr = autocorr[len(autocorr)//2:]
        autocorr = autocorr / autocorr[0]  # Normalize
        
        # Find first peak after zero (period)
        # Look for first local maximum after first minimum
        if len(autocorr) > 20:
            # Find first minimum
            min_idx = np.argmin(autocorr[:20])
            # Find next maximum
            search_window = autocorr[min_idx:min_idx+50]
            if len(search_window) > 10:
                max_idx = np.argmax(search_window[:10])
                peak_value = search_window[max_idx]
                return float(peak_value)
        
        return 0.0
    
    def visualize_trajectories(self, expert_data: Dict, violator_data: Dict):
        """
        Visualize expert vs violator trajectories.
        
        Parameters
        ----------
        expert_data, violator_data : Dict
            Data dictionaries from generate methods.
        """
        import matplotlib.pyplot as plt
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # Plot sample trajectories
        sample_expert = expert_data['trajectories'][0]
        sample_violator = violator_data['trajectories'][0]
        times = expert_data['times']
        
        # Angle comparison
        axes[0, 0].plot(times, sample_expert[:, 2], 'b-', label='Expert', linewidth=2)
        axes[0, 0].plot(times, sample_violator[:, 2], 'r-', label='Violator', alpha=0.7)
        axes[0, 0].axhline(y=0.35, color='g', linestyle='--', label='Safe limit')
        axes[0, 0].axhline(y=-0.35, color='g', linestyle='--')
        axes[0, 0].set_xlabel('Time (s)')
        axes[0, 0].set_ylabel('Pole Angle θ (rad)')
        axes[0, 0].set_title('Pole Angle: Expert vs Violator')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Phase portrait (θ vs θ̇)
        axes[0, 1].plot(sample_expert[:, 2], sample_expert[:, 3], 'b-', alpha=0.7, label='Expert')
        axes[0, 1].plot(sample_violator[:, 2], sample_violator[:, 3], 'r-', alpha=0.7, label='Violator')
        axes[0, 1].set_xlabel('Angle θ (rad)')
        axes[0, 1].set_ylabel('Angular Velocity θ̇ (rad/s)')
        axes[0, 1].set_title('Phase Portrait')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # Cart position
        axes[1, 0].plot(times, sample_expert[:, 0], 'b-', label='Expert')
        axes[1, 0].plot(times, sample_violator[:, 0], 'r-', label='Violator')
        axes[1, 0].set_xlabel('Time (s)')
        axes[1, 0].set_ylabel('Cart Position x (m)')
        axes[1, 0].set_title('Cart Position')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # Periodicity scores histogram
        expert_scores = [m['periodic_score'] for m in expert_data['metadata']]
        violator_scores = [m['periodic_score'] for m in violator_data['metadata']]
        
        axes[1, 1].hist(expert_scores, bins=20, alpha=0.7, label='Expert', color='blue')
        axes[1, 1].hist(violator_scores, bins=20, alpha=0.7, label='Violator', color='red')
        axes[1, 1].set_xlabel('Periodicity Score')
        axes[1, 1].set_ylabel('Count')
        axes[1, 1].set_title('Distribution of Periodicity Scores')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        filename = os.path.join(os.path.dirname(__file__), '../..', 'visualizations/pendulum_trajectories.png')
        plt.savefig(filename, dpi=150, bbox_inches='tight')
        plt.show()