import gymnasium as gym
from gymnasium import spaces
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Rectangle, Circle
from scipy.integrate import solve_ivp
import warnings
from typing import Tuple, Optional, Dict, Any


class InvertedPendulumEnv(gym.Env):
    """
    A physics-accurate inverted pendulum environment implementing the full nonlinear dynamics
    with support for constrained control and real-time visualization.
    """
    
    def __init__(self, 
                 config: Optional[Dict[str, Any]] = None):
        """
        Initialize the inverted pendulum environment.
        
        Args:
            config : dict, optional
                Configuration dictionary with physical parameters and simulation settings.
                If None, uses default parameters.
        """

        super().__init__()
        
        # Default configuration
        self.default_config = {
            # Physical parameters
            'cart_mass': 1.0,           # kg
            'pole_mass': 0.1,           # kg  
            'pole_length': 0.5,         # m 
            'gravity': 9.81,            # m/s^2
            'cart_friction': 0.1,       # N·s/m
            'pole_friction': 0.01,      # N·m·s/rad
            
            # Simulation parameters
            'max_episode_steps': 500,   # max episode length
            'dt': 0.02,                 # s (simulation timestep)
            'max_force': 10.0,          # N (maximum applied force)
            'max_cart_position': 2.4,   # m (track boundaries)
            'max_pole_angle': np.pi,    # rad (maximum pole angle in absolute value)
            
            # Safety parameters
            'safe_cart_position': 2.0,  # m (safety threshold for cost function)
            'safe_pole_angle': np.deg2rad(20), # rad (safety angle threshold)
            
            # Visualization
            'render_mode': 'human',     # 'human' for animation, 'rgb_array' for frames
        }
        
        # Merge provided config with defaults
        self.config = self.default_config.copy()
        if config is not None:
            self.config.update(config)

        # Define observation space 
        self.observation_space = spaces.Box(
            low=np.array([
                -self.config['max_cart_position'],
                -np.inf,
                -self.config['max_pole_angle'], 
                -np.inf
            ]),
            high=np.array([
                self.config['max_cart_position'],
                np.inf,
                self.config['max_pole_angle'],
                np.inf
            ]),
            dtype=np.float64
        )

        # Define action space 
        self.action_space = spaces.Box(
            low=-self.config['max_force'],
            high=self.config['max_force'],
            shape=(1,),
            dtype=np.float64 
        )
            
        # Initialize state
        self.state = None
        self.time = 0.0
        self.steps = 0
        self.max_steps = self.config['max_episode_steps']  # 500 steps, 10 seconds at dt=0.02
        
        # Visualization components
        self.fig = None
        self.ax = None
        self.cart_patch = None
        self.pole_line = None
        self.mass_circle = None
        self.trace_line = None
        self.trace_data = []
        
        # Reset environment
        self.reset()
        
    def _dynamics(self, t: float, y: np.ndarray, force: float) -> np.ndarray:
        """
        Compute the derivatives of the state vector using the nonlinear dynamics.
        
        Args:
            t : float
                Current time (unused, for compatibility with ODE solvers)
            y : np.ndarray
                State vector [x, x_dot, theta, theta_dot]
            force : float
                Applied force to the cart
            
        Returns:
            dy_dt : np.ndarray
                Derivative of state vector [x_dot, x_ddot, theta_dot, theta_ddot]
        """
        x, x_dot, theta, theta_dot = y
        m_c = self.config['cart_mass']
        m_p = self.config['pole_mass']
        l = self.config['pole_length']
        g = self.config['gravity']
        mu_c = self.config['cart_friction']
        mu_p = self.config['pole_friction']
        
        # Precompute trigonometric terms
        sin_theta = np.sin(theta)
        cos_theta = np.cos(theta)
        sin2_theta = sin_theta ** 2
        
        # Common denominator
        denom = m_c + m_p * sin2_theta
        
        # Cart acceleration (Equation 1 from report)
        x_ddot = (
            force 
            - mu_c * np.sign(x_dot) if x_dot != 0 else 0.0
            - m_p * l * theta_dot ** 2 * sin_theta
            + m_p * g * sin_theta * cos_theta
            - (mu_p * theta_dot * cos_theta) / l
        ) / denom
        
        # Pole angular acceleration (Equation 2 from report)
        theta_ddot = (
            g * (m_c + m_p) * sin_theta
            - ((m_c + m_p) * mu_p * theta_dot) / (m_p * l)
            + cos_theta * (force - (mu_c * np.sign(x_dot) if x_dot != 0 else 0.0))
            - m_p * l * theta_dot ** 2 * sin_theta * cos_theta
        ) / (l * denom)
        
        return np.array([x_dot, x_ddot, theta_dot, theta_ddot])
    
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """
        Take one step in the environment.
        
        Args:
            action : np.ndarray
                Force applied to the cart (will be clipped to [-max_force, max_force])
            
        Returns:
            state : np.ndarray
                New state vector [x, x_dot, theta, theta_dot]
            reward : float
                Reward for this step
            done : bool
                Whether the episode has terminated
            truncated: bool
                Whether episode should be truncated (max steps reached)
            info : dict
                Additional information (cost, constraint violation, etc.)
        """
        # Clip action
        force = np.clip(action, -self.config['max_force'], self.config['max_force'])[0]
        
        # Solve ODE for one timestep
        t_span = (0, self.config['dt'])
        solution_object = solve_ivp(
            lambda t, y: self._dynamics(t, y, force),
            t_span,
            self.state,
            t_eval=[self.config['dt']],
            method='RK45'
        )
        
        # Update state
        self.state = solution_object.y[:, 0]
        self.time += self.config['dt']
        self.steps += 1
        
        # Check termination conditions
        done = self._is_done()
        truncated = self.steps >= self.max_steps
        
        # Compute reward and cost
        reward = self._get_reward(force)
        cost = self._get_cost()
        
        # Additional info
        info = {
            'force_applied': force,
            'cost': cost,
            'time': self.time,
            'steps': self.steps,
            'timeout': truncated 
        }
        
        return self.state.copy(), reward, done, truncated, info
    
    def _get_reward(self, force: float) -> float:
        """Compute reward based on current state."""
        x, _, theta, _ = self.state
        
        # Dense reward encouraging stabilization at center
        reward = 1.0
        if theta < 0:
            theta_ccw = theta if abs(theta) < np.pi else 2*np.pi - abs(theta)
        else:
            theta_ccw = theta if abs(theta) < np.pi else abs(theta) - 2*np.pi
        reward -= (theta_ccw / self.config['max_pole_angle']) ** 2
        reward -= 0.1 * (x / self.config['max_cart_position']) ** 2
        reward -= 0.001 * (force / self.config['max_force']) ** 2
        
        return max(reward, 0)  # Ensure non-negative reward
    
    def _get_cost(self) -> float:
        """Compute cost based on boundary proximity."""
        x, _, theta, _ = self.state
        x_max = self.config['max_cart_position']
        x_safe = self.config['safe_cart_position']
        theta_max = np.pi 
        theta_safe = self.config['safe_pole_angle']
        
        # # Linear cost when beyond safe boundary
        # if abs(x) <= x_safe:
        #     position_cost = 0.0
        # else:
        #     position_cost = (abs(x) - x_safe) / (x_max - x_safe)

        # Linear cost when beyond angle limit 
        # convert from range [-2*pi, 0] to (-pi, pi]
        if theta < 0:
            theta_ccw = theta if abs(theta) < np.pi else 2*np.pi - abs(theta)
        else:
            theta_ccw = theta if abs(theta) < np.pi else abs(theta) - 2*np.pi
            
        if abs(theta_ccw) <= theta_safe:
            angle_cost = 0.0 
        else:
            angle_cost = (abs(theta_ccw) - theta_safe) / (theta_max - theta_safe)

        return angle_cost 
    
    def _is_done(self) -> bool:
        """Check if episode should terminate."""
        x, _, theta, _ = self.state
        cart_out_of_bounds = abs(x) > self.config['max_cart_position']
        
        return cart_out_of_bounds
    
    
    def reset(self, seed: int = None, options: Optional[Dict] = None) -> Tuple[np.ndarray, Dict]:
        """
        Reset the environment to initial state.
        
        Args:
            seed: seed for initialization 
            options : Dict | None 
                If not None, then should contain custom initial state [x, x_dot, theta, theta_dot]. 
                If None, uses random initialization.
            
        Returns:
            state : np.ndarray
                Initial state vector
        """
        super().reset(seed=seed)
        if options is not None and "initial_state" in options:
            self.state = options["initial_state"].copy() 
        else:
            # Random initialization near unstable equilibrium
            x0 = np.random.uniform(-1.0, 1.0)
            theta0 = np.random.uniform(-0.2, 0.2)
            self.state = np.array([x0, 0.0, theta0, 0.0])
        
        self.time = 0.0
        self.steps = 0
        self.trace_data = []

        info = {}
        
        return self.state.copy(), info 
    
    def get_state(self) -> np.ndarray:
        """Get current state vector."""
        return self.state.copy()
    
    def set_state(self, state: np.ndarray):
        """Set current state vector."""
        self.state = state.copy()
    
    def render(self, mode: Optional[str] = None):
        """
        Render the current state of the environment.
        
        Args:
            mode : str, optional
                Rendering mode ('human' for animation, 'rgb_array' for frame array)
        """
        if mode is None:
            mode = self.config['render_mode']
            
        if mode == 'human':
            self._render_human()
        elif mode == 'rgb_array':
            return self._render_rgb_array()
        else:
            raise ValueError(f"Unsupported render mode: {mode}")
    
    def _render_human(self):
        """Render environment using matplotlib animation."""
        if self.fig is None:
            self._init_render()
        
        self._update_render()
        plt.pause(0.001)
    
    def _init_render(self):
        """Initialize matplotlib rendering components."""
        self.fig, self.ax = plt.subplots(figsize=(10, 6))
        self.ax.set_xlim(-3.0, 3.0)
        self.ax.set_ylim(-0.5, 1.5)
        self.ax.set_aspect('equal')
        self.ax.grid(True, alpha=0.3)
        self.ax.set_xlabel('Position (m)')
        self.ax.set_ylabel('Height (m)')
        self.ax.set_title('Inverted Pendulum Environment')
        
        # Create cart
        cart_width, cart_height = 0.4, 0.2
        self.cart_patch = Rectangle(
            (-cart_width/2, -cart_height/2), cart_width, cart_height,
            facecolor='blue', alpha=0.7
        )
        self.ax.add_patch(self.cart_patch)
        
        # Create pole
        self.pole_line, = self.ax.plot([], [], 'brown', linewidth=3)
        
        # Create pole mass
        self.mass_circle = Circle((0, 0), 0.05, facecolor='red', alpha=0.7)
        self.ax.add_patch(self.mass_circle)
        
        # Create trace
        self.trace_line, = self.ax.plot([], [], 'g--', alpha=0.5, linewidth=1)
        
        # Add track boundaries
        self.ax.axvline(-self.config['max_cart_position'], color='r', linestyle='--', alpha=0.5, label='Track Boundary')
        self.ax.axvline(self.config['max_cart_position'], color='r', linestyle='--', alpha=0.5)
        self.ax.axvline(-self.config['safe_cart_position'], color='orange', linestyle=':', alpha=0.5, label='Safety Threshold')
        self.ax.axvline(self.config['safe_cart_position'], color='orange', linestyle=':', alpha=0.5)
        
        self.ax.legend()
    
    def _update_render(self):
        """Update rendering components with current state."""
        if self.state is None:
            return
            
        x, _, theta, _ = self.state
        l = self.config['pole_length']
        
        # Update cart position
        cart_width = 0.4
        self.cart_patch.set_xy((x - cart_width/2, -0.1))
        
        # Update pole position
        pole_x = [x, x - l * np.sin(theta)]
        pole_y = [0, l * np.cos(theta)]
        self.pole_line.set_data(pole_x, pole_y)
        
        # Update mass position
        self.mass_circle.center = (x - l * np.sin(theta), l * np.cos(theta))
        
        # Update trace
        self.trace_data.append((x - l * np.sin(theta), l * np.cos(theta)))
        if len(self.trace_data) > 100:  # Keep last 100 points
            self.trace_data.pop(0)
        
        if self.trace_data:
            trace_x, trace_y = zip(*self.trace_data)
            self.trace_line.set_data(trace_x, trace_y)
        
        # Update title with state information
        if theta < 0:
            theta_ccw = theta if abs(theta) < np.pi else 2*np.pi - abs(theta)
        else:
            theta_ccw = theta if abs(theta) < np.pi else abs(theta) - 2*np.pi
        self.ax.set_title(
            f'Inverted Pendulum | '
            f'x: {x:.2f}m, θ: {np.degrees(theta_ccw):.1f}° | '
            f'Time: {self.time:.1f}s'
        )
    
    def _render_rgb_array(self) -> np.ndarray:
        """Render environment as an RGB array for video recording."""
        if self.fig is None:
            self._init_render()
        
        self._update_render()
        
        # Convert matplotlib figure to RGB array
        self.fig.canvas.draw()

        # Get the RGBA buffer
        buf = np.frombuffer(self.fig.canvas.buffer_rgba(), dtype=np.uint8)

        # Get the actual canvas dimensions
        width, height = self.fig.canvas.get_width_height()

        # Reshape and convert RGBA to RGB
        buf = buf.reshape((height, width, 4))  # RGBA format
        rgb_buf = buf[:, :, :3]  # Extract RGB channels, discard alpha
        
        return rgb_buf
    
    def close(self):
        """Close rendering resources."""
        if self.fig is not None:
            plt.close(self.fig)
            self.fig = None
            self.ax = None


# Demonstration and testing
if __name__ == "__main__":
    print("Testing Inverted Pendulum Environment...")
    
    # Create environment
    env = InvertedPendulumEnv()
    
    # Test manual control
    print("Manual control test - applying small oscillating force")
    state, _ = env.reset(options={"initial_state": np.array([0.0, 0.0, 0.1, 0.0])})
    env.render()
    
    # Simple oscillating control for demonstration
    for i in range(500):
        force = 1.0 * np.sin(i * 0.1)  # Oscillating force
        state, reward, done, truncated, info = env.step(np.array([force]))
        
        print(f"Step {i}: x={state[0]:.2f}, θ={np.degrees(state[2]):.1f}°, "
              f"reward={reward:.3f}, cost={info['cost']:.3f}")
        
        env.render()
        
        # if done:
        #     print("Episode terminated!")
        #     break
    
    env.close()
    print("Test completed successfully!")