import torch
import torch.nn as nn
import numpy as np

class LinearManifold(nn.Module):
    """Simple linear manifold in HSV space: w^T h + b = 0"""
    
    def __init__(self, hsv_dim=3):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(hsv_dim))
        self.bias = nn.Parameter(torch.randn(1))
        
    def forward(self, hsv_points):
        """Signed distance to manifold"""
        return torch.matmul(hsv_points, self.weight) + self.bias
    
    def project_to_manifold(self, hsv_points):
        """Project points to manifold surface"""
        distances = self.forward(hsv_points)
        return hsv_points - distances.unsqueeze(-1) * self.weight / (torch.norm(self.weight)**2 + 1e-8)


class CircularManifold(nn.Module):
    """Circular manifold in HSV space: (h-h0)^2 + (s-s0)^2 - r^2 = 0"""
    
    def __init__(self, hsv_dim=3):
        super().__init__()
        # Circle center in Hue-Saturation plane
        self.center = nn.Parameter(torch.tensor([0.5, 0.5]))  # [h_center, s_center]
        self.radius = nn.Parameter(torch.tensor([0.3]))  # circle radius
        # Value plane offset
        self.v_offset = nn.Parameter(torch.tensor([0.0]))
        
    def forward(self, hsv_points):
        """Signed distance to circular manifold"""
        h, s, v = hsv_points[:, 0], hsv_points[:, 1], hsv_points[:, 2]

        # Handle circular hue distance correctly
        hue_diff = torch.abs(h - self.center[0])
        hue_distance = torch.minimum(hue_diff, 1.0 - hue_diff) 
        
        # Distance from circle in Hue-Saturation plane
        circle_dist = torch.sqrt(hue_distance**2 + (s - self.center[1])**2) - self.radius
        
        # Combine with value component
        value_dist = torch.sqrt((v - self.v_offset)**2)
        
        # Combined distance (circle dominates, value adds small influence)
        return circle_dist + 0.1 * value_dist
    
    def get_circle_points(self, n_points=100):
        """Get points on the circle for visualization"""
        angles = torch.linspace(0, 2*np.pi, n_points)
        h = self.center[0] + self.radius * torch.cos(angles)
        s = self.center[1] + self.radius * torch.sin(angles)
        v = torch.ones_like(h) * self.v_offset
        return torch.stack([h, s, v], dim=1).detach().numpy()
    

class CylindricalManifold(nn.Module):
    """Manifold that respects the cylindrical topology of HSV space"""
    
    def __init__(self, hsv_dim=3):
        super().__init__()
        # Parameters define a "safe tube" in cylindrical coordinates
        self.hue_center = nn.Parameter(torch.tensor([0.5]))  # Center of safe hue band
        self.hue_width = nn.Parameter(torch.tensor([0.2]))   # Width of safe hue band  
        self.sat_center = nn.Parameter(torch.tensor([0.6]))  # Optimal saturation
        self.sat_tolerance = nn.Parameter(torch.tensor([0.3]))  # Saturation tolerance
        self.value_center = nn.Parameter(torch.tensor([0.5]))  # Optimal value
        self.value_tolerance = nn.Parameter(torch.tensor([0.4]))  # Value tolerance
        
    def forward(self, hsv_points):
        h, s, v = hsv_points[:, 0], hsv_points[:, 1], hsv_points[:, 2]
        
        # Circular distance for hue (respects 0-1 wrap-around)
        hue_diff = torch.abs(h - self.hue_center)
        hue_dist = torch.minimum(hue_diff, 1.0 - hue_diff) - self.hue_width
        
        # Standard distances for saturation and value
        sat_dist = torch.abs(s - self.sat_center) - self.sat_tolerance
        val_dist = torch.abs(v - self.value_center) - self.value_tolerance
        
        # Combined distance - positive means outside safe region
        return torch.maximum(hue_dist, torch.maximum(sat_dist, val_dist))
    
    def get_constrained_params(self):
        """Get the constrained parameter values"""
        return {
            'hue_center': torch.sigmoid(self.hue_center).item(),
            'hue_width': (torch.sigmoid(self.hue_width) * 0.4).item(),
            'sat_center': torch.sigmoid(self.sat_center).item(),
            'sat_tolerance': (torch.sigmoid(self.sat_tolerance) * 0.4).item(),
            'value_center': torch.sigmoid(self.value_center).item(),
            'value_tolerance': (torch.sigmoid(self.value_tolerance) * 0.4).item()
        }
    
    def get_safe_boundary(self, n_angles=100, n_height=50):
        """Get points on the safe boundary for visualization"""
        params = self.get_constrained_params()
        hue_center = params['hue_center']
        hue_width = params['hue_width']
        sat_center = params['sat_center']
        sat_tolerance = params['sat_tolerance']
        value_center = params['value_center']
        value_tolerance = params['value_tolerance']

        angles = torch.linspace(0, 2*np.pi, n_angles)
        heights = torch.linspace(0, 1, n_height)

        # Create tube surface in cylindrical coordinates
        tube_points = []
        tube_colors = []

        for angle in angles:
            for height in heights:
                # Hue varies around the circle
                h = (hue_center + hue_width * torch.cos(angle)) % 1.0
                
                # Saturation at the boundary
                s = sat_center + sat_tolerance * torch.sin(angle)
                
                # Value varies along the height
                v = value_center + value_tolerance * (height - 0.5) * 2

                # Clamp to valid HSV range
                h = torch.clamp(h, 0.0, 1.0)
                s = torch.clamp(s, 0.0, 1.0)
                v = torch.clamp(v, 0.0, 1.0)
                
                point = torch.stack([h, s, v])
                tube_points.append(point)
                
                # Color the tube by hue for visual appeal
                enhanced_s = max(s.item(), 0.6)  # Ensure minimum saturation for visibility
                enhanced_v = max(v.item(), 0.7)  # Ensure minimum brightness
                tube_colors.append([h.item(), enhanced_s, enhanced_v])  # HSV to RGB-like for plotting

        tube_points = torch.stack(tube_points).detach().numpy() # [300,3]
        tube_colors = np.array(tube_colors) # [300,3]
        
        return tube_points, tube_colors 
    
    def get_tube_wireframe(self, n_points=50):
        """Get wireframe lines for the tube visualization"""
        # Create multiple horizontal circles at different heights
        heights = [0.0, 0.5, 1.0]  # Bottom, middle, top of value range
        wireframe_points = []
        
        for height in heights:
            angles = torch.linspace(0, 2*np.pi, n_points)
            v = self.value_center + self.value_tolerance * (height - 0.5) * 2
            
            for angle in angles:
                h = (self.hue_center + self.hue_width * torch.cos(angle)) % 1.0
                s = self.sat_center + self.sat_tolerance * torch.sin(angle)
                wireframe_points.append([h.item(), s.item(), v.item()])
        
        return np.array(wireframe_points)


class StarManifold(nn.Module):
    """Star-shaped manifold for more visual interest"""
    
    def __init__(self, hsv_dim=3):
        super().__init__()
        self.center = nn.Parameter(torch.tensor([0.5, 0.5]))
        self.base_radius = nn.Parameter(torch.tensor([0.25]))
        self.amplitude = nn.Parameter(torch.tensor([0.1]))  # star point amplitude
        self.frequency = nn.Parameter(torch.tensor([5.0]))  # number of points
        self.v_offset = nn.Parameter(torch.tensor([0.0]))
        
    def forward(self, hsv_points):
        """Signed distance to star manifold"""
        h, s, v = hsv_points[:, 0], hsv_points[:, 1], hsv_points[:, 2]
        
        # Convert to polar coordinates relative to center
        dx = h - self.center[0]
        dy = s - self.center[1]
        angle = torch.atan2(dy, dx)
        radius = torch.sqrt(dx**2 + dy**2)
        
        # Star-shaped radius
        star_radius = self.base_radius + self.amplitude * torch.sin(self.frequency * angle)
        
        # Distance from star boundary
        star_dist = radius - star_radius
        
        # Combine with value
        value_dist = v - self.v_offset
        
        return star_dist + 0.1 * value_dist
    
    def get_star_points(self, n_points=200):
        """Get points on the star for visualization"""
        angles = torch.linspace(0, 2*np.pi, n_points)
        star_radius = self.base_radius + self.amplitude * torch.sin(self.frequency * angles)
        h = self.center[0] + star_radius * torch.cos(angles)
        s = self.center[1] + star_radius * torch.sin(angles)
        v = torch.ones_like(h) * self.v_offset
        return torch.stack([h, s, v], dim=1).detach().numpy()