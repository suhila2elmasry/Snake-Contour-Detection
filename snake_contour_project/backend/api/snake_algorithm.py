import matplotlib
matplotlib.use('Agg')  # Set non-interactive backend FIRST
import numpy as np
import cv2
import matplotlib.pyplot as plt
import base64
from io import BytesIO
from skimage.filters import sobel, gaussian
from skimage.util import img_as_float

class GreedySnake:
    def __init__(self, image, initial_contour, alpha=0.3, beta=0.5, gamma=1.5,
                 max_iterations=200, convergence_threshold=0.5):
        """
        Initialize Greedy Snake Algorithm
        
        Parameters:
        - image: Input image
        - initial_contour: Initial contour points (n, 2)
        - alpha: Elasticity parameter (0-1)
        - beta: Stiffness parameter (0-1)
        - gamma: External energy weight (0-5) - higher = stronger edge attraction
        - max_iterations: Maximum iterations
        - convergence_threshold: Convergence criteria
        """
        self.image = image
        self.gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        
        # Compute edge map using Sobel
        self.edge_map = self._compute_edge_map_sobel()
        
        # Snake parameters
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.max_iterations = max_iterations
        self.convergence_threshold = convergence_threshold
        
        # Initialize contour
        self.contour = np.array(initial_contour, dtype=np.float32)
        self.num_points = len(self.contour)
        
        # Initialize energy matrices
        self.contour_energy = []
        self.convergence_history = []
        
        print(f"Snake initialized with {self.num_points} points")
        print(f"Edge map - min: {self.edge_map.min():.3f}, max: {self.edge_map.max():.3f}")
        print(f"Parameters: alpha={alpha}, beta={beta}, gamma={gamma}")
        
    def _compute_edge_map_sobel(self):
        """Compute edge map using Sobel filter"""
        # Convert to float
        img_float = img_as_float(self.gray)
        
        # Gaussian smoothing
        img_smooth = gaussian(img_float, sigma=2.0)
        
        # Compute Sobel edge map
        edge_map = sobel(img_smooth)
        
        # Normalize to 0-1
        if edge_map.max() > edge_map.min():
            edge_map = (edge_map - edge_map.min()) / (edge_map.max() - edge_map.min() + 1e-6)
        
        # CRITICAL: Invert so edges have LOW energy (attractive)
        # Edge map has HIGH values at edges (1.0), we want LOW values (0.0) for attraction
        edge_map = 1.0 - edge_map
        
        print(f"Sobel edge map (inverted) - min: {edge_map.min():.3f}, max: {edge_map.max():.3f}")
        print(f"  -> At edges: LOW values (attractive)")
        print(f"  -> At background: HIGH values")
        
        # Save debug image
        debug_img = (edge_map * 255).astype(np.uint8)
        cv2.imwrite('debug_edge_map.png', debug_img)
        
        return edge_map
    
    def _compute_internal_energy(self, points):
        """Compute internal energy (continuity + curvature)"""
        n = len(points)
        internal_energy = np.zeros(n)
        
        # Compute continuity energy (first derivative) - distance between points
        continuity = np.zeros(n)
        for i in range(n):
            prev = points[(i-1) % n]
            curr = points[i]
            # Distance to previous point
            continuity[i] = np.linalg.norm(curr - prev)
        
        # Normalize continuity
        if continuity.max() > 0:
            continuity = continuity / continuity.max()
        
        # Compute curvature energy (second derivative)
        curvature = np.zeros(n)
        for i in range(n):
            prev = points[(i-1) % n]
            curr = points[i]
            nxt = points[(i+1) % n]
            
            # Curvature: magnitude of second derivative
            # |prev - 2*curr + next|
            curvature[i] = np.linalg.norm(prev - 2*curr + nxt)
        
        # Normalize curvature
        if curvature.max() > 0:
            curvature = curvature / curvature.max()
        
        # Combine energies
        internal_energy = self.alpha * continuity + self.beta * curvature
        
        return internal_energy
    
    def _compute_external_energy(self, point):
        """Compute external energy from edge map"""
        x, y = int(round(point[0])), int(round(point[1]))
        
        # Check boundaries
        if x < 0 or x >= self.edge_map.shape[1] or y < 0 or y >= self.edge_map.shape[0]:
            return 1.0  # High energy outside image
        
        # Return edge value (LOW at edges = attractive)
        return self.edge_map[y, x]
    
    def _find_neighborhood_minimum(self, point_idx, window_size=5):
        """Find minimum energy point in neighborhood (greedy search)"""
        center = self.contour[point_idx]
        min_energy = float('inf')
        best_point = center.copy()
        
        half_window = window_size // 2
        
        # Search in neighborhood
        for dx in range(-half_window, half_window + 1):
            for dy in range(-half_window, half_window + 1):
                candidate = center + [dx, dy]
                
                # Check if candidate is within image boundaries
                x, y = int(round(candidate[0])), int(round(candidate[1]))
                if x < 0 or x >= self.edge_map.shape[1] or y < 0 or y >= self.edge_map.shape[0]:
                    continue
                
                # Compute internal energy contribution
                temp_contour = self.contour.copy()
                temp_contour[point_idx] = candidate
                internal_energy = self._compute_internal_energy(temp_contour)[point_idx]
                
                # Compute external energy
                external_energy = self._compute_external_energy(candidate)
                
                # Total energy - gamma can be >1 to give more weight to edges
                total_energy = internal_energy + self.gamma * external_energy
                
                if total_energy < min_energy:
                    min_energy = total_energy
                    best_point = candidate.copy()
        
        return best_point, min_energy
    
    def evolve(self):
        """Evolve snake using greedy algorithm"""
        print("Starting snake evolution...")
        
        for iteration in range(self.max_iterations):
            total_movement = 0
            iteration_energy = []
            
            # Update each point (greedy search)
            for i in range(self.num_points):
                best_point, min_energy = self._find_neighborhood_minimum(i, window_size=5)
                
                # Calculate movement
                movement = np.linalg.norm(best_point - self.contour[i])
                total_movement += movement
                
                # Update point
                self.contour[i] = best_point
                iteration_energy.append(min_energy)
            
            # Store average energy
            avg_energy = np.mean(iteration_energy)
            self.contour_energy.append(avg_energy)
            
            # Check convergence
            avg_movement = total_movement / self.num_points
            self.convergence_history.append(avg_movement)
            
            # Print progress
            if iteration % 10 == 0:
                print(f"Iteration {iteration:3d}: movement={avg_movement:.4f}, energy={avg_energy:.4f}")
            
            if avg_movement < self.convergence_threshold:
                print(f"Converged after {iteration} iterations")
                break
        
        print(f"Evolution complete. Final energy: {self.contour_energy[-1]:.4f}")
        return self.contour
    
    def get_chain_code(self):
        """Generate Freeman chain code"""
        chain_code = []
        n = len(self.contour)
        
        # Direction mapping (Freeman chain code: 0-7)
        directions = {
            (0, 1): 0,   # Right
            (1, 1): 1,   # Down-Right
            (1, 0): 2,   # Down
            (1, -1): 3,  # Down-Left
            (0, -1): 4,  # Left
            (-1, -1): 5, # Up-Left
            (-1, 0): 6,  # Up
            (-1, 1): 7   # Up-Right
        }
        
        for i in range(n):
            p1 = self.contour[i]
            p2 = self.contour[(i + 1) % n]
            
            # Calculate direction
            dx = int(round(p2[0] - p1[0]))
            dy = int(round(p2[1] - p1[1]))
            
            # Find closest direction
            if dx != 0 or dy != 0:
                best_dir = None
                min_dist = float('inf')
                
                for (ddx, ddy), code in directions.items():
                    dist = np.sqrt((dx - ddx)**2 + (dy - ddy)**2)
                    if dist < min_dist:
                        min_dist = dist
                        best_dir = code
                
                if best_dir is not None:
                    chain_code.append(best_dir)
        
        return chain_code
    
    def compute_perimeter(self):
        """Compute perimeter of contour"""
        perimeter = 0
        n = len(self.contour)
        
        for i in range(n):
            p1 = self.contour[i]
            p2 = self.contour[(i + 1) % n]
            perimeter += np.linalg.norm(p2 - p1)
        
        return perimeter
    
    def compute_area(self):
        """Compute area inside contour using shoelace formula"""
        x = self.contour[:, 0]
        y = self.contour[:, 1]
        
        # Shoelace formula
        area = 0.5 * abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))
        return area
    
    def get_visualization(self):
        """Create visualization of the snake evolution"""
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # Original image with final contour
        if len(self.image.shape) == 3:
            axes[0].imshow(cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB))
        else:
            axes[0].imshow(self.image, cmap='gray')
        
        contour_closed = np.vstack([self.contour, self.contour[0]])
        axes[0].plot(contour_closed[:, 0], contour_closed[:, 1], 'r-', linewidth=2)
        axes[0].plot(self.contour[:, 0], self.contour[:, 1], 'ro', markersize=3)
        axes[0].set_title('Final Contour')
        axes[0].axis('off')
        
        # Edge map
        edge_display = (self.edge_map * 255).astype(np.uint8)
        axes[1].imshow(edge_display, cmap='gray')
        axes[1].plot(contour_closed[:, 0], contour_closed[:, 1], 'r-', linewidth=2)
        axes[1].set_title('Edge Map (dark = edges)')
        axes[1].axis('off')
        
        # Energy convergence
        if self.contour_energy:
            axes[2].plot(self.contour_energy, 'b-', linewidth=2)
            axes[2].set_xlabel('Iteration')
            axes[2].set_ylabel('Average Energy')
            axes[2].set_title('Energy Convergence')
            axes[2].grid(True)
        
        plt.tight_layout()
        
        # Convert plot to base64
        buffer = BytesIO()
        plt.savefig(buffer, format='png', dpi=100, bbox_inches='tight')
        buffer.seek(0)
        image_base64 = base64.b64encode(buffer.getvalue()).decode()
        plt.close(fig)
        
        return image_base64