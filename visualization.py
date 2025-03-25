import matplotlib.pyplot as plt
import numpy as np
import os
import torch

class SnakeVisualizer:
    def __init__(self, save_dir='visualization'):
        """
        Initialize the snake game visualizer
        Args:
            save_dir: Directory to save visualizations
        """
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)

    def save_grid_image(self, grid, filename):
        """
        Save a grid as a grayscale image with exact pixel size
        Args:
            grid: Tensor or numpy array representing the game state
            filename: Name of the file to save
        """
        if isinstance(grid, torch.Tensor):
            grid = grid.detach().cpu().numpy()
            
        # Convert to numpy array and scale to 0-255 range
        # -1 (snake) -> 0 (black)
        # 0 (empty) -> 128 (gray)
        # 1 (food) -> 255 (white)
        # intermediate values for trail are scaled accordingly
        img = ((grid + 1) * 127.5).astype(np.uint8)
        
        # Save as PNG with exact 32x32 pixel size (no interpolation)
        plt.figure(figsize=(1, 1), dpi=32)  # Force exact 32x32 pixels
        plt.imshow(img, cmap='gray', interpolation='none')
        plt.axis('off')
        plt.margins(0, 0)
        plt.savefig(os.path.join(self.save_dir, filename), dpi=32, bbox_inches='tight', pad_inches=0)
        plt.close()

    def save_training_curve(self, losses, filename='training_curve.png'):
        """
        Plot and save the training loss curve
        Args:
            losses: List of loss values
            filename: Name of the file to save
        """
        plt.figure(figsize=(10, 5))
        plt.semilogy(losses)  # Use logarithmic scale for y-axis
        plt.title('Training Loss (log scale)')
        plt.xlabel('Iteration')
        plt.ylabel('Loss (log)')
        plt.grid(True, which="both", ls="-")
        plt.savefig(os.path.join(self.save_dir, filename))
        plt.close()

    def save_sample_game_states(self, states, filename='game_visualization.png'):
        """
        Visualize and save sample game states for monitoring training progress
        Args:
            states: List of game states (tensors or numpy arrays)
            filename: Name of the file to save
        """
        num_samples = min(len(states), 6)  # Show at most 6 frames
        indices = np.linspace(0, len(states) - 1, num_samples).astype(int)
        
        plt.figure(figsize=(15, 3))
        for i, idx in enumerate(indices):
            plt.subplot(1, num_samples, i + 1)
            state = states[idx]
            if isinstance(state, torch.Tensor):
                state = state.detach().cpu().numpy()
            plt.imshow(state, cmap='viridis')
            plt.title(f"Step {idx}")
            plt.axis('off')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.save_dir, filename))
        plt.close()
