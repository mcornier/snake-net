import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt
from snake_game import SnakeGame
from snake_model import SnakeNet
import os
import argparse

class SnakeDataset(Dataset):
    def __init__(self, data):
        self.data = data
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        state, action, next_state = self.data[idx]
        
        # Add channel dimension
        state = state.unsqueeze(0)  
        next_state = next_state.unsqueeze(0)
        
        return state, action, next_state

def generate_dataset(num_episodes=100, max_steps=1000, mode='auto'):
    game = SnakeGame(size=32)
    dataset = game.generate_dataset(num_episodes, max_steps, mode)
    game.close()
    return dataset

def save_sample_game_states(states, filename='game_visualization.png'):
    """
    Visualize and save sample game states for monitoring training progress.
    """
    num_samples = min(len(states), 6)  # Show at most 6 frames
    indices = np.linspace(0, len(states) - 1, num_samples).astype(int)
    
    plt.figure(figsize=(15, 3))
    for i, idx in enumerate(indices):
        plt.subplot(1, num_samples, i + 1)
        plt.imshow(states[idx].numpy(), cmap='viridis')
        plt.title(f"Step {idx}")
        plt.axis('off')
    
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()

def plot_training_curve(losses, filename='training_curve.png'):
    """
    Plot and save the training loss curve.
    """
    plt.figure(figsize=(10, 5))
    plt.plot(losses)
    plt.title('Training Loss')
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.grid(True)
    plt.savefig(filename)
    plt.close()

def train(model, dataset, num_epochs=50, batch_size=32, learning_rate=0.001, save_dir='models'):
    """
    Train the SnakeNet model using the provided dataset.
    """
    # Create data loader
    train_loader = DataLoader(SnakeDataset(dataset), batch_size=batch_size, shuffle=True)
    
    # Custom loss function that emphasizes larger errors
    def custom_loss(output, target):
        # Calculate absolute difference and clamp to prevent NaN
        epsilon = 1e-6  # small constant to prevent division by zero
        diff = torch.abs(output - target)
        diff = torch.clamp(diff, min=epsilon, max=10.0)  # limit max error to prevent explosion
        
        # Apply custom transformation element-wise:
        # - Pour diff < 1 : diff^0.7 pour amplifier (moins agressif que sqrt)
        # - Pour diff >= 1 : diff pour ne pas réduire
        small_errors = diff < 1
        transformed_diff = torch.zeros_like(diff)
        transformed_diff[small_errors] = diff[small_errors] ** 0.7  # less aggressive amplification
        transformed_diff[~small_errors] = diff[~small_errors]
        
        # Average over all elements
        return transformed_diff.mean()
    
    # Loss function and optimizer
    criterion = custom_loss
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # Create directory for saved models if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)
    
    # Training loop
    all_losses = []
    best_loss = float('inf')
    
    for epoch in range(num_epochs):
        epoch_loss = 0
        
        for batch_idx, (state, action, target) in enumerate(train_loader):
            # Move data to device
            device = next(model.parameters()).device
            state = state.to(device)
            action = action.to(device)
            target = target.to(device)
            
            # Forward pass
            output = model(state, action)
            loss = criterion(output, target)
            
            # Backward pass and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Track loss
            epoch_loss += loss.item()
            
            if batch_idx % 10 == 0:
                print(f"Epoch {epoch+1}/{num_epochs}, Batch {batch_idx}/{len(train_loader)}, Loss: {loss.item():.6f}")
        
        # Average loss for the epoch
        avg_loss = epoch_loss / len(train_loader)
        all_losses.append(avg_loss)
        
        print(f"Epoch {epoch+1}/{num_epochs} completed. Average Loss: {avg_loss:.6f}")
        
        # Save model if it's the best so far
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), os.path.join(save_dir, 'snake_model_best.pt'))
        
        # Periodically save checkpoints
        if (epoch + 1) % 5 == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_loss,
            }, os.path.join(save_dir, f'snake_model_checkpoint_epoch_{epoch+1}.pt'))
    
    # Save final model
    torch.save(model.state_dict(), os.path.join(save_dir, 'snake_model_final.pt'))
    
    # Plot training curve
    plot_training_curve(all_losses)
    
    return all_losses

def save_grid_image(grid, filename):
    """Save a grid as a grayscale image with exact pixel size"""
    # Convert to numpy array and scale to 0-255 range
    # -1 (snake) -> 0 (black)
    # 0 (empty) -> 128 (gray)
    # 1 (food) -> 255 (white)
    # intermediate values for trail are scaled accordingly
    img = ((grid.numpy() + 1) * 127.5).astype(np.uint8)
    
    # Save as PNG with exact 32x32 pixel size (no interpolation)
    plt.figure(figsize=(1, 1), dpi=32)  # Force exact 32x32 pixels
    plt.imshow(img, cmap='gray', interpolation='none')
    plt.axis('off')
    plt.margins(0, 0)
    plt.savefig(filename, dpi=32, bbox_inches='tight', pad_inches=0)
    plt.close()

def evaluate_model(model, num_games=5, max_steps=1000):
    """
    Evaluate the trained model by having it play complete games and visualizing the results.
    Additionally generate test cases for specific scenarios.
    """
    game = SnakeGame(size=32)
    device = next(model.parameters()).device
    
    # Create test directory if it doesn't exist
    os.makedirs('test_results', exist_ok=True)

    print("\nGenerating test cases...")
    
    print("Test Case 1: Empty grid with food -> Place snake")
    empty_grid = torch.zeros((32, 32), device=device)
    empty_grid[16, 16] = 1  # Place food in center
    prediction = model.predict_next_state(empty_grid, torch.tensor([0, 1, 0, 0], device=device))  # Moving right
    save_grid_image(empty_grid, 'test_results/test1_input.png')
    save_grid_image(prediction, 'test_results/test1_output.png')
    print("Test 1 saved")
    
    print("\nTest Case 2: Snake + input -> Next state")
    initial_state = game.reset()
    prediction = model.predict_next_state(initial_state, torch.tensor([0, 1, 0, 0], device=device))  # Moving right
    save_grid_image(initial_state, 'test_results/test2_input.png')
    save_grid_image(prediction, 'test_results/test2_output.png')
    print("Test 2 saved")
    
    print("\nTest Case 3: Long snake -> Next state")
    game.length = 3  # Set snake length to 3
    for _ in range(3):  # Move right to grow snake
        state, _ = game.step([0, 1, 0, 0])
    prediction = model.predict_next_state(state, torch.tensor([0, 1, 0, 0], device=device))  # Moving right
    save_grid_image(state, 'test_results/test3_input.png')
    save_grid_image(prediction, 'test_results/test3_output.png')
    print("Test 3 saved")
    print("\nAll test cases saved in test_results/")
    
    # Regular game evaluation
    for game_idx in range(num_games):
        initial_state = game.reset()
        game_states = model.play_game(initial_state, max_steps)
        save_sample_game_states(game_states, f'game_{game_idx+1}_visualization.png')
        print(f"Game {game_idx+1}: Completed {len(game_states)} steps")
    
    game.close()

def interactive_play(model, max_steps=1000):
    """
    Interactive play mode where a pre-trained model generates the next state 
    based on human input.
    """
    game = SnakeGame(size=32)
    initial_state = game.reset()
    current_state = initial_state
    
    print("Interactive Snake Game - Press arrow keys to control the snake, ESC to quit")
    game.render()
    
    # Initialize direction
    current_direction = torch.zeros(4)
    current_direction[1] = 1  # Start moving right
    
    step = 0
    running = True
    clock = pygame.time.Clock()
    
    while running and step < max_steps:
        # Handle input
        direction = None
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
                break
                
            if event.type == pygame.KEYDOWN:
                # Gestion des touches directionnelles avec vérification des directions opposées
                if event.key == pygame.K_UP:
                    # Ne pas aller vers le haut si on va vers le bas
                    if not (current_direction[2] == 1):
                        direction = torch.zeros(4)
                        direction[0] = 1
                elif event.key == pygame.K_RIGHT:
                    # Ne pas aller à droite si on va à gauche
                    if not (current_direction[3] == 1):
                        direction = torch.zeros(4)
                        direction[1] = 1
                elif event.key == pygame.K_DOWN:
                    # Ne pas aller vers le bas si on va vers le haut
                    if not (current_direction[0] == 1):
                        direction = torch.zeros(4)
                        direction[2] = 1
                elif event.key == pygame.K_LEFT:
                    # Ne pas aller à gauche si on va à droite
                    if not (current_direction[1] == 1):
                        direction = torch.zeros(4)
                        direction[3] = 1
                elif event.key == pygame.K_ESCAPE:
                    running = False
                    break
        
        # If no new direction, continue in current direction
        if direction is None:
            direction = current_direction
        else:
            current_direction = direction
        
        # Use model to predict next state
        next_state = model.predict_next_state(current_state, direction)
        
        # Update the game state
        current_state = next_state
        
        # Render the predicted state
        game.board = current_state.numpy()
        game.render()
        
        step += 1
        
        # Check for game over conditions
        if torch.sum(current_state == -1) == 0:
            print("Game over! Snake disappeared.")
            running = False
            break
        
        # Control game speed
        clock.tick(10)  # 10 FPS
    
    game.close()

def main():
    parser = argparse.ArgumentParser(description='Train and evaluate SnakeNet')
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'evaluate', 'interactive'],
                       help='Operation mode: train, evaluate, or interactive')
    parser.add_argument('--episodes', type=int, default=100, help='Number of episodes for dataset generation')
    parser.add_argument('--steps', type=int, default=500, help='Maximum steps per episode')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=50, help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--model_path', type=str, default='models/snake_model_best.pt',
                       help='Path to saved model for evaluation or interactive play')
    
    args = parser.parse_args()
    
    # Create the model (board_size 32 => latent_dim 1024, hidden_dim 512)
    model = SnakeNet(board_size=32, latent_dim=1024, hidden_dim=512)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    
    if args.mode == 'train':
        print(f"Generating dataset with {args.episodes} episodes...")
        dataset = generate_dataset(num_episodes=args.episodes, max_steps=args.steps)
        print(f"Dataset size: {len(dataset)} samples")
        
        print(f"Training model for {args.epochs} epochs...")
        train(model, dataset, num_epochs=args.epochs, batch_size=args.batch_size, learning_rate=args.lr)
        
        print("Training complete. Evaluating model...")
        evaluate_model(model)
        
    elif args.mode == 'evaluate':
        print(f"Loading model from {args.model_path}...")
        model.load_state_dict(torch.load(args.model_path, map_location=device))
        model.eval()
        
        print("Evaluating model...")
        evaluate_model(model)
        
    elif args.mode == 'interactive':
        print(f"Loading model from {args.model_path}...")
        model.load_state_dict(torch.load(args.model_path, map_location=device))
        model.eval()
        
        print("Starting interactive play...")
        interactive_play(model)

if __name__ == "__main__":
    main()
