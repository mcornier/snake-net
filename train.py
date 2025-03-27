import torch
import torch.optim as optim 
from torch.utils.data import Dataset, DataLoader
import os
import pygame
import numpy as np
from snake_game import SnakeGame
from visualization import SnakeVisualizer

class SnakeDataset(Dataset):
    def __init__(self, data, device=None):
        self.device = device
        # Move data to specified device if provided
        if device is not None:
            self.data = [(state.to(device), action.to(device), next_state.to(device)) 
                        for state, action, next_state in data]
        else:
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

def train(model, dataset, num_epochs=50, batch_size=32, learning_rate=0.001, save_dir='models', device=None):
    """
    Train the SnakeNet model using the provided dataset.
    
    Args:
        model: The SnakeNet model to train
        dataset: Training dataset
        num_epochs: Number of epochs to train for
        batch_size: Batch size for training
        learning_rate: Learning rate for optimizer
        save_dir: Directory to save model checkpoints
        device: Device to train on (cpu or cuda). If None, uses model's current device.
    """
    if device is None:
        device = next(model.parameters()).device
    # Create data loader with data on correct device
    train_dataset = SnakeDataset(dataset, device=device)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    # Custom loss function that emphasizes larger errors - ensure it's on correct device
    def custom_loss(output, target):
        diff = output - target
        # Add epsilon to avoid division by zero
        epsilon = 1e-6
        # Use smooth L1 loss for stability
        abs_diff = torch.abs(diff)
        quadratic = 0.5 * diff ** 2
        linear = abs_diff - 0.5
        
        # Apply custom scaling for small errors (using stable operations)
        small_errors = abs_diff < 1
        loss = torch.where(small_errors, 
                          quadratic * (torch.clamp(abs_diff + epsilon, min=epsilon) ** -0.3),
                          linear)
        
        return loss.mean()
    
    # Initialize visualizer for training monitoring
    visualizer = SnakeVisualizer()
    
    # Loss function and optimizer
    criterion = custom_loss
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # Learning rate scheduler that reduces LR on plateau
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3, verbose=True)
    
    # Create directory for saved models if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)
    
    # Training loop
    all_losses = []
    best_loss = float('inf')
    
    for epoch in range(num_epochs):
        epoch_loss = 0
        epoch_states = []  # Store some states for visualization
        
        for batch_idx, (state, action, target) in enumerate(train_loader):
            # Data is already on device from dataset initialization
            # Just ensure model is on correct device
            model = model.to(device)
            
            # Forward pass
            output = model(state, action)
            loss = criterion(output, target)
            
            # Backward pass and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Track loss
            epoch_loss += loss.item()
            
            # Store some states for visualization
            if batch_idx % 50 == 0:  # Save every 50th batch
                # Save input state, prediction, and target for visualization
                visualizer.save_grid_image(
                    state[0, 0].cpu(),
                    f'epoch_{epoch+1}_batch_{batch_idx}_input.png'
                )
                visualizer.save_grid_image(
                    output[0, 0].detach().cpu(),
                    f'epoch_{epoch+1}_batch_{batch_idx}_prediction.png'
                )
                visualizer.save_grid_image(
                    target[0, 0].cpu(),
                    f'epoch_{epoch+1}_batch_{batch_idx}_target.png'
                )
            
            if batch_idx % 10 == 0:
                current_lr = optimizer.param_groups[0]['lr']
                print(f"Epoch {epoch+1}/{num_epochs}, Batch {batch_idx}/{len(train_loader)}, Loss: {loss.item():.6f}, LR: {current_lr:.6f}")
        
        # Average loss for the epoch
        avg_loss = epoch_loss / len(train_loader)
        all_losses.append(avg_loss)
        
        current_lr = optimizer.param_groups[0]['lr']
        print(f"Epoch {epoch+1}/{num_epochs} completed. Average Loss: {avg_loss:.6f}, LR: {current_lr:.6f}")
        
        # Step the scheduler based on average loss
        scheduler.step(avg_loss)
        
        # Save model if it's the best so far
        if avg_loss < best_loss:
            best_loss = avg_loss
            # Save model state
            model_state = model.state_dict()
            torch.save(model_state, os.path.join(save_dir, 'snake_model_best.pt'))
        
        # Save training curve at each epoch
        visualizer.save_training_curve(all_losses, 'training_curve.png')
        
        # Periodically save checkpoints
        if (epoch + 1) % 5 == 0:
            # Save checkpoint
            # Save checkpoint with device-agnostic state
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_loss,
                'device': str(device)
            }
            torch.save(checkpoint, os.path.join(save_dir, f'snake_model_checkpoint_epoch_{epoch+1}.pt'))
    
    # Save final model
    # Save final model state
    torch.save(model.state_dict(), os.path.join(save_dir, 'snake_model_final.pt'))
    
    return all_losses

def train_rl(model, num_episodes=1000, max_steps=1000, learning_rate=0.00005, save_dir='models', device=None):
    """
    Train the SnakeNet model using reinforcement learning by comparing with actual game states.
    
    Args:
        model: The SnakeNet model to train
        num_episodes: Number of episodes to train for
        max_steps: Maximum steps per episode
        learning_rate: Learning rate for optimizer
        save_dir: Directory to save model checkpoints
        device: Device to train on (cpu or cuda)
    """
    if device is None:
        device = next(model.parameters()).device
    
    # Initialize two pygame windows
    pygame.init()
    screen_size = 256
    # Position windows side by side
    os.environ['SDL_VIDEO_WINDOW_POS'] = '50,50'
    game_screen = pygame.display.set_mode((screen_size, screen_size))
    pygame.display.set_caption('Snake Game')
    
    os.environ['SDL_VIDEO_WINDOW_POS'] = f'{50 + screen_size + 10},50'
    model_screen = pygame.display.set_mode((screen_size, screen_size))
    pygame.display.set_caption('Model Prediction')
    
    # Initialize game and optimizer
    game = SnakeGame(size=32)
    game.screen = game_screen  # Use the first window for the game
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # Custom loss function
    def custom_loss(output, target):
        diff = output - target
        epsilon = 1e-6
        abs_diff = torch.abs(diff)
        quadratic = 0.5 * diff ** 2
        linear = abs_diff - 0.5
        small_errors = abs_diff < 1
        loss = torch.where(small_errors, 
                          quadratic * (torch.clamp(abs_diff + epsilon, min=epsilon) ** -0.3),
                          linear)
        return loss.mean()
    
    # Training loop
    all_losses = []
    best_loss = float('inf')
    
    for episode in range(num_episodes):
        state = game.reset()
        episode_loss = 0
        steps = 0
        
        while steps < max_steps:
            # Handle pygame events
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    return all_losses
            
            # Generate action using search mode
            action = np.zeros(4)
            food_pos = np.where(game.board == 1)
            head_pos = game.snake[0]
            
            # Calculate direction to food
            dy = food_pos[0][0] - head_pos[0]
            dx = food_pos[1][0] - head_pos[1]
            
            # Choose action that moves towards food
            if abs(dx) > abs(dy):
                if dx > 0:
                    action[1] = 1  # right
                else:
                    action[3] = 1  # left
            else:
                if dy > 0:
                    action[2] = 1  # down
                else:
                    action[0] = 1  # up
            
            # Get next state from game
            next_state, done = game.step(action)
            game.render()  # Render game state
            
            # Get model prediction
            model.eval()
            with torch.no_grad():
                state_tensor = state.clone().to(device).unsqueeze(0).unsqueeze(0)
                action_tensor = torch.FloatTensor(action).to(device).unsqueeze(0)
                prediction = model(state_tensor, action_tensor)
                
                # Render model prediction
                pred_array = prediction.squeeze().cpu().numpy()
                model_screen.fill((128, 128, 128))
                
                for i in range(32):
                    for j in range(32):
                        val = pred_array[i, j]
                        if val <= -0.8:  # Snake body
                            color = (0, 0, 0)
                        elif val >= 0.8:  # Food
                            color = (255, 255, 255)
                        elif val < 0:  # Trail
                            gray_val = int(128 + val * 128)
                            gray_val = max(0, min(255, gray_val))
                            color = (gray_val, gray_val, gray_val)
                        else:
                            continue
                        
                        rect = pygame.Rect(j * 8, i * 8, 8, 8)
                        pygame.draw.rect(model_screen, color, rect)
                
                pygame.display.flip()
            
            # Train model
            model.train()
            optimizer.zero_grad()
            
            # Calculate loss comparing prediction to actual next state
            loss = custom_loss(prediction.squeeze(), next_state.to(device))
            loss.backward()
            optimizer.step()
            
            episode_loss += loss.item()
            
            if done:
                break
            
            state = next_state
            steps += 1
            
            # Control visualization speed
            pygame.time.wait(50)  # 50ms delay for visualization
        
        # Average loss for the episode
        avg_loss = episode_loss / steps
        all_losses.append(avg_loss)
        
        print(f"Episode {episode+1}/{num_episodes} completed. Average Loss: {avg_loss:.6f}")
        
        # Save model if it's the best so far
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), os.path.join(save_dir, 'snake_model_rl_best.pt'))
        
        # Save checkpoint every 50 episodes
        if (episode + 1) % 50 == 0:
            checkpoint = {
                'episode': episode,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_loss,
                'device': str(device)
            }
            torch.save(checkpoint, os.path.join(save_dir, f'snake_model_rl_checkpoint_episode_{episode+1}.pt'))
    
    # Save final model
    torch.save(model.state_dict(), os.path.join(save_dir, 'snake_model_rl_final.pt'))
    pygame.quit()
    
    return all_losses
