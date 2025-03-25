import torch
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import os
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
