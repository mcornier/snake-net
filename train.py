import torch
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import os
from snake_game import SnakeGame
from visualization import SnakeVisualizer

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

def train(model, dataset, num_epochs=50, batch_size=32, learning_rate=0.001, save_dir='models'):
    """
    Train the SnakeNet model using the provided dataset.
    """
    # Create data loader
    train_loader = DataLoader(SnakeDataset(dataset), batch_size=batch_size, shuffle=True)
    
    # Custom loss function that emphasizes larger errors
    def custom_loss(output, target):
        diff = output - target
        diff = torch.clamp(diff, min=-10.0, max=10.0)
        transformed_diff = torch.sinh(torch.abs(diff))  # always >= 0
        return transformed_diff.sum() / output.size(0)  # Normalize per sample
    
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
                print(f"Epoch {epoch+1}/{num_epochs}, Batch {batch_idx}/{len(train_loader)}, Loss: {loss.item():.6f}")
        
        # Average loss for the epoch
        avg_loss = epoch_loss / len(train_loader)
        all_losses.append(avg_loss)
        
        print(f"Epoch {epoch+1}/{num_epochs} completed. Average Loss: {avg_loss:.6f}")
        
        # Step the scheduler based on average loss
        scheduler.step(avg_loss)
        
        # Save model if it's the best so far
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), os.path.join(save_dir, 'snake_model_best.pt'))
        
        # Save training curve at each epoch
        visualizer.save_training_curve(all_losses, 'training_curve.png')
        
        # Periodically save checkpoints
        if (epoch + 1) % 5 == 0:
            # Save checkpoint
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_loss,
            }, os.path.join(save_dir, f'snake_model_checkpoint_epoch_{epoch+1}.pt'))
    
    # Save final model
    torch.save(model.state_dict(), os.path.join(save_dir, 'snake_model_final.pt'))
    
    return all_losses
