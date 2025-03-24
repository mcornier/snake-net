import torch
import torch.nn as nn
import torch.nn.functional as F

class SelfAttention(nn.Module):
    def __init__(self, embed_dim):
        super(SelfAttention, self).__init__()
        self.query = nn.Linear(embed_dim, embed_dim)
        self.key = nn.Linear(embed_dim, embed_dim)
        self.value = nn.Linear(embed_dim, embed_dim)
        self.scale = torch.sqrt(torch.FloatTensor([embed_dim]))
        
    def forward(self, x):
        # x: [batch_size, sequence_length, embed_dim]
        batch_size = x.shape[0]
        
        Q = self.query(x)  # [batch_size, sequence_length, embed_dim]
        K = self.key(x)    # [batch_size, sequence_length, embed_dim]
        V = self.value(x)  # [batch_size, sequence_length, embed_dim]
        
        # Compute attention scores
        energy = torch.matmul(Q, K.permute(0, 2, 1)) / self.scale.to(x.device)
        
        # Get attention weights
        attention = F.softmax(energy, dim=-1)
        
        # Apply attention to values
        x = torch.matmul(attention, V)
        
        return x

class CNNEncoder(nn.Module):
    def __init__(self, input_channels=1, latent_dim=1024):
        super(CNNEncoder, self).__init__()
        # Input: 32x32
        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=3, stride=2, padding=1)   # 16x16
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)    # 8x8
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1)   # 4x4
        self.conv4 = nn.Conv2d(128, 256, kernel_size=2, stride=2, padding=0)  # 2x2
        
        # Calculate flattened size (2x2x256 = 1024)
        self.flattened_size = 2 * 2 * 256
        # Final linear layer to get to the latent dimension
        self.fc = nn.Linear(self.flattened_size, latent_dim)
        
    def forward(self, x):
        # x: [batch_size, channels, height, width]
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        
        # Flatten (batch_size, 128, 4, 4) -> (batch_size, 2048) puis projection vers latent_dim (1024)
        x = x.reshape(x.size(0), self.flattened_size)
        
        # Project to latent dimension
        x = self.fc(x)
        
        return x

class CNNDecoder(nn.Module):
    def __init__(self, latent_dim=1024, output_channels=1):
        super(CNNDecoder, self).__init__()
        # Start from 2x2x256 to match encoder
        self.fc = nn.Linear(latent_dim, 2 * 2 * 256)
        
        self.deconv1 = nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1)  # 4x4
        self.deconv2 = nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1)   # 8x8
        self.deconv3 = nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1)    # 16x16
        self.deconv4 = nn.ConvTranspose2d(32, output_channels, kernel_size=4, stride=2, padding=1)  # 32x32
        
    def forward(self, x):
        # Project and reshape
        x = self.fc(x)
        x = x.view(x.size(0), 256, 2, 2)  # Match encoder's final shape
        
        # Deconvolution layers
        x = F.relu(self.deconv1(x))  # 4x4
        x = F.relu(self.deconv2(x))  # 8x8
        x = F.relu(self.deconv3(x))  # 16x16
        x = self.deconv4(x)          # 32x32
        
        return x

class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, dropout=0.1):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

class SnakeNet(nn.Module):
    def __init__(self, board_size=32, latent_dim=1024, hidden_dim=512):
        super(SnakeNet, self).__init__()
        self.board_size = board_size
        
        # CNN Encoder
        self.encoder = CNNEncoder(input_channels=1, latent_dim=latent_dim)
        
        # Direction embedding
        self.direction_embedding = nn.Linear(4, 64)
        
        # Combine board encoding and direction
        self.combiner = nn.Linear(latent_dim + 64, latent_dim)
        
        # Self-attention
        self.attention = SelfAttention(latent_dim)
        
        # MLP
        self.mlp = MLP(latent_dim, hidden_dim, latent_dim)
        
        # CNN Decoder
        self.decoder = CNNDecoder(latent_dim, output_channels=1)
        
    def forward(self, board, direction):
        # board: [batch_size, 1, height, width]
        # direction: [batch_size, 4]
        
        # Encode board
        board_encoded = self.encoder(board)
        
        # Encode direction
        dir_encoded = self.direction_embedding(direction)
        
        # Combine board and direction
        combined = torch.cat([board_encoded, dir_encoded], dim=1)
        combined = self.combiner(combined)
        
        # Add batch dimension for attention (treating each sample as a "sequence" of length 1)
        combined = combined.unsqueeze(1)
        
        # Apply self-attention
        attended = self.attention(combined)
        
        # Remove the "sequence" dimension
        attended = attended.squeeze(1)
        
        # Apply MLP
        processed = self.mlp(attended)
        
        # Decode to board state
        output = self.decoder(processed)
        
        return output
    
    def predict_next_state(self, current_state, direction):
        """
        Predicts the next state given the current state and direction.
        
        Args:
            current_state: Tensor of shape [32, 32] representing the current board
            direction: Tensor of shape [4] representing the direction
            
        Returns:
            Predicted next state Tensor of shape [32, 32]
        """
        # Add batch and channel dimensions
        current_state = current_state.unsqueeze(0).unsqueeze(0)
        direction = direction.unsqueeze(0)
        
        # Forward pass
        with torch.no_grad():
            next_state = self.forward(current_state, direction)
        
        # Remove batch and channel dimensions
        next_state = next_state.squeeze(0).squeeze(0)
        
        return next_state
    
    def play_game(self, initial_state, max_steps=1000):
        """
        Auto-plays the game by recursively predicting the next state.
        
        Args:
            initial_state: Tensor of shape [64, 64] representing the initial board
            max_steps: Maximum number of steps to simulate
            
        Returns:
            List of game states
        """
        states = [initial_state]
        current_state = initial_state
        
        for _ in range(max_steps):
            # Simple policy: go in the direction where there's food
            snake_pos = (current_state == -1).nonzero()
            head_pos = snake_pos[0]  # First snake block is the head
            
            food_pos = (current_state == 1).nonzero()
            if len(food_pos) == 0:
                break  # No food found, end game
                
            food_pos = food_pos[0]
            
            # Calculate direction vector to food
            direction_vector = food_pos - head_pos
            
            # Determine direction (up, right, down, left)
            direction = torch.zeros(4)
            if torch.abs(direction_vector[0]) > torch.abs(direction_vector[1]):
                # Move horizontally
                if direction_vector[0] > 0:
                    direction[1] = 1  # right
                else:
                    direction[3] = 1  # left
            else:
                # Move vertically
                if direction_vector[1] > 0:
                    direction[2] = 1  # down
                else:
                    direction[0] = 1  # up
            
            # Predict next state
            next_state = self.predict_next_state(current_state, direction)
            states.append(next_state)
            current_state = next_state
            
            # Check for game over (no snake or no movement)
            if torch.sum(current_state == -1) == 0 or torch.equal(current_state, states[-2]):
                break
                
        return states
