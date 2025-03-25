import pygame
import torch
from snake_game import SnakeGame

class SnakeInterface:
    def __init__(self, game, model=None):
        """
        Initialize the snake game interface
        Args:
            game: SnakeGame instance
            model: Optional trained model for interactive mode
        """
        self.game = game
        self.model = model
        
    def play_human(self, save_dataset=True):
        """
        Human play mode with optional dataset generation
        """
        dataset = []
        running = True
        clock = pygame.time.Clock()
        state = self.game.reset()
        current_direction = [0, 1, 0, 0]  # Start moving right
        episode = 0
        steps_in_episode = 0
        
        while running:
            action = None
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_UP and not current_direction[2]:
                        action = [1, 0, 0, 0]
                    elif event.key == pygame.K_RIGHT and not current_direction[3]:
                        action = [0, 1, 0, 0]
                    elif event.key == pygame.K_DOWN and not current_direction[0]:
                        action = [0, 0, 1, 0]
                    elif event.key == pygame.K_LEFT and not current_direction[1]:
                        action = [0, 0, 0, 1]
                    elif event.key == pygame.K_ESCAPE:
                        running = False
            
            if action is None:
                action = current_direction
            else:
                current_direction = action
            
            action_tensor = torch.FloatTensor(action)
            next_state, _ = self.game.step(action)  # Ignore game over status
            
            if save_dataset:
                dataset.append((state, action_tensor, next_state))
            
            state = next_state
            steps_in_episode += 1
            
            # Display game
            self.game.render()
            clock.tick(10)  # 10 FPS for human playability
            
        return dataset if save_dataset else None

    def play_interactive(self, max_steps=1000):
        """
        Interactive play mode where a pre-trained model generates the next state
        based on human input and feeds its own output back as input.
        """
        if self.model is None:
            raise ValueError("Model required for interactive mode")
        
        print("Interactive Snake Game - Press arrow keys to control the snake, ESC to quit")
        state = torch.zeros((32, 32))  # Start with empty board
        state[16, 16] = -1  # Place initial snake head
        state[16, 15] = 1  # Place initial food
        
        device = next(self.model.parameters()).device
        current_direction = torch.zeros(4, dtype=torch.float32)
        current_direction[1] = 1  # Start moving right
        
        step = 0
        running = True
        clock = pygame.time.Clock()
        
        # Initialize game window
        self.game.board = state.numpy()
        self.game.render()
        
        while running and step < max_steps:
            direction = None
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                    break
                    
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_UP:
                        direction = torch.zeros(4, dtype=torch.float32)
                        direction[0] = 1
                    elif event.key == pygame.K_RIGHT:
                        direction = torch.zeros(4, dtype=torch.float32)
                        direction[1] = 1
                    elif event.key == pygame.K_DOWN:
                        direction = torch.zeros(4, dtype=torch.float32)
                        direction[2] = 1
                    elif event.key == pygame.K_LEFT:
                        direction = torch.zeros(4, dtype=torch.float32)
                        direction[3] = 1
                    elif event.key == pygame.K_ESCAPE:
                        running = False
                        break
            
            if direction is None:
                direction = current_direction
            else:
                current_direction = direction
                
            # Get next state from model
            next_state = self.model.predict_next_state(state, current_direction)
            state = next_state
            
            # Update display
            self.game.board = state.numpy()
            self.game.render()
            
            step += 1
            clock.tick(10)  # 10 FPS
