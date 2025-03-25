import pygame
import numpy as np
import torch

class SnakeGame:
    def __init__(self, size=32):
        self.size = size
        self.reset()
        
        # Pygame initialization for visualization
        pygame.init()
        self.screen_size = 256  # 8 pixels per cell with a 32x32 grid
        self.screen = pygame.display.set_mode((self.screen_size, self.screen_size))
        pygame.display.set_caption('Snake Game')
        
    def reset(self):
        self.board = np.zeros((self.size, self.size))
        # Initial snake position (middle of the board)
        self.snake = [(self.size // 2, self.size // 2)]
        self.direction = [1, 0]  # Start moving right
        self.board[self.snake[0]] = -1
        self.length = 1  # Track snake length
        self.place_food()
        self.game_over = False
        return self.get_state()
    
    def place_food(self):
        empty = np.where(self.board == 0)
        if len(empty[0]) > 0:
            idx = np.random.randint(len(empty[0]))
            self.board[empty[0][idx], empty[1][idx]] = 1
    
    def step(self, action):
        # action: [up, right, down, left]
        if isinstance(action, torch.Tensor):
            action = action.detach().cpu().numpy()
        
        # Update direction based on action (adjusted for correct matrix coordinates)
        if np.argmax(action) == 0:  # up
            self.direction = [-1, 0]  # Moving left in matrix coordinates
        elif np.argmax(action) == 1:  # right
            self.direction = [0, 1]  # Moving down
        elif np.argmax(action) == 2:  # down
            self.direction = [1, 0]  # Moving right
        elif np.argmax(action) == 3:  # left
            self.direction = [0, -1]  # Moving up
        
        # Move snake
        head = self.snake[0]
        new_head = (head[0] + self.direction[0], head[1] + self.direction[1])
        
        # Check for collision with walls
        if (new_head[0] < 0 or new_head[0] >= self.size or
            new_head[1] < 0 or new_head[1] >= self.size):
            self.game_over = True
            return self.get_state(), True
        
        # Check for collision with self
        if self.board[new_head] == -1:
            self.game_over = True
            return self.get_state(), True
        
        # Check if food was eaten
        ate_food = self.board[new_head] == 1
        
        # Move snake
        self.snake.insert(0, new_head)
        self.board[new_head] = -1  # Head is always -1
        
        # Update body segments
        for i, pos in enumerate(self.snake[1:], 1):
            if i < self.length:  # Si le segment fait partie du corps actuel
                self.board[pos] = -1  # Corps solide en noir
            else:  # Segments qui dépassent la longueur deviennent la trainée
                if self.board[pos] == -1:  # Only update if it's still solid
                    self.board[pos] = -0.5  # Start fade out
        
        # Update previous trail (segments au-delà de la longueur)
        mask = (self.board < 0) & (self.board > -1)  # Find all trail positions
        self.board[mask] = self.board[mask] * 0.5  # Fade out by half
        
        if ate_food:
            self.length += 1  # Increase snake length when food is eaten
            self.place_food()
        elif len(self.snake) > self.length:
            # Only remove tail if we're longer than we should be
            tail = self.snake.pop()
        
        return self.get_state(), False
    
    def get_state(self):
        return torch.FloatTensor(self.board)
    
    def render(self):
        self.screen.fill((128, 128, 128))  # Fond gris pour 0
        
        for i in range(self.size):
            for j in range(self.size):
                # Valeur de la cellule
                val = self.board[i, j]
                
                # Calcul de la couleur basé sur la valeur
                if val == -1:  # Snake - noir
                    color = (0, 0, 0)
                elif val == 1:  # Food - blanc
                    color = (255, 255, 255)
                elif val == 0:  # Empty - gris
                    continue  # Déjà rempli par le fond
                else:
                    # Niveau de gris pour les valeurs intermédiaires
                    # Map de -1 à 1 vers 0 à 255
                    gray_val = int(128 + val * 128)
                    gray_val = max(0, min(255, gray_val))  # Clamp entre 0 et 255
                    color = (gray_val, gray_val, gray_val)
                
                # Draw 8x8 pixel block
                rect = pygame.Rect(j * 8, i * 8, 8, 8)
                pygame.draw.rect(self.screen, color, rect)
        
        pygame.display.flip()

    def close(self):
        pygame.quit()

    def generate_dataset(self, num_episodes=100, max_steps=1000, mode='auto', max_samples=50000):
        dataset = []
        running = True
        episode = 0
        
        clock = pygame.time.Clock()
        
        while running and episode < num_episodes:
            # Vider la queue d'événements
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                    break
            
            state = self.reset()
            current_direction = np.array([0, 1, 0, 0])  # Start moving right
            steps = 0
            
            while steps < max_steps and running:
                # Vérifier les événements pygame
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        running = False
                        break
                if mode == 'auto':
                    # Simple automatic movement pattern (circles or back-and-forth)
                    action = np.zeros(4)
                    time_step = steps % 16
                    if time_step < 4:
                        action[1] = 1  # right
                    elif time_step < 8:
                        action[2] = 1  # down
                    elif time_step < 12:
                        action[3] = 1  # left
                    else:
                        action[0] = 1  # up
                elif mode == 'random' or mode == 'search':
                    # Get valid moves (can't go backwards)
                    possible_actions = []
                    
                    # Check which moves are valid (not opposite to current direction)
                    if current_direction[2] != 1:  # If not going down, can go up
                        possible_actions.append([1, 0, 0, 0])
                    if current_direction[3] != 1:  # If not going left, can go right
                        possible_actions.append([0, 1, 0, 0])
                    if current_direction[0] != 1:  # If not going up, can go down
                        possible_actions.append([0, 0, 1, 0])
                    if current_direction[1] != 1:  # If not going right, can go left
                        possible_actions.append([0, 0, 0, 1])
                    
                    if mode == 'search':
                        # Find food position
                        food_pos = np.where(self.board == 1)
                        head_pos = self.snake[0]
                        
                        # Calculate direction to food
                        dy = food_pos[0][0] - head_pos[0]
                        dx = food_pos[1][0] - head_pos[1]
                        
                        # Filter actions that move closer to food
                        better_actions = []
                        for action in possible_actions:
                            # Check each possible move and see if it gets us closer to food
                            if ((dy < 0 and np.array_equal(action, [1, 0, 0, 0])) or   # Food is up
                                (dy > 0 and np.array_equal(action, [0, 0, 1, 0])) or   # Food is down
                                (dx > 0 and np.array_equal(action, [0, 1, 0, 0])) or   # Food is right
                                (dx < 0 and np.array_equal(action, [0, 0, 0, 1]))):    # Food is left
                                better_actions.append(action)
                        
                        # If we have actions that move closer to food, use those
                        if better_actions:
                            action = better_actions[np.random.randint(len(better_actions))]
                        else:
                            # If no direct path, use any valid move
                            action = possible_actions[np.random.randint(len(possible_actions))]
                    else:
                        # Random mode: randomly choose from valid moves
                        if np.random.random() < 0.7 and any(np.array_equal(current_direction, a) for a in possible_actions):
                            action = current_direction
                        else:
                            action = possible_actions[np.random.randint(len(possible_actions))]
                    
                    current_direction = action
                else:
                    # Manual control for human player
                    action = np.zeros(4)
                    for event in pygame.event.get():
                        if event.type == pygame.KEYDOWN:
                            if event.key == pygame.K_UP:
                                action[0] = 1
                            elif event.key == pygame.K_RIGHT:
                                action[1] = 1
                            elif event.key == pygame.K_DOWN:
                                action[2] = 1
                            elif event.key == pygame.K_LEFT:
                                action[3] = 1
                
                next_state, done = self.step(action)
                # Limiter la taille du dataset
                if len(dataset) < max_samples:
                    dataset.append((state, torch.FloatTensor(action), next_state))
                
                state = next_state
                self.render()
                
                if done:
                    break
                
                # Control speed based on mode
                if mode == 'auto' or mode == 'random' or mode == 'search':
                    clock.tick(200)  # Fast for automatic generation (200 FPS)
                else:
                    clock.tick(10)  # Slower for human visualization (10 FPS)
                
                steps += 1
            
            episode += 1
            if len(dataset) >= max_samples:
                print(f"Dataset size limit reached ({max_samples} samples)")
                break
        
        return dataset
