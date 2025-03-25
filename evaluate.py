import os
import torch
from snake_game import SnakeGame
from visualization import SnakeVisualizer

class SnakeEvaluator:
    def __init__(self, model, visualizer=None):
        """
        Initialize the snake game evaluator
        Args:
            model: Trained SnakeNet model
            visualizer: Optional SnakeVisualizer instance
        """
        self.model = model
        self.visualizer = visualizer or SnakeVisualizer('test_results')
        os.makedirs('test_results', exist_ok=True)
        
    def generate_test_cases(self):
        """
        Generate and save test cases for specific scenarios
        """
        game = SnakeGame(size=32)
        device = next(self.model.parameters()).device
        
        print("\nGenerating test cases...")
        
        # Test Case 1: Empty grid with food -> Place snake
        print("Test Case 1: Empty grid with food -> Place snake")
        empty_grid = torch.zeros((32, 32), device=device)
        empty_grid[16, 16] = 1  # Place food in center
        prediction = self.model.predict_next_state(
            empty_grid, 
            torch.tensor([0, 1, 0, 0], dtype=torch.float32, device=device)  # Moving right
        )
        self.visualizer.save_grid_image(empty_grid, 'test1_input.png')
        self.visualizer.save_grid_image(prediction, 'test1_output.png')
        print("Test 1 saved")
        
        # Test Case 2: Snake + input -> Next state
        print("\nTest Case 2: Snake + input -> Next state")
        initial_state = game.reset()
        prediction = self.model.predict_next_state(
            initial_state,
            torch.tensor([0, 1, 0, 0], dtype=torch.float32, device=device)  # Moving right
        )
        self.visualizer.save_grid_image(initial_state, 'test2_input.png')
        self.visualizer.save_grid_image(prediction, 'test2_output.png')
        print("Test 2 saved")
        
        # Test Case 3: Long snake -> Next state
        print("\nTest Case 3: Long snake -> Next state")
        game.length = 3  # Set snake length to 3
        for _ in range(3):  # Move right to grow snake
            state, _ = game.step([0, 1, 0, 0])
        prediction = self.model.predict_next_state(
            state,
            torch.tensor([0, 1, 0, 0], dtype=torch.float32, device=device)  # Moving right
        )
        self.visualizer.save_grid_image(state, 'test3_input.png')
        self.visualizer.save_grid_image(prediction, 'test3_output.png')
        print("Test 3 saved")
        
        print("\nAll test cases saved in test_results/")
        game.close()
        
    def evaluate_model(self, num_games=5, max_steps=1000):
        """
        Evaluate the trained model by playing complete games
        """
        game = SnakeGame(size=32)
        
        for game_idx in range(num_games):
            initial_state = game.reset()
            game_states = []
            current_state = initial_state
            
            for step in range(max_steps):
                # Predict next state using model
                device = next(self.model.parameters()).device
                current_state = current_state.to(device)
                
                # Simple policy: go in direction of food
                snake_pos = (current_state == -1).nonzero()
                head_pos = snake_pos[0]  # First snake block is the head
                
                food_pos = (current_state == 1).nonzero()
                if len(food_pos) == 0:
                    break  # No food found, end game
                    
                food_pos = food_pos[0]
                
                # Calculate direction vector to food
                direction_vector = food_pos - head_pos
                
                # Determine direction (up, right, down, left)
                direction = torch.zeros(4, device=device)
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
                next_state = self.model.predict_next_state(current_state, direction)
                game_states.append(next_state)
                current_state = next_state
                
                # Debug info
                print(f"\nStep {step}:")
                print(f"Snake positions: {(current_state == -1).nonzero().tolist()}")
                print(f"Food position: {(current_state == 1).nonzero().tolist()}")
                print(f"Moving direction: {direction.tolist()}")
                
                # Save state visualization
                self.visualizer.save_grid_image(
                    current_state,
                    f'game_{game_idx+1}_step_{step}.png'
                )
                
                # Check for game over (no snake or no movement)
                game_over = False
                
                if torch.sum(current_state == -1) == 0:
                    print("Game over: Snake disappeared")
                    game_over = True
                elif len(game_states) > 1 and torch.equal(current_state, game_states[-2]):
                    print("Game over: No movement")
                    game_over = True
                
                if game_over:
                    break
            
            # Save visualization of game progression
            self.visualizer.save_sample_game_states(
                game_states,
                f'game_{game_idx+1}_visualization.png'
            )
            print(f"Game {game_idx+1}: Completed {len(game_states)} steps")
        
        game.close()
