import argparse
import pygame
import torch
from snake_game import SnakeGame
from snake_model import SnakeNet
from train import train, generate_dataset, evaluate_model, interactive_play
import os

def main():
    parser = argparse.ArgumentParser(description='Snake Neural Network Simulator')
    parser.add_argument('--mode', type=str, default='play',
                        choices=['play', 'train', 'simulate', 'interactive', 'random', 'search'],
                        help='Mode to run: play (human controls), train (train model), simulate (model plays), interactive (human + model), random/search (generate datasets)')
    parser.add_argument('--size', type=int, default=64, help='Size of the game board (default: 64x64)')
    parser.add_argument('--episodes', type=int, default=100, help='Number of episodes for training (default: 100)')
    parser.add_argument('--steps', type=int, default=500, help='Maximum steps per episode (default: 500)')
    parser.add_argument('--epochs', type=int, default=50, help='Number of training epochs (default: 50)')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size for training (default: 32)')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate (default: 0.001)')
    parser.add_argument('--model-path', type=str, default='models/snake_model_best.pt',
                       help='Path to saved model for simulation or interactive mode')
    
    args = parser.parse_args()
    
    if args.mode in ['random', 'search']:
        # Random play mode with dataset generation
        game = SnakeGame(size=args.size)
        dataset = []
        
        print(f"Generating {args.mode} dataset with {args.episodes} episodes...")
        dataset = game.generate_dataset(num_episodes=args.episodes, max_steps=args.steps, mode=args.mode)
        
        # Save final dataset
        os.makedirs('datasets', exist_ok=True)
        final_path = f'datasets/{args.mode}_gameplay.pt'
        torch.save(dataset, final_path)
        print(f"Random dataset saved to {final_path} with {len(dataset)} samples")
        
        game.close()
        
    elif args.mode == 'play':
        # Human play mode with dataset generation
        game = SnakeGame(size=args.size)
        dataset = []
        
        running = True
        clock = pygame.time.Clock()
        
        state = game.reset()
        
        # Initialize direction
        current_direction = [0, 1, 0, 0]  # Start moving right
        
        # Create directory for saving datasets if it doesn't exist
        os.makedirs('datasets', exist_ok=True)
        
        episode = 0
        steps_in_episode = 0
        
        while running:
            # Handle input
            action = None
            
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    # Gestion des touches directionnelles avec vérification des directions opposées
                    if event.key == pygame.K_UP:
                        # Ne pas aller vers le haut si on va vers le bas
                        if not (current_direction[2] == 1):
                            action = [1, 0, 0, 0]
                    elif event.key == pygame.K_RIGHT:
                        # Ne pas aller à droite si on va à gauche
                        if not (current_direction[3] == 1):
                            action = [0, 1, 0, 0]
                    elif event.key == pygame.K_DOWN:
                        # Ne pas aller vers le bas si on va vers le haut
                        if not (current_direction[0] == 1):
                            action = [0, 0, 1, 0]
                    elif event.key == pygame.K_LEFT:
                        # Ne pas aller à gauche si on va à droite
                        if not (current_direction[1] == 1):
                            action = [0, 0, 0, 1]
                    elif event.key == pygame.K_ESCAPE:
                        running = False
            
            # If no new action, continue in current direction
            if action is None:
                action = current_direction
            else:
                current_direction = action
            
            # Convert action to tensor for dataset
            action_tensor = torch.FloatTensor(action)
            
            # Update game state
            next_state, done = game.step(action)
            
            # Add to dataset
            dataset.append((state, action_tensor, next_state))
            state = next_state
            
            steps_in_episode += 1
            
            if done:
                print(f"Game over! Episode {episode + 1} completed with {steps_in_episode} steps")
                # Save dataset after each episode
                torch.save(dataset, f'datasets/human_gameplay_episode_{episode + 1}.pt')
                print(f"Dataset saved with {len(dataset)} samples")
                
                # Reset for next episode
                state = game.reset()
                current_direction = [0, 1, 0, 0]  # Reset direction
                episode += 1
                steps_in_episode = 0
            
            # Render the game
            game.render()
            
            # Control game speed
            clock.tick(10)  # 10 FPS
        
        # Save final dataset
        if dataset:
            final_path = 'datasets/human_gameplay_final.pt'
            torch.save(dataset, final_path)
            print(f"Final dataset saved to {final_path} with {len(dataset)} samples")
        
        game.close()
        
    elif args.mode == 'train':
        print("Training mode activated")
        
        # Create model
        model = SnakeNet(board_size=args.size, latent_dim=4096, hidden_dim=2048)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model.to(device)
        
        print(f"Generating dataset with {args.episodes} episodes...")
        dataset = generate_dataset(num_episodes=args.episodes, max_steps=args.steps)
        print(f"Dataset size: {len(dataset)} samples")
        
        print(f"Training model for {args.epochs} epochs...")
        train(model, dataset, num_epochs=args.epochs, batch_size=args.batch_size, learning_rate=args.lr)
        
        print("Training complete. Evaluating model...")
        evaluate_model(model)
        
    elif args.mode == 'simulate':
        print("Simulation mode activated")
        
        # Create and load model
        model = SnakeNet(board_size=args.size, latent_dim=4096, hidden_dim=2048)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model.to(device)
        
        # Check if model file exists
        if not os.path.exists(args.model_path):
            print(f"Error: Model file {args.model_path} not found.")
            print("Please train a model first or specify a valid model path.")
            return
        
        print(f"Loading model from {args.model_path}...")
        model.load_state_dict(torch.load(args.model_path, map_location=device))
        model.eval()
        
        # Run simulation
        evaluate_model(model, num_games=5, max_steps=args.steps)
        
    elif args.mode == 'interactive':
        print("Interactive mode activated")
        
        # Create and load model
        model = SnakeNet(board_size=args.size, latent_dim=4096, hidden_dim=2048)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model.to(device)
        
        # Check if model file exists
        if not os.path.exists(args.model_path):
            print(f"Error: Model file {args.model_path} not found.")
            print("Please train a model first or specify a valid model path.")
            return
        
        print(f"Loading model from {args.model_path}...")
        model.load_state_dict(torch.load(args.model_path, map_location=device))
        model.eval()
        
        # Start interactive play
        interactive_play(model, max_steps=args.steps)

if __name__ == "__main__":
    main()
