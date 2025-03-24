import argparse
import pygame
import torch
from snake_game import SnakeGame
from snake_model import SnakeNet
from train import train, generate_dataset, evaluate_model, interactive_play
import os

def main():
    parser = argparse.ArgumentParser(description='Snake Neural Network Simulator')
    parser.add_argument('--mode', type=str, default='train',
                        choices=['play', 'train', 'evaluate', 'interactive', 'random', 'search'],
                        help='Mode to run: play (human), train (train model), evaluate (evaluate model), interactive (human + model), random/search (generate datasets)')
    parser.add_argument('--episodes', type=int, default=100, help='Number of episodes for training (default: 100)')
    parser.add_argument('--steps', type=int, default=500, help='Maximum steps per episode (default: 500)')
    parser.add_argument('--epochs', type=int, default=50, help='Number of training epochs (default: 50)')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size for training (default: 32)')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate (default: 0.001)')
    parser.add_argument('--model-path', type=str, default='models/snake_model_best.pt',
                       help='Path to saved model for evaluation or interactive mode')
    
    args = parser.parse_args()
    
    # Create the model with same dimensions as saved model
    model = SnakeNet(board_size=32, latent_dim=1024)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    
    if args.mode in ['random', 'search']:
        # Random/Search play mode with dataset generation
        game = SnakeGame(size=32)  # Fixed size to 32x32
        print(f"Generating {args.mode} dataset with {args.episodes} episodes...")
        dataset = game.generate_dataset(num_episodes=args.episodes, max_steps=args.steps, mode=args.mode)
        
        # Save dataset
        os.makedirs('datasets', exist_ok=True)
        final_path = f'datasets/{args.mode}_gameplay.pt'
        torch.save(dataset, final_path)
        print(f"Dataset saved to {final_path} with {len(dataset)} samples")
        game.close()
        
    elif args.mode == 'play':
        # Human play mode with dataset generation
        game = SnakeGame(size=32)  # Fixed size to 32x32
        dataset = []
        
        running = True
        clock = pygame.time.Clock()
        state = game.reset()
        current_direction = [0, 1, 0, 0]  # Start moving right
        
        os.makedirs('datasets', exist_ok=True)
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
            next_state, done = game.step(action)
            dataset.append((state, action_tensor, next_state))
            state = next_state
            steps_in_episode += 1
            
            if done:
                print(f"Episode {episode + 1} completed with {steps_in_episode} steps")
                torch.save(dataset, f'datasets/human_gameplay_episode_{episode + 1}.pt')
                state = game.reset()
                current_direction = [0, 1, 0, 0]
                episode += 1
                steps_in_episode = 0
            
            game.render()
            clock.tick(10)
        
        if dataset:
            torch.save(dataset, 'datasets/human_gameplay_final.pt')
        game.close()
        
    elif args.mode == 'train':
        print("Training mode activated")
        
        print("Loading existing datasets...")
        dataset = []
        for file in os.listdir('datasets'):
            if file.endswith('.pt'):
                print(f"Loading {file}...")
                data = torch.load(f'datasets/{file}')
                dataset.extend(data)
        print(f"Total dataset size: {len(dataset)} samples")
        
        print(f"Training model for {args.epochs} epochs...")
        train(model, dataset, num_epochs=args.epochs, batch_size=args.batch_size, learning_rate=args.lr)
        
        print("Training complete. Evaluating model...")
        evaluate_model(model)
        
    elif args.mode == 'evaluate':
        print("Evaluation mode activated")
        
        # Check if model file exists
        if not os.path.exists(args.model_path):
            print(f"Error: Model file {args.model_path} not found.")
            print("Please train a model first or specify a valid model path.")
            return
        
        print(f"Loading model from {args.model_path}...")
        model.load_state_dict(torch.load(args.model_path, map_location=device))
        model.eval()
        
        print("Starting evaluation...")
        evaluate_model(model)
        
    elif args.mode == 'interactive':
        print("Interactive mode activated")
        
        # Check if model file exists
        if not os.path.exists(args.model_path):
            print(f"Error: Model file {args.model_path} not found.")
            print("Please train a model first or specify a valid model path.")
            return
        
        print(f"Loading model from {args.model_path}...")
        model.load_state_dict(torch.load(args.model_path, map_location=device))
        model.eval()
        
        print("Starting interactive play...")
        interactive_play(model, max_steps=args.steps)

if __name__ == "__main__":
    main()
