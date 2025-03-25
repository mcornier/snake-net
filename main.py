import argparse
import torch
import os
import sys
from snake_game import SnakeGame
from snake_model import SnakeNet
from train import train, generate_dataset
from interface import SnakeInterface

def main():
    parser = argparse.ArgumentParser(description='Snake Neural Network Simulator')
    parser.add_argument('--mode', type=str, default='train',
                        choices=['play', 'train', 'interactive', 'random', 'search'],
                        help='Mode to run: play (human), train (train model), interactive (human + model), random/search (generate datasets)')
    parser.add_argument('--episodes', type=int, default=100, help='Number of episodes for training (default: 100)')
    parser.add_argument('--steps', type=int, default=500, help='Maximum steps per episode (default: 500)')
    parser.add_argument('--epochs', type=int, default=50, help='Number of training epochs (default: 50)')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size for training (default: 32)')
    parser.add_argument('--lr', type=float, default=0.0001, help='Learning rate (default: 0.0001)')
    parser.add_argument('--model-path', type=str, default='models/snake_model_best.pt',
                       help='Path to saved model for interactive mode')
    parser.add_argument('--device', type=str, default='cuda', choices=['cpu', 'cuda'],
                       help='Device to use for training/inference (default: cuda)')
    
    args = parser.parse_args()
    
    # Create the model with same dimensions as saved model
    model = SnakeNet(board_size=32, latent_dim=1024)
    # Validate CUDA availability
    if args.device == 'cuda' and not torch.cuda.is_available():
        print("Warning: CUDA requested but not available. Falling back to CPU.")
        device = torch.device('cpu')
    else:
        device = torch.device(args.device)
    print(f"Using device: {device}")
    model.to(device)
    
    if args.mode in ['random', 'search']:
        # Random/Search play mode with dataset generation
        dataset = generate_dataset(num_episodes=args.episodes, max_steps=args.steps, mode=args.mode)
        
        # Save dataset
        os.makedirs('datasets', exist_ok=True)
        final_path = f'datasets/{args.mode}_gameplay.pt'
        torch.save(dataset, final_path)
        print(f"Dataset saved to {final_path} with {len(dataset)} samples")
        
    elif args.mode == 'play':
        # Human play mode with dataset generation
        game = SnakeGame(size=32)
        interface = SnakeInterface(game)
        dataset = interface.play_human(save_dataset=True)
        
        if dataset:
            os.makedirs('datasets', exist_ok=True)
            torch.save(dataset, 'datasets/human_gameplay_final.pt')
        
    elif args.mode == 'train':
        print("Training mode activated")
        
        print("Loading existing datasets...")
        dataset = []
        for file in os.listdir('datasets'):
            if file.endswith('.pt'):
                print(f"Loading {file}...")
                data = torch.load(f'datasets/{file}')
                # Move loaded data to device immediately
                data = [(state.to(device), action.to(device), next_state.to(device)) 
                       for state, action, next_state in data]
                dataset.extend(data)
        print(f"Total dataset size: {len(dataset)} samples")
        
        # For training mode, check if --model-path was provided in command line args
        cmd_args = parser.parse_args()
        model_path_provided = False
        for arg in sys.argv[1:]:
            if arg.startswith('--model-path'):
                model_path_provided = True
                break
            
        if model_path_provided:
            if os.path.exists(args.model_path):
                print(f"Loading existing model from {args.model_path} to continue training...")
                model.load_state_dict(torch.load(args.model_path, map_location=device))
            else:
                print(f"Warning: Model file {args.model_path} not found, starting with a new model...")
        else:
            print("Starting with a new model...")
        
        print(f"Training model for {args.epochs} epochs...")
        train(model, dataset, num_epochs=args.epochs, batch_size=args.batch_size, learning_rate=args.lr, device=device)
        
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
        game = SnakeGame(size=32)
        interface = SnakeInterface(game, model)
        interface.play_interactive(max_steps=args.steps)

if __name__ == "__main__":
    main()
