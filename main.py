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
                        choices=['train', 'evaluate', 'interactive'],
                        help='Mode to run: train (train model), evaluate (evaluate model), interactive (human + model)')
    parser.add_argument('--episodes', type=int, default=100, help='Number of episodes for training (default: 100)')
    parser.add_argument('--steps', type=int, default=500, help='Maximum steps per episode (default: 500)')
    parser.add_argument('--epochs', type=int, default=50, help='Number of training epochs (default: 50)')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size for training (default: 32)')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate (default: 0.001)')
    parser.add_argument('--model-path', type=str, default='models/snake_model_best.pt',
                       help='Path to saved model for evaluation or interactive mode')
    
    args = parser.parse_args()
    
    # Create the model with same dimensions as saved model
    model = SnakeNet(board_size=32, latent_dim=4096, hidden_dim=2048)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    
    if args.mode == 'train':
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
