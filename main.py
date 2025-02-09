import argparse
import torch
import torch.nn.functional as F
from torch_geometric.datasets import MNISTSuperpixels
from torch_geometric.data import DataLoader
import torch_geometric.transforms as T
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np

from models import Net, NetWithPE, LaplacianPE

def train(model, train_loader, optimizer, device):
    model.train()
    total_loss = 0
    
    for data in tqdm(train_loader, desc='Training', leave=False):
        data = data.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, data.y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * data.num_graphs
    
    return total_loss / len(train_loader.dataset)

def validate(model, loader, device):
    model.eval()
    total_loss = 0
    correct = 0
    
    for data in tqdm(loader, desc='Validating', leave=False):
        data = data.to(device)
        with torch.no_grad():
            output = model(data)
            loss = F.nll_loss(output, data.y)
            total_loss += loss.item() * data.num_graphs
            pred = output.max(dim=1)[1]
            correct += pred.eq(data.y).sum().item()
    
    accuracy = correct / len(loader.dataset)
    avg_loss = total_loss / len(loader.dataset)
    return avg_loss, accuracy

def plot_training_curves(k_results, save_path='training_curves.png'):
    plt.figure(figsize=(12, 6))
    
    for k, results in k_results.items():
        epochs = range(1, len(results['train_loss']) + 1)
        plt.plot(epochs, results['train_loss'], 
                label=f'Train Loss (k={k})', linestyle='-')
        plt.plot(epochs, results['val_loss'], 
                label=f'Val Loss (k={k})', linestyle='--')
    
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss for Different k Values')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--k_values', nargs='+', type=int, default=[0,3, 5, 8], 
                      help='List of k values for positional encoding dimensions')
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size')
    parser.add_argument('--lr', type=float, default=0.01, help='Learning rate')
    parser.add_argument('--device', type=str, default='cuda', help='Device to use')
    parser.add_argument('--data_dir', type=str, default='data', help='Data directory')
    args = parser.parse_args()
    
    # Set device
    device = torch.device(args.device if torch.cuda.is_available() and args.device == 'cuda' else 'cpu')
    print(f'Using device: {device}')
    
    # Results dictionary to store metrics for each k
    k_results = {}
    
    # Train and evaluate for each k value
    for k in args.k_values:
        print(f'\nTraining model with k={k}')
        
        # Define transforms
        transform = T.Compose([
            LaplacianPE(k=k),
            T.NormalizeFeatures()
        ])
        
        # Load datasets
        train_dataset = MNISTSuperpixels(
            root=args.data_dir, 
            train=True,
            transform=transform
        )
        
        test_dataset = MNISTSuperpixels(
            root=args.data_dir, 
            train=False,
            transform=transform
        )
        
        # Create data loaders
        train_loader = DataLoader(
            train_dataset, 
            batch_size=args.batch_size, 
            shuffle=True
        )
        test_loader = DataLoader(
            test_dataset, 
            batch_size=args.batch_size
        )
        
        # Initialize model and optimizer
        model = NetWithPE(num_features=1, pos_enc_dim=k).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
        
        # Initialize metrics storage
        k_results[k] = {
            'train_loss': [],
            'val_loss': [],
            'train_acc': [],
            'val_acc': []
        }
        
        # Training loop
        best_val_acc = 0
        for epoch in range(1, args.epochs + 1):
            # Train
            train_loss = train(model, train_loader, optimizer, device)
            
            # Validate
            val_loss, val_acc = validate(model, test_loader, device)
            train_loss_val, train_acc = validate(model, train_loader, device)
            
            # Store metrics
            k_results[k]['train_loss'].append(train_loss_val)
            k_results[k]['val_loss'].append(val_loss)
            k_results[k]['train_acc'].append(train_acc)
            k_results[k]['val_acc'].append(val_acc)
            
            # Save best model
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                torch.save(model.state_dict(), f'best_model_k{k}.pt')
            
            print(f'Epoch: {epoch:02d}, '
                  f'Train Loss: {train_loss:.4f}, '
                  f'Val Loss: {val_loss:.4f}, '
                  f'Train Acc: {train_acc:.4f}, '
                  f'Val Acc: {val_acc:.4f}, '
                  f'Best Val Acc: {best_val_acc:.4f}')
    
    # Plot results
    plot_training_curves(k_results)
    
    # Print final results
    print('\nFinal Results:')
    for k in args.k_values:
        best_val_acc = max(k_results[k]['val_acc'])
        print(f'k={k}: Best Validation Accuracy: {best_val_acc:.4f}')

if __name__ == '__main__':
    main()