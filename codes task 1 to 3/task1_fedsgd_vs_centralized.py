"""
Task 1: FedSGD vs Centralized SGD
Demonstrates theoretical equivalence between FedSGD (K=1) and centralized training
"""
import torch
import os
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
import numpy as np
import time
import json
import matplotlib.pyplot as plt
from utils import (SimpleCNN, get_device, set_seed, split_data_iid, 
                   get_model_params, set_model_params, average_weights, evaluate_model)

torch.set_default_dtype(torch.float64)   # ← ADD THIS LINE
class FedSGDTrainer:
    """Federated SGD trainer with K=1 local step"""
    
    def __init__(self, model, train_dataset, test_dataset, num_clients=3, 
                 lr=0.01, batch_size=64, device='cpu'):
        self.model = model
        self.num_clients = num_clients
        self.lr = lr
        self.batch_size = batch_size
        self.device = device
        
        # Split data IID across clients
        self.client_dict = split_data_iid(train_dataset, num_clients)
        self.client_loaders = {}
        self.client_sizes = {}
        
        for client_id in range(num_clients):
            client_subset = Subset(train_dataset, self.client_dict[client_id])
            # Use entire local dataset as one batch for true FedSGD
            self.client_loaders[client_id] = DataLoader(
                client_subset, batch_size=len(client_subset), shuffle=False
            )
            self.client_sizes[client_id] = len(client_subset)
        
        self.test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False)
        self.criterion = nn.CrossEntropyLoss()
        
    def local_update(self, client_id, global_params):
        set_model_params(self.model, global_params)
        self.model.train()
        self.model.zero_grad()            # clear old grads first

        data, target = next(iter(self.client_loaders[client_id]))
        data, target = data.to(self.device), target.to(self.device)
        
        self.model.zero_grad()
        output = self.model(data)
        loss = self.criterion(output, target)
        loss.backward()
        
        # CORRECT WAY: apply update AND assign back
        updated_params = []
        with torch.no_grad():
            for param in self.model.parameters():
                if param.grad is not None:
                    # Apply update directly to param.data
                    param.data = param.data - self.lr * param.grad
                    updated_params.append(param.data.clone())  # clone the updated value
        
        self.model.zero_grad()  # already clean
        return updated_params
    
   
    def train_round(self):
        """Execute one round of FedSGD"""
        global_params = get_model_params(self.model)
        
        # Collect updates from all clients
        client_params = []
        sizes = []
        for client_id in range(self.num_clients):
            params = self.local_update(client_id, global_params)
            client_params.append(params)
            sizes.append(self.client_sizes[client_id])
        
        # Average updates (weighted by data size)
        avg_params = average_weights(client_params, sizes)
        set_model_params(self.model, avg_params)
        
    def evaluate(self):
        """Evaluate global model"""
        return evaluate_model(self.model, self.test_loader, self.device)


class CentralizedTrainer:
    """Centralized SGD trainer"""
    
    def __init__(self, model, train_dataset, test_dataset, lr=0.01, device='cpu'):
        self.model = model
        self.lr = lr
        self.device = device
        
        # Use entire dataset as one batch for equivalence
        self.train_loader = DataLoader(
            train_dataset, batch_size=len(train_dataset), shuffle=False
        )
        self.test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False)
        self.criterion = nn.CrossEntropyLoss()

   

    def train_step(self):
        self.model.train()
        self.model.zero_grad()            # clear old grads first
    
        data, target = next(iter(self.train_loader))
        data, target = data.to(self.device), target.to(self.device)
        
        self.model.zero_grad()
        output = self.model(data)
        loss = self.criterion(output, target)
        loss.backward()
        
        with torch.no_grad():
            for param in self.model.parameters():
                if param.grad is not None:
                    param.data -= self.lr * param.grad  # direct update
        
        self.model.zero_grad()

    def evaluate(self):
        """Evaluate model"""
        return evaluate_model(self.model, self.test_loader, self.device)


def compare_fedsgd_centralized(num_rounds=1000, num_clients=3, lr=0.1, dataset='cifar10',
                               early_stopping=True, patience=200, min_delta=0.1):
    """
    Compare FedSGD and Centralized SGD
    
    Args:
        num_rounds: Number of training rounds
        num_clients: Number of clients for FedSGD
        lr: Learning rate
        dataset: 'cifar10' or 'mnist'
        early_stopping: Enable early stopping
        patience: Rounds to wait without improvement
        min_delta: Minimum improvement to reset patience
    """
    set_seed(42)
    # Inside compare_fedsgd_centralized(), right after set_seed(42)
    
    device = get_device()
    print(f"Using device: {device}")
    
    # Use CIFAR-10 standard normalization
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        transforms.Lambda(lambda x: x.double())   # ← ADD
    ])

    
    # Load dataset
    if dataset == 'cifar10':
        train_data = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
        test_data = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
        num_classes = 10
        input_channels = 3
    else:  # mnist
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        train_data = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
        test_data = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
        num_classes = 10
        input_channels = 1
    
    # Use subset for faster training
    train_subset_size = 6000
    train_indices = np.random.choice(len(train_data), train_subset_size, replace=False)
    train_subset = Subset(train_data, train_indices)

    print(f"\nDataset: {dataset.upper()}")
    print(f"Training samples: {len(train_subset)}")
    print(f"Test samples: {len(test_data)}")
    print(f"Num clients: {num_clients}")
    print(f"Learning rate: {lr}\n")
    
    # Initialize models with same weights
    set_seed(42)
    model_fedsgd = SimpleCNN(num_classes=num_classes, input_channels=input_channels).to(device)
    set_seed(42)
    model_central = SimpleCNN(num_classes=num_classes, input_channels=input_channels).to(device)
    model_fedsgd = model_fedsgd.double()
    model_central = model_central.double()
    # Count parameters
    num_params = sum(p.numel() for p in model_fedsgd.parameters())
    print(f"Total model parameters: {num_params:,}")
    
    # Verify initial weights are identical
    fedsgd_init = get_model_params(model_fedsgd)
    central_init = get_model_params(model_central)
    init_diff = sum([torch.norm(p1 - p2).item() for p1, p2 in zip(fedsgd_init, central_init)])
    print(f"Initial weight difference: {init_diff:.10f}\n")
    
    # Initialize trainers
    fedsgd_trainer = FedSGDTrainer(
        model_fedsgd, train_subset, test_data, num_clients=num_clients, 
        lr=lr, device=device
    )
    central_trainer = CentralizedTrainer(
        model_central, train_subset, test_data, lr=lr, device=device
    )
    
    # Training tracking
    fedsgd_accuracies = []
    central_accuracies = []
    fedsgd_losses = []
    central_losses = []
    weight_differences = []
    
    best_acc = 0.0
    best_round = 0
    wait = 0
    start_time_global = time.time()

    results = {
        "num_clients": num_clients,
        "num_params": num_params,
        "fedsgd_total_params": num_params * num_clients,
        "centralized_params": num_params,
        "rounds": []
    }

    print("Round | FedSGD Acc | Central Acc | Weight Diff | Time")
    print("-" * 65)
    
    for round_idx in range(num_rounds):
        round_start = time.time()
        
        # Train both
        fedsgd_trainer.train_round()
        central_trainer.train_step()

        # Evaluate
        fedsgd_acc, fedsgd_loss = fedsgd_trainer.evaluate()
        central_acc, central_loss = central_trainer.evaluate()

        # Compute weight difference
        fedsgd_params = get_model_params(model_fedsgd)
        central_params = get_model_params(model_central)
        weight_diff = sum([torch.norm(p1 - p2).item() for p1, p2 in zip(fedsgd_params, central_params)])

        # Timing
        round_time = time.time() - round_start

        # Record stats
        fedsgd_accuracies.append(fedsgd_acc)
        central_accuracies.append(central_acc)
        fedsgd_losses.append(fedsgd_loss)
        central_losses.append(central_loss)
        weight_differences.append(weight_diff)

        results["rounds"].append({
            "round": round_idx + 1,
            "fedsgd_acc": fedsgd_acc,
            "fedsgd_loss": fedsgd_loss,
            "central_acc": central_acc,
            "central_loss": central_loss,
            "weight_diff": weight_diff,
            "round_time_sec": round_time
        })

        if (round_idx + 1) % 10 == 0 or round_idx < 5:
            print(f"{round_idx+1:5d} | {fedsgd_acc:9.2f}% | {central_acc:10.2f}% | {weight_diff:.10f} | {round_time:.2f}s")

        # Early stopping
        if early_stopping:
            if fedsgd_acc - best_acc > min_delta:
                best_acc = fedsgd_acc
                best_round = round_idx + 1
                wait = 0
            else:
                wait += 1
                if wait >= patience:
                    print(f"\n🛑 Early stopping at round {round_idx+1} (no improvement for {patience} rounds)")
                    break

    # Total time
    total_time = time.time() - start_time_global
    print(f"\n⏱️  Total training time: {total_time:.2f}s ({total_time/60:.2f} minutes)")

    results["early_stopping"] = {
        "enabled": early_stopping,
        "patience": patience,
        "min_delta": min_delta,
        "best_round": best_round,
        "best_fedsgd_acc": best_acc,
        "total_time_sec": total_time
    }

    # Plot results
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    # Accuracy comparison
    axes[0].plot(fedsgd_accuracies, 'o-', label='FedSGD', markersize=3)
    axes[0].plot(central_accuracies, 's-', label='Centralized', markersize=3)
    axes[0].set_xlabel('Round')
    axes[0].set_ylabel('Test Accuracy (%)')
    axes[0].set_title('Test Accuracy Comparison')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Loss comparison
    axes[1].plot(fedsgd_losses, 'o-', label='FedSGD', markersize=3)
    axes[1].plot(central_losses, 's-', label='Centralized', markersize=3)
    axes[1].set_xlabel('Round')
    axes[1].set_ylabel('Test Loss')
    axes[1].set_title('Test Loss Comparison')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    # Weight difference
    axes[2].plot(weight_differences, 'o-', color='red', markersize=3)
    axes[2].set_xlabel('Round')
    axes[2].set_ylabel('L2 Weight Difference')
    axes[2].set_title('Model Parameter Divergence')
    axes[2].set_yscale('log')
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('task1_fedsgd_vs_centralized.png', dpi=300, bbox_inches='tight')
    
    # Save results
    with open("task1_results.json", "w") as f:
        json.dump(results, f, indent=4)
    
    print(f"\nPlot saved as 'task1_fedsgd_vs_centralized.png'")
    print("Results saved to 'task1_results.json'")
    
    # Summary
    print(f"\n{'='*65}")
    print("SUMMARY")
    print(f"{'='*65}")
    print(f"Final FedSGD accuracy: {fedsgd_accuracies[-1]:.2f}%")
    print(f"Final Centralized accuracy: {central_accuracies[-1]:.2f}%")
    print(f"Final weight difference: {weight_differences[-1]:.10f}")
    print(f"Average weight difference: {np.mean(weight_differences):.10f}")
    print(f"Max weight difference: {np.max(weight_differences):.10f}")
    
    # Check equivalence
    is_equivalent = weight_differences[-1] < 1e-4
    print(f"\nConclusion: FedSGD and Centralized SGD {'ARE' if is_equivalent else 'are NOT'} equivalent")
    print(f"(Weight difference threshold: 1e-4)")
    # Right before the final return statement
    print(f"\n{'='*65}")
    print("SAVING MODELS")
    print(f"{'='*65}")

    # Create save directory
    os.makedirs('saved_models', exist_ok=True)

    # Save FedSGD model
    torch.save({
        'model_state_dict': model_fedsgd.state_dict(),
        'final_accuracy': fedsgd_accuracies[-1],
        'final_loss': fedsgd_losses[-1],
        'num_clients': num_clients,
        'learning_rate': lr,
        'dataset': dataset
    }, 'saved_models/fedsgd_final.pth')

    # Save Centralized model
    torch.save({
        'model_state_dict': model_central.state_dict(),
        'final_accuracy': central_accuracies[-1],
        'final_loss': central_losses[-1],
        'learning_rate': lr,
        'dataset': dataset
    }, 'saved_models/central_final.pth')

    print("✓ Models saved to 'saved_models/' directory")

if __name__ == "__main__":
    # Run comparison
    compare_fedsgd_centralized(
        num_rounds=1000,      # Increased for learning
        num_clients=3,
        lr=0.1,              # Increased for full-batch
        dataset='cifar10',
        early_stopping=True,
        patience=200,         # Stop if no improvement for 50 rounds
        min_delta=0.1        # Minimum 0.1% improvement
    )