"""
Task 4.1: FedProx - Local Regularization
Implements FedProx algorithm with proximal term to mitigate data heterogeneity
"""
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
import numpy as np
import matplotlib.pyplot as plt
import json
import time
import os
from utils import (SimpleCNN, get_device, set_seed, split_data_dirichlet,
                   get_model_params, set_model_params, average_weights,
                   evaluate_model, compute_weight_divergence)


class FedProxTrainer:
    """
    FedProx trainer with proximal term regularization
    Loss = CrossEntropy + (mu/2) * ||theta - theta_global||^2
    """
    
    def __init__(self, model, train_dataset, test_dataset, client_dict,
                 local_epochs=5, lr=0.01, batch_size=64, mu=0.01, device='cpu'):
        """
        Args:
            mu: Proximal term coefficient (0 = vanilla FedAvg)
        """
        self.model = model
        self.local_epochs = local_epochs
        self.lr = lr
        self.batch_size = batch_size
        self.mu = mu  # Proximal term coefficient
        self.device = device
        self.num_clients = len(client_dict)
        
        # Setup client loaders
        self.client_loaders = {}
        self.client_sizes = {}
        
        for client_id, indices in client_dict.items():
            client_subset = Subset(train_dataset, indices)
            self.client_loaders[client_id] = DataLoader(
                client_subset, batch_size=batch_size, shuffle=True,
                pin_memory=True if device.type == 'cuda' else False,
                num_workers=0  # Set to 0 for Windows compatibility
            )
            self.client_sizes[client_id] = len(indices)
        
        self.test_loader = DataLoader(
            test_dataset, batch_size=256, shuffle=False,
            pin_memory=True if device.type == 'cuda' else False,
            num_workers=0  # Set to 0 for Windows compatibility
        )
        self.criterion = nn.CrossEntropyLoss()
        
        # Track best model
        self.best_accuracy = 0.0
        self.best_model_state = None
    
    def compute_proximal_term(self, local_params, global_params):
        """
        Compute proximal term: (mu/2) * ||theta_local - theta_global||^2
        """
        proximal_term = 0.0
        for local_param, global_param in zip(local_params, global_params):
            proximal_term += torch.sum((local_param - global_param) ** 2)
        return (self.mu / 2.0) * proximal_term
    
    def local_update(self, client_id, global_params):
        """
        Perform K local epochs with FedProx proximal term
        """
        set_model_params(self.model, global_params)
        self.model.to(self.device)  # Ensure model is on GPU
        self.model.train()
        
        # Store global params for proximal term (detached, no gradient, on device)
        global_params_fixed = [param.detach().clone().to(self.device) for param in global_params]
        
        optimizer = torch.optim.SGD(self.model.parameters(), lr=self.lr)
        
        local_losses = []
        
        for epoch in range(self.local_epochs):
            epoch_loss = 0.0
            num_batches = 0
            
            for data, target in self.client_loaders[client_id]:
                data, target = data.to(self.device), target.to(self.device)
                
                # Convert to double precision if model is double
                if next(self.model.parameters()).dtype == torch.float64:
                    data = data.double()
                
                optimizer.zero_grad()
                
                # Forward pass
                output = self.model(data)
                
                # Data loss (cross-entropy)
                data_loss = self.criterion(output, target)
                
                # Proximal term: (mu/2) * ||theta - theta_global||^2
                if self.mu > 0:
                    current_params = list(self.model.parameters())
                    proximal_loss = self.compute_proximal_term(current_params, global_params_fixed)
                    total_loss = data_loss + proximal_loss
                else:
                    total_loss = data_loss
                
                # Backward pass
                total_loss.backward()
                optimizer.step()
                
                epoch_loss += total_loss.item()
                num_batches += 1
            
            avg_epoch_loss = epoch_loss / num_batches
            local_losses.append(avg_epoch_loss)
        
        return get_model_params(self.model), local_losses
    
    def train_round(self):
        """Execute one round of FedProx with all clients"""
        global_params = get_model_params(self.model)
        
        # Collect updates from all clients
        client_params = []
        client_models = []
        sizes = []
        all_local_losses = []
        
        for client_id in range(self.num_clients):
            params, local_losses = self.local_update(client_id, global_params)
            client_params.append(params)
            client_models.append(params)
            sizes.append(self.client_sizes[client_id])
            all_local_losses.append(local_losses)
        
        # Compute weight divergence before aggregation
        divergence = compute_weight_divergence(client_models, global_params)
        
        # Weighted average
        avg_params = average_weights(client_params, sizes)
        set_model_params(self.model, avg_params)
        
        # Average local losses across clients
        avg_local_loss = np.mean([np.mean(losses) for losses in all_local_losses])
        
        return {
            'divergence': divergence,
            'avg_local_loss': avg_local_loss
        }
    
    def evaluate(self):
        """Evaluate global model"""
        acc, loss = evaluate_model(self.model, self.test_loader, self.device)
        
        # Update best model if needed
        if acc > self.best_accuracy:
            self.best_accuracy = acc
            self.best_model_state = {
                'model_state_dict': self.model.state_dict(),
                'accuracy': acc,
                'loss': loss
            }
        
        return acc, loss
    
    def save_best_model(self, save_path):
        """Save the best model encountered during training"""
        if self.best_model_state is not None:
            torch.save(self.best_model_state, save_path)
            return True
        return False


def compare_fedavg_fedprox(num_rounds=30, num_clients=5, local_epochs=5,
                           alpha=0.1, mu_values=[0.0, 0.1, 0.5, 0.9],
                           lr=0.01, dataset='cifar10', use_double=True):
    """
    Compare FedAvg (mu=0) vs FedProx with different mu values
    
    Args:
        num_rounds: Number of communication rounds (max 30 as per requirement)
        alpha: Dirichlet parameter (0.1 for high heterogeneity)
        mu_values: List of proximal term coefficients (0.0 = FedAvg baseline)
    """
    set_seed(42)
    device = get_device()
    
    if use_double:
        torch.set_default_dtype(torch.float64)
    
    print(f"\n{'='*70}")
    print("TASK 4.1: FedProx - Proximal Term Regularization")
    print(f"{'='*70}")
    print(f"Using device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    else:
        print("WARNING: GPU not available, using CPU (will be slower)")
    print(f"Dataset: {dataset.upper()}")
    print(f"Clients: {num_clients}")
    print(f"Rounds: {num_rounds}")
    print(f"Local epochs: {local_epochs}")
    print(f"Learning rate: {lr}")
    print(f"Data heterogeneity: α={alpha} (Dirichlet)")
    print(f"Mu values: {mu_values} (0.0 = FedAvg baseline)")
    print(f"Double precision: {use_double}\n")
    
    # Create checkpoint directory
    os.makedirs('checkpoints/task4_1_fedprox', exist_ok=True)
    
    # Load dataset
    if dataset == 'cifar10':
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        ])
        train_data = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
        test_data = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
        num_classes = 10
        input_channels = 3
    else:
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        train_data = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
        test_data = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
        num_classes = 10
        input_channels = 1
    
    # Create non-IID data split using Dirichlet
    set_seed(42)
    client_dict = split_data_dirichlet(train_data, num_clients, alpha=alpha, num_classes=num_classes)
    
    # Show data distribution
    print(f"Data distribution (α={alpha}, highly heterogeneous):")
    if hasattr(train_data, 'targets'):
        labels = np.array(train_data.targets)
    else:
        labels = np.array([train_data[i][1] for i in range(len(train_data))])
    
    print("Client |" + "|".join([f" C{i:2d} " for i in range(num_classes)]) + "| Total")
    print("-" * (7 + 6*num_classes + 8))
    for client_id in range(num_clients):
        indices = client_dict[client_id]
        client_labels = labels[indices]
        counts = [np.sum(client_labels == c) for c in range(num_classes)]
        total = len(indices)
        print(f"  {client_id:2d}   |" + "|".join([f"{c:4d} " for c in counts]) + f"| {total:5d}")
    print()
    
    # Storage for all results
    all_results = {
        "experiment": "fedprox_comparison",
        "dataset": dataset,
        "num_clients": num_clients,
        "num_rounds": num_rounds,
        "local_epochs": local_epochs,
        "learning_rate": lr,
        "alpha": alpha,
        "mu_values": mu_values,
        "use_double": use_double,
        "results": {}
    }
    
    results = {}
    
    # Run experiment for each mu value
    for mu in mu_values:
        method_name = "FedAvg" if mu == 0.0 else f"FedProx(μ={mu})"
        print(f"\n{'='*60}")
        print(f"Training: {method_name}")
        print(f"{'='*60}")
        
        # Initialize model
        set_seed(42)
        model = SimpleCNN(num_classes=num_classes, input_channels=input_channels)
        if use_double:
            model = model.double()
        model = model.to(device)  # Move to GPU after dtype conversion
        
        num_params = sum(p.numel() for p in model.parameters())
        print(f"Model parameters: {num_params:,}")
        print(f"Model device: {next(model.parameters()).device}\n")
        
        # Initialize trainer
        trainer = FedProxTrainer(
            model=model,
            train_dataset=train_data,
            test_dataset=test_data,
            client_dict=client_dict,
            local_epochs=local_epochs,
            lr=lr,
            batch_size=64,
            mu=mu,
            device=device
        )
        
        # Training metrics
        accuracies = []
        test_losses = []
        train_losses = []
        divergences = []
        round_times = []
        
        start_time = time.time()
        
        # Training loop
        print(f"Round | Test Acc | Test Loss | Train Loss | Divergence | Time")
        print("-" * 70)
        
        for round_idx in range(num_rounds):
            round_start = time.time()
            
            # Train round
            round_info = trainer.train_round()
            
            # Evaluate
            test_acc, test_loss = trainer.evaluate()
            
            # Record metrics
            accuracies.append(test_acc)
            test_losses.append(test_loss)
            train_losses.append(round_info['avg_local_loss'])
            divergences.append(round_info['divergence'])
            round_times.append(time.time() - round_start)
            
            # Print progress
            if (round_idx + 1) % 5 == 0 or round_idx < 3:
                print(f"{round_idx+1:5d} | {test_acc:7.2f}% | {test_loss:9.4f} | "
                      f"{round_info['avg_local_loss']:10.4f} | {round_info['divergence']:10.4f} | "
                      f"{round_times[-1]:4.2f}s")
        
        total_time = time.time() - start_time
        
        # Save best model checkpoint
        checkpoint_path = f"checkpoints/task4_1_fedprox/best_model_mu_{mu}.pt"
        saved = trainer.save_best_model(checkpoint_path)
        
        print(f"\n✓ Training completed in {total_time:.2f}s ({total_time/60:.2f} min)")
        print(f"  Final Accuracy: {accuracies[-1]:.2f}%")
        print(f"  Best Accuracy: {trainer.best_accuracy:.2f}%")
        print(f"  Final Test Loss: {test_losses[-1]:.4f}")
        print(f"  Average Divergence: {np.mean(divergences):.4f}")
        if saved:
            print(f"  Best model saved: {checkpoint_path}")
        
        # Store results
        results[mu] = {
            'accuracies': accuracies,
            'test_losses': test_losses,
            'train_losses': train_losses,
            'divergences': divergences,
            'round_times': round_times
        }
        
        all_results["results"][f"mu_{mu}"] = {
            "mu": mu,
            "method": method_name,
            "final_accuracy": float(accuracies[-1]),
            "best_accuracy": float(trainer.best_accuracy),
            "final_test_loss": float(test_losses[-1]),
            "avg_divergence": float(np.mean(divergences)),
            "total_time_sec": float(total_time),
            "accuracies": [float(x) for x in accuracies],
            "test_losses": [float(x) for x in test_losses],
            "train_losses": [float(x) for x in train_losses],
            "divergences": [float(x) for x in divergences]
        }
        
        # Clear GPU cache between runs
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            print(f"  GPU memory cleared for next run")
    
    # Add comparison summary to JSON
    fedavg_final_acc = results[0.0]['accuracies'][-1]
    comparison_summary = {
        "baseline_fedavg_accuracy": float(fedavg_final_acc),
        "improvements": {}
    }
    
    for mu in mu_values:
        method = "FedAvg" if mu == 0.0 else f"FedProx(μ={mu})"
        final_acc = results[mu]['accuracies'][-1]
        best_acc = max(results[mu]['accuracies'])
        improvement = final_acc - fedavg_final_acc
        
        comparison_summary["improvements"][f"mu_{mu}"] = {
            "method": method,
            "final_accuracy": float(final_acc),
            "best_accuracy": float(best_acc),
            "improvement_vs_fedavg": float(improvement),
            "improvement_percentage": f"{improvement:+.2f}%"
        }
    
    all_results["comparison_summary"] = comparison_summary
    
    # Save results to JSON
    json_path = 'task4_1_fedprox_results.json'
    with open(json_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\n✅ Results saved to '{json_path}'")
    
    # Create comprehensive plots
    plot_fedprox_comparison(results, mu_values, alpha, num_rounds)
    
    # Summary table
    print(f"\n{'='*70}")
    print("SUMMARY: FedProx vs FedAvg")
    print(f"{'='*70}")
    print(f"{'Method':>20} | {'Final Acc':>10} | {'Best Acc':>10} | {'Final Loss':>11} | {'Avg Div':>10}")
    print("-" * 70)
    for mu in mu_values:
        method = "FedAvg" if mu == 0.0 else f"FedProx(μ={mu})"
        final_acc = results[mu]['accuracies'][-1]
        best_acc = max(results[mu]['accuracies'])
        final_loss = results[mu]['test_losses'][-1]
        avg_div = np.mean(results[mu]['divergences'])
        print(f"{method:>20} | {final_acc:>9.2f}% | {best_acc:>9.2f}% | {final_loss:>11.4f} | {avg_div:>10.4f}")
    
    print(f"\n{'='*70}")
    print("OBSERVATIONS:")
    print(f"{'='*70}")
    fedavg_acc = results[0.0]['accuracies'][-1]
    print(f"FedAvg (baseline) final accuracy: {fedavg_acc:.2f}%")
    
    for mu in [m for m in mu_values if m > 0]:
        fedprox_acc = results[mu]['accuracies'][-1]
        improvement = fedprox_acc - fedavg_acc
        print(f"FedProx(μ={mu}) improvement: {improvement:+.2f}% "
              f"({'✓ Better' if improvement > 0 else '✗ Worse'})")
    
    return results, all_results


def plot_fedprox_comparison(results, mu_values, alpha, num_rounds):
    """
    Create comprehensive comparison plots
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    rounds = np.arange(1, num_rounds + 1)
    
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    
    for idx, mu in enumerate(mu_values):
        label = "FedAvg (baseline)" if mu == 0.0 else f"FedProx (μ={mu})"
        color = colors[idx % len(colors)]
        linewidth = 3 if mu == 0.0 else 2
        linestyle = '-' if mu == 0.0 else '--'
        
        # Test Accuracy
        axes[0, 0].plot(rounds, results[mu]['accuracies'], 
                       label=label, color=color, linewidth=linewidth, 
                       linestyle=linestyle, alpha=0.8)
        
        # Test Loss
        axes[0, 1].plot(rounds, results[mu]['test_losses'],
                       label=label, color=color, linewidth=linewidth,
                       linestyle=linestyle, alpha=0.8)
        
        # Train Loss (local)
        axes[1, 0].plot(rounds, results[mu]['train_losses'],
                       label=label, color=color, linewidth=linewidth,
                       linestyle=linestyle, alpha=0.8)
        
        # Client Divergence
        axes[1, 1].plot(rounds, results[mu]['divergences'],
                       label=label, color=color, linewidth=linewidth,
                       linestyle=linestyle, alpha=0.8)
    
    # Formatting
    axes[0, 0].set_title('Test Accuracy (%)', fontsize=14, fontweight='bold')
    axes[0, 0].set_xlabel('Communication Round', fontsize=12)
    axes[0, 0].set_ylabel('Accuracy (%)', fontsize=12)
    axes[0, 0].legend(fontsize=10)
    axes[0, 0].grid(True, alpha=0.3, linestyle='--')
    
    axes[0, 1].set_title('Test Loss', fontsize=14, fontweight='bold')
    axes[0, 1].set_xlabel('Communication Round', fontsize=12)
    axes[0, 1].set_ylabel('Loss', fontsize=12)
    axes[0, 1].legend(fontsize=10)
    axes[0, 1].grid(True, alpha=0.3, linestyle='--')
    
    axes[1, 0].set_title('Training Loss (Local Avg)', fontsize=14, fontweight='bold')
    axes[1, 0].set_xlabel('Communication Round', fontsize=12)
    axes[1, 0].set_ylabel('Loss', fontsize=12)
    axes[1, 0].legend(fontsize=10)
    axes[1, 0].grid(True, alpha=0.3, linestyle='--')
    
    axes[1, 1].set_title('Client Drift (Weight Divergence)', fontsize=14, fontweight='bold')
    axes[1, 1].set_xlabel('Communication Round', fontsize=12)
    axes[1, 1].set_ylabel('L2 Divergence', fontsize=12)
    axes[1, 1].legend(fontsize=10)
    axes[1, 1].grid(True, alpha=0.3, linestyle='--')
    
    plt.suptitle(f'FedProx vs FedAvg: Data Heterogeneity (α={alpha})', 
                 fontsize=16, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    
    plot_path = 'task4_1_fedprox.png'
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ Plot saved as '{plot_path}'")


if __name__ == "__main__":
    print(f"\nStarting Task 4.1 @ {time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    results, all_results = compare_fedavg_fedprox(
        num_rounds=30,
        num_clients=5,
        local_epochs=5,
        alpha=0.1,  # High heterogeneity
        mu_values=[0.0, 0.1, 0.5, 0.9],  # 0.0 = FedAvg baseline
        lr=0.01,
        dataset='cifar10',
        use_double=True
    )
    
    print("\n" + "="*70)
    print("Task 4.1 Completed!")
    print("="*70)
    print("\nGenerated files:")
    print("  → task4_1_fedprox_results.json (complete metrics)")
    print("  → task4_1_fedprox.png (comparison plots)")
    print("  → checkpoints/task4_1_fedprox/best_model_mu_*.pt (best models)")
    print("\nNext: Implement Task 4.2 (SCAFFOLD), 4.3 (FedGH), 4.4 (FedSAM)")

