"""
Task 3: Exploring Data Heterogeneity Impact
Study how label heterogeneity (Dirichlet distribution) affects FedAvg performance
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


class FedAvgHeterogeneity:
    """FedAvg trainer for heterogeneity experiments"""
    
    def __init__(self, model, train_dataset, test_dataset, client_dict,
                 local_epochs=5, lr=0.01, batch_size=64, device='cpu'):
        self.model = model
        self.local_epochs = local_epochs
        self.lr = lr
        self.batch_size = batch_size
        self.device = device
        self.num_clients = len(client_dict)
        
        # Setup client loaders
        self.client_loaders = {}
        self.client_sizes = {}
        
        for client_id, indices in client_dict.items():
            client_subset = Subset(train_dataset, indices)
            self.client_loaders[client_id] = DataLoader(
                client_subset, batch_size=batch_size, shuffle=True
            )
            self.client_sizes[client_id] = len(indices)
        
        self.test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False)
        self.criterion = nn.CrossEntropyLoss()
    
    def local_update(self, client_id, global_params):
        """Perform K local epochs of training on client data"""
        set_model_params(self.model, global_params)
        self.model.train()
        
        optimizer = torch.optim.SGD(self.model.parameters(), lr=self.lr)
        
        for epoch in range(self.local_epochs):
            for data, target in self.client_loaders[client_id]:
                data, target = data.to(self.device), target.to(self.device)
                # Convert to double precision
                data = data.double()
                
                optimizer.zero_grad()
                output = self.model(data)
                loss = self.criterion(output, target)
                loss.backward()
                optimizer.step()
        
        return get_model_params(self.model)
    
    def train_round(self):
        """Execute one round of FedAvg with all clients"""
        global_params = get_model_params(self.model)
        
        # Collect updates from all clients
        client_params = []
        client_models = []
        sizes = []
        
        for client_id in range(self.num_clients):
            params = self.local_update(client_id, global_params)
            client_params.append(params)
            client_models.append(params)
            sizes.append(self.client_sizes[client_id])
        
        # Compute weight divergence
        divergence = compute_weight_divergence(client_models, global_params)
        
        # Average updates
        avg_params = average_weights(client_params, sizes)
        set_model_params(self.model, avg_params)
        
        return divergence
    
    def evaluate(self):
        """Evaluate global model with double precision support"""
        self.model.eval()
        correct = 0
        total = 0
        loss_sum = 0.0
        criterion = nn.CrossEntropyLoss()
        
        with torch.no_grad():
            for data, target in self.test_loader:
                data, target = data.to(self.device), target.to(self.device)
                data = data.double()  # Convert to double for model
                
                output = self.model(data)
                loss_sum += criterion(output, target).item()
                pred = output.argmax(dim=1)
                correct += pred.eq(target).sum().item()
                total += target.size(0)
        
        accuracy = 100. * correct / total
        avg_loss = loss_sum / len(self.test_loader)
        
        return accuracy, avg_loss


def analyze_data_distribution(client_dict, dataset, num_classes=10):
    """Analyze and visualize data distribution across clients"""
    # Get labels
    if hasattr(dataset, 'targets'):
        labels = np.array(dataset.targets)
    elif hasattr(dataset, 'labels'):
        labels = np.array(dataset.labels)
    else:
        labels = np.array([dataset[i][1] for i in range(len(dataset))])
    
    num_clients = len(client_dict)
    distribution = np.zeros((num_clients, num_classes))
    
    for client_id, indices in client_dict.items():
        client_labels = labels[indices]
        for cls in range(num_classes):
            distribution[client_id, cls] = np.sum(client_labels == cls)
    
    return distribution


def experiment_heterogeneity(num_rounds=50, num_clients=5, local_epochs=5,
                             alpha_values=[100, 1.0, 0.2, 0.05], lr=0.01, 
                             dataset='cifar10', num_classes=10,
                             save_checkpoints=True, checkpoint_interval=10):
    """
    Experiment: Study impact of data heterogeneity using Dirichlet distribution
    
    Args:
        alpha_values: List of Dirichlet concentration parameters
                     (larger = more IID, smaller = more heterogeneous)
        save_checkpoints: Whether to save model checkpoints
        checkpoint_interval: Save checkpoint every N rounds
    """
    set_seed(42)
    device = get_device()
    print(f"Using device: {device}")
    
    # Create directories for checkpoints
    if save_checkpoints:
        os.makedirs('checkpoints/task3', exist_ok=True)
    
    # Load dataset with proper normalization
    if dataset == 'cifar10':
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        ])
        train_data = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
        test_data = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
        input_channels = 3
    else:
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        train_data = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
        test_data = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
        input_channels = 1
    
    print(f"\n{'='*60}")
    print("EXPERIMENT: Impact of Data Heterogeneity")
    print(f"{'='*60}")
    print(f"Dataset: {dataset.upper()}")
    print(f"Clients: {num_clients}")
    print(f"Rounds: {num_rounds}")
    print(f"Local epochs: {local_epochs}")
    print(f"Alpha values: {alpha_values}")
    print(f"  (larger α = more IID, smaller α = more heterogeneous)\n")
    
    # Overall results storage
    all_results = {
        "experiment": "data_heterogeneity",
        "dataset": dataset,
        "num_clients": num_clients,
        "num_rounds": num_rounds,
        "local_epochs": local_epochs,
        "learning_rate": lr,
        "alpha_values": alpha_values,
        "results": {}
    }
    
    results = {}
    distributions = {}
    
    for alpha in alpha_values:
        print(f"\n{'='*50}")
        print(f"Training with α={alpha}")
        print(f"{'='*50}")
        
        # Create non-IID data split using Dirichlet
        set_seed(42)
        client_dict = split_data_dirichlet(
            train_data, num_clients, alpha=alpha, num_classes=num_classes
        )
        
        # Analyze distribution
        distribution = analyze_data_distribution(client_dict, train_data, num_classes)
        distributions[alpha] = distribution
        
        print("\nData distribution across clients (samples per class):")
        print("Client |" + "|".join([f" C{i:2d} " for i in range(num_classes)]) + "| Total")
        print("-" * (7 + 6*num_classes + 8))
        for client_id in range(num_clients):
            counts = distribution[client_id].astype(int)
            total = int(np.sum(counts))
            print(f"  {client_id:2d}   |" + "|".join([f"{c:4d} " for c in counts]) + f"| {total:5d}")
        
        # Initialize model with double precision
        set_seed(42)
        model = SimpleCNN(num_classes=num_classes, input_channels=input_channels).to(device)
        model = model.double()  # Convert to double precision
        
        # Count parameters
        num_params = sum(p.numel() for p in model.parameters())
        print(f"\nModel parameters: {num_params:,}")
        
        trainer = FedAvgHeterogeneity(
            model, train_data, test_data, client_dict,
            local_epochs=local_epochs, lr=lr, device=device
        )
        
        accuracies = []
        losses = []
        divergences = []
        round_times = []
        
        alpha_results = {
            "alpha": float(alpha),
            "data_type": "IID" if alpha >= 10 else "Non-IID",
            "num_params": num_params,
            "data_distribution": distribution.tolist(),
            "rounds": []
        }
        
        start_time_total = time.time()
        
        for round_idx in range(num_rounds):
            round_start = time.time()
            
            divergence = trainer.train_round()
            acc, loss = trainer.evaluate()
            
            round_time = time.time() - round_start
            
            accuracies.append(acc)
            losses.append(loss)
            divergences.append(divergence)
            round_times.append(round_time)
            
            # Store round details
            alpha_results["rounds"].append({
                "round": round_idx + 1,
                "accuracy": float(acc),
                "loss": float(loss),
                "divergence": float(divergence),
                "time_sec": float(round_time)
            })
            
            if (round_idx + 1) % 10 == 0:
                print(f"Round {round_idx+1}/{num_rounds}: Acc={acc:.2f}%, Loss={loss:.4f}, "
                      f"Div={divergence:.4f}, Time={round_time:.2f}s")
            
            # Save checkpoint
            if save_checkpoints and (round_idx + 1) % checkpoint_interval == 0:
                checkpoint_path = f'checkpoints/task3/alpha_{alpha}_round_{round_idx+1}.pt'
                torch.save({
                    'round': round_idx + 1,
                    'alpha': alpha,
                    'model_state_dict': model.state_dict(),
                    'accuracy': acc,
                    'loss': loss,
                    'divergence': divergence
                }, checkpoint_path)
        
        total_time = time.time() - start_time_total
        
        # Save final model
        if save_checkpoints:
            final_path = f'checkpoints/task3/alpha_{alpha}_final.pt'
            torch.save({
                'round': num_rounds,
                'alpha': alpha,
                'model_state_dict': model.state_dict(),
                'accuracy': accuracies[-1],
                'loss': losses[-1],
                'divergence': divergences[-1],
                'all_accuracies': accuracies,
                'all_losses': losses,
                'all_divergences': divergences
            }, final_path)
            print(f"Final model saved to {final_path}")
        
        # Summary statistics
        alpha_results["summary"] = {
            "final_accuracy": float(accuracies[-1]),
            "best_accuracy": float(max(accuracies)),
            "final_loss": float(losses[-1]),
            "avg_divergence": float(np.mean(divergences)),
            "max_divergence": float(np.max(divergences)),
            "total_time_sec": float(total_time),
            "avg_round_time_sec": float(np.mean(round_times))
        }
        
        results[alpha] = {
            'accuracies': accuracies,
            'losses': losses,
            'divergences': divergences,
            'round_times': round_times
        }
        
        all_results["results"][f"alpha_{alpha}"] = alpha_results
        
        print(f"\nFinal accuracy: {accuracies[-1]:.2f}%")
        print(f"Best accuracy: {max(accuracies):.2f}%")
        print(f"Average divergence: {np.mean(divergences):.4f}")
        print(f"Total time: {total_time:.2f}s ({total_time/60:.2f} min)")
    
    # Save all results to JSON
    json_path = 'task3_heterogeneity_results.json'
    with open(json_path, 'w') as f:
        json.dump(all_results, f, indent=4)
    print(f"\n✅ Results saved to '{json_path}'")
    
    # Plot results
    fig = plt.figure(figsize=(18, 12))
    gs = fig.add_gridspec(4, len(alpha_values), hspace=0.3, wspace=0.3)
    
    # Top row: Data distributions
    for idx, alpha in enumerate(alpha_values):
        ax = fig.add_subplot(gs[0, idx])
        distribution = distributions[alpha]
        
        im = ax.imshow(distribution.T, aspect='auto', cmap='YlOrRd', interpolation='nearest')
        ax.set_xlabel('Client ID')
        ax.set_ylabel('Class')
        ax.set_title(f'α={alpha}\n({"IID" if alpha >= 10 else "Non-IID"})')
        ax.set_xticks(range(num_clients))
        ax.set_yticks(range(num_classes))
        plt.colorbar(im, ax=ax, label='Sample count')
    
    # Second row: Accuracy curves
    ax_acc = fig.add_subplot(gs[1, :])
    for alpha in alpha_values:
        label = f'α={alpha} ({"IID" if alpha >= 10 else "Non-IID"})'
        ax_acc.plot(results[alpha]['accuracies'], label=label, linewidth=2)
    ax_acc.set_xlabel('Communication Round')
    ax_acc.set_ylabel('Test Accuracy (%)')
    ax_acc.set_title('Impact of Data Heterogeneity on Model Performance')
    ax_acc.legend()
    ax_acc.grid(True, alpha=0.3)
    
    # Third row: Loss curves
    ax_loss = fig.add_subplot(gs[2, :])
    for alpha in alpha_values:
        label = f'α={alpha}'
        ax_loss.plot(results[alpha]['losses'], label=label, linewidth=2)
    ax_loss.set_xlabel('Communication Round')
    ax_loss.set_ylabel('Test Loss')
    ax_loss.set_title('Loss Convergence vs Data Heterogeneity')
    ax_loss.legend()
    ax_loss.grid(True, alpha=0.3)
    
    # Fourth row: Divergence
    ax_div = fig.add_subplot(gs[3, :])
    for alpha in alpha_values:
        label = f'α={alpha}'
        ax_div.plot(results[alpha]['divergences'], label=label, linewidth=2)
    ax_div.set_xlabel('Communication Round')
    ax_div.set_ylabel('Weight Divergence')
    ax_div.set_title('Client Drift vs Data Heterogeneity')
    ax_div.legend()
    ax_div.grid(True, alpha=0.3)
    
    plt.savefig('task3_heterogeneity.png', dpi=300, bbox_inches='tight')
    print(f"✅ Plot saved as 'task3_heterogeneity.png'")
    
    # Summary table
    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")
    print(f"{'Alpha':>10} | {'Type':>10} | {'Final Acc':>10} | {'Best Acc':>10} | {'Avg Div':>10}")
    print("-" * 70)
    for alpha in alpha_values:
        data_type = "IID" if alpha >= 10 else "Non-IID"
        final_acc = results[alpha]['accuracies'][-1]
        best_acc = max(results[alpha]['accuracies'])
        avg_div = np.mean(results[alpha]['divergences'])
        print(f"{alpha:>10.2f} | {data_type:>10} | {final_acc:>9.2f}% | {best_acc:>9.2f}% | {avg_div:>10.4f}")
    
    return results, distributions, all_results


if __name__ == "__main__":
    results, distributions, all_results = experiment_heterogeneity(
        num_rounds=100,
        num_clients=5,
        local_epochs=5,
        alpha_values=[100, 1.0, 0.2, 0.05],
        lr=0.01,
        dataset='cifar10',
        num_classes=10,
        save_checkpoints=True,
        checkpoint_interval=20
    )