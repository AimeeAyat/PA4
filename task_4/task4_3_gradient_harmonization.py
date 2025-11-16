"""
Task 4.3: Gradient Harmonization (FedGH)
Implements FedGH algorithm to harmonize conflicting client gradients
Reference: Recent 2023 research on gradient harmonization
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


def flatten_params(params):
    """Flatten list of parameter tensors into single vector"""
    return torch.cat([p.flatten() for p in params])


def unflatten_params(flat_params, reference_params):
    """Unflatten vector back to list of tensors matching reference structure"""
    unflat = []
    idx = 0
    for param in reference_params:
        numel = param.numel()
        unflat.append(flat_params[idx:idx+numel].view(param.shape))
        idx += numel
    return unflat


def harmonize_gradients(client_updates, client_weights=None, device='cpu'):
    """
    Harmonize conflicting gradients using pairwise projection
    
    Args:
        client_updates: List of client update vectors (each is list of tensors)
        client_weights: Optional weights for each client (e.g., data sizes)
        device: Device to perform computations on
        
    Returns:
        harmonized_updates: List of harmonized update vectors
        num_conflicts: Number of conflicts detected and resolved
    """
    num_clients = len(client_updates)
    
    if num_clients <= 1:
        return client_updates, 0
    
    # Flatten all client updates to vectors
    flat_updates = []
    for update in client_updates:
        flat = flatten_params(update).to(device)
        flat_updates.append(flat)
    
    # Convert to tensor for easier manipulation [num_clients, param_dim]
    updates_tensor = torch.stack(flat_updates)
    
    # Track conflicts
    num_conflicts = 0
    conflict_pairs = []
    
    # Find all conflicting pairs (negative dot product)
    for i in range(num_clients):
        for j in range(i + 1, num_clients):
            g_i = updates_tensor[i]
            g_j = updates_tensor[j]
            
            # Compute dot product
            dot_product = torch.dot(g_i, g_j).item()
            
            # If negative, gradients are in conflict (angle > 90°)
            if dot_product < 0:
                num_conflicts += 1
                conflict_pairs.append((i, j, dot_product))
    
    # Harmonize conflicting pairs
    if num_conflicts > 0:
        # Process conflicts (we'll modify in place)
        for i, j, dot_prod in conflict_pairs:
            g_i = updates_tensor[i].clone()
            g_j = updates_tensor[j].clone()
            
            # Compute norms
            norm_i_sq = torch.sum(g_i ** 2)
            norm_j_sq = torch.sum(g_j ** 2)
            
            # Avoid division by zero
            if norm_i_sq > 1e-10 and norm_j_sq > 1e-10:
                # Project g_i onto orthogonal complement of g_j
                # g_i' = g_i - ((g_i · g_j) / ||g_j||^2) * g_j
                dot_ij = torch.dot(g_i, g_j)
                projection_i = (dot_ij / norm_j_sq) * g_j
                g_i_new = g_i - projection_i
                
                # Project g_j onto orthogonal complement of g_i
                # g_j' = g_j - ((g_i · g_j) / ||g_i||^2) * g_i
                projection_j = (dot_ij / norm_i_sq) * g_i
                g_j_new = g_j - projection_j
                
                # Update in tensor
                updates_tensor[i] = g_i_new
                updates_tensor[j] = g_j_new
    
    # Unflatten back to original structure
    harmonized_updates = []
    reference = client_updates[0]
    
    for flat_update in updates_tensor:
        unflat = unflatten_params(flat_update, reference)
        harmonized_updates.append(unflat)
    
    return harmonized_updates, num_conflicts


class FedGHTrainer:
    """
    FedAvg with Gradient Harmonization
    
    Key: Harmonize conflicting client updates before aggregation
    """
    
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
                client_subset, batch_size=batch_size, shuffle=True,
                pin_memory=True if device.type == 'cuda' else False,
                num_workers=0
            )
            self.client_sizes[client_id] = len(indices)
        
        self.test_loader = DataLoader(
            test_dataset, batch_size=256, shuffle=False,
            pin_memory=True if device.type == 'cuda' else False,
            num_workers=0
        )
        self.criterion = nn.CrossEntropyLoss()
        
        # Track best model
        self.best_accuracy = 0.0
        self.best_model_state = None
    
    def local_update(self, client_id, global_params):
        """
        Perform K local epochs of standard FedAvg training
        """
        set_model_params(self.model, global_params)
        self.model.to(self.device)
        self.model.train()
        
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
                loss = self.criterion(output, target)
                
                # Backward pass
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item()
                num_batches += 1
            
            avg_epoch_loss = epoch_loss / num_batches
            local_losses.append(avg_epoch_loss)
        
        return get_model_params(self.model), local_losses
    
    def train_round(self):
        """
        Execute one round of FedAvg with Gradient Harmonization
        """
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
        
        # Compute weight divergence before harmonization
        divergence_before = compute_weight_divergence(client_models, global_params)
        
        # Compute client updates (deltas from global model)
        client_updates = []
        for params in client_params:
            update = [client_param - global_param 
                     for client_param, global_param in zip(params, global_params)]
            client_updates.append(update)
        
        # GRADIENT HARMONIZATION: Resolve conflicts
        harmonized_updates, num_conflicts = harmonize_gradients(
            client_updates, client_weights=sizes, device=self.device
        )
        
        # Reconstruct client params from harmonized updates
        harmonized_params = []
        for update in harmonized_updates:
            params = [global_param + upd 
                     for global_param, upd in zip(global_params, update)]
            harmonized_params.append(params)
        
        # Weighted average of harmonized parameters
        avg_params = average_weights(harmonized_params, sizes)
        set_model_params(self.model, avg_params)
        
        # Compute divergence after harmonization
        divergence_after = compute_weight_divergence(harmonized_params, global_params)
        
        # Average local losses across clients
        avg_local_loss = np.mean([np.mean(losses) for losses in all_local_losses])
        
        return {
            'divergence': divergence_after,
            'divergence_before': divergence_before,
            'avg_local_loss': avg_local_loss,
            'num_conflicts': num_conflicts
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


def compare_with_harmonization(num_rounds=30, num_clients=5, local_epochs=5,
                                alpha=0.05, lr=0.01, dataset='cifar10', use_double=True):
    """
    Compare FedAvg with Gradient Harmonization
    
    Args:
        num_rounds: Number of communication rounds (max 30)
        alpha: Dirichlet parameter (0.05 for very high heterogeneity)
    """
    set_seed(42)
    device = get_device()
    
    if use_double:
        torch.set_default_dtype(torch.float64)
    
    print(f"\n{'='*70}")
    print("TASK 4.3: Gradient Harmonization (FedGH)")
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
    print(f"Data heterogeneity: α={alpha} (Dirichlet - very high)")
    print(f"Double precision: {use_double}\n")
    
    # Create checkpoint directory
    os.makedirs('checkpoints/task4_3_fedgh', exist_ok=True)
    
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
    print(f"Data distribution (α={alpha}, extremely heterogeneous):")
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
        "experiment": "fedgh_comparison",
        "dataset": dataset,
        "num_clients": num_clients,
        "num_rounds": num_rounds,
        "local_epochs": local_epochs,
        "learning_rate": lr,
        "alpha": alpha,
        "use_double": use_double,
        "results": {}
    }
    
    results = {}
    
    # Train FedGH (FedAvg + Gradient Harmonization)
    print(f"\n{'='*60}")
    print(f"Training: FedGH (FedAvg + Gradient Harmonization)")
    print(f"{'='*60}")
    
    # Initialize model
    set_seed(42)
    model = SimpleCNN(num_classes=num_classes, input_channels=input_channels)
    if use_double:
        model = model.double()
    model = model.to(device)
    
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {num_params:,}")
    print(f"Model device: {next(model.parameters()).device}")
    print(f"Harmonization: O(M²) pairwise gradient conflict resolution\n")
    
    # Initialize trainer
    trainer = FedGHTrainer(
        model=model,
        train_dataset=train_data,
        test_dataset=test_data,
        client_dict=client_dict,
        local_epochs=local_epochs,
        lr=lr,
        batch_size=64,
        device=device
    )
    
    # Training metrics
    accuracies = []
    test_losses = []
    train_losses = []
    divergences = []
    divergences_before = []
    conflicts = []
    round_times = []
    
    start_time = time.time()
    
    # Training loop
    print(f"Round | Test Acc | Test Loss | Train Loss | Conflicts | Div Before | Div After | Time")
    print("-" * 95)
    
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
        divergences_before.append(round_info['divergence_before'])
        conflicts.append(round_info['num_conflicts'])
        round_times.append(time.time() - round_start)
        
        # Print progress
        if (round_idx + 1) % 5 == 0 or round_idx < 3:
            print(f"{round_idx+1:5d} | {test_acc:7.2f}% | {test_loss:9.4f} | "
                  f"{round_info['avg_local_loss']:10.4f} | {round_info['num_conflicts']:9d} | "
                  f"{round_info['divergence_before']:10.4f} | {round_info['divergence']:9.4f} | "
                  f"{round_times[-1]:4.2f}s")
    
    total_time = time.time() - start_time
    
    # Save best model checkpoint
    checkpoint_path = "checkpoints/task4_3_fedgh/best_model_fedgh.pt"
    saved = trainer.save_best_model(checkpoint_path)
    
    print(f"\n✓ Training completed in {total_time:.2f}s ({total_time/60:.2f} min)")
    print(f"  Final Accuracy: {accuracies[-1]:.2f}%")
    print(f"  Best Accuracy: {trainer.best_accuracy:.2f}%")
    print(f"  Final Test Loss: {test_losses[-1]:.4f}")
    print(f"  Average Divergence: {np.mean(divergences):.4f}")
    print(f"  Total Conflicts Resolved: {sum(conflicts)}")
    print(f"  Average Conflicts per Round: {np.mean(conflicts):.1f}")
    if saved:
        print(f"  Best model saved: {checkpoint_path}")
    
    # Store results
    results['FedGH'] = {
        'accuracies': accuracies,
        'test_losses': test_losses,
        'train_losses': train_losses,
        'divergences': divergences,
        'divergences_before': divergences_before,
        'conflicts': conflicts,
        'round_times': round_times
    }
    
    all_results["results"]["FedGH"] = {
        "method": "FedGH (Gradient Harmonization)",
        "final_accuracy": float(accuracies[-1]),
        "best_accuracy": float(trainer.best_accuracy),
        "final_test_loss": float(test_losses[-1]),
        "avg_divergence": float(np.mean(divergences)),
        "avg_divergence_before_harmonization": float(np.mean(divergences_before)),
        "total_conflicts": int(sum(conflicts)),
        "avg_conflicts_per_round": float(np.mean(conflicts)),
        "total_time_sec": float(total_time),
        "accuracies": [float(x) for x in accuracies],
        "test_losses": [float(x) for x in test_losses],
        "train_losses": [float(x) for x in train_losses],
        "divergences": [float(x) for x in divergences],
        "divergences_before": [float(x) for x in divergences_before],
        "conflicts": [int(x) for x in conflicts]
    }
    
    # Clear GPU cache
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        print(f"  GPU memory cleared")
    
    # Save results to JSON
    json_path = 'task4_3_fedgh_results.json'
    with open(json_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\n✅ Results saved to '{json_path}'")
    
    # Create plots
    plot_fedgh_results(results, alpha, num_rounds)
    
    # Summary
    print(f"\n{'='*70}")
    print("SUMMARY: FedGH Performance")
    print(f"{'='*70}")
    print(f"Final Accuracy: {accuracies[-1]:.2f}%")
    print(f"Best Accuracy: {trainer.best_accuracy:.2f}%")
    print(f"Final Test Loss: {test_losses[-1]:.4f}")
    print(f"Average Divergence (after harmonization): {np.mean(divergences):.4f}")
    print(f"Average Divergence (before harmonization): {np.mean(divergences_before):.4f}")
    print(f"Divergence Reduction: {(np.mean(divergences_before) - np.mean(divergences)):.4f}")
    print(f"Total Conflicts Resolved: {sum(conflicts)}")
    print(f"Conflict Resolution Rate: {sum(conflicts) / (num_rounds * num_clients * (num_clients-1) / 2) * 100:.1f}%")
    
    return results, all_results


def plot_fedgh_results(results, alpha, num_rounds):
    """
    Create comprehensive plots for FedGH
    """
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    rounds = np.arange(1, num_rounds + 1)
    
    fedgh_data = results['FedGH']
    
    # Test Accuracy
    axes[0, 0].plot(rounds, fedgh_data['accuracies'], 
                   label='FedGH', color='#ff7f0e', linewidth=2.5, alpha=0.8)
    axes[0, 0].set_title('Test Accuracy (%)', fontsize=14, fontweight='bold')
    axes[0, 0].set_xlabel('Communication Round', fontsize=12)
    axes[0, 0].set_ylabel('Accuracy (%)', fontsize=12)
    axes[0, 0].legend(fontsize=10)
    axes[0, 0].grid(True, alpha=0.3, linestyle='--')
    
    # Test Loss
    axes[0, 1].plot(rounds, fedgh_data['test_losses'],
                   label='FedGH', color='#ff7f0e', linewidth=2.5, alpha=0.8)
    axes[0, 1].set_title('Test Loss', fontsize=14, fontweight='bold')
    axes[0, 1].set_xlabel('Communication Round', fontsize=12)
    axes[0, 1].set_ylabel('Loss', fontsize=12)
    axes[0, 1].legend(fontsize=10)
    axes[0, 1].grid(True, alpha=0.3, linestyle='--')
    
    # Train Loss (local)
    axes[0, 2].plot(rounds, fedgh_data['train_losses'],
                   label='FedGH', color='#ff7f0e', linewidth=2.5, alpha=0.8)
    axes[0, 2].set_title('Training Loss (Local Avg)', fontsize=14, fontweight='bold')
    axes[0, 2].set_xlabel('Communication Round', fontsize=12)
    axes[0, 2].set_ylabel('Loss', fontsize=12)
    axes[0, 2].legend(fontsize=10)
    axes[0, 2].grid(True, alpha=0.3, linestyle='--')
    
    # Client Divergence (Before vs After Harmonization)
    axes[1, 0].plot(rounds, fedgh_data['divergences_before'],
                   label='Before Harmonization', color='#d62728', linewidth=2, alpha=0.7, linestyle='--')
    axes[1, 0].plot(rounds, fedgh_data['divergences'],
                   label='After Harmonization', color='#2ca02c', linewidth=2.5, alpha=0.8)
    axes[1, 0].set_title('Client Drift (Harmonization Effect)', fontsize=14, fontweight='bold')
    axes[1, 0].set_xlabel('Communication Round', fontsize=12)
    axes[1, 0].set_ylabel('L2 Divergence', fontsize=12)
    axes[1, 0].legend(fontsize=10)
    axes[1, 0].grid(True, alpha=0.3, linestyle='--')
    
    # Number of Conflicts per Round
    axes[1, 1].bar(rounds, fedgh_data['conflicts'], color='#9467bd', alpha=0.7, width=0.8)
    axes[1, 1].axhline(y=np.mean(fedgh_data['conflicts']), color='r', linestyle='--', 
                      linewidth=2, label=f'Avg: {np.mean(fedgh_data["conflicts"]):.1f}')
    axes[1, 1].set_title('Gradient Conflicts Detected', fontsize=14, fontweight='bold')
    axes[1, 1].set_xlabel('Communication Round', fontsize=12)
    axes[1, 1].set_ylabel('Number of Conflicts', fontsize=12)
    axes[1, 1].legend(fontsize=10)
    axes[1, 1].grid(True, alpha=0.3, linestyle='--', axis='y')
    
    # Cumulative Conflicts
    cumulative_conflicts = np.cumsum(fedgh_data['conflicts'])
    axes[1, 2].plot(rounds, cumulative_conflicts, color='#8c564b', linewidth=2.5, alpha=0.8)
    axes[1, 2].fill_between(rounds, 0, cumulative_conflicts, color='#8c564b', alpha=0.3)
    axes[1, 2].set_title('Cumulative Conflicts Resolved', fontsize=14, fontweight='bold')
    axes[1, 2].set_xlabel('Communication Round', fontsize=12)
    axes[1, 2].set_ylabel('Total Conflicts', fontsize=12)
    axes[1, 2].grid(True, alpha=0.3, linestyle='--')
    
    plt.suptitle(f'FedGH: Gradient Harmonization (α={alpha})', 
                 fontsize=16, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    
    plot_path = 'task4_3_fedgh.png'
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ Plot saved as '{plot_path}'")


if __name__ == "__main__":
    print(f"\nStarting Task 4.3 @ {time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    results, all_results = compare_with_harmonization(
        num_rounds=30,
        num_clients=5,
        local_epochs=5,
        alpha=0.05,  # Very high heterogeneity (worse than 0.1)
        lr=0.01,
        dataset='cifar10',
        use_double=True
    )
    
    print("\n" + "="*70)
    print("Task 4.3 Completed!")
    print("="*70)
    print("\nGenerated files:")
    print("  → task4_3_fedgh_results.json (complete metrics + conflicts)")
    print("  → task4_3_fedgh.png (6-panel comparison plots)")
    print("  → checkpoints/task4_3_fedgh/best_model_fedgh.pt (best model)")
    print("\nNext: Compare FedGH vs FedAvg vs FedProx vs SCAFFOLD")

