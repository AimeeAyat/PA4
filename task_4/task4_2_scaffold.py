"""
Task 4.2: SCAFFOLD - Control Variates for Client Drift
Implements SCAFFOLD algorithm using control variates to correct client drift
Reference: Karimireddy et al., ICML 2020
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


class SCAFFOLDTrainer:
    """
    SCAFFOLD trainer with control variates
    
    Key idea: Use control variates c_global and c_local[i] to correct client drift
    Gradient adjustment: grad = grad + (c_local[i] - c_global)
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
                num_workers=6
            )
            self.client_sizes[client_id] = len(indices)
        
        self.test_loader = DataLoader(
            test_dataset, batch_size=256, shuffle=False,
            pin_memory=True if device.type == 'cuda' else False,
            num_workers=6
        )
        self.criterion = nn.CrossEntropyLoss()
        
        # Initialize control variates (zero initially)
        # c_global: global control variate (same shape as model parameters)
        self.c_global = [torch.zeros_like(param).to(device) for param in model.parameters()]
        
        # c_local: local control variate for each client
        self.c_local = {}
        for client_id in range(self.num_clients):
            self.c_local[client_id] = [torch.zeros_like(param).to(device) 
                                       for param in model.parameters()]
        
        # Track best model
        self.best_accuracy = 0.0
        self.best_model_state = None
    
    def local_update(self, client_id, global_params):
        """
        Perform K local epochs with SCAFFOLD control variate correction
        
        Key: Adjust gradient by (c_local[i] - c_global) at each step
        """
        set_model_params(self.model, global_params)
        self.model.to(self.device)
        self.model.train()
        
        # Store initial global params for control update
        theta_global = [param.detach().clone().to(self.device) for param in global_params]
        
        # Get client's control variate and global control variate
        c_i = self.c_local[client_id]
        c_global = self.c_global
        
        # Compute control difference (c_i - c_global) once
        c_diff = [c_i_param - c_g_param for c_i_param, c_g_param in zip(c_i, c_global)]
        
        optimizer = torch.optim.SGD(self.model.parameters(), lr=self.lr)
        
        local_losses = []
        total_steps = 0
        
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
                
                # Backward pass (compute standard gradients)
                loss.backward()
                
                # SCAFFOLD: Apply control variate correction to gradients
                # grad = grad + (c_local[i] - c_global)
                with torch.no_grad():
                    for param, correction in zip(self.model.parameters(), c_diff):
                        if param.grad is not None:
                            param.grad.data += correction
                
                # Update weights with corrected gradients
                optimizer.step()
                
                epoch_loss += loss.item()
                num_batches += 1
                total_steps += 1
            
            avg_epoch_loss = epoch_loss / num_batches
            local_losses.append(avg_epoch_loss)
        
        # Get final local model parameters
        theta_local = get_model_params(self.model)
        
        # Update client's control variate
        # c_i^{t+1} = c_global + (1 / (K * lr)) * (theta_global - theta_local)
        # Equivalent to: c_i^{t+1} = c_i - (1 / (K * lr)) * (theta_global - theta_local)
        c_i_new = []
        K = self.local_epochs
        num_steps = total_steps
        
        for c_g, theta_g, theta_l in zip(c_global, theta_global, theta_local):
            # Option a: c_i^{t+1} = c_i - (c_global - c_i) - (1/(K*lr)) * (theta_global - theta_local)
            # Simplified: c_i^{t+1} = c_global + (1/(num_steps * lr)) * (theta_l - theta_g)
            delta_theta = theta_l - theta_g
            c_i_new_param = c_g - (1.0 / (num_steps * self.lr)) * delta_theta
            c_i_new.append(c_i_new_param)
        
        # Update client's control variate
        self.c_local[client_id] = c_i_new
        
        return theta_local, c_i_new, local_losses
    
    def train_round(self):
        """Execute one round of SCAFFOLD with all clients"""
        global_params = get_model_params(self.model)
        
        # Collect updates from all clients
        client_params = []
        client_controls = []
        client_models = []
        sizes = []
        all_local_losses = []
        
        for client_id in range(self.num_clients):
            params, c_i_new, local_losses = self.local_update(client_id, global_params)
            client_params.append(params)
            client_controls.append(c_i_new)
            client_models.append(params)
            sizes.append(self.client_sizes[client_id])
            all_local_losses.append(local_losses)
        
        # Compute weight divergence before aggregation
        divergence = compute_weight_divergence(client_models, global_params)
        
        # Weighted average of model parameters
        avg_params = average_weights(client_params, sizes)
        set_model_params(self.model, avg_params)
        
        # Update global control variate
        # c_global^{t+1} = (1/M) * sum(c_i^{t+1})
        # This is the average of all client control variates
        c_global_new = []
        for idx in range(len(self.c_global)):
            c_avg = torch.zeros_like(self.c_global[idx])
            total_size = sum(sizes)
            
            for client_id in range(self.num_clients):
                weight = sizes[client_id] / total_size
                c_avg += weight * client_controls[client_id][idx]
            
            c_global_new.append(c_avg)
        
        self.c_global = c_global_new
        
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
                'loss': loss,
                'c_global': [c.cpu().clone() for c in self.c_global],
                'c_local': {cid: [c.cpu().clone() for c in cv] 
                           for cid, cv in self.c_local.items()}
            }
        
        return acc, loss
    
    def save_best_model(self, save_path):
        """Save the best model encountered during training"""
        if self.best_model_state is not None:
            torch.save(self.best_model_state, save_path)
            return True
        return False


def compare_with_scaffold(num_rounds=30, num_clients=5, local_epochs=5,
                          alpha=0.1, lr=0.01, dataset='cifar10', use_double=True):
    """
    Compare FedAvg, FedProx, and SCAFFOLD
    
    Args:
        num_rounds: Number of communication rounds (max 30)
        alpha: Dirichlet parameter (0.1 for high heterogeneity)
    """
    set_seed(42)
    device = get_device()
    
    if use_double:
        torch.set_default_dtype(torch.float64)
    
    print(f"\n{'='*70}")
    print("TASK 4.2: SCAFFOLD - Control Variates for Client Drift")
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
    print(f"Double precision: {use_double}\n")
    
    # Create checkpoint directory
    os.makedirs('checkpoints/task4_2_scaffold', exist_ok=True)
    
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
        "experiment": "scaffold_comparison",
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
    
    # Train SCAFFOLD
    print(f"\n{'='*60}")
    print(f"Training: SCAFFOLD")
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
    print(f"Communication overhead: 2x (model + control variates)\n")
    
    # Initialize trainer
    trainer = SCAFFOLDTrainer(
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
    checkpoint_path = "checkpoints/task4_2_scaffold/best_model_scaffold.pt"
    saved = trainer.save_best_model(checkpoint_path)
    
    print(f"\n✓ Training completed in {total_time:.2f}s ({total_time/60:.2f} min)")
    print(f"  Final Accuracy: {accuracies[-1]:.2f}%")
    print(f"  Best Accuracy: {trainer.best_accuracy:.2f}%")
    print(f"  Final Test Loss: {test_losses[-1]:.4f}")
    print(f"  Average Divergence: {np.mean(divergences):.4f}")
    if saved:
        print(f"  Best model saved: {checkpoint_path}")
    
    # Store results
    results['SCAFFOLD'] = {
        'accuracies': accuracies,
        'test_losses': test_losses,
        'train_losses': train_losses,
        'divergences': divergences,
        'round_times': round_times
    }
    
    all_results["results"]["SCAFFOLD"] = {
        "method": "SCAFFOLD",
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
    
    # Clear GPU cache
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        print(f"  GPU memory cleared")
    
    # Save results to JSON
    json_path = 'task4_2_scaffold_results.json'
    with open(json_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\n✅ Results saved to '{json_path}'")
    
    # Create plots
    plot_scaffold_results(results, alpha, num_rounds)
    
    # Summary
    print(f"\n{'='*70}")
    print("SUMMARY: SCAFFOLD Performance")
    print(f"{'='*70}")
    print(f"Final Accuracy: {accuracies[-1]:.2f}%")
    print(f"Best Accuracy: {trainer.best_accuracy:.2f}%")
    print(f"Final Test Loss: {test_losses[-1]:.4f}")
    print(f"Average Divergence: {np.mean(divergences):.4f}")
    print(f"Communication overhead: 2x per round (model + control variates)")
    
    return results, all_results


def plot_scaffold_results(results, alpha, num_rounds):
    """
    Create comprehensive plots for SCAFFOLD
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    rounds = np.arange(1, num_rounds + 1)
    
    scaffold_data = results['SCAFFOLD']
    
    # Test Accuracy
    axes[0, 0].plot(rounds, scaffold_data['accuracies'], 
                   label='SCAFFOLD', color='#2ca02c', linewidth=2.5, alpha=0.8)
    axes[0, 0].set_title('Test Accuracy (%)', fontsize=14, fontweight='bold')
    axes[0, 0].set_xlabel('Communication Round', fontsize=12)
    axes[0, 0].set_ylabel('Accuracy (%)', fontsize=12)
    axes[0, 0].legend(fontsize=10)
    axes[0, 0].grid(True, alpha=0.3, linestyle='--')
    
    # Test Loss
    axes[0, 1].plot(rounds, scaffold_data['test_losses'],
                   label='SCAFFOLD', color='#2ca02c', linewidth=2.5, alpha=0.8)
    axes[0, 1].set_title('Test Loss', fontsize=14, fontweight='bold')
    axes[0, 1].set_xlabel('Communication Round', fontsize=12)
    axes[0, 1].set_ylabel('Loss', fontsize=12)
    axes[0, 1].legend(fontsize=10)
    axes[0, 1].grid(True, alpha=0.3, linestyle='--')
    
    # Train Loss (local)
    axes[1, 0].plot(rounds, scaffold_data['train_losses'],
                   label='SCAFFOLD', color='#2ca02c', linewidth=2.5, alpha=0.8)
    axes[1, 0].set_title('Training Loss (Local Avg)', fontsize=14, fontweight='bold')
    axes[1, 0].set_xlabel('Communication Round', fontsize=12)
    axes[1, 0].set_ylabel('Loss', fontsize=12)
    axes[1, 0].legend(fontsize=10)
    axes[1, 0].grid(True, alpha=0.3, linestyle='--')
    
    # Client Divergence
    axes[1, 1].plot(rounds, scaffold_data['divergences'],
                   label='SCAFFOLD', color='#2ca02c', linewidth=2.5, alpha=0.8)
    axes[1, 1].set_title('Client Drift (Weight Divergence)', fontsize=14, fontweight='bold')
    axes[1, 1].set_xlabel('Communication Round', fontsize=12)
    axes[1, 1].set_ylabel('L2 Divergence', fontsize=12)
    axes[1, 1].legend(fontsize=10)
    axes[1, 1].grid(True, alpha=0.3, linestyle='--')
    
    plt.suptitle(f'SCAFFOLD: Data Heterogeneity (α={alpha})', 
                 fontsize=16, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    
    plot_path = 'task4_2_scaffold.png'
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ Plot saved as '{plot_path}'")


if __name__ == "__main__":
    print(f"\nStarting Task 4.2 @ {time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    results, all_results = compare_with_scaffold(
        num_rounds=30,
        num_clients=5,
        local_epochs=5,
        alpha=0.1,  # High heterogeneity
        lr=0.01,
        dataset='cifar10',
        use_double=True
    )
    
    print("\n" + "="*70)
    print("Task 4.2 Completed!")
    print("="*70)
    print("\nGenerated files:")
    print("  → task4_2_scaffold_results.json (complete metrics)")
    print("  → task4_2_scaffold.png (comparison plots)")
    print("  → checkpoints/task4_2_scaffold/best_model_scaffold.pt (best model)")
    print("\nNext: Compare SCAFFOLD vs FedAvg vs FedProx")

