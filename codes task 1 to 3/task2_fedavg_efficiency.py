"""
Task 2: FedAvg Implementation - COMPLETE VERSION
Addresses all requirements: drift, regret, communication cost, stability analysis
"""
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
import numpy as np
import matplotlib.pyplot as plt
import time
import json
from utils import (SimpleCNN, get_device, set_seed, split_data_iid,
                   get_model_params, set_model_params, average_weights, 
                   evaluate_model, compute_weight_divergence)


class FedAvgTrainer:
    """Federated Averaging trainer - Research-grade implementation"""
    
    def __init__(self, model, train_dataset, test_dataset, num_clients=10,
                 local_epochs=5, local_steps=None, lr=0.01, batch_size=64, 
                 momentum=0.0, weight_decay=0.0, device='cpu', seed=42):
        self.model = model
        self.num_clients = num_clients
        self.local_epochs = local_epochs
        self.local_steps = local_steps
        self.lr = lr
        self.batch_size = batch_size
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.device = device
        self.seed = seed
        
        # Split data IID
        self.client_dict = split_data_iid(train_dataset, num_clients)
        self.client_loaders = {}
        self.client_sizes = {}
        
        for client_id in range(num_clients):
            client_subset = Subset(train_dataset, self.client_dict[client_id])
            # CRITICAL: shuffle=False for deterministic local epochs
            self.client_loaders[client_id] = DataLoader(
                client_subset, batch_size=batch_size, shuffle=False
            )
            self.client_sizes[client_id] = len(client_subset)
        
        self.test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False)
        self.criterion = nn.CrossEntropyLoss()
        self.total_data = sum(self.client_sizes.values())
        
        # Track best loss for regret calculation
        self.best_test_loss = float('inf')
        
    def local_update(self, client_id, global_params, round_idx):
        """Perform K local epochs with fresh optimizer"""
        set_model_params(self.model, global_params)
        self.model.train()
        
        # Fresh optimizer per round (critical!)
        optimizer = torch.optim.SGD(
            self.model.parameters(), 
            lr=self.lr, 
            momentum=self.momentum,
            weight_decay=self.weight_decay
        )
        
        steps = 0
        max_steps = self.local_steps or (self.local_epochs * len(self.client_loaders[client_id]))
        
        for epoch in range(999):  # safety
            for data, target in self.client_loaders[client_id]:
                if steps >= max_steps:
                    break
                data, target = data.to(self.device), target.to(self.device)
                
                optimizer.zero_grad()
                output = self.model(data)
                loss = self.criterion(output, target)
                loss.backward()
                optimizer.step()
                steps += 1
            if steps >= max_steps:
                break
                
        return get_model_params(self.model)
    
    def train_round(self, client_fraction=1.0, round_idx=0):
        """One FedAvg round with sampling"""
        set_seed(self.seed + round_idx)  # deterministic per round
        
        global_params = get_model_params(self.model)
        
        num_selected = max(1, int(client_fraction * self.num_clients))
        selected_clients = np.random.choice(
            self.num_clients, num_selected, replace=False
        )
        
        client_params = []
        client_models = []
        sizes = []
        
        for client_id in selected_clients:
            params = self.local_update(client_id, global_params, round_idx)
            client_params.append(params)
            client_models.append(params)
            sizes.append(self.client_sizes[client_id])
        
        # Client drift (before aggregation)
        divergence = compute_weight_divergence(client_models, global_params)
        
        # Weighted average
        avg_params = average_weights(client_params, sizes)
        set_model_params(self.model, avg_params)
        
        return {
            'divergence': divergence,
            'selected_clients': len(selected_clients),
            'total_clients': self.num_clients
        }
    
    def evaluate(self):
        """Evaluate and track best loss for regret"""
        acc, loss = evaluate_model(self.model, self.test_loader, self.device)
        self.best_test_loss = min(self.best_test_loss, loss)
        return acc, loss
    
    def compute_regret(self, current_loss):
        """
        Regret metric: R_t = L(θ_t) - L(θ*)
        We approximate L(θ*) as the best loss seen so far
        """
        return current_loss - self.best_test_loss


def compute_stability_metrics(accuracies, losses, window=10):
    """
    Compute stability metrics to detect oscillations and convergence
    
    Returns:
        - variance: variance in accuracy over last 'window' rounds
        - oscillation_count: number of sign changes in accuracy gradient
        - convergence_rate: exponential moving average decay rate
    """
    acc_array = np.array(accuracies)
    loss_array = np.array(losses)
    
    # Variance in recent rounds (higher = more unstable)
    recent_acc_var = np.var(acc_array[-window:]) if len(acc_array) >= window else 0.0
    recent_loss_var = np.var(loss_array[-window:]) if len(loss_array) >= window else 0.0
    
    # Count oscillations (sign changes in gradient)
    acc_diff = np.diff(acc_array)
    oscillations = np.sum(np.diff(np.sign(acc_diff)) != 0) if len(acc_diff) > 1 else 0
    
    # Measure convergence rate (fit exponential to recent loss)
    if len(loss_array) >= window:
        x = np.arange(window)
        y = loss_array[-window:]
        # Simple linear fit to log(loss) gives decay rate
        try:
            coeffs = np.polyfit(x, y, 1)
            convergence_rate = abs(coeffs[0])  # slope magnitude
        except:
            convergence_rate = 0.0
    else:
        convergence_rate = 0.0
    
    return {
        'accuracy_variance': float(recent_acc_var),
        'loss_variance': float(recent_loss_var),
        'oscillation_count': int(oscillations),
        'convergence_rate': float(convergence_rate)
    }


def experiment_local_epochs(num_rounds=100, num_clients=10, K_values=[1, 5, 10, 20],
                            lr=0.01, dataset='cifar10', use_double=False):
    set_seed(42)
    device = get_device()
    if use_double:
        torch.set_default_dtype(torch.float64)
    
    # Load dataset
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    train_data = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    test_data = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    
    print(f"\n{'='*70}")
    print("EXPERIMENT 1: Local Epochs (K) - FedAvg")
    print(f"{'='*70}")
    
    results = {
        "experiment": "local_epochs",
        "dataset": dataset,
        "num_clients": num_clients,
        "num_rounds": num_rounds,
        "lr": lr,
        "use_double": use_double,
        "model_params": sum(p.numel() for p in SimpleCNN(10, 3).parameters()),
        "total_train_samples": len(train_data),
        "configs": {}
    }
    
    for K in K_values:
        print(f"\nRunning K={K}...")
        set_seed(42)
        model = SimpleCNN(10, 3).to(device)
        if use_double:
            model = model.double()
            
        trainer = FedAvgTrainer(
            model, train_data, test_data, num_clients=num_clients,
            local_epochs=K, lr=lr, batch_size=64, device=device, seed=42
        )
        
        stats = {
            'accuracies': [], 'losses': [], 'divergences': [],
            'regrets': [], 'round_times': [], 'selected_clients': []
        }
        
        start_time = time.time()
        for r in range(num_rounds):
            round_start = time.time()
            round_info = trainer.train_round(client_fraction=1.0, round_idx=r)
            acc, loss = trainer.evaluate()
            
            # Compute regret
            regret = trainer.compute_regret(loss)
            
            stats['accuracies'].append(acc)
            stats['losses'].append(loss)
            stats['divergences'].append(round_info['divergence'])
            stats['regrets'].append(regret)
            stats['selected_clients'].append(round_info['selected_clients'])
            stats['round_times'].append(time.time() - round_start)
            
            if (r+1) % 20 == 0:
                print(f"  Round {r+1:3d} | Acc: {acc:.2f}% | Loss: {loss:.4f} | "
                      f"Drift: {round_info['divergence']:.4f} | Regret: {regret:.4f}")
        
        total_time = time.time() - start_time
        
        # Compute stability metrics
        stability = compute_stability_metrics(stats['accuracies'], stats['losses'])
        
        stats['total_time_sec'] = total_time
        stats['avg_round_time'] = np.mean(stats['round_times'])
        stats['final_accuracy'] = stats['accuracies'][-1]
        stats['final_regret'] = stats['regrets'][-1]
        stats['stability_metrics'] = stability
        
        results["configs"][f"K={K}"] = stats
        print(f"  K={K} → Final Acc: {stats['final_accuracy']:.2f}% | "
              f"Regret: {stats['final_regret']:.4f} | "
              f"Oscillations: {stability['oscillation_count']} | "
              f"Time: {total_time:.1f}s")
    
    # Save JSON
    with open("task2_local_epochs_results.json", "w") as f:
        json.dump(results, f, indent=2)
    
    # Enhanced plot with regret and stability
    plot_fedavg_comprehensive(results, experiment_type="local_epochs", 
                             title_suffix=f" - K ∈ {K_values}")
    plt.savefig('task2_local_epochs.png', dpi=300, bbox_inches='tight')
    print("Results saved to task2_local_epochs_results.json and plot")
    
    return results


def experiment_client_sampling(num_rounds=200, num_clients=20, K=5,
                               fractions=[1.0, 0.5, 0.2, 0.1], lr=0.05, 
                               dataset='cifar10', use_double=True):
    """
    Experiment 2: Impact of Client Sampling Fraction (n/N)
    """
    set_seed(42)
    device = get_device()
    
    if use_double:
        torch.set_default_dtype(torch.float64)
    
    # CIFAR-10 standard normalization
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    
    train_data = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    test_data = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    
    print(f"\n{'='*70}")
    print("EXPERIMENT 2: Client Sampling Fraction (Partial Participation)")
    print(f"{'='*70}")
    print(f"Dataset: {dataset.upper()} | Clients: {num_clients} | K: {K} | Rounds: {num_rounds}")
    print(f"Sampling fractions: {fractions} → {[(f, int(f*num_clients)) for f in fractions]} clients")
    print(f"Learning rate: {lr} | Double precision: {use_double}")
    
    # Model parameter count (for comm cost)
    dummy_model = SimpleCNN(num_classes=10, input_channels=3)
    model_params = sum(p.numel() for p in dummy_model.parameters())
    bytes_per_param = 8 if use_double else 4
    mb_per_model = model_params * bytes_per_param / (1024 * 1024)
    
    results = {
        "experiment": "client_sampling",
        "dataset": dataset,
        "num_clients": num_clients,
        "K": K,
        "num_rounds": num_rounds,
        "lr": lr,
        "use_double": use_double,
        "model_params": int(model_params),
        "mb_per_full_model_upload": round(mb_per_model, 3),
        "mb_per_full_model_download": round(mb_per_model, 3),
        "total_train_samples": len(train_data),
        "configs": {}
    }
    
    for frac in fractions:
        n_selected = max(1, int(frac * num_clients))
        label = f"frac={frac:.2f} ({n_selected}/{num_clients})"
        print(f"\n→ Running {label}...")
        
        set_seed(42)
        model = SimpleCNN(10, 3).to(device)
        if use_double:
            model = model.double()
        
        trainer = FedAvgTrainer(
            model=model,
            train_dataset=train_data,
            test_dataset=test_data,
            num_clients=num_clients,
            local_epochs=K,
            lr=lr,
            batch_size=64,
            device=device,
            seed=42
        )
        
        stats = {
            'accuracies': [],
            'losses': [],
            'divergences': [],
            'regrets': [],
            'round_times': [],
            'selected_per_round': [],
            'comm_cost_per_round_MB': [],
            'cumulative_comm_MB': []
        }
        
        total_comm = 0.0
        start_time = time.time()
        
        for r in range(num_rounds):
            round_start = time.time()
            
            round_info = trainer.train_round(client_fraction=frac, round_idx=r)
            acc, loss = trainer.evaluate()
            regret = trainer.compute_regret(loss)
            
            selected = round_info['selected_clients']
            upload_mb = selected * mb_per_model
            download_mb = mb_per_model  # server broadcasts to all selected
            round_comm_mb = upload_mb + download_mb
            total_comm += round_comm_mb
            
            stats['accuracies'].append(acc)
            stats['losses'].append(loss)
            stats['divergences'].append(round_info['divergence'])
            stats['regrets'].append(regret)
            stats['selected_per_round'].append(selected)
            stats['round_times'].append(time.time() - round_start)
            stats['comm_cost_per_round_MB'].append(round(round_comm_mb, 3))
            stats['cumulative_comm_MB'].append(round(total_comm, 3))
            
            if (r + 1) % 25 == 0 or r < 5:
                print(f"  Round {r+1:3d} | Acc: {acc:5.2f}% | Loss: {loss:.4f} | "
                      f"Drift: {round_info['divergence']:.4f} | Regret: {regret:.4f} | "
                      f"Clients: {selected} | Comm: {round_comm_mb:.1f} MB")
        
        total_time = time.time() - start_time
        
        # Compute stability metrics
        stability = compute_stability_metrics(stats['accuracies'], stats['losses'])
        
        stats.update({
            'total_time_sec': round(total_time, 2),
            'avg_round_time_sec': round(np.mean(stats['round_times']), 3),
            'final_accuracy': round(stats['accuracies'][-1], 2),
            'final_divergence': round(stats['divergences'][-1], 4),
            'final_regret': round(stats['regrets'][-1], 4),
            'total_communication_MB': round(total_comm, 2),
            'avg_comm_per_round_MB': round(np.mean(stats['comm_cost_per_round_MB']), 3),
            'stability_metrics': stability
        })
        
        results["configs"][f"{frac:.2f}"] = stats
        
        print(f"  ✓ Done → Acc: {stats['final_accuracy']}% | "
              f"Regret: {stats['final_regret']:.4f} | "
              f"Oscillations: {stability['oscillation_count']} | "
              f"Total Comm: {total_comm:.1f} MB")
    
    # Save full results
    with open("task2_client_sampling_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to 'task2_client_sampling_results.json'")
    
    # Enhanced plot
    plot_fedavg_comprehensive(results, experiment_type="client_sampling",
                             title_suffix=f" (K={K}, N={num_clients})")
    plt.savefig('task2_client_sampling.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("Plot saved as 'task2_client_sampling.png'")
    
    return results


def plot_fedavg_comprehensive(results, experiment_type="local_epochs", title_suffix=""):
    """
    Enhanced plotting with regret, drift, and stability analysis
    """
    fig, axes = plt.subplots(3, 2, figsize=(16, 12))
    
    configs = results["configs"]
    first_config = next(iter(configs.values()))
    rounds = np.arange(1, len(first_config['accuracies']) + 1)
    
    for label, data in configs.items():
        # Row 1: Accuracy and Loss
        axes[0,0].plot(rounds, data['accuracies'], label=label, linewidth=2.5, alpha=0.8)
        axes[0,1].plot(rounds, data['losses'], label=label, linewidth=2.5, alpha=0.8)
        
        # Row 2: Drift and Regret
        axes[1,0].plot(rounds, data['divergences'], label=label, linewidth=2.5, alpha=0.8)
        axes[1,1].plot(rounds, data['regrets'], label=label, linewidth=2.5, alpha=0.8)
        
        # Row 3: Communication cost and Stability (rolling std)
        if experiment_type == "client_sampling":
            axes[2,0].plot(rounds, data['cumulative_comm_MB'], label=label, linewidth=2.5, alpha=0.8)
        else:
            # For local epochs, show cumulative time
            axes[2,0].plot(rounds, np.cumsum(data['round_times']), label=label, linewidth=2.5, alpha=0.8)
        
        # Rolling standard deviation of accuracy (stability indicator)
        window = 10
        if len(data['accuracies']) >= window:
            rolling_std = [np.std(data['accuracies'][max(0, i-window):i+1]) 
                          for i in range(len(data['accuracies']))]
            axes[2,1].plot(rounds, rolling_std, label=label, linewidth=2.5, alpha=0.8)
    
    # Titles and labels
    axes[0,0].set_title('Test Accuracy (%)', fontsize=14, fontweight='bold')
    axes[0,1].set_title('Test Loss', fontsize=14, fontweight='bold')
    axes[1,0].set_title('Client Drift (L2 from Global)', fontsize=14, fontweight='bold')
    axes[1,1].set_title('Regret: L(θ_t) - L(θ*)', fontsize=14, fontweight='bold')
    
    if experiment_type == "client_sampling":
        axes[2,0].set_title('Cumulative Communication (MB)', fontsize=14, fontweight='bold')
    else:
        axes[2,0].set_title('Wall-clock Time (sec)', fontsize=14, fontweight='bold')
    
    axes[2,1].set_title('Accuracy Stability (Rolling Std)', fontsize=14, fontweight='bold')
    
    for ax in axes.flat:
        ax.set_xlabel('Communication Round', fontsize=12)
        ax.legend(fontsize=10, loc='best')
        ax.grid(True, alpha=0.3, linestyle='--')
    
    exp_name = "Local Epochs (K)" if experiment_type == "local_epochs" else "Client Sampling"
    plt.suptitle(f'FedAvg: {exp_name} Analysis{title_suffix}', 
                 fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout(rect=[0, 0, 1, 0.99])


if __name__ == "__main__":
    print(f"Starting Task 2 Experiments @ {time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Experiment 1: Vary K
    results_K = experiment_local_epochs(
        num_rounds=100,
        num_clients=10,
        K_values=[1, 5, 10, 20],
        lr=0.01,
        dataset='cifar10',
        use_double=True
    )
    
    # Experiment 2: Vary sampling fraction
    results_frac = experiment_client_sampling(
        num_rounds=100,
        num_clients=10,
        K=5,
        fractions=[1.0, 0.5, 0.2, 0.1],
        lr=0.01,
        dataset='cifar10',
        use_double=True
    )
    
    print("\n" + "="*70)
    print("All Task 2 experiments completed!")
    print("="*70)
    print("\nOutputs:")
    print("  → task2_local_epochs_results.json (with regret & stability)")
    print("  → task2_client_sampling_results.json (with regret & stability)")
    print("  → task2_local_epochs.png (6-panel comprehensive plot)")
    print("  → task2_client_sampling.png (6-panel comprehensive plot)")
    print("\nKey Metrics Tracked:")
    print("  ✓ Accuracy (test set)")
    print("  ✓ Loss (test set)")
    print("  ✓ Client Drift (L2 divergence from global)")
    print("  ✓ Regret (L(θ_t) - L(θ*))")
    print("  ✓ Communication Cost (MB)")
    print("  ✓ Stability (oscillations, variance, convergence rate)")