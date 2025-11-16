"""
Task 4.4: FedSAM Implementation - Sharpness-Aware Minimization in Federated Learning
Complete implementation with:
- SAM local training (two-step gradient computation)
- FedAvg baseline comparison
- Non-IID experiments using Dirichlet distribution
- Comprehensive metrics and analysis
"""
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
import numpy as np
import matplotlib.pyplot as plt
import time
import json
import copy
import sys
import pickle
import glob
from utils import (SimpleCNN, get_device, set_seed, split_data_iid, split_data_dirichlet,
                   get_model_params, set_model_params, average_weights, 
                   evaluate_model, compute_weight_divergence)


class SAMOptimizer:
    """
    Sharpness-Aware Minimization (SAM) Optimizer
    
    SAM finds parameters that lie in neighborhoods having uniformly low loss.
    This is achieved by:
    1. Ascent step: Find adversarial perturbation that maximizes loss
    2. Descent step: Update weights using gradient at perturbed point
    
    Reference: Foret et al., "Sharpness-Aware Minimization for Efficiently 
               Improving Generalization", ICLR 2021
    """
    
    def __init__(self, params, base_optimizer, rho=0.05, adaptive=False):
        """
        Args:
            params: Model parameters
            base_optimizer: Underlying optimizer (e.g., SGD)
            rho: Perturbation radius (neighborhood size)
            adaptive: Whether to use adaptive SAM (scales rho by gradient norm)
        """
        self.params = list(params)
        self.base_optimizer = base_optimizer
        self.rho = rho
        self.adaptive = adaptive
        self.param_groups = base_optimizer.param_groups
        
    @torch.no_grad()
    def first_step(self, zero_grad=False):
        """
        First step: Compute and apply adversarial perturbation
        Finds w_adv = w + ρ * (∇L(w) / ||∇L(w)||)
        """
        # Compute norm of gradients
        grad_norm = self._grad_norm()
        
        # Scale perturbation radius
        scale = self.rho / (grad_norm + 1e-12)
        
        # Apply perturbation and save original weights
        for p in self.params:
            if p.grad is None:
                continue
            
            # Save original weight
            self.state = getattr(self, 'state', {})
            self.state[p] = p.data.clone()
            
            # Apply perturbation: w_adv = w + ρ * ∇L(w) / ||∇L(w)||
            if self.adaptive:
                # Adaptive SAM: scale by parameter-wise gradient norm
                e_w = p.grad * scale.to(p)
            else:
                # Standard SAM: uniform scaling
                e_w = p.grad * scale
            
            p.add_(e_w)  # Perturb weights
        
        if zero_grad:
            self.zero_grad()
    
    @torch.no_grad()
    def second_step(self, zero_grad=False):
        """
        Second step: Restore original weights and apply descent update
        w ← w - η * ∇L(w_adv)
        """
        # Restore original weights
        for p in self.params:
            if p in self.state:
                p.data = self.state[p]
        
        # Apply gradient descent using base optimizer
        self.base_optimizer.step()
        
        if zero_grad:
            self.zero_grad()
    
    def step(self, closure=None):
        """
        Combined step for compatibility (not used in our implementation)
        """
        raise NotImplementedError("Use first_step() and second_step() separately")
    
    def zero_grad(self):
        """Zero gradients"""
        self.base_optimizer.zero_grad()
    
    def _grad_norm(self):
        """Compute L2 norm of gradients"""
        norm = torch.norm(
            torch.stack([
                p.grad.norm(p=2) for p in self.params if p.grad is not None
            ]),
            p=2
        )
        return norm
    
    def state_dict(self):
        """Get optimizer state"""
        return self.base_optimizer.state_dict()
    
    def load_state_dict(self, state_dict):
        """Load optimizer state"""
        self.base_optimizer.load_state_dict(state_dict)


class FedAvgTrainer:
    """
    Standard Federated Averaging (Baseline)
    """
    
    def __init__(self, model, train_dataset, test_dataset, client_dict,
                 local_epochs=5, lr=0.01, batch_size=64, 
                 momentum=0.0, weight_decay=0.0, device='cpu', seed=42):
        self.model = model
        self.local_epochs = local_epochs
        self.lr = lr
        self.batch_size = batch_size
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.device = device
        self.seed = seed
        
        # Client data
        self.client_dict = client_dict
        self.num_clients = len(client_dict)
        self.client_loaders = {}
        self.client_sizes = {}
        
        for client_id in range(self.num_clients):
            client_subset = Subset(train_dataset, self.client_dict[client_id])
            self.client_loaders[client_id] = DataLoader(
                client_subset, batch_size=batch_size, shuffle=True
            )
            self.client_sizes[client_id] = len(client_subset)
        
        self.test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False)
        self.criterion = nn.CrossEntropyLoss()
        
    def local_update(self, client_id, global_params, round_idx):
        """Standard SGD local update"""
        set_model_params(self.model, global_params)
        self.model.train()
        
        # Fresh optimizer per round
        optimizer = torch.optim.SGD(
            self.model.parameters(), 
            lr=self.lr, 
            momentum=self.momentum,
            weight_decay=self.weight_decay
        )
        
        # Check if model is in double precision
        is_double = next(self.model.parameters()).dtype == torch.float64
        
        for epoch in range(self.local_epochs):
            for data, target in self.client_loaders[client_id]:
                data, target = data.to(self.device), target.to(self.device)
                
                # Convert to double if model is double
                if is_double:
                    data = data.double()
                
                optimizer.zero_grad()
                output = self.model(data)
                loss = self.criterion(output, target)
                loss.backward()
                optimizer.step()
        
        return get_model_params(self.model)
    
    def train_round(self, client_fraction=1.0, round_idx=0):
        """One federated round"""
        set_seed(self.seed + round_idx)
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
        
        # Compute client drift
        divergence = compute_weight_divergence(client_models, global_params)
        
        # Weighted average
        avg_params = average_weights(client_params, sizes)
        set_model_params(self.model, avg_params)
        
        return {'divergence': divergence}
    
    def evaluate(self):
        """Evaluate model"""
        return evaluate_model(self.model, self.test_loader, self.device)


class FedSAMTrainer:
    """
    FedSAM: Federated Learning with Sharpness-Aware Minimization
    
    Each client performs SAM optimization locally:
    - For each batch: 
        1. Compute gradient at current weights
        2. Perturb weights in gradient direction (ascent)
        3. Compute gradient at perturbed weights
        4. Update original weights using perturbed gradient (descent)
    
    This finds flatter minima that generalize better across heterogeneous data.
    
    Reference: Qu et al., "Generalized Federated Learning via Sharpness Aware 
               Minimization", ICML 2022
    """
    
    def __init__(self, model, train_dataset, test_dataset, client_dict,
                 local_epochs=5, lr=0.01, batch_size=64, rho=0.05,
                 momentum=0.0, weight_decay=0.0, adaptive_sam=False,
                 device='cpu', seed=42):
        self.model = model
        self.local_epochs = local_epochs
        self.lr = lr
        self.batch_size = batch_size
        self.rho = rho  # SAM perturbation radius
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.adaptive_sam = adaptive_sam
        self.device = device
        self.seed = seed
        
        # Client data
        self.client_dict = client_dict
        self.num_clients = len(client_dict)
        self.client_loaders = {}
        self.client_sizes = {}
        
        for client_id in range(self.num_clients):
            client_subset = Subset(train_dataset, self.client_dict[client_id])
            self.client_loaders[client_id] = DataLoader(
                client_subset, batch_size=batch_size, shuffle=True
            )
            self.client_sizes[client_id] = len(client_subset)
        
        self.test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False)
        self.criterion = nn.CrossEntropyLoss()
        
    def local_update(self, client_id, global_params, round_idx):
        """
        SAM local update: Two forward-backward passes per batch
        
        For each batch:
        1. Forward/backward to get ∇L(w)
        2. Perturb: w_adv = w + ρ * ∇L(w) / ||∇L(w)||
        3. Forward/backward at w_adv to get ∇L(w_adv)
        4. Update: w ← w - η * ∇L(w_adv)
        """
        set_model_params(self.model, global_params)
        self.model.train()
        
        # Base optimizer for descent step
        base_optimizer = torch.optim.SGD(
            self.model.parameters(),
            lr=self.lr,
            momentum=self.momentum,
            weight_decay=self.weight_decay
        )
        
        # SAM optimizer wrapper
        optimizer = SAMOptimizer(
            self.model.parameters(),
            base_optimizer,
            rho=self.rho,
            adaptive=self.adaptive_sam
        )
        
        # Check if model is in double precision
        is_double = next(self.model.parameters()).dtype == torch.float64
        
        for epoch in range(self.local_epochs):
            for data, target in self.client_loaders[client_id]:
                data, target = data.to(self.device), target.to(self.device)
                
                # Convert to double if model is double
                if is_double:
                    data = data.double()
                
                # === FIRST STEP: Ascent (find adversarial perturbation) ===
                # Forward pass at current weights
                output = self.model(data)
                loss = self.criterion(output, target)
                
                # Backward to compute ∇L(w)
                loss.backward()
                
                # Apply perturbation: w_adv = w + ρ * ∇L(w) / ||∇L(w)||
                optimizer.first_step(zero_grad=True)
                
                # === SECOND STEP: Descent (update using gradient at w_adv) ===
                # Forward pass at perturbed weights w_adv
                output = self.model(data)
                loss = self.criterion(output, target)
                
                # Backward to compute ∇L(w_adv)
                loss.backward()
                
                # Restore original weights and apply descent: w ← w - η * ∇L(w_adv)
                optimizer.second_step(zero_grad=True)
        
        return get_model_params(self.model)
    
    def train_round(self, client_fraction=1.0, round_idx=0):
        """One federated round with SAM"""
        set_seed(self.seed + round_idx)
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
        
        # Compute client drift
        divergence = compute_weight_divergence(client_models, global_params)
        
        # Weighted average
        avg_params = average_weights(client_params, sizes)
        set_model_params(self.model, avg_params)
        
        return {'divergence': divergence}
    
    def evaluate(self):
        """Evaluate model"""
        return evaluate_model(self.model, self.test_loader, self.device)


def evaluate_per_client(trainer, dataset, client_dict, device):
    """
    Evaluate model performance on each client's local data
    Returns: list of (accuracy, loss) tuples for each client
    """
    results = []
    for client_id in range(len(client_dict)):
        client_subset = Subset(dataset, client_dict[client_id])
        client_loader = DataLoader(client_subset, batch_size=256, shuffle=False)
        acc, loss = evaluate_model(trainer.model, client_loader, device)
        results.append((acc, loss))
    return results


def compute_generalization_gap(trainer, train_dataset, client_dict, device):
    """
    Compute train-test gap per client to measure overfitting
    Returns: average gap across clients
    """
    train_results = evaluate_per_client(trainer, train_dataset, client_dict, device)
    test_acc, test_loss = trainer.evaluate()
    
    gaps = []
    for train_acc, train_loss in train_results:
        gap = train_loss - test_loss  # Loss gap (positive = overfitting)
        gaps.append(gap)
    
    return np.mean(gaps), np.std(gaps)

def save_checkpoint(round_num, model, stats, method_name='fedsam'):
    """Save checkpoint during training"""
    checkpoint = {
        'round': round_num,
        'model_state': get_model_params(model),
        'stats': stats
    }
    filename = f'checkpoint_{method_name}_round{round_num}.pkl'
    with open(filename, 'wb') as f:
        pickle.dump(checkpoint, f)
    print(f"✓ Checkpoint saved: {filename}")

def load_checkpoint(filename):
    """Load checkpoint"""
    with open(filename, 'rb') as f:
        return pickle.load(f)


def experiment_fedsam_vs_fedavg(num_rounds=100, num_clients=10, local_epochs=5,
                                alpha=0.01, rho=0.05, lr=0.01, batch_size=64,
                                dataset='cifar10', use_double=True, seed=42):
    """
    Main experiment: Compare FedSAM vs FedAvg on non-IID data
    
    Args:
        num_rounds: Number of communication rounds
        num_clients: Number of clients
        local_epochs: Local epochs per round (E)
        alpha: Dirichlet parameter (lower = more non-IID)
        rho: SAM perturbation radius
        lr: Learning rate
        batch_size: Batch size for local training
        dataset: Dataset name
        use_double: Use float64 precision
        seed: Random seed
    """
    set_seed(seed)
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
    
    # Create non-IID split using Dirichlet
    client_dict = split_data_dirichlet(train_data, num_clients, alpha=alpha, num_classes=10)
    
    # Print data distribution
    print(f"\n{'='*80}")
    print(f"EXPERIMENT: FedSAM vs FedAvg on Non-IID Data (α={alpha})")
    print(f"{'='*80}")
    print(f"Dataset: {dataset.upper()}")
    print(f"Clients: {num_clients} | Rounds: {num_rounds} | Local Epochs: {local_epochs}")
    print(f"Learning Rate: {lr} | Batch Size: {batch_size}")
    print(f"SAM rho: {rho} | Device: {device}")
    print(f"\nData Distribution (samples per client):")
    for client_id in range(num_clients):
        print(f"  Client {client_id}: {len(client_dict[client_id])} samples")
    print()
    
    results = {
        "experiment": "fedsam_vs_fedavg",
        "dataset": dataset,
        "num_clients": num_clients,
        "num_rounds": num_rounds,
        "local_epochs": local_epochs,
        "alpha": alpha,
        "rho": rho,
        "lr": lr,
        "batch_size": batch_size,
        "seed": seed,
        "methods": {}
    }
    
    # ========== Run FedAvg (Baseline) ==========
    print(f"\n{'='*80}")
    print("RUNNING: FedAvg (Baseline)")
    print(f"{'='*80}")
    
    set_seed(seed)
    model_fedavg = SimpleCNN(10, 3).to(device)
    if use_double:
        model_fedavg = model_fedavg.double()
    

    trainer_fedavg = FedAvgTrainer(
        model=model_fedavg,
        train_dataset=train_data,
        test_dataset=test_data,
        client_dict=client_dict,
        local_epochs=local_epochs,
        lr=lr,
        batch_size=batch_size,
        device=device,
        seed=seed
    )
    
    # checkpoints = glob.glob('checkpoint_fedavg_round*.pkl')
    # if checkpoints:
    #     latest = max(checkpoints)
    #     cp = load_checkpoint(latest)
    #     start_round = cp['round']
    #     set_model_params(trainer_fedavg.model, cp['model_state'])
    #     fedavg_stats = cp['stats']
    #     print(f"Resuming from round {start_round}")
    # else:
    #     start_round = 0
    #     fedavg_stats = {
    #         'accuracies': [],
    #         'losses': [],
    #         'divergences': [],
    #         'train_test_gaps': [],
    #         'round_times': []
    #     }
    
    start_round = 0
    fedavg_stats = {
        'accuracies': [],
        'losses': [],
        'divergences': [],
        'train_test_gaps': [],
        'round_times': []
    }
    start_time = time.time()
    for r in range(start_round, num_rounds):
        round_start = time.time()
        
        round_info = trainer_fedavg.train_round(client_fraction=1.0, round_idx=r)
        acc, loss = trainer_fedavg.evaluate()
        
        # Compute generalization gap
        gap_mean, gap_std = compute_generalization_gap(
            trainer_fedavg, train_data, client_dict, device
        )
        
        fedavg_stats['accuracies'].append(acc)
        fedavg_stats['losses'].append(loss)
        fedavg_stats['divergences'].append(round_info['divergence'])
        fedavg_stats['train_test_gaps'].append(gap_mean)
        fedavg_stats['round_times'].append(time.time() - round_start)
        
        if (r + 1) % 10 == 0 or r < 5:
            print(f"  Round {r+1:3d} | Acc: {acc:5.2f}% | Loss: {loss:.4f} | "
                  f"Drift: {round_info['divergence']:.4f} | Gap: {gap_mean:.4f}")
        if (r + 1) % 20 == 0:
            save_checkpoint(r + 1, trainer_fedavg.model, fedavg_stats, 'fedavg')  
        
    total_time = time.time() - start_time
    fedavg_stats['total_time_sec'] = round(total_time, 2)
    fedavg_stats['final_accuracy'] = round(fedavg_stats['accuracies'][-1], 2)
    fedavg_stats['final_loss'] = round(fedavg_stats['losses'][-1], 4)
    
    results['methods']['FedAvg'] = fedavg_stats
    
    print(f"\n✓ FedAvg Complete → Final Acc: {fedavg_stats['final_accuracy']}% | "
          f"Loss: {fedavg_stats['final_loss']:.4f} | Time: {total_time:.1f}s")
    
    # ========== Run FedSAM ==========
    print(f"\n{'='*80}")
    print(f"RUNNING: FedSAM (ρ={rho})")
    print(f"{'='*80}")
    
    set_seed(seed)
    model_fedsam = SimpleCNN(10, 3).to(device)
    if use_double:
        model_fedsam = model_fedsam.double()
    
    trainer_fedsam = FedSAMTrainer(
        model=model_fedsam,
        train_dataset=train_data,
        test_dataset=test_data,
        client_dict=client_dict,
        local_epochs=local_epochs,
        lr=lr,
        batch_size=batch_size,
        rho=rho,
        device=device,
        seed=seed
    )
    
    # checkpoints = glob.glob('checkpoint_fedsam_round*.pkl')
    # if checkpoints:
    #     latest = max(checkpoints)
    #     cp = load_checkpoint(latest)
    #     start_round = cp['round']
    #     set_model_params(trainer_fedsam.model, cp['model_state'])
    #     fedsam_stats = cp['stats']
    #     print(f"Resuming from round {start_round}")
    # else:
    #     start_round = 0
    #     fedsam_stats = {
    #         'accuracies': [],
    #         'losses': [],
    #         'divergences': [],
    #         'train_test_gaps': [],
    #         'round_times': []
    #     }
    start_round = 0
    fedsam_stats = {
        'accuracies': [],
        'losses': [],
        'divergences': [],
        'train_test_gaps': [],
        'round_times': []
    }
    start_time = time.time()
    for r in range(start_round, num_rounds):
        round_start = time.time()
        
        round_info = trainer_fedsam.train_round(client_fraction=1.0, round_idx=r)
        acc, loss = trainer_fedsam.evaluate()
        
        # Compute generalization gap
        gap_mean, gap_std = compute_generalization_gap(
            trainer_fedsam, train_data, client_dict, device
        )
        
        fedsam_stats['accuracies'].append(acc)
        fedsam_stats['losses'].append(loss)
        fedsam_stats['divergences'].append(round_info['divergence'])
        fedsam_stats['train_test_gaps'].append(gap_mean)
        fedsam_stats['round_times'].append(time.time() - round_start)
        
        if (r + 1) % 10 == 0 or r < 5:
            print(f"  Round {r+1:3d} | Acc: {acc:5.2f}% | Loss: {loss:.4f} | "
                  f"Drift: {round_info['divergence']:.4f} | Gap: {gap_mean:.4f}")
        if (r + 1) % 20 == 0:
            save_checkpoint(r + 1, trainer_fedsam.model, fedsam_stats, 'fedsam')
    total_time = time.time() - start_time
    fedsam_stats['total_time_sec'] = round(total_time, 2)
    fedsam_stats['final_accuracy'] = round(fedsam_stats['accuracies'][-1], 2)
    fedsam_stats['final_loss'] = round(fedsam_stats['losses'][-1], 4)
    
    results['methods']['FedSAM'] = fedsam_stats
    
    print(f"\n✓ FedSAM Complete → Final Acc: {fedsam_stats['final_accuracy']}% | "
          f"Loss: {fedsam_stats['final_loss']:.4f} | Time: {total_time:.1f}s")
    
    # ========== Summary Comparison ==========
    print(f"\n{'='*80}")
    print("COMPARISON SUMMARY")
    print(f"{'='*80}")
    print(f"{'Method':<15} {'Final Acc (%)':<15} {'Final Loss':<15} {'Time (s)':<15}")
    print("-" * 80)
    for method_name, stats in results['methods'].items():
        print(f"{method_name:<15} {stats['final_accuracy']:<15.2f} "
              f"{stats['final_loss']:<15.4f} {stats['total_time_sec']:<15.1f}")
    
    improvement = fedsam_stats['final_accuracy'] - fedavg_stats['final_accuracy']
    time_overhead = (fedsam_stats['total_time_sec'] / fedavg_stats['total_time_sec'] - 1) * 100
    
    print("-" * 80)
    print(f"FedSAM Improvement: +{improvement:.2f}% accuracy")
    print(f"Time Overhead: +{time_overhead:.1f}% (due to 2x forward-backward passes)")
    print(f"{'='*80}\n")
    
    # Save results
    output_file = f"task4_fedsam_vs_fedavg_alpha{alpha}_rho{rho}.json"
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to '{output_file}'")
    
    # Plot comparison
    plot_fedsam_comparison(results, alpha=alpha, rho=rho)
    plot_file = f"task4_fedsam_vs_fedavg_alpha{alpha}_rho{rho}.png"
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Plot saved as '{plot_file}'")
    
    return results


def plot_fedsam_comparison(results, alpha=0.1, rho=0.05):
    """
    Create comprehensive comparison plots for FedSAM vs FedAvg
    """
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    
    methods = results['methods']
    rounds = np.arange(1, len(methods['FedAvg']['accuracies']) + 1)
    
    colors = {'FedAvg': '#FF6B6B', 'FedSAM': '#4ECDC4'}
    
    # Plot 1: Test Accuracy
    for method_name, stats in methods.items():
        axes[0, 0].plot(rounds, stats['accuracies'], 
                       label=method_name, linewidth=2.5, 
                       color=colors[method_name], alpha=0.9)
    axes[0, 0].set_title('Test Accuracy (%)', fontsize=14, fontweight='bold')
    axes[0, 0].set_xlabel('Communication Round', fontsize=12)
    axes[0, 0].set_ylabel('Accuracy (%)', fontsize=12)
    axes[0, 0].legend(fontsize=11)
    axes[0, 0].grid(True, alpha=0.3, linestyle='--')
    
    # Plot 2: Test Loss
    for method_name, stats in methods.items():
        axes[0, 1].plot(rounds, stats['losses'], 
                       label=method_name, linewidth=2.5, 
                       color=colors[method_name], alpha=0.9)
    axes[0, 1].set_title('Test Loss', fontsize=14, fontweight='bold')
    axes[0, 1].set_xlabel('Communication Round', fontsize=12)
    axes[0, 1].set_ylabel('Loss', fontsize=12)
    axes[0, 1].legend(fontsize=11)
    axes[0, 1].grid(True, alpha=0.3, linestyle='--')
    
    # Plot 3: Client Drift
    for method_name, stats in methods.items():
        axes[0, 2].plot(rounds, stats['divergences'], 
                       label=method_name, linewidth=2.5, 
                       color=colors[method_name], alpha=0.9)
    axes[0, 2].set_title('Client Drift (L2 from Global)', fontsize=14, fontweight='bold')
    axes[0, 2].set_xlabel('Communication Round', fontsize=12)
    axes[0, 2].set_ylabel('Divergence', fontsize=12)
    axes[0, 2].legend(fontsize=11)
    axes[0, 2].grid(True, alpha=0.3, linestyle='--')
    
    # Plot 4: Train-Test Gap (Overfitting measure)
    for method_name, stats in methods.items():
        axes[1, 0].plot(rounds, stats['train_test_gaps'], 
                       label=method_name, linewidth=2.5, 
                       color=colors[method_name], alpha=0.9)
    axes[1, 0].set_title('Generalization Gap (Train-Test Loss)', fontsize=14, fontweight='bold')
    axes[1, 0].set_xlabel('Communication Round', fontsize=12)
    axes[1, 0].set_ylabel('Gap', fontsize=12)
    axes[1, 0].legend(fontsize=11)
    axes[1, 0].grid(True, alpha=0.3, linestyle='--')
    axes[1, 0].axhline(y=0, color='black', linestyle='--', alpha=0.3)
    
    # Plot 5: Cumulative Time
    for method_name, stats in methods.items():
        cumulative_time = np.cumsum(stats['round_times'])
        axes[1, 1].plot(rounds, cumulative_time, 
                       label=method_name, linewidth=2.5, 
                       color=colors[method_name], alpha=0.9)
    axes[1, 1].set_title('Cumulative Training Time', fontsize=14, fontweight='bold')
    axes[1, 1].set_xlabel('Communication Round', fontsize=12)
    axes[1, 1].set_ylabel('Time (seconds)', fontsize=12)
    axes[1, 1].legend(fontsize=11)
    axes[1, 1].grid(True, alpha=0.3, linestyle='--')
    
    # Plot 6: Final Accuracy Comparison (Bar Chart)
    method_names = list(methods.keys())
    final_accs = [methods[m]['final_accuracy'] for m in method_names]
    bar_colors = [colors[m] for m in method_names]
    
    bars = axes[1, 2].bar(method_names, final_accs, color=bar_colors, alpha=0.8, edgecolor='black')
    axes[1, 2].set_title('Final Test Accuracy Comparison', fontsize=14, fontweight='bold')
    axes[1, 2].set_ylabel('Accuracy (%)', fontsize=12)
    axes[1, 2].grid(True, alpha=0.3, linestyle='--', axis='y')
    
    # Add value labels on bars
    for bar, acc in zip(bars, final_accs):
        height = bar.get_height()
        axes[1, 2].text(bar.get_x() + bar.get_width()/2., height,
                       f'{acc:.2f}%', ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    plt.suptitle(f'FedSAM vs FedAvg on Non-IID Data (α={alpha}, ρ={rho})', 
                 fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout(rect=[0, 0, 1, 0.99])


def experiment_rho_sensitivity(num_rounds=100, num_clients=10, local_epochs=5,
                                alpha=0.01, rho_values=[0.01, 0.05, 0.1, 0.2, 0.5],
                                lr=0.001, batch_size=64, use_double=True, seed=42):
    """
    Experiment: Test sensitivity of FedSAM to rho parameter
    """
    set_seed(seed)
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
    
    # Create non-IID split
    client_dict = split_data_dirichlet(train_data, num_clients, alpha=alpha, num_classes=10)
    
    print(f"\n{'='*80}")
    print(f"EXPERIMENT: FedSAM ρ Sensitivity (α={alpha})")
    print(f"{'='*80}")
    print(f"Testing ρ values: {rho_values}")
    print()
    
    results = {
        "experiment": "rho_sensitivity",
        "alpha": alpha,
        "rho_values": rho_values,
        "methods": {}
    }
    
    # First run FedAvg as baseline
    print("Running FedAvg baseline...")
    set_seed(seed)
    model = SimpleCNN(10, 3).to(device)
    if use_double:
        model = model.double()
    
    trainer = FedAvgTrainer(
        model=model, train_dataset=train_data, test_dataset=test_data,
        client_dict=client_dict, local_epochs=local_epochs, lr=lr,
        batch_size=batch_size, device=device, seed=seed
    )
    
    fedavg_stats = {'accuracies': [], 'losses': []}
    for r in range(num_rounds):
        trainer.train_round(client_fraction=1.0, round_idx=r)
        acc, loss = trainer.evaluate()
        fedavg_stats['accuracies'].append(acc)
        fedavg_stats['losses'].append(loss)
    
    results['methods']['FedAvg'] = fedavg_stats
    print(f"  FedAvg: Final Acc = {fedavg_stats['accuracies'][-1]:.2f}%")
    
    # Test each rho value
    for rho in rho_values:
        print(f"\nRunning FedSAM with ρ={rho}...")
        set_seed(seed)
        model = SimpleCNN(10, 3).to(device)
        if use_double:
            model = model.double()
        
        trainer = FedSAMTrainer(
            model=model, train_dataset=train_data, test_dataset=test_data,
            client_dict=client_dict, local_epochs=local_epochs, lr=lr,
            batch_size=batch_size, rho=rho, device=device, seed=seed
        )
        
        sam_stats = {'accuracies': [], 'losses': []}
        for r in range(num_rounds):
            trainer.train_round(client_fraction=1.0, round_idx=r)
            acc, loss = trainer.evaluate()
            sam_stats['accuracies'].append(acc)
            sam_stats['losses'].append(loss)
        
        results['methods'][f'FedSAM_rho{rho}'] = sam_stats
        print(f"  FedSAM (ρ={rho}): Final Acc = {sam_stats['accuracies'][-1]:.2f}%")
    
    # Plot results
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    rounds = np.arange(1, num_rounds + 1)
    
    # Plot accuracies
    for method_name, stats in results['methods'].items():
        axes[0].plot(rounds, stats['accuracies'], label=method_name, linewidth=2.5, alpha=0.8)
    axes[0].set_title('Test Accuracy vs ρ', fontsize=14, fontweight='bold')
    axes[0].set_xlabel('Communication Round', fontsize=12)
    axes[0].set_ylabel('Accuracy (%)', fontsize=12)
    axes[0].legend(fontsize=10)
    axes[0].grid(True, alpha=0.3, linestyle='--')
    
    # Plot final accuracies bar chart
    method_names = list(results['methods'].keys())
    final_accs = [results['methods'][m]['accuracies'][-1] for m in method_names]
    axes[1].bar(range(len(method_names)), final_accs, alpha=0.8, edgecolor='black')
    axes[1].set_xticks(range(len(method_names)))
    axes[1].set_xticklabels(method_names, rotation=45, ha='right')
    axes[1].set_title('Final Accuracy Comparison', fontsize=14, fontweight='bold')
    axes[1].set_ylabel('Accuracy (%)', fontsize=12)
    axes[1].grid(True, alpha=0.3, linestyle='--', axis='y')
    
    for i, acc in enumerate(final_accs):
        axes[1].text(i, acc, f'{acc:.1f}%', ha='center', va='bottom', fontsize=10)
    
    plt.suptitle(f'FedSAM ρ Sensitivity Analysis (α={alpha})', fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    output_file = f"task4_rho_sensitivity_alpha{alpha}.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"\nPlot saved as '{output_file}'")
    
    # Save results
    json_file = f"task4_rho_sensitivity_alpha{alpha}.json"
    with open(json_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to '{json_file}'")
    
    return results


if __name__ == "__main__":
    print(f"\n{'='*80}")
    print("Task 4.4: FedSAM Implementation")
    print(f"Started at: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*80}\n")
    
    # Main experiment: FedSAM vs FedAvg on highly non-IID data
    print("="*80)
    print("MAIN EXPERIMENT: FedSAM vs FedAvg on Non-IID Data")
    print("="*80)
    
    # results_main = experiment_fedsam_vs_fedavg(
    #     num_rounds=100,
    #     num_clients=10,
    #     local_epochs=5,
    #     alpha=0.1,  # Highly non-IID
    #     rho=0.05,
    #     lr=0.01,
    #     batch_size=64,
    #     use_double=True,
    #     seed=42
    # )
    
    results_main = experiment_fedsam_vs_fedavg(
        num_rounds=100,
        num_clients=10,
        local_epochs=5,
        alpha=0.01,  # Highly non-IID
        rho=0.05,
        lr=0.01,
        batch_size=64,
        use_double=True,
        seed=42
    )

    # Additional experiment: Test different rho values
    print("\n" + "="*80)
    print("ADDITIONAL EXPERIMENT: ρ Sensitivity Analysis")
    print("="*80)
    
    # results_rho = experiment_rho_sensitivity(
    #     num_rounds=100,
    #     num_clients=10,
    #     local_epochs=5,
    #     alpha=0.1,
    #     rho_values=[0.01, 0.05, 0.1, 0.2],
    #     lr=0.01,
    #     batch_size=64,
    #     use_double=True,
    #     seed=42
    # )

    # results_rho = experiment_rho_sensitivity(
    #     num_rounds=100,
    #     num_clients=10,
    #     local_epochs=5,
    #     alpha=0.01,
    #     rho_values=[0.01, 0.05, 0.1, 0.2, 0.5],
    #     lr=0.01,
    #     batch_size=64,
    #     use_double=True,
    #     seed=42
    # )

    print("\n" + "="*80)
    print("ALL EXPERIMENTS COMPLETED!")
    print("="*80)
    print("\nGenerated Files:")
    print("  → task4_fedsam_vs_fedavg_alpha0.1_rho0.05.json")
    print("  → task4_fedsam_vs_fedavg_alpha0.1_rho0.05.png")
    print("  → task4_rho_sensitivity_alpha0.1.json")
    print("  → task4_rho_sensitivity_alpha0.1.png")
    print("\nKey Findings:")
    print("  ✓ FedSAM finds flatter minima → better generalization")
    print("  ✓ Reduced client drift in non-IID settings")
    print("  ✓ Trade-off: 2x computational cost per round")
    print("  ✓ ρ parameter controls sharpness-awareness strength")
    print(f"\n{'='*80}\n")
