"""
Utility functions for federated learning experiments
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
import numpy as np
from typing import List, Dict, Tuple
import copy


class SimpleCNN(nn.Module):
    """
    Simple CNN architecture for CIFAR-10
    Architecture: Conv1 -> ReLU -> Pool -> Conv2 -> ReLU -> Pool -> FC1 -> ReLU -> FC2
    """
    def __init__(self, num_classes=10, input_channels=3):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=3, padding=1 , dtype=torch.float64) 
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1 , dtype=torch.float64)
        self.fc1 = nn.Linear(64 * 8 * 8, 128 , dtype=torch.float64)
        self.fc2 = nn.Linear(128, num_classes , dtype=torch.float64)
        
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 64 * 8 * 8)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


def get_device():
    """Get available device (GPU if available, else CPU)"""
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def set_seed(seed=42):
    """Set random seeds for reproducibility"""
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)


def split_data_iid(dataset, num_clients):
    """
    Split dataset into IID partitions for clients
    
    Args:
        dataset: PyTorch dataset
        num_clients: Number of clients
        
    Returns:
        dict: Dictionary mapping client_id to list of sample indices
    """
    num_items = len(dataset)
    items_per_client = num_items // num_clients
    client_dict = {}
    all_idxs = np.arange(num_items)
    # np.random.shuffle(all_idxs)
    
    for i in range(num_clients):
        client_dict[i] = all_idxs[i*items_per_client:(i+1)*items_per_client].tolist()
    
    return client_dict


def split_data_dirichlet(dataset, num_clients, alpha=0.5, num_classes=10):
    """
    Split dataset using Dirichlet distribution for non-IID data
    
    Args:
        dataset: PyTorch dataset with targets attribute
        num_clients: Number of clients
        alpha: Dirichlet concentration parameter (smaller = more heterogeneous)
        num_classes: Number of classes in dataset
        
    Returns:
        dict: Dictionary mapping client_id to list of sample indices
    """
    # Get labels
    if hasattr(dataset, 'targets'):
        labels = np.array(dataset.targets)
    elif hasattr(dataset, 'labels'):
        labels = np.array(dataset.labels)
    else:
        labels = np.array([dataset[i][1] for i in range(len(dataset))])
    
    client_dict = {i: [] for i in range(num_clients)}
    
    # For each class, sample proportions from Dirichlet and distribute
    for k in range(num_classes):
        idx_k = np.where(labels == k)[0]
        np.random.shuffle(idx_k)
        
        # Sample proportions from Dirichlet
        proportions = np.random.dirichlet(np.repeat(alpha, num_clients))
        
        # Balance proportions to ensure each client gets some data
        proportions = np.array([p * (len(idx_k)) for p in proportions])
        proportions = proportions.astype(int)
        
        # Distribute samples according to proportions
        start_idx = 0
        for client_id in range(num_clients):
            end_idx = start_idx + proportions[client_id]
            client_dict[client_id].extend(idx_k[start_idx:end_idx].tolist())
            start_idx = end_idx
    
    return client_dict


def get_model_params(model):
    """Get model parameters as a list of tensors"""
    return [param.data.clone() for param in model.parameters()]


def set_model_params(model, params):
    """Set model parameters from a list of tensors"""
    for model_param, param in zip(model.parameters(), params):
        model_param.data = param.clone()


def average_weights(weights_list, weights_sizes=None):
    """
    Average model weights
    
    Args:
        weights_list: List of model parameters (each is a list of tensors)
        weights_sizes: Optional list of weights for weighted average (e.g., data sizes)
        
    Returns:
        List of averaged parameters
    """
    if weights_sizes is None:
        weights_sizes = [1.0] * len(weights_list)
    
    total_size = sum(weights_sizes)
    weights_sizes = [w / total_size for w in weights_sizes]
    
    avg_params = []
    for param_idx in range(len(weights_list[0])):
        avg_param = torch.zeros_like(weights_list[0][param_idx])
        for client_idx, client_params in enumerate(weights_list):
            avg_param += weights_sizes[client_idx] * client_params[param_idx]
        avg_params.append(avg_param)
    
    return avg_params

def evaluate_model(model, dataloader, device):
    """
    Evaluate model on given dataloader
    
    Returns:
        tuple: (accuracy, loss)
    """
    model.eval()
    correct = 0
    total = 0
    loss_sum = 0.0
    criterion = nn.CrossEntropyLoss()
    
    # Check if model is in double precision
    is_double = next(model.parameters()).dtype == torch.float64
    
    with torch.no_grad():
        for data, target in dataloader:
            data, target = data.to(device), target.to(device)
            
            # Convert to double if model is double
            if is_double:
                data = data.double()
            
            output = model(data)
            loss_sum += criterion(output, target).item()
            pred = output.argmax(dim=1)
            correct += pred.eq(target).sum().item()
            total += target.size(0)
    
    accuracy = 100. * correct / total
    avg_loss = loss_sum / len(dataloader)
    
    return accuracy, avg_loss

# def evaluate_model(model, dataloader, device):
#     """
#     Evaluate model on given dataloader
    
#     Returns:
#         tuple: (accuracy, loss)
#     """
#     model.eval()
#     correct = 0
#     total = 0
#     loss_sum = 0.0
#     criterion = nn.CrossEntropyLoss()
    
#     with torch.no_grad():
#         for data, target in dataloader:
#             data, target = data.to(device), target.to(device)
#             output = model(data)
#             loss_sum += criterion(output, target).item()
#             pred = output.argmax(dim=1)
#             correct += pred.eq(target).sum().item()
#             total += target.size(0)
    
#     accuracy = 100. * correct / total
#     avg_loss = loss_sum / len(dataloader)
    
#     return accuracy, avg_loss


def compute_weight_divergence(client_models, global_model):
    """
    Compute average L2 distance between client models and global model
    
    Args:
        client_models: List of client model parameters
        global_model: Global model parameters
        
    Returns:
        float: Average weight divergence
    """
    divergences = []
    for client_params in client_models:
        divergence = 0.0
        for global_param, client_param in zip(global_model, client_params):
            divergence += torch.norm(client_param - global_param).item() ** 2
        divergences.append(np.sqrt(divergence))
    
    return np.mean(divergences)


def compute_gradient_norm(model, dataloader, device):
    """
    Compute gradient norm on given data
    
    Returns:
        float: L2 norm of gradient
    """
    model.train()
    model.zero_grad()
    criterion = nn.CrossEntropyLoss()
    
    total_loss = 0.0
    for data, target in dataloader:
        data, target = data.to(device), target.to(device)
        output = model(data)
        loss = criterion(output, target)
        total_loss += loss
    
    avg_loss = total_loss / len(dataloader)
    avg_loss.backward()
    
    grad_norm = 0.0
    for param in model.parameters():
        if param.grad is not None:
            grad_norm += torch.norm(param.grad).item() ** 2
    
    model.zero_grad()
    return np.sqrt(grad_norm)


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