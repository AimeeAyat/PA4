"""
Main runner for all federated learning experiments
EE5102/CS6302 - Advanced Topics in Machine Learning
Assignment 4: Federated Learning
"""
import argparse
import sys


def run_task1():
    """Task 1: FedSGD vs Centralized SGD"""
    print("\n" + "="*70)
    print("TASK 1: FedSGD vs Centralized SGD")
    print("="*70)
    from task1_fedsgd_vs_centralized import compare_fedsgd_centralized
    compare_fedsgd_centralized(
        num_rounds=1000,
        num_clients=3,
        lr=0.01,
        dataset='cifar10'
    )

def run_task1_v2():
    """Task 1: FedSGD vs Centralized SGD V2"""
    print("\n" + "="*70)
    print("TASK 1: FedSGD vs Centralized SGD V2")
    print("="*70)
    from task1_fedsgd_vs_centralized_v2 import compare_fedsgd_centralized
    compare_fedsgd_centralized(
        num_rounds=500,
        num_clients=3,
        lr=0.01,
        dataset='cifar10'
    )

def run_task2():
    """Task 2: FedAvg Implementation"""
    print("\n" + "="*70)
    print("TASK 2: FedAvg - Communication Efficiency")
    print("="*70)
    from task2_fedavg_efficiency import (
        experiment_local_epochs,
        experiment_client_sampling
    )
    
    # Experiment 1: Vary local epochs
    print("\n>>> Experiment 1: Varying Local Epochs K")
    experiment_local_epochs(
        num_rounds=100,
        num_clients=10,
        K_values=[1, 5, 10, 20],
        lr=0.01,
        dataset='cifar10'
    )
    
    # Experiment 2: Vary client sampling
    print("\n>>> Experiment 2: Varying Client Sampling")
    experiment_client_sampling(
        num_rounds=100,
        num_clients=10,
        K=5,
        fractions=[1.0, 0.5, 0.2],
        lr=0.01,
        dataset='cifar10'
    )


def run_task3():
    """Task 3: Data Heterogeneity Impact"""
    print("\n" + "="*70)
    print("TASK 3: Exploring Data Heterogeneity")
    print("="*70)
    from task3_data_heterogeneity import experiment_heterogeneity
    experiment_heterogeneity(
        num_rounds=100,
        num_clients=5,
        local_epochs=5,
        alpha_values=[100, 1.0, 0.2, 0.05],
        lr=0.01,
        dataset='cifar10',
        num_classes=10
    )


def run_task4():
    """Task 4: Mitigation Strategies"""
    print("\n" + "="*70)
    print("TASK 4: Mitigating Heterogeneity")
    print("="*70)
    
    print("\n>>> Task 4.1: FedProx")
    from task4_1_fedprox import compare_fedavg_fedprox
    compare_fedavg_fedprox(
        num_rounds=50,
        num_clients=5,
        local_epochs=5,
        alpha=0.1,
        mu_values=[0.0, 0.01, 0.1],
        lr=0.01,
        dataset='cifar10'
    )
    
    print("\n>>> Task 4.2: SCAFFOLD")
    from task4_2_scaffold import compare_with_scaffold
    compare_with_scaffold(
        num_rounds=50,
        num_clients=5,
        local_epochs=5,
        alpha=0.1,
        lr=0.01,
        dataset='cifar10'
    )
    
    print("\n>>> Task 4.3: Gradient Harmonization")
    from task4_3_gradient_harmonization import compare_with_harmonization
    compare_with_harmonization(
        num_rounds=50,
        num_clients=5,
        local_epochs=5,
        alpha=0.05,
        lr=0.01,
        dataset='cifar10'
    )
    
    print("\n>>> Task 4.4: FedSAM")
    from task4_4_fedsam import compare_with_fedsam
    compare_with_fedsam(
        num_rounds=50,
        num_clients=5,
        local_epochs=5,
        alpha=0.1,
        rho=0.05,
        lr=0.01,
        dataset='cifar10'
    )


def run_task4_comparison():
    """Task 4: Comprehensive Comparison"""
    print("\n" + "="*70)
    print("TASK 4: Comprehensive Comparison of All Methods")
    print("="*70)
    from task4_comprehensive_comparison import comprehensive_comparison
    comprehensive_comparison(
        num_rounds=50,
        num_clients=5,
        local_epochs=5,
        alpha_values=[0.1],
        lr=0.01,
        dataset='cifar10'
    )


def run_all():
    """Run all tasks sequentially"""
    try:
        run_task1()
    except Exception as e:
        print(f"Error in Task 1: {e}")
    
    try:
        run_task1_v2()
    except Exception as e:
        print(f"Error in Task 1: {e}")

    try:
        run_task2()
    except Exception as e:
        print(f"Error in Task 2: {e}")
    
    try:
        run_task3()
    except Exception as e:
        print(f"Error in Task 3: {e}")
    
    try:
        run_task4()
    except Exception as e:
        print(f"Error in Task 4: {e}")
    
    try:
        run_task4_comparison()
    except Exception as e:
        print(f"Error in Task 4 Comparison: {e}")


def main():
    parser = argparse.ArgumentParser(
        description='Federated Learning Experiments - Assignment 4'
    )
    parser.add_argument(
        '--task',
        type=str,
        choices=['1', '1-v2', '2', '3', '4', '4-comp', 'all'],
        default='all',
        help='Which task to run (default: all)'
    )
    
    args = parser.parse_args()
    
    print("\n" + "="*70)
    print("EE5102/CS6302 - Advanced Topics in Machine Learning")
    print("Assignment 4: Federated Learning")
    print("="*70)
    
    if args.task == '1':
        run_task1()
    if args.task == '1-v2':
        run_task1_v2()
    elif args.task == '2':
        run_task2()
    elif args.task == '3':
        run_task3()
    elif args.task == '4':
        run_task4()
    elif args.task == '4-comp':
        run_task4_comparison()
    elif args.task == 'all':
        run_all()
    
    print("\n" + "="*70)
    print("ALL EXPERIMENTS COMPLETED!")
    print("="*70)
    print("\nGenerated files:")
    print("- task1_fedsgd_vs_centralized.png")
    print("- task2_local_epochs.png")
    print("- task2_client_sampling.png")
    print("- task3_heterogeneity.png")
    print("- task4_1_fedprox.png")
    print("- task4_2_scaffold.png")
    print("- task4_3_fedgh.png")
    print("- task4_4_fedsam.png")
    print("- task4_comparison_alpha_*.png")
    print("\nCheck the plots for visual results!")


if __name__ == "__main__":
    main()