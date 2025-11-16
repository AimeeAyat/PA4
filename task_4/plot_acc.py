import matplotlib.pyplot as plt
import numpy as np

# -------------------------------
# Data from your table
# -------------------------------
methods = [
    "FedAvg (α=0.1)",
    "FedScaffolding (α=0.1)",
    "FedProx (α=0.1)",
    "FedSAM ρ=0.05 (α=0.1)",
    "FedGH (α=0.01)",
    "FedAvg (α=0.01)"
]

accuracies = [
    63.48,
    68.21,
    63.84,
    65.08,
    59.01,
    50.10
]

# Colors (optional, all default if removed)
colors = [
    "#4e79a7", "#59a14f", "#9c755f", "#f28e2b",
    "#e15759", "#76b7b2"
]

# -------------------------------
# Plot
# -------------------------------
plt.figure(figsize=(12, 6))

bars = plt.bar(methods, accuracies)

# Annotate accuracy on each bar
for bar, acc in zip(bars, accuracies):
    plt.text(bar.get_x() + bar.get_width()/2, 
             bar.get_height() + 0.5, 
             f"{acc:.2f}%", 
             ha='center', va='bottom', fontsize=10)

plt.ylabel("Accuracy (%)")
plt.title("Accuracy Comparison Across FL Methods")
plt.xticks(rotation=45, ha='right')

plt.tight_layout()
plt.show()
