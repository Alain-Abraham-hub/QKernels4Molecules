# ===============================================================
# ⚛️ Quantum Walk Embedding Preprocessing on TUDataset (PROTEINS)
# ===============================================================

import torch
import numpy as np
import networkx as nx
from scipy.linalg import expm
from torch_geometric.datasets import TUDataset
from torch_geometric.transforms import NormalizeFeatures
from torch_geometric.utils import to_networkx

# ---------------------------------------------------------------
# Step 1: Load and Normalize the Dataset
# ---------------------------------------------------------------
dataset = TUDataset(root='data/TUDataset', name='PROTEINS', transform=NormalizeFeatures())
print(f"Dataset Loaded: {dataset}")
print(f"Number of graphs: {len(dataset)}")

# ---------------------------------------------------------------
# Step 2: Define Helper Function — Build Graph Matrices
# ---------------------------------------------------------------
def build_graph_matrices(data):
    """Convert PyG graph to NetworkX and compute adjacency, degree, Laplacian, and transition matrix."""
    G = to_networkx(data, to_undirected=True)
    G.remove_nodes_from(list(nx.isolates(G)))  # remove isolated nodes

    A = nx.adjacency_matrix(G).todense()
    A = np.array(A, dtype=float)
    
    # Degree matrix
    D = np.diag(np.sum(A, axis=1))
    
    # Normalized Laplacian (for quantum Hamiltonian)
    degrees = np.sum(A, axis=1).ravel() + 1e-8  # flatten to 1D
    D_inv_sqrt = np.diag(1.0 / np.sqrt(degrees))
    L_norm = np.eye(A.shape[0]) - D_inv_sqrt @ A @ D_inv_sqrt

    # Classical transition matrix (for comparison)
    D_inv = np.diag(1.0 / degrees)
    P = D_inv @ A  # random walk transition

    return A, D, L_norm, P

# ---------------------------------------------------------------
# Step 3: Define Quantum Walk Embedding Function
# ---------------------------------------------------------------
def quantum_walk_embedding(L_norm, time_steps=5):
    """
    Compute quantum walk embedding using unitary time evolution.
    Uses time-averaged probability distributions over nodes.
    """
    n = L_norm.shape[0]
    H = L_norm  # Hamiltonian (can also use adjacency A)
    psi0 = np.zeros((n, 1))
    psi0[0, 0] = 1.0  # initial state: start at node 0

    probs_over_time = []

    for t in range(1, time_steps + 1):
        U_t = expm(-1j * H * t)        # unitary evolution
        psi_t = U_t @ psi0             # evolved state
        probs = np.abs(psi_t) ** 2     # probability distribution
        probs_over_time.append(probs.flatten())

    # Time-averaged node probabilities (embedding)
    embedding = np.mean(probs_over_time, axis=0)
    return embedding

# ---------------------------------------------------------------
# Step 4: Generate Quantum Walk Features for Entire Dataset
# ---------------------------------------------------------------
quantum_walk_features = []
labels = []
max_nodes = 0  # To track the largest graph size

# First pass: compute embeddings and find max number of nodes
for idx, data in enumerate(dataset):
    try:
        A, D, L_norm, P = build_graph_matrices(data)
        embedding = quantum_walk_embedding(L_norm, time_steps=5)
        quantum_walk_features.append(embedding)
        labels.append(int(data.y))
        max_nodes = max(max_nodes, embedding.shape[0])
    except Exception as e:
        print(f"Skipping graph {idx} due to error: {e}")

print(f"Max number of nodes across all graphs: {max_nodes}")

# ---------------------------------------------------------------
# Step 5: Pad embeddings to fixed size
# ---------------------------------------------------------------
padded_features = np.array([
    np.pad(embedding, (0, max_nodes - len(embedding))) 
    for embedding in quantum_walk_features
])

labels = np.array(labels)

print(f"\n✅ Extracted {len(padded_features)} quantum walk embeddings.")
print(f"Feature vector shape: {padded_features.shape}")

# ---------------------------------------------------------------
# Step 6: Save for Later Use (optional)
# ---------------------------------------------------------------
np.savez("protein_quantum_walk_features.npz", X=padded_features, y=labels)
print("\n💾 Saved quantum walk features to 'protein_quantum_walk_features.npz'")