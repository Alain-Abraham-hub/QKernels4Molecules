# ===============================================================
# feature_extraction.py
# Extract quantum + classical graph features and save as .npz
# ===============================================================

import numpy as np
from sklearn.preprocessing import StandardScaler
from pathlib import Path

def extract_quantum_walk_features(G, times=None):
    import networkx as nx
    from scipy import linalg as la
    from scipy.stats import skew, kurtosis, entropy

    if times is None:
        times = np.linspace(0.1, 10.0, 30)
    n_nodes = G.number_of_nodes()
    if n_nodes == 0:
        return np.zeros(100)
    
    features = []
    try:
        L = nx.normalized_laplacian_matrix(G).todense()
        L = np.array(L, dtype=np.complex128)

        all_probs = []
        for t in times:
            try:
                U_t = la.expm(-1j * L * t)
                probs = np.abs(np.diag(U_t)) ** 2
                all_probs.append(probs)
            except:
                all_probs.append(np.zeros(n_nodes))
        all_probs = np.array(all_probs)
        avg_probs = all_probs.mean(axis=0)

        # Top-k probabilities
        top_k = min(30, n_nodes)
        sorted_probs = np.sort(avg_probs)[::-1][:top_k]
        features.extend(sorted_probs)
        features.extend([0] * (30 - len(sorted_probs)))

        # Statistical moments
        features.append(np.mean(avg_probs))
        features.append(np.std(avg_probs))
        features.append(skew(avg_probs))
        features.append(kurtosis(avg_probs))
        features.append(entropy(avg_probs + 1e-10))

        # Temporal dynamics
        prob_changes = np.diff(all_probs, axis=0)
        features.append(np.mean(np.abs(prob_changes)))
        features.append(np.std(np.abs(prob_changes)))
        features.append(np.max(np.abs(prob_changes)))

        # Return probability
        return_probs = [all_probs[i, 0] if n_nodes > 0 else 0 for i in range(len(times))]
        features.append(np.mean(return_probs))
        features.append(np.std(return_probs))

        # Spectral features
        eigenvalues = la.eigvalsh(L)
        eigenvalues = np.sort(eigenvalues)[:15]
        features.extend(eigenvalues)
        features.extend([0] * (15 - len(eigenvalues)))

        # Coherence
        ipr = np.sum(avg_probs ** 4)
        features.append(ipr)

        # Pad to 100
        features.extend([0] * (100 - len(features)))
        return np.array(features[:100])
    except:
        return np.zeros(100)

def extract_classical_graph_features(G):
    import networkx as nx
    from scipy.stats import skew

    n_nodes = G.number_of_nodes()
    if n_nodes == 0:
        return np.zeros(80)
    
    features = []
    try:
        # Basic
        features.append(n_nodes)
        features.append(G.number_of_edges())
        features.append(nx.density(G))
        
        # Degrees
        degrees = [G.degree(n) for n in G.nodes()]
        deg_hist, _ = np.histogram(degrees, bins=15, density=True)
        features.extend(deg_hist)
        features.append(np.mean(degrees))
        features.append(np.std(degrees))
        features.append(np.max(degrees))
        features.append(skew(degrees))

        # Clustering
        clustering = list(nx.clustering(G).values())
        clust_hist, _ = np.histogram(clustering, bins=10, density=True)
        features.extend(clust_hist)
        features.append(np.mean(clustering))
        features.append(nx.transitivity(G))

        # Centrality
        if n_nodes <= 100:
            betweenness = list(nx.betweenness_centrality(G).values())
        else:
            betweenness = list(nx.betweenness_centrality(G, k=50).values())
        features.append(np.mean(betweenness))
        features.append(np.max(betweenness))

        # Triangles
        triangles = list(nx.triangles(G).values())
        features.append(np.sum(triangles))
        features.append(np.mean(triangles))

        # Assortativity
        try:
            features.append(nx.degree_assortativity_coefficient(G))
        except:
            features.append(0)

        # Connected components
        features.append(nx.number_connected_components(G))

        # Path-based
        if nx.is_connected(G):
            features.append(nx.average_shortest_path_length(G))
            features.append(nx.diameter(G))
        else:
            largest_cc = max(nx.connected_components(G), key=len)
            G_largest = G.subgraph(largest_cc)
            if G_largest.number_of_nodes() > 1:
                features.append(nx.average_shortest_path_length(G_largest))
                try:
                    features.append(nx.diameter(G_largest))
                except:
                    features.append(0)
            else:
                features.extend([0, 0])

        # Pad to 80
        features.extend([0] * (80 - len(features)))
        return np.array(features[:80])
    except:
        return np.zeros(80)

def extract_features_for_dataset(dataset_name='PROTEINS'):
    print(f"\n🔬 Extracting features for {dataset_name}")
    from torch_geometric.datasets import TUDataset
    from torch_geometric.utils import to_networkx

    dataset = TUDataset(root='./data/TUDataset', name=dataset_name)
    X_quantum, X_classical, y = [], [], []

    for idx, data in enumerate(dataset):
        G = to_networkx(data, to_undirected=True)
        X_quantum.append(extract_quantum_walk_features(G))
        X_classical.append(extract_classical_graph_features(G))
        y.append(int(data.y.item()))
    
    X_quantum = StandardScaler().fit_transform(np.array(X_quantum))
    X_classical = StandardScaler().fit_transform(np.array(X_classical))
    y = np.array(y)

    output_path = f'{dataset_name.lower()}_hybrid_features.npz'
    np.savez(output_path, X_quantum=X_quantum, X_classical=X_classical, y=y)
    print(f"Features saved to {output_path}")
    return output_path

if __name__ == "__main__":
    datasets = ['PROTEINS', 'MUTAG', 'AIDS', 'NCI1', 'PTC_MR']
    for ds in datasets:
        extract_features_for_dataset(ds)
