import numpy as np 

def calculate_cav(X_core, core_labels, cav = 'dist'):
    """
    Calculate the cluster assignment vectors (CAV) for the core nodes.
    
    Parameters:
    - X_core: The feature matrix for the core nodes.
    - core_labels: The labels assigned to the core nodes.
    - cav: The type of CAV to calculate ('dist' or 'ind').
    
    Returns:
    - cluster_assignment_vectors: A list of CAVs for each core node.
    """
    cluster_assignment_vectors = []

    if cav == 'dist':
        centroids = np.array([X_core[core_labels == i].mean(axis=0) for i in np.unique(core_labels)])
        for i in range(len(core_labels)):
            vec = [np.linalg.norm(X_core[i] - centroids[j]) for j in range(len(centroids))]
            cluster_assignment_vectors.append(np.array(vec).astype('float64'))

    elif cav == 'ind':
        for i in range(len(core_labels)):
            vec = [-1 if core_labels[i] == j else 0 for j in range(len(np.unique(core_labels)))]
            cluster_assignment_vectors.append(np.array(vec).astype('float64'))

    return cluster_assignment_vectors