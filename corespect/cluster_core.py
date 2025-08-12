
from sklearn.cluster import KMeans
import numpy as np
from . import louvain_setup, densify
from .cav import calculate_cav
from collections import deque

import networkx as nx

import hnswlib

from sklearn.manifold import SpectralEmbedding

# import hdbscan 
from hdbscan import all_points_membership_vectors, HDBSCAN

def get_kNN(X, q=15):
    """
    Generate a k-nearest neighbors graph from the input data.
    :param X: Input data (numpy array).
    :param q: Number of nearest neighbors.
    :return: k-nearest neighbors list and distances.
    """
    n = X.shape[0]
    dim = X.shape[1]
    p = hnswlib.Index(space='l2', dim=dim)
    p.init_index(max_elements=n, ef_construction=200, M=64)
    p.add_items(X)
    p.set_ef(2*q)

    labels, dists = p.knn_query(X, k=q+1)
    knn_list = labels[:, 1:]
    knn_dists = dists[:, 1:]

    return knn_list, knn_dists


def k_means(X,core_nodes,cav, cluster_algo_params):
    # Check if required parameters are provided
    allowed_params = ['k', 'choose_min_obj']
    for key in cluster_algo_params.keys():
        if key not in allowed_params:
            raise ValueError(f"Unwanted parameter found: {key}")
    
    if 'k' not in cluster_algo_params:
        raise ValueError("Parameter 'k' is required for k-means clustering")
    k = cluster_algo_params['k']
    choose_min_obj = cluster_algo_params.get('choose_min_obj', True)
    
    # Perform k-means clustering
    X_core = X[core_nodes]

    if choose_min_obj:
        min_obj_val = float('inf')

        for rounds in range(20):

            kmeans = KMeans(n_clusters=k, n_init=1, max_iter=1000)
            kmeans.fit(X_core)

            centroids = kmeans.cluster_centers_
            obj_val = kmeans.inertia_
            labels_km = kmeans.labels_

            if rounds == 0 or obj_val < min_obj_val:
                min_obj_val = obj_val
                best_centroids = centroids
                best_labels_km = labels_km

        centroids = best_centroids
        core_labels = best_labels_km

    else:
        kmeans = KMeans(n_clusters=k)
        kmeans.fit(X_core)
        centroids = kmeans.cluster_centers_
        core_labels = kmeans.labels_

    cluster_assignment_vectors = calculate_cav(X_core, core_labels, cav=cav)

    print("Clustered core using k-means with cav:", cav)

    return core_labels, cluster_assignment_vectors


def spectral_clustering(X, core_nodes, cav, cluster_algo_params):
    # Check if required parameters are provided
    allowed_params = ['k', 'choose_min_obj']
    for key in cluster_algo_params.keys():
        if key not in allowed_params:
            raise ValueError(f"Unwanted parameter found: {key}")
    
    if 'k' not in cluster_algo_params:
        raise ValueError("Parameter 'k' is required for Spectral Clustering")
    k = cluster_algo_params['k']
    choose_min_obj = cluster_algo_params.get('choose_min_obj', True)
    
    # Perform spectral embedding
    X_core = X[core_nodes]

    SE = SpectralEmbedding(n_components=k, affinity='nearest_neighbors', n_neighbors=15)
    X_core = SE.fit_transform(X_core)

    if choose_min_obj:
        min_obj_val = float('inf')

        for rounds in range(20):

            kmeans = KMeans(n_clusters=k, n_init=1, max_iter=1000)
            kmeans.fit(X_core)

            centroids = kmeans.cluster_centers_
            obj_val = kmeans.inertia_
            labels_km = kmeans.labels_

            if rounds == 0 or obj_val < min_obj_val:
                min_obj_val = obj_val
                best_centroids = centroids
                best_labels_km = labels_km

        centroids = best_centroids
        core_labels = best_labels_km

    else:
        kmeans = KMeans(n_clusters=k)
        kmeans.fit(X_core)
        centroids = kmeans.cluster_centers_
        core_labels = kmeans.labels_

    cluster_assignment_vectors = calculate_cav(X_core, core_labels, cav=cav)

    print("Clustered core using spectral-clustering with cav:", cav)

    return core_labels, cluster_assignment_vectors

def hdbscan(X,core_nodes,cav,cluster_algo_params):
    # Check if required parameters are provided
    allowed_params = ['metric', 'min_cluster_size', 'min_samples', 'alpha']
    for key in cluster_algo_params.keys():
        if key not in allowed_params:
            raise ValueError(f"Unwanted parameter found: {key}")

    min_cluster_size = cluster_algo_params.get('min_cluster_size', 10)
    min_samples = cluster_algo_params.get('min_samples', min_cluster_size)
    metric = cluster_algo_params.get('metric', 'l2')
    alpha = cluster_algo_params.get('alpha', 1.1)
    
    # Perform HDBSCAN clustering
    X_core = X[core_nodes]

    clusterer = HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        metric=metric,
        alpha=alpha,
        prediction_data=True
    ).fit(X_core)


    soft_probs = all_points_membership_vectors(clusterer)
    core_labels = np.argmax(soft_probs, axis=1)


    our_k= len(set(core_labels)) - (1 if -1 in core_labels else 0)

    cluster_assignment_vectors=[]
    
    for i in range(len(core_nodes)):
        vec = []
        for j in range(our_k):
            if core_labels[i] == j:
                vec.append(-1)
            else:
                vec.append(0)
        cluster_assignment_vectors.append(np.array(vec).astype('float64'))

    core_labels_final=np.ones(len(core_nodes)) * -1

    label_map = {lbl: i for i, lbl in enumerate(set(core_labels))}

    for i in range(len(core_nodes)):
        core_labels_final[i] = label_map[core_labels[i]]

    cluster_assignment_vectors = calculate_cav(X_core, core_labels_final, cav=cav)

    print("Clustered core using HDBSCAN with cav:", cav)

    return core_labels_final,cluster_assignment_vectors

