
from sklearn.cluster import KMeans
import numpy as np
from . import densify, louvain_setup
from collections import deque

import networkx as nx

import hnswlib

from hdbscan import HDBSCAN, all_points_membership_vectors


def get_kNN(X, q=15):
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


def k_means(X,true_k,choose_min_obj=True):

    if choose_min_obj:
        min_obj_val = float('inf')

        for rounds in range(20):

            kmeans = KMeans(n_clusters=true_k, n_init=1, max_iter=1000)
            kmeans.fit(X)

            centroids = kmeans.cluster_centers_
            obj_val = kmeans.inertia_
            labels_km = kmeans.labels_

            if rounds == 0 or obj_val < min_obj_val:
                min_obj_val = obj_val
                best_centroids = centroids
                best_labels_km = labels_km

        centroids = best_centroids
        labels_km = best_labels_km

    else:
        kmeans = KMeans(n_clusters=true_k)
        kmeans.fit(X)
        centroids = kmeans.cluster_centers_
        labels_km = kmeans.labels_


    return labels_km


def spectral_clustering(X,true_k,choose_min_obj=True):

    from sklearn.manifold import SpectralEmbedding

    SE = SpectralEmbedding(n_components=true_k, affinity='nearest_neighbors', n_neighbors=15)
    X_SE = SE.fit_transform(X)

    return k_means(X_SE, true_k, choose_min_obj=choose_min_obj)


from . import louvain_setup
from collections import deque

import networkx as nx



M=200
ef_construction=200
ef=200

import hnswlib
def get_kNN(X, q=15):
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


def louvain(X,ng_num=15,resolution=1.0):


    n=X.shape[0]

    G= nx.DiGraph()
    knn_list, _ = get_kNN(X, q=ng_num)
    for i in range(n):
        for j in knn_list[i, :]:
            if i != j:
                G.add_edge(i, j, weight=1)


#    Gp=densify.Densify_v0(G, Gp, core_nodes, ng_num)


    total_partition= louvain_setup.louvain_partitions(G, weight="weight", resolution=resolution)
    partition_ = deque(total_partition, maxlen=1).pop()
    label_map= louvain_setup.partition_to_label(partition_)

    louvain_labels=[]
    for ell in  range(n):
        louvain_labels.append(label_map[ell])


    return louvain_labels


def hdbscan(X,min_sample=10,min_cluster_size=10):
    
    clusterer = HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min_sample,
        metric='l2',
        alpha=1.1,
        prediction_data = True
    ).fit(X)


    soft_probs = all_points_membership_vectors(clusterer)
    soft_labels = np.argmax(soft_probs, axis=1)


    return soft_labels