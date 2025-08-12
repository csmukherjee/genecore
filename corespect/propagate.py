


import numpy as np
import hnswlib
from numba import njit

import networkx as nx

M=200
ef_construction=200
ef=200

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



def CDNN_layer_efficient(X, nodes_this_round, final_labels, ng_num):
    cores = []
    n = len(final_labels)
    for ell in range(n):
        if final_labels[ell] != -1:
            cores.append(ell)

    dim = X.shape[1]

    p = hnswlib.Index(space='l2', dim=dim)
    p.init_index(max_elements=len(cores),
                 ef_construction=ef_construction,
                 M=M)

    labels = np.array(cores, dtype=np.int64)

    p.add_items(X[cores], labels)
    p.set_ef(ef)

    knn_list, knn_dists = p.knn_query(X[nodes_this_round], k=ng_num)

    return knn_list, knn_dists


# We obtain ng_num nearest neighbors to each of the current partitions, and then use this to obtain the final layer-i to layer-i-1 edges. This is to deal with weakness of ANN algorithms in dealing with OOD queries.

def CDNN_layer(X, nodes_this_round, final_labels, ng_num):
    all_neighbors = []
    all_distances = []

    # Build current_partition

    true_k = len(set(final_labels)) - 1

    n = len(final_labels)
    current_partition = [[] for _ in range(true_k)]
    for ell in range(n):
        val = final_labels[ell]
        if val != -1:
            current_partition[val].append(ell)

    # done

    dim = X.shape[1]
    queries = X[nodes_this_round]

    for cluster_nodes in current_partition:
        if len(cluster_nodes) == 0:
            continue

        # 1) Extract data for this cluster
        cluster_data = X[cluster_nodes]

        # 2) Build an HNSW index on the fly
        p = hnswlib.Index(space='l2', dim=dim)
        p.init_index(max_elements=len(cluster_nodes),
                     ef_construction=ef_construction,
                     M=M)

        # 3) Add items, using the *actual* indices as labels
        labels = np.array(cluster_nodes, dtype=np.int64)
        p.add_items(cluster_data, labels)

        # 4) Set query‐time ef
        p.set_ef(ef)

        # 5) Query all rem_nodes at once
        k_ = min(2 * ng_num, len(cluster_nodes))
        # print('len cluster_nodes:', len(cluster_nodes), 'k_:', k_)
        nbrs, dists = p.knn_query(queries, k=k_)

        all_neighbors.append(nbrs)
        all_distances.append(dists)

    # 6) Stack results from every cluster: shape = (Q, C·ng_num)
    all_neighbors = np.concatenate(all_neighbors, axis=1)
    all_distances = np.concatenate(all_distances, axis=1)

    # 7) For each query, pick the top-ng_num closest across *all* clusters
    idx_sort = np.argsort(all_distances, axis=1)[:, :ng_num]

    Q = len(nodes_this_round)
    knn_list = np.zeros((Q, ng_num), dtype=np.int64)
    knn_dists = np.zeros((Q, ng_num), dtype=all_distances.dtype)

    for i in range(Q):
        sel = idx_sort[i]
        knn_list[i] = all_neighbors[i, sel]
        knn_dists[i] = all_distances[i, sel]

    return knn_list, knn_dists


def ng_heat_kernel(distances):
    n_points = len(distances)
    sigmas = np.zeros(n_points)
    target = np.zeros(n_points)
    rhos = np.zeros(n_points)

    P_vec = [[] for _ in range(n_points)]

    # Step 1: Compute rho_i (local connectivity)
    for i in range(n_points):

        rhos[i] = distances[i][0] if len(distances[i]) > 0 else 0  # Minimum nonzero distance

        if rhos[i] != min(distances[i]):
            raise KeyError(f"0-th index is not closest, {rhos[i]:.3f} {min(distances[i]):.3f}")

    # Step 2: Solve for sigma_i using binary search
    def find_sigma(i, target_i):

        lo, hi = 1e-8, 100000.0  # Search range for sigma
        for _ in range(64):  # Binary search
            sigma = (lo + hi) / 2.0
            weights = np.exp(-(distances[i] - rhos[i]) / sigma)
            if np.sum(weights) > target_i:
                hi = sigma
            else:
                lo = sigma
        return (lo + hi) / 2.0

    for i in range(n_points):
        if len(distances[i]) > 0:
            c1 = len(distances[i])

            if c1 == 1:
                target[i] = 1

            elif c1 <= 5:
                target[i] = 1 + c1 / 2

            else:
                target[i] = np.log2(c1)

            sigmas[i] = find_sigma(i, target[i])

    # Step 3: Compute edge weights
    for i in range(n_points):
        for d in distances[i]:
            weight = np.exp(-(d - rhos[i]) / sigmas[i])

            P_vec[i].append(weight)

    P_vec = np.array(P_vec)

    assert np.shape(P_vec) == np.shape(distances)

    for i in range(n_points):
        P_vec[i] = P_vec[i] / sum(P_vec[i])

    return P_vec


@njit
def pseudo_centroid_distances(labels, nodes_this_round, P_vec, ng_list, dist_to_centroids=None, node_to_rank=None):
    n0, k0 = np.shape(ng_list)
    true_k_temp = len(dist_to_centroids[0])

    for i in range(n0):

        u = nodes_this_round[i]

        dists_temp = np.zeros(true_k_temp, dtype=np.float64)
        for j in range(k0):
            v = ng_list[i, j]

            for ell in range(true_k_temp):
                # print(ell)
                # print(v)
                # print(node_to_rank[v])
                # print(dist_to_centroids[node_to_rank[v]])

                dists_temp[ell] += P_vec[i, j] * dist_to_centroids[node_to_rank[v]][ell]

        #            dists_temp+=P_vec[i,j]*dist_to_centroids[node_to_rank[v]]

        c_idx = np.argmin(dists_temp)
        labels[u] = c_idx

        dist_to_centroids.append(dists_temp)

    return labels, dist_to_centroids





def CDNN(X,sorted_points,core_nodes,final_labels,layer_ratio,cluster_assignment_vectors,ng_num):

    num_step= len(layer_ratio) - 1
    n=X.shape[0]

    node_to_rank = -1 * np.zeros(n).astype(int)
    rank_counter = 0
    for node in core_nodes:
        node_to_rank[node] = rank_counter
        rank_counter += 1

    # Current_partition is used only for calculating layer-wise CDNN

    # Find nearest centroid layer by layer for periphery nodes
    for rnd in range(num_step):

        nodes_this_round = sorted_points[int(layer_ratio[rnd] * n):int(layer_ratio[rnd + 1] * n)].astype(int)

        # We create the CDNN graph layer-by-layer
        ng_list, distances = CDNN_layer(X, nodes_this_round, final_labels, ng_num)

        # Get the delegation weights.
        P_vec = ng_heat_kernel(distances)

        # Obtain the final cluster allocation through pseudo distance calculation
        final_labels, cluster_assignment_vectors = pseudo_centroid_distances(final_labels, nodes_this_round, P_vec, ng_list,
                                                                    cluster_assignment_vectors, node_to_rank)

        for peri_node in nodes_this_round:
            node_to_rank[peri_node] = rank_counter
            rank_counter += 1


    print("Finished propagation with CDNN")

    return final_labels



def label_prop(X,sorted_points,core_nodes,final_labels,layer_ratio,cluster_assignment_vectors,ng_num):

    from  sklearn.semi_supervised import LabelPropagation as LP

    lp_model=LP(kernel='knn', n_neighbors=ng_num, max_iter=1000, tol=1e-3)
    lp_model.fit(X,final_labels)
    final_labels= lp_model.transduction_

    print("Finished propagation with scikit-lean label propagation")

    return final_labels


#----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- -----
#Recursive majority addition

def majority_addition(G,cnum,final_labels,remaining_nodes,eps=0):

    added_nodes=set()
    for ell in remaining_nodes:
        votes=np.zeros(cnum)
        for u in G.neighbors(ell):
            if final_labels[u]!=-1:
                    votes[final_labels[u]]+=1

        if max(votes)> (0.5-eps)*G.out_degree(ell):
            final_labels[ell]=np.argmax(votes)
            added_nodes.add(ell)

    remaining_nodes=remaining_nodes-added_nodes

    return final_labels,remaining_nodes


def max_addition(G,cnum,final_labels,remaining_nodes,eps=0):

    added_nodes=set()

    #Calculate max.
    max_list=[]
    for ell in remaining_nodes:
        votes=np.zeros(cnum)
        for u in G.neighbors(ell):
            if final_labels[u]!=-1:
                    votes[final_labels[u]]+=1


        max_list.append(max(votes))


    #Add nodes with max votes greater than median of max_list
    for ell in remaining_nodes:
        votes = np.zeros(cnum)
        for u in G.neighbors(ell):
            if final_labels[u] != -1:
                votes[final_labels[u]] += 1


        thr= np.percentile(max_list, 75) + eps

        if max(votes)> thr:
            final_labels[ell] = np.argmax(votes)
            added_nodes.add(ell)


    remaining_nodes=remaining_nodes-added_nodes

    return final_labels,remaining_nodes




def recursive_majority(X,sorted_points,core_nodes,final_labels,layer_ratio,cluster_assignment_vectors,ng_num):


    n=X.shape[0]


    G= nx.DiGraph()
    knn_list, _ = get_kNN(X, q=ng_num)
    for i in range(n):
        for j in knn_list[i, :]:
            if i != j:
                G.add_edge(i, j, weight=1)

    remaining_nodes=set(G.nodes)-set(core_nodes)


    n0=np.sum(final_labels != -1)
    rounds=0
    tolerance=int(0.005*G.number_of_nodes())

    cnum=len(set(final_labels))-1
    eps=0
    stage=0 #Set stage=1 to go back to normal recursive majority

    while True:
        n0 = np.sum(final_labels != -1)

        final_labels,remaining_nodes=majority_addition(G,cnum,final_labels,remaining_nodes,eps)

        rounds+= 1
        n1=np.sum(final_labels != -1)

        if n1-n0<tolerance:
            break



    print(f"Finished propagation with recursive majority addition in {rounds} rounds and {len(remaining_nodes)} remaining nodes.")
    return final_labels


def adaptive_majority(X,sorted_points,core_nodes,final_labels,layer_ratio,cluster_assignment_vectors,ng_num):


    n=X.shape[0]


    G= nx.DiGraph()
    knn_list, _ = get_kNN(X, q=ng_num)
    for i in range(n):
        for j in knn_list[i, :]:
            if i != j:
                G.add_edge(i, j, weight=1)

    remaining_nodes=set(G.nodes)-set(core_nodes)


    n0=np.sum(final_labels != -1)
    rounds=0
    tolerance=int(0.005*G.number_of_nodes())

    cnum=len(set(final_labels))-1
    eps=0
    stage=0 #Set stage=1 to go back to normal recursive majority

    while True and stage<2:

        n0 = np.sum(final_labels != -1)
        if stage==0:
            final_labels,remaining_nodes=majority_addition(G,cnum,final_labels,remaining_nodes,eps)

        elif stage==1:
            final_labels, remaining_nodes = max_addition(G, cnum, final_labels, remaining_nodes, eps)
            stage+=1

        rounds+= 1
        n1=np.sum(final_labels != -1)

        print("Stage, eps, n1, n0", stage, eps,n1,n0)


        if n1-n0<tolerance:
            stage+=1

        else:
            stage=max(0,stage-1)

        if stage==2:
            break

        if n0>0.95*G.number_of_nodes():
            break


    print(f"Finished propagation with recursive majority addition in {rounds} rounds and {len(remaining_nodes)} remaining nodes.")
    return final_labels

def majority_addition_adaptive(G,cnum,final_labels,remaining_nodes):

    n=G.number_of_nodes()
    hmap=np.zeros(n)
    for i in range(n):
        if final_labels[i]!=-1:
            hmap[i]=1

    H= nx.DiGraph()
    for u,v in G.edges():
        if hmap[u]==1 and hmap[v]==1:
            H.add_edge(u,v,weight=G.edges[u,v]['weight'])

    in_degs=np.zeros(n)
    for u in H.nodes():
        in_degs[u]=H.degree(u)


    added_nodes=set()
    for ell in remaining_nodes:
        temp=0
        votes=np.zeros(cnum)
        for u in G.neighbors(ell):
            if final_labels[u]!=-1:
                    temp+=1
                    votes[final_labels[u]]+=1

        if temp>0.5*G.out_degree(ell) and max(votes)> 0.5*temp:

            final_labels[ell]=np.argmax(votes)
            added_nodes.add(ell)

    remaining_nodes=remaining_nodes-added_nodes

    return final_labels,remaining_nodes



def recursive_majority_adaptive(X,sorted_points,core_nodes,final_labels,layer_ratio,cluster_assignment_vectors,ng_num):


    n=X.shape[0]


    G= nx.DiGraph()
    knn_list, _ = get_kNN(X, q=ng_num)
    for i in range(n):
        for j in knn_list[i, :]:
            if i != j:
                G.add_edge(i, j, weight=1)

    remaining_nodes=set(G.nodes)-set(core_nodes)


    n0=np.sum(final_labels != -1)
    rounds=0
    tolerance=int(0.01*G.number_of_nodes())

    cnum=len(set(final_labels))-1
    while True:
        final_labels,remaining_nodes=majority_addition_adaptive(G,cnum,final_labels,remaining_nodes)

        rounds+= 1
        n1=np.sum(final_labels != -1)

        if n1-n0<tolerance or rounds>10:
            break


    print(f"Finished propagation with recursive majority addition in {rounds} rounds and {len(remaining_nodes)} remaining nodes.")
    return final_labels