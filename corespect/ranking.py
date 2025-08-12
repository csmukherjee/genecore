import numpy as np
import hnswlib
import networkx as nx
from numba import njit
from numba.typed import List
from . import clustering_algo as cs_algo


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


def init_random_walk(G,init_walk_len=10):
    n = G.number_of_nodes()
    v_cover=np.ones((n))

    hmap={}
    t=0
    for u in G.nodes():
        hmap[u]=t
        t+=1


    in_list = [[] for _ in range(n)]
    degs=np.zeros((n))
    for u in G.nodes():
        degs[hmap[u]]=G.out_degree(u)



    deg_list=[]
    for u in G.nodes():
        x=[]
        for v in G.predecessors(u):
            wt_uv=G.edges[v,u]['weight']
            x.append(wt_uv/degs[hmap[v]])
            in_list[hmap[u]].append(hmap[v])
        deg_list.append(np.array(x))


    #Now change v_cover:
    for ell in range(init_walk_len):

        v_cover_n=np.zeros((n))
        for i in range(n):
            v_cover_n[i]=sum(deg_list[i]*v_cover[in_list[i]])

        v_cover=v_cover_n.copy()

    init_score={}

    #Add init_score back to the graph
    for u in G.nodes():
        init_score[u]=np.float32(v_cover[hmap[u]])

    return init_score


@njit
def ascending_walk(ng_list,init_score,times):



    n0=len(init_score)
    v_cover=np.zeros((n0))

    walker=[]
    walker_ng_list=[]

    for ell in range(n0):
        x=[]
        for v in ng_list[ell]:
            if init_score[v]>init_score[ell]:
                x.append(v)


        walker.append(x)
        walker_ng_list.append(len(x))



    v_cover_n=np.zeros((n0))

    for rounds in range(times):

        #print("rounds=",rounds)

        v_cover=np.zeros((n0))

        for u in range(n0):
            v=u
            while walker_ng_list[v]>0:
                v= walker[v][np.random.randint(0, walker_ng_list[v])]

            v_cover[u]=init_score[v]

        v_cover_n=v_cover_n+v_cover


    v_cover_n=v_cover_n/times

    return v_cover_n



def FLOW_rank_optimized(X,init_score,r):

    times=200

    knn_list, _=get_kNN(X, r)
    G=nx.DiGraph()
    for i in range(len(knn_list)):
        for j in knn_list[i]:
            G.add_edge(i,j,weight=1)

    n=X.shape[0]



    ng_list=[]

    #Preparing for numba. ng_list is a n*r matrix.
    for ell in range(n):
      x=[]
      for v in G.neighbors(ell):
        x.append(int(v))

      ng_list.append(List(x))

    init_score_numba=np.zeros((n))
    for u in init_score:
        init_score_numba[u]=init_score[u]

    v_cover_n=ascending_walk(ng_list,init_score_numba,times)

    final_score={}
    for u in G.nodes():
        final_score[u]=init_score[u]/(v_cover_n[u]+0.00001)




    return final_score


def FlowRank(X,q=20,r=20):


    #Get initial density estimation.

    knn_list, _=get_kNN(X, q)
    G=nx.DiGraph()
    for i in range(len(knn_list)):
        for j in knn_list[i]:
            G.add_edge(i,j,weight=1)


    init_score=init_random_walk(G)

    final_score=FLOW_rank_optimized(X,init_score,r)


    return final_score

#FlowRank based on number of hops
@njit
def ascending_walk_count_hops(ng_list,init_score,times):



    n0=len(init_score)
    v_cover=np.zeros((n0))

    walker=[]
    walker_ng_list=[]

    for ell in range(n0):
        x=[]
        for v in ng_list[ell]:
            if init_score[v]>init_score[ell]:
                x.append(v)


        walker.append(x)
        walker_ng_list.append(len(x))



    v_cover_n=np.zeros((n0))

    for rounds in range(times):

        #print("rounds=",rounds)

        v_cover=np.zeros((n0))

        for u in range(n0):
            c=0
            v=u
            while walker_ng_list[v]>0:
                v= walker[v][np.random.randint(0, walker_ng_list[v])]
                c+=1

            v_cover[u]=c

        v_cover_n=v_cover_n+v_cover


    v_cover_n=v_cover_n/times

    return v_cover_n


def FLOW_rank_optimized_hops(X,init_score,r):

    times=200

    knn_list, _=get_kNN(X, r)
    G=nx.DiGraph()
    for i in range(len(knn_list)):
        for j in knn_list[i]:
            G.add_edge(i,j,weight=1)

    n=X.shape[0]



    ng_list=[]

    #Preparing for numba. ng_list is a n*r matrix.
    for ell in range(n):
      x=[]
      for v in G.neighbors(ell):
        x.append(int(v))

      ng_list.append(List(x))

    init_score_numba=np.zeros((n))
    for u in init_score:
        init_score_numba[u]=init_score[u]

    v_cover_n=ascending_walk_count_hops(ng_list,init_score_numba,times)

    final_score={}
    for u in G.nodes():
        final_score[u]=-v_cover_n[u]




    return final_score



def FlowRank_count_hops(X,q=20,r=20):


    #Get initial density estimation.

    knn_list, _=get_kNN(X, q)
    G=nx.DiGraph()
    for i in range(len(knn_list)):
        for j in knn_list[i]:
            G.add_edge(i,j,weight=1)


    init_score=init_random_walk(G)

    final_score=FLOW_rank_optimized_hops(X,init_score,r)


    return final_score


#Graph with varying search radius



#Partition and then FlowRank


import numpy as np
from collections import defaultdict


def assign_score_bins_per_cluster(cluster_labels: np.ndarray, score_dict: dict) -> dict:
    n = len(cluster_labels)
    assert n == len(score_dict), "Mismatch in number of points"

    # Group points by cluster
    clusters = defaultdict(list)
    for idx, cluster in enumerate(cluster_labels):
        clusters[cluster].append(idx)

    score_bins = {}

    for cluster_id, indices in clusters.items():
        # Get scores for this cluster
        scores = [(i, score_dict[i]) for i in indices]
        scores.sort(key=lambda x: x[1], reverse=True)

        k = len(scores)
        num_bins = 30
        bin_size = max(1, k // num_bins)

        for rank, (i, _) in enumerate(scores):
            bin_id = rank // bin_size
            bin_id = min(bin_id, num_bins - 1)  # handle edge cases
            score_bins[i] = 1.0 - (1/num_bins * bin_id)

    return score_bins


# import clustering_algo as cs_algo
def Filtered_FlowRank(X,q=20,r=20):


    first_labels=cs_algo.louvain(X,ng_num=15,resolution=0.3)
    first_score=FlowRank(X,q,r)

    #Re-Rank based on first_labels and first_score
    final_score=assign_score_bins_per_cluster(first_labels,first_score)

    return final_score



#Standard rankings: Degree and PageRank
def PageRank(X,q=20,r=20):

    alpha=0.85

    knn_list, _=get_kNN(X, q)
    G=nx.DiGraph()
    for i in range(len(knn_list)):
        for j in knn_list[i]:
            G.add_edge(i,j,weight=1)

    pagerank_scores = nx.pagerank(G,alpha=alpha)

    return pagerank_scores

