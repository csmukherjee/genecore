import numpy as np
import copy
import numba
from numba import njit


def calc_KNN(pairwise_dist_matrix, k):
    #Returns pairwise distances matrix with entries 0 if not in the k-nearest neighbors of each other
    #pairwise_dist_matrix: numpy array of shape (n_samples, n_samples)
    #k: int, number of nearest neighbors to consider
    #returns: numpy array of shape (n_samples, n_samples)
    KNN = np.zeros_like(pairwise_dist_matrix)
    for i in range(pairwise_dist_matrix.shape[0]):
        idx = np.argsort(pairwise_dist_matrix[i])[:k+1]
        KNN[i][idx] = pairwise_dist_matrix[i][idx]
    
    return KNN

def spectral_2D_embedding(pairwise_dist_matrix, n_components=2):
    #pairwise_dist_matrix: numpy array of shape (n_samples, n_samples)
    #n_components: int, number of dimensions to embed the data into
    #returns: numpy array of shape (n_samples, n_components)

    #use TruncatedSVD 
    from sklearn.decomposition import TruncatedSVD
    svd = TruncatedSVD(n_components=n_components)
    embedded = svd.fit_transform(pairwise_dist_matrix)
    print("Spectral embedding shape: ", embedded.shape)
    return embedded
    

def make_epochs_per_sample(weights, n_epochs):
    result = -1.0 * np.ones(weights.shape[0], dtype=np.float64)
    n_samples = n_epochs * (weights / weights.max())
    result[n_samples > 0] = float(n_epochs) / np.float64(n_samples[n_samples > 0])
    return result

@numba.njit()
def clip(val):
    if val > 4.0:
        return 4.0
    elif val < -4.0:
        return -4.0
    else:
        return val


# @numba.njit(
#     "f4(f4[::1],f4[::1])",
#     fastmath=True,
#     cache=True,
#     locals={
#         "result": numba.types.float32,
#         "diff": numba.types.float32,
#         "dim": numba.types.intp,
#         "i": numba.types.intp,
#     },
# )
# def rdist(x, y): #squared euclidean distance
#     result = 0.0
#     dim = x.shape[0]
#     for i in range(dim):
#         diff = x[i] - y[i]
#         result += diff * diff
#
#     return result


def wij(w, fr_i, fr_j):
    return 1.0 * w * fr_i * fr_j 



@njit
def _optimize_layout_euclidean_single_epoch(
    coordinates, #(nx2 array)
    #weights[i] = w(edge_from[i], edge_to[i])
    weights, # (w(i,j)) 1d array of length |E|
    edge_from, # 1d array of length |E|
    edge_to, # 1d array of length |E| 
    flow_rank, # 1d array of length |V| (flowrank value
    epochs_per_sample,
    epoch_of_next_sample,
    epochs_per_negative_sample,
    epoch_of_next_negative_sample,
    n,
    alpha, #learning rate       
    #can use func find_ab_params as well 
    dim,
    a,
    b,
    repulsion_ratio,
):
    n_vertices = coordinates.shape[0]
    
    for i in numba.prange(epochs_per_sample.shape[0]):
        
        #Attraction
        if epoch_of_next_sample[i] <= n:
            #print('doing attraction')
            j = edge_from[i]
            k = edge_to[i]
            
            #2d coordinates of the points j and k
            current = coordinates[j]
            other = coordinates[k]
            #print('coordinates from: ', current, ' coordinates to: ', other)
            #replaced rdist with inline
            result = 0.0
            dim = current.shape[0]
            #print('dim: ', dim)
            for dd in range(dim):
                diff = current[dd] - other[dd]
                result += diff * diff

            dist_squared = result

            if dist_squared > 0.0:
                grad_coeff = -2.0 * a * b * pow(dist_squared, b - 1.0)
                grad_coeff /= a * pow(dist_squared, b) + 1.0
                # *********************************** #
                # Multiply by the function of w(i,j), fr(i) and fr(j)
                #grad_coeff *=1.0 *weights[i]*flow_rank[j]*flow_rank[k]

                # *********************************** #
            else:
                grad_coeff = 0.0

            for d in range(dim):
                grad_d = clip(grad_coeff * (current[d] - other[d]))
                current[d] += grad_d * alpha #update of location

            epoch_of_next_sample[i] += epochs_per_sample[i]    
            #Repulsion
            n_neg_samples = int(
                (n - epoch_of_next_negative_sample[i]) / epochs_per_negative_sample[i]
            )

            for p in range(n_neg_samples):
                # They have random seed for each vertex (If too slow maybe)
                # k = tau_rand_int(rng_state_per_sample[j]) % n_vertices
                k = np.random.randint(n_vertices)
                # random point k's 2D-coordinates
                other = coordinates[k]

                #replaced rdist with inline
                result = 0.0
                dim = current.shape[0]
                for dd in range(dim):
                    diff = current[dd] - other[dd]
                    result += diff * diff
                
                dist_squared = result

                if dist_squared > 0.0:
                    grad_coeff = 2.0 * b
                    grad_coeff /= (0.001 + dist_squared) * (
                        a * pow(dist_squared, b) + 1
                    )
                    # *********************************** #
                    # Multiply by the function of w(i,j), fr(i) and fr(j)
                    #grad_coeff *= 1 - wij(weights[i],flow_rank[j],flow_rank[k])
                    #grad_coeff *= 1.0- weights[i]*flow_rank[j]*flow_rank[k]
                    # *********************************** #
                elif j == k:
                    continue
                else:
                    grad_coeff = 0.0

                for d in range(dim):
                    if grad_coeff > 0.0:
                        grad_d = clip(grad_coeff * (current[d] - other[d]))
                    else:
                        grad_d = 0
                    current[d] += grad_d * alpha

            epoch_of_next_negative_sample[i] += (
                n_neg_samples * epochs_per_negative_sample[i]
            )




#only need parameters (coordinates, weights, edge_from, edge_to, flow_rank)
def umap_optimize_layout_euclidean(
        coordinates, #(nx2 array)
        #weights[i] = w(edge_from[i], edge_to[i])
        weights, # (w(i,j)) 1d array of length |E|
        edge_from, # 1d array of length |E|
        edge_to, # 1d array of length |E| 
        flow_rank, # 1d array of length |V| (flowrank value of each node)
        total_epochs = 200, #number of total epochs to run
        initial_alpha = 1, #learning rate       
        #can use func find_ab_params as well 
        a = 1.579,
        b = 0.895,
        repulsion_ratio = 5.0, #how many repulsion compared to one attraction
):
    alpha = initial_alpha
    dim = coordinates.shape[1]
    
    epochs_per_sample = make_epochs_per_sample(weights, total_epochs)
    epochs_per_negative_sample = epochs_per_sample / repulsion_ratio
    epoch_of_next_negative_sample = epochs_per_negative_sample.copy()
    #epoch_of_next_negative_sample = copy.deepcopy(epoch_per_negative_sample)
    
    epoch_of_next_sample = epochs_per_sample.copy()
    #epoch_of_next_sample = copy.deepcopy(epochs_per_sample)
    


    #Optimize
    embedding_list = [] # list of all intermediate steps of the coordinates change
    #embedding_list.append(coordinates.copy())
    for n in range(total_epochs):


        _optimize_layout_euclidean_single_epoch(
            coordinates, #(nx2 array)
            #weights[i] = w(edge_from[i], edge_to[i])
            weights, # (w(i,j)) 1d array of length |E|
            edge_from, # 1d array of length |E|
            edge_to, # 1d array of length |E| 
            flow_rank, # 1d array of length |V| (flowrank value of each node)
            epochs_per_sample,
            epoch_of_next_sample,
            epochs_per_negative_sample,
            epoch_of_next_negative_sample,
            n,
            alpha, #learning rate       
            #can use func find_ab_params as well 
            dim,
            a,
            b,
            repulsion_ratio,
        )

        alpha = initial_alpha * (1.0 - (float(n) / float(total_epochs)))
        #embedding_list.append(coordinates.copy())
    
    
    return coordinates
    #return embedding_list
    


import importlib
def run_custom_umap(edge_list_final,weights_final,n,epochs=400, precomputed=None):
    edge_from=[]
    edge_to=[]
    for i,j in edge_list_final:
        edge_from.append(int(i))
        edge_to.append(int(j))


    #edge_from, edge_to = map(lambda x: np.array(x, dtype=int), zip(*G1.edges()))
    if precomputed is not None:
        embedding = precomputed
    else:
        edge_from=np.array(edge_from)
        edge_to=np.array(edge_to)

        coordinates=np.random.uniform(low=-1, high=1, size=(n,2))

        print(type(coordinates))
        flow_rank=np.ones((n))
  
        embedding=umap_optimize_layout_euclidean(
            coordinates, #(nx2 array)
            #weights[i] = w(edge_from[i], edge_to[i])
            weights_final, # (w(i,j)) 1d array of length |E|
            edge_from, # 1d array of length |E|
            edge_to, # 1d array of length |E|
            flow_rank, # 1d array of length |V| (flowrank value of each node)
            total_epochs=epochs)


    return embedding

# def find_ab_params(spread=1, min_dist=0.1):
#     """Fit a, b params for the differentiable curve used in lower
#     dimensional fuzzy simplicial complex construction. We want the
#     smooth curve (from a pre-defined family with simple gradient) that
#     best matches an offset exponential decay.
#     """
#
#     def curve(x, a, b):
#         return 1.0 / (1.0 + a * x ** (2 * b))
#
#     xv = np.linspace(0, spread * 3, 300)
#     yv = np.zeros(xv.shape)
#     yv[xv < min_dist] = 1.0
#     yv[xv >= min_dist] = np.exp(-(xv[xv >= min_dist] - min_dist) / spread)
#     params, covar = curve_fit(curve, xv, yv)
#     return params[0], params[1]
