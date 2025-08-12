""" CoreSPECT Implementation"""

from . import ranking, cluster_core, propagate
from .cav import calculate_cav
import numpy as np


class Corespect():
    def __init__(
        self, 
        q=40, 
        r=40,
        ng_num=20, 
        ranking_algo='FlowRank', 
        layer_ratio=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0], 
        propagate_algo='CDNN',
        cav='dist',
    ):
        # Attributes of the CoreSPECT (CDNN)
        self.q = q
        self.r = r
        self.ng_num = ng_num
        self.ranking_algo = ranking_algo
        self.layer_ratio = layer_ratio
        self.cluster_algo = None
        self.propagate_algo = propagate_algo
        self.num_step = len(layer_ratio) - 1
        self.cav = cav
        
        # Attributes of the Data and Core Clustering Algorithm(To be set later)
        self.cluster_assignment_vectors = None
        self.sorted_points = None
        self.core_labels = None
        self.core_nodes = None
        self.final_labels = None
        self.X = None
        self.n = None
        self.scores = None
        self.cluster_algo = None
        self.cluster_algo_params = None
        

    def _calc_rank(self):
        #If scores are not provided, rank using ranking_algo
        if self.scores is None:
            if hasattr(ranking, self.ranking_algo):
                func = getattr(ranking, self.ranking_algo)
                if callable(func):
                    scores=func(self.X, self.q, self.r)
                else:
                    raise TypeError(f"{self.ranking_algo} is not callable")

            else:
                raise KeyError(f"{self.ranking_algo} is not a valid ranking algorithm")
            print(f"Obtained vertex ranking using {self.ranking_algo}")

        else:
            print("Ranking with passed scores")

        self.sorted_points = np.array(sorted(scores, key=scores.get, reverse=True)).astype(int)

        if self.layer_ratio is not None:
            top_frac = self.layer_ratio[0] #Todo: make this parameter free.

        else:
            raise KeyError("The C-P partitions cannot be None")
        
        self.core_nodes = self.sorted_points[0:int(top_frac * self.n)]

    def _cluster_core_nodes(self):
        # Cluster the core_nodes using cluster_algo
        if hasattr(cluster_core, self.cluster_algo):
            func = getattr(cluster_core, self.cluster_algo)
            if callable(func):
                self.core_labels, self.cluster_assignment_vectors = func(self.X, core_nodes=self.core_nodes, cav=self.cav, cluster_algo_params=self.cluster_algo_params)
            else:
                raise TypeError(f"{self.cluster_algo} is not callable")

        else:
            raise KeyError(f"{self.cluster_algo} is not a valid clustering algorithm")


        # Start generating final labels
        self.final_labels = -1 * np.ones(self.n)
        self.final_labels[self.core_nodes] = self.core_labels
        self.final_labels = self.final_labels.astype(int)

        print("Number of clusters found in the core:", len(set(self.core_labels)))

    def _propagate_labels(self):
        # Label the rest (or some of the rest) of the points using the propagate_algo algorithm
        if hasattr(propagate, self.propagate_algo):
            func = getattr(propagate, self.propagate_algo)
            if callable(func):
                self.final_labels = func(self.X, self.sorted_points, self.core_nodes, self.final_labels, self.layer_ratio, self.cluster_assignment_vectors, self.ng_num)
            else:
                raise TypeError(f"{self.propagate_algo} is not callable")

        else:
            raise KeyError(f"{self.propagate_algo} is not a valid propagation algorithm")


    def find_core(self, X, scores=None):
        self.X = X
        self.n = X.shape[0]

        self._calc_rank()
        
        return self.core_nodes
    
    def propagate_labels(self, core_labels):
        self.core_labels = core_labels
        self.final_labels = -1 * np.ones(self.n)
        self.final_labels[self.core_nodes] = self.core_labels

        self.cluster_assignment_vectors = calculate_cav(X_core = self.X[self.core_nodes], core_labels = self.core_labels, cav = self.cav)
        self._propagate_labels()

        return self.final_labels

    def fit_predict(self, X, scores=None, cluster_algo='k_means', **cluster_algo_params):
        self.X = X
        self.n = X.shape[0]
        self.scores = scores
        self.cluster_algo = cluster_algo
        self.cluster_algo_params = cluster_algo_params

        self._calc_rank()
        self._cluster_core_nodes()
        self._propagate_labels()
        
        return self.final_labels
    

