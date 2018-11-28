#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Nov 28, 2018
@author: Quentin Lutz <qlutz@enst.fr>
"""

try:
    from numba import njit
except ImportError:
    def njit(func):
        return func

import numpy as np
from scipy import sparse


class NormalizedSignedGraph:
    """
    A class of graph specialized for Louvain algorithm.

    Attributes
    ----------
    n_nodes: the number of nodes in the graph
    TODO// Define parameters
    """

    def __init__(self, adj_matrix, pos_node_weights='degree', neg_node_weights='degree'):
        """

        Parameters
        ----------
        adj_matrix: adjacency matrix of the graph in a SciPy sparse matrix
        TODO// Define parameters
        """
        self.n_nodes = adj_matrix.shape[0]
        self.pos_norm_adj = adj_matrix[adj_matrix > 0]
        self.neg_norm_adj = - adj_matrix[adj_matrix < 0]
        tot_sum = self.pos_norm_adj.sum() + self.neg_norm_adj.sum()
        self.pos_norm_adj = self.pos_norm_adj / tot_sum
        self.neg_norm_adj = self.neg_norm_adj / tot_sum
        self.pos_node_weights = self._check_node_weights(pos_node_weights, 'pos')
        self.neg_node_weights = self._check_node_weights(neg_node_weights, 'neg')

    def _check_node_weights(self, node_weights, adj_matrix):
        if type(node_weights) == np.ndarray:
            if len(node_weights) != self.n_nodes:
                raise ValueError('The number of node weights should match the number of nodes.')
            if any(node_weights < np.zeros(self.n_nodes)):
                raise ValueError('All node weights should be non-negative.')
        elif type(node_weights) == str:
            if node_weights == 'degree':
                if adj_matrix == 'pos':
                    node_weights = self.pos_norm_adj.dot(np.ones(self.n_nodes))
                elif adj_matrix == 'neg':
                    node_weights = self.neg_norm_adj.dot(np.ones(self.n_nodes))
            elif node_weights == 'uniform':
                node_weights = np.ones(self.n_nodes) / self.n_nodes
            else:
                raise ValueError('Unknown distribution type.')
        else:
            raise TypeError(
                'Node weights must be a known distribution ("degree" or "uniform" string) or a custom NumPy array.')
        return node_weights

    def aggregate(self, membership):
        """
        Aggregates nodes belonging to the same clusters.

        Parameters
        ----------
        membership: scipy sparse matrix of shape n_nodes x n_clusters

        Returns
        -------
        the aggregated graph
        """
        self.pos_norm_adj = membership.T.dot(self.pos_norm_adj.dot(membership)).tocsr()
        self.neg_norm_adj = membership.T.dot(self.neg_norm_adj.dot(membership)).tocsr()
        self.pos_node_weights = np.array(membership.T.dot(self.pos_node_weights).T)
        self.neg_node_weights = np.array(membership.T.dot(self.neg_node_weights).T)
        self.n_nodes = self.pos_norm_adj.shape[0]
        return self


class GreedyModularity:
    """
    A greedy modularity optimizer.

    Attributes
    ----------
    score_: total increase of modularity after fitting
    labels_: partition of the nodes. labels[node] = cluster_index
    """

    graph_type = NormalizedSignedGraph

    def __init__(self, resolution=1., tol=0., shuffle_nodes=False):
        """

        Parameters
        ----------
        resolution: modularity resolution
        tol: minimum modularity increase to enter a new optimization pass
        shuffle_nodes: whether to shuffle the nodes before beginning an optimization pass
        """
        self.resolution = resolution
        self.tol = tol
        self.shuffle_nodes = shuffle_nodes
        self.score_ = None
        self.labels_ = None

    def fit(self, graph: NormalizedSignedGraph):
        """
        Iterates over the nodes of the graph and moves them to the cluster of highest increase among their neighbors.
        Parameters
        ----------
        graph: the graph to cluster

        Returns
        -------
        self

        """
        increase = True
        total_increase = 0.

        labels: np.ndarray = np.arange(graph.n_nodes)
        pos_clusters_proba: np.ndarray = graph.pos_node_weights.copy()
        pos_self_loops: np.ndarray = graph.pos_norm_adj.diagonal()

        while increase:
            increase = False
            pass_increase = 0.

            if self.shuffle_nodes:
                nodes = np.random.permutation(np.arange(graph.n_nodes))
            else:
                nodes = range(graph.n_nodes)

            for node in nodes:
                node_cluster: int = labels[node]

                # positive modularity

                pos_neighbor_weights: np.ndarray = graph.pos_norm_adj.data[
                                           graph.pos_norm_adj.indptr[node]:graph.pos_norm_adj.indptr[node + 1]]
                pos_neighbors: np.ndarray = graph.pos_norm_adj.indices[
                                        graph.pos_norm_adj.indptr[node]:graph.pos_norm_adj.indptr[node + 1]]
                pos_neighbors_clusters: np.ndarray = labels[pos_neighbors]
                pos_unique_clusters: list = list(set(pos_neighbors_clusters.tolist()) - {node_cluster})
                pos_n_clusters: int = len(pos_unique_clusters)

                if pos_n_clusters > 0:
                    pos_node_proba: float = graph.pos_node_weights[node]
                    pos_node_ratio: float = self.resolution * pos_node_proba

                    # node_weights of connections to all other nodes in original cluster
                    out_delta: float = (pos_self_loops[node] - pos_neighbor_weights.dot(pos_neighbors_clusters == node_cluster))
                    # proba to choose (node, other_neighbor) among original cluster
                    out_delta += pos_node_ratio * (pos_clusters_proba[node_cluster] - pos_node_proba)

                    local_delta: np.ndarray = np.full(pos_n_clusters, out_delta)

                    for index_cluster, cluster in enumerate(pos_unique_clusters):
                        # node_weights of connections to all other nodes in candidate cluster
                        in_delta: float = pos_neighbor_weights.dot(pos_neighbors_clusters == cluster)
                        # proba to choose (node, other_neighbor) among new cluster
                        in_delta -= pos_node_ratio * pos_clusters_proba[cluster]

                        local_delta[index_cluster] += in_delta

                    # negative modularity
                    # TODO// Negative modularity

                    best_delta: float = 2 * max(local_delta)
                    if best_delta > 0:
                        pass_increase += best_delta
                        best_cluster = pos_unique_clusters[local_delta.argmax()]

                        pos_clusters_proba[node_cluster] -= pos_node_proba
                        pos_clusters_proba[best_cluster] += pos_node_proba
                        labels[node] = best_cluster

            total_increase += pass_increase
            if pass_increase > self.tol:
                increase = True

        self.score_ = total_increase
        _, self.labels_ = np.unique(labels, return_inverse=True)

        return self


@njit
def fit_core(shuffle_nodes, n_nodes, node_weights, resolution, self_loops, tol, indptr, indices, weights):
    """

    Parameters
    ----------
    shuffle_nodes: if True, a random permutation of the node is done. The natural order is used otherwise
    n_nodes: number of nodes in the graph
    edge_weights: the edge weights in the graph
    adjacency: the adjacency matrix without weights
    node_weights: the node weights in the graph
    resolution: the resolution for the Louvain modularity
    self_loops: the weights of the self loops for each node
    tol: the minimum desired increase for each maximization pass
    indptr: the indptr array from the Scipy CSR adjacency matrix
    indices: the indices array from the Scipy CSR adjacency matrix
    weights: the data array from the Scipy CSR adjacency matrix

    Returns
    -------
    a tuple consisting of:
        -the labels found by the algorithm
        -the score of the algorithm (total modularity increase)
    """
    increase = True
    total_increase = 0

    labels: np.ndarray = np.arange(n_nodes)
    clusters_proba: np.ndarray = node_weights.copy()

    local_cluster_weights = np.full(n_nodes, 0.0)
    nodes = np.arange(n_nodes)
    while increase:
        increase = False
        pass_increase = 0.

        if shuffle_nodes:
            nodes = np.random.permutation(np.arange(n_nodes))

        for node in nodes:
            node_cluster = labels[node]

            for k in range(indptr[node], indptr[node + 1]):
                local_cluster_weights[labels[indices[k]]] += weights[k]

            unique_clusters = set(labels[indices[indptr[node]:indptr[node + 1]]])
            unique_clusters.discard(node_cluster)

            if len(unique_clusters):
                node_proba = node_weights[node]
                node_ratio = resolution * node_proba

                # neighbors_weights of connections to all other nodes in original cluster
                out_delta = self_loops[node] - local_cluster_weights[node_cluster]

                # proba to choose (node, other_neighbor) among original cluster
                out_delta += node_ratio * (clusters_proba[node_cluster] - node_proba)

                best_delta = 0.0
                best_cluster = node_cluster

                for cluster in unique_clusters:
                    # neighbors_weights of connections to all other nodes in candidate cluster
                    in_delta = local_cluster_weights[
                        cluster]  # np.sum(neighbors_weights[neighbors_clusters == cluster])
                    local_cluster_weights[cluster] = 0.0
                    # proba to choose (node, other_neighbor) among new cluster
                    in_delta -= node_ratio * clusters_proba[cluster]
                    local_delta = 2 * (out_delta + in_delta)
                    if local_delta > best_delta:
                        best_delta = local_delta
                        best_cluster = cluster

                if best_delta > 0:
                    pass_increase += best_delta
                    clusters_proba[node_cluster] -= node_proba
                    clusters_proba[best_cluster] += node_proba
                    labels[node] = best_cluster
            local_cluster_weights[node_cluster] = 0.0

        total_increase += pass_increase
        if pass_increase > tol:
            increase = True
    return labels, total_increase


class GreedyModularityJiT:
    """
    A greedy modularity optimizer using Numba for enhanced performance.

    Tested with Numba v0.40.1.

    Attributes
    ----------
    score_: total increase of modularity after fitting
    labels_: partition of the nodes. labels[node] = cluster_index
    """

    def __init__(self, resolution=1., tol=0., shuffle_nodes=False):
        """

        Parameters
        ----------
        resolution: modularity resolution
        tol: minimum modularity increase to enter a new optimization pass
        shuffle_nodes: whether to shuffle the nodes before beginning an optimization pass
        """
        self.resolution = resolution
        self.tol = tol
        self.shuffle_nodes = shuffle_nodes
        self.score_ = None
        self.labels_ = None

    def fit(self, graph: NormalizedSignedGraph):
        """
        Iterates over the nodes of the graph and moves them to the cluster of highest increase among their neighbors.
        Parameters
        ----------
        graph: the graph to cluster

        Returns
        -------
        self

        """
        self_loops: np.ndarray = graph.norm_adj.diagonal()

        res_labels, total_increase = fit_core(self.shuffle_nodes,
                                              graph.n_nodes,
                                              graph.node_weights,
                                              self.resolution,
                                              self_loops,
                                              self.tol,
                                              graph.norm_adj.indptr,
                                              graph.norm_adj.indices,
                                              graph.norm_adj.data)

        self.score_ = total_increase
        _, self.labels_ = np.unique(res_labels, return_inverse=True)

        return self


class Louvain:
    """
    Macro algorithm for Louvain clustering.

    Several versions of the Greedy Modularity Maximization are available.
    Those include a pure Python version which is used by default.
    A Numba version named 'GreedyModularityJiT' is also available.

    Attributes
    ----------
    labels_: partition of the nodes. labels[node] = cluster_index
    iteration_count_: number of aggregations performed during the last run of the "fit" method

    Example
    -------
    >>>louvain = Louvain()
    >>>graph = sparse.identity(3, format='csr')
    >>>louvain.fit(graph).labels_
        array([0, 1, 2])
    >>>louvain_jit = Louvain(algorithm=GreedyModularityJiT())
    >>>louvain_jit.fit(graph).labels_
        array([0, 1, 2])
    """

    def __init__(self, algorithm=GreedyModularity, tol=0., max_agg_iter: int = -1, verbose=0):
        """

        Parameters
        ----------
        algorithm: the fixed level optimization algorithm, requires a fit method and score_ and labels_ attributes.
        tol: the minimum modularity increase to keep aggregating.
        max_agg_iter: the maximum number of aggregations to perform, a negative value is interpreted as no limit
        verbose: enables verbosity
        """
        self.algorithm = algorithm()
        self.graph_type = algorithm.graph_type
        self.tol = tol
        if type(max_agg_iter) != int:
            raise TypeError('The maximum number of iterations should be a integer')
        self.max_agg_iter = max_agg_iter
        self.verbose = verbose
        self.labels_ = None
        self.iteration_count_ = None

    def fit(self, adj_matrix: sparse.csr_matrix, node_weights="degree"):
        """
        Alternates local optimization and aggregation until convergence.
        Parameters
        ----------
        adj_matrix: adjacency matrix of the graph to cluster
        node_weights: node node_weights distribution to be used in the second term of the modularity

        Returns
        -------
        self
        """
        if type(adj_matrix) != sparse.csr_matrix:
            raise TypeError('The adjacency matrix should be in a scipy compressed sparse row (csr) format.')
        # check that the graph is not directed
        if adj_matrix.shape[0] != adj_matrix.shape[1]:
            raise ValueError('The adjacency matrix should be square.')
        if (adj_matrix != adj_matrix.T).nnz != 0:
            raise ValueError('The graph should not be directed. Please fit a symmetric adjacency matrix.')
        graph = self.graph_type(adj_matrix, node_weights)
        membership = sparse.identity(graph.n_nodes, format='csr')
        increase = True
        iteration_count = 0
        if self.verbose:
            print("Starting with", graph.n_nodes, "nodes")
        while increase:
            iteration_count += 1
            self.algorithm.fit(graph)
            if self.algorithm.score_ <= self.tol:
                increase = False
            else:
                row = np.arange(graph.n_nodes)
                col = self.algorithm.labels_
                data = np.ones(graph.n_nodes)
                agg_membership = sparse.csr_matrix((data, (row, col)))
                membership = membership.dot(agg_membership)
                graph.aggregate(agg_membership)

                if graph.n_nodes == 1:
                    break
            if self.verbose:
                print("Iteration", iteration_count, "completed with", graph.n_nodes, "clusters")
            if iteration_count == self.max_agg_iter:
                break

        self.iteration_count_ = iteration_count
        self.labels_ = np.squeeze(np.asarray(membership.argmax(axis=1)))
        _, self.labels_ = np.unique(self.labels_, return_inverse=True)
        return self
