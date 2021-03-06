#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Nov 6 2018

Authors:
Nathan De Lara <ndelara@enst.fr>

Quality metrics for graph embeddings
"""

import numpy as np

from scipy import sparse
from scipy.stats import hmean


def dot_modularity(adjacency_matrix, embedding: np.ndarray, features=None, resolution=1., weights='degree',
                   return_all: bool=False):
    """
    Difference of the weighted average dot product between embeddings of pairs of neighbors in the graph
    (fit term) and pairs of nodes in the graph (diversity term).

    :math:`Q = \\sum_{ij}(\\dfrac{A_{ij}}{w} - \\gamma \\dfrac{d_id_j}{w^2})x_i^Tx_j`

    This metric is normalized to lie between -1 and 1.
    If the embeddings are normalized, this reduces to the cosine modularity.

    Parameters
    ----------
    adjacency_matrix: sparse.csr_matrix or np.ndarray
        the adjacency matrix of the graph
    embedding: np.ndarray
        the embedding to evaluate, embedding[i] must represent the embedding of node i
    features: None or np.ndarray
        For bipartite graphs, features should be the embedding of the second part
    resolution: float
        scaling for first-order approximation
    weights: ``'degree'`` or ``'uniform'``
        prior distribution on the nodes

    return_all: bool, default = ``False``
        whether to return (fit, diversity) or fit - diversity

    Returns
    -------
    dot_modularity: a float or a tuple of floats.
    """
    if type(adjacency_matrix) == sparse.csr_matrix:
        adj_matrix = adjacency_matrix
    elif sparse.isspmatrix(adjacency_matrix) or type(adjacency_matrix) == np.ndarray:
        adj_matrix = sparse.csr_matrix(adjacency_matrix)
    else:
        raise TypeError(
            "The argument must be a NumPy array or a SciPy Sparse matrix.")
    n_nodes, m_nodes = adj_matrix.shape
    total_weight: float = adjacency_matrix.data.sum()

    if features is None:
        if n_nodes != m_nodes:
            raise ValueError('feature cannot be None for non-square adjacency matrices.')
        else:
            normalization = np.linalg.norm(embedding) ** 2 / np.sqrt(n_nodes * m_nodes)
            features = embedding
    else:
        normalization = np.linalg.norm(embedding.dot(features.T)) / np.sqrt(n_nodes * m_nodes)

    if weights == 'degree':
        wou = adj_matrix.dot(np.ones(m_nodes)) / total_weight
        win = adj_matrix.T.dot(np.ones(n_nodes)) / total_weight
    elif weights == 'uniform':
        wou = np.ones(n_nodes) / n_nodes
        win = np.ones(m_nodes) / m_nodes
    else:
        raise ValueError('weights must be degree or uniform.')

    fit = (np.multiply(embedding, adjacency_matrix.dot(features))).sum() / (total_weight * normalization)
    diversity = (embedding.T.dot(wou)).dot(features.T.dot(win)) / normalization

    if return_all:
        return fit, resolution * diversity
    else:
        return fit - resolution * diversity


def hscore(adjacency_matrix, embedding: np.ndarray, order='second', return_all: bool=False):
    """Harmonic mean of fit and diversity with respect to first or second order node similarity.

    Parameters
    ----------
    adjacency_matrix: sparse.csr_matrix or np.ndarray
        the adjacency matrix of the graph
    embedding: np.ndarray
        the embedding to evaluate, embedding[i] must represent the embedding of node i
    order: \'first\' or \'second\'.
        The order of the node similarity metric to use. First-order corresponds to edges weights while second-order
        corresponds to the weights of the edges in the normalized cocitation graph.
    return_all: bool, default = ``False``
        whether to return (fit, diversity) or hmean(fit, diversity)

    Returns
    -------
    hscore: a float or a tuple of floats.

    """
    if type(adjacency_matrix) == sparse.csr_matrix:
        adj_matrix = adjacency_matrix
    elif sparse.isspmatrix(adjacency_matrix) or type(adjacency_matrix) == np.ndarray:
        adj_matrix = sparse.csr_matrix(adjacency_matrix)
    else:
        raise TypeError(
            "The argument must be a NumPy array or a SciPy Sparse matrix.")
    n_nodes, m_nodes = adj_matrix.shape
    if order == 'first' and (n_nodes != m_nodes):
        raise ValueError('For fist order similarity, the adjacency matrix must be square.')
    total_weight = adj_matrix.data.sum()
    # out-degree vector
    dou = adj_matrix.dot(np.ones(m_nodes))
    # in-degree vector
    din = adj_matrix.T.dot(np.ones(n_nodes))

    pdhou, pdhin = np.zeros(n_nodes), np.zeros(m_nodes)
    pdhou[dou.nonzero()] = 1 / np.sqrt(dou[dou.nonzero()])
    pdhin[din.nonzero()] = 1 / np.sqrt(din[din.nonzero()])

    normalization = np.linalg.norm(embedding.T * np.sqrt(dou)) ** 2
    if order == 'first':
        fit = (np.multiply(embedding, adjacency_matrix.dot(embedding))).sum()
        fit /= total_weight * (np.linalg.norm(embedding) ** 2 / n_nodes)
    elif order == 'second':
        fit = np.linalg.norm(adj_matrix.T.dot(embedding).T * pdhin) ** 2 / normalization
    else:
        raise ValueError('The similarity order should be \'first\' or \'second\'.')
    diversity = (np.linalg.norm(embedding.T.dot(dou))) ** 2 / total_weight
    diversity = 1 - diversity / normalization

    if return_all:
        return fit, diversity
    else:
        if np.isclose(fit, 0.) or np.isclose(diversity, 0.):
            return 0.
        else:
            return hmean([fit, diversity])
