"""
Implementation of Precision-Recall Cover from the pseudocode in Appendix A.3.

Fasil Cheema and Ruth Urner. Precision recall cover: A method for assessing
generative models. In International Conference on Artificial Intelligence and
Statistics, pages 6571–6594. PMLR, 2023.
"""

import numpy as np
from sklearn.neighbors import NearestNeighbors


def k_nearest_neighbor_distances(sample_set, k):
    neighbors = NearestNeighbors(n_neighbors=k).fit(sample_set)
    distances, _ = neighbors.kneighbors(sample_set)
    return distances[:, -1]


def precision_recall_cover(P, Q, k=3, C=3):
    Q_beta = []
    k_prime = C * k
    r_Q = k_nearest_neighbor_distances(Q, k_prime)

    for i, y in enumerate(Q):
        val = pr_cover_indicator(y=y, r_y=r_Q[i], P=P, k=k)
        if val == 1:
            Q_beta.append(y)

    PC = len(Q_beta) / len(Q)
    return PC


def pr_cover_indicator(P, y, r_y, k):
    i = 0
    for x in P:
        if np.linalg.norm(y - x) <= r_y:
            i += 1
        if i >= k:
            return 1
    return 0
