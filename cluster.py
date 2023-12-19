import numpy as np
from matrix_closure import matrix_closure 

def cluster(similarity_matrix, threshold, include_single_clusters=False):
    closure = matrix_closure(similarity_matrix > threshold)
    triu = np.triu(closure, k=1)
    indexes_to_suppress = np.sum(triu, axis=0) > 0
    clusters_expansion = np.delete(closure + np.diag(np.repeat(True, closure.shape[0])), indexes_to_suppress, axis=0)
    if (include_single_clusters):
        return clusters_expansion
    # single_template_cluster_indexes = np.array((np.sum(clusters_expansion, axis=1) <= 1).T)[0]
    single_template_cluster_indexes = np.sum(clusters_expansion, axis=1) <= 1
    proper_clusters_expansion = np.delete(clusters_expansion, single_template_cluster_indexes.T, axis=0)
    return proper_clusters_expansion

def cocluster(cooccurrence_matrix, corr_thresholds=(0.8, 0.8), include_single_clusters=(False, True), order=(True, True)):
    rows_corr = np.corrcoef(cooccurrence_matrix)
    row_clusters = cluster(rows_corr, corr_thresholds[0], include_single_clusters[0])

    if order[0]:
        row_clusters_size = np.sum(row_clusters, axis=1)
        row_clusters = row_clusters[row_clusters_size.argsort()[::-1]]

    cooccurrence_row_clusters_to_cols = row_clusters @ cooccurrence_matrix
    col_indexes_to_suppress = np.where((np.sum(cooccurrence_row_clusters_to_cols, axis=0) == 0))[1]
    reduced_cols_expansion = np.delete(np.diag(np.repeat(True, cooccurrence_row_clusters_to_cols.shape[1])), col_indexes_to_suppress, axis=0)
    cooccurrence_row_clusters_to_reduced_cols = cooccurrence_row_clusters_to_cols @ reduced_cols_expansion.T

    reduced_cols_corr = np.corrcoef(cooccurrence_row_clusters_to_reduced_cols.T)
    reduced_col_clusters = cluster(reduced_cols_corr, corr_thresholds[1], include_single_clusters=[1])
    col_clusters = reduced_col_clusters @ reduced_cols_expansion

    if order[1]:
        col_clusters_size = np.sum(col_clusters, axis=1)
        col_clusters = col_clusters[col_clusters_size.argsort()[::-1]]

    cluster_cooccurrence = cooccurrence_row_clusters_to_cols @ col_clusters.T

    return row_clusters, col_clusters, cluster_cooccurrence