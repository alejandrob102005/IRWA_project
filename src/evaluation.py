import numpy as np
from indexing import search_tfidf

# P@K
def precision_at_k(doc_score, k):
    
    if k <= 0:
        return 0.0
    top_k_scores = doc_score[:k]
    return float(np.sum(top_k_scores) / k)


# R@K
def recall_at_k(doc_score, k, total_relevant):
    
    if total_relevant == 0 or k <= 0:
        return 0.0
    top_k_scores = doc_score[:k]
    return float(np.sum(top_k_scores) / total_relevant)

# F1-Score@K
def f1_score_at_k(doc_score, k, total_relevant):
    
    p_k = precision_at_k(doc_score, k)
    r_k = recall_at_k(doc_score, k, total_relevant)
    if p_k + r_k == 0:
        return 0.0
    return float(2 * (p_k * r_k) / (p_k + r_k))

# Average Precision@K
def average_precision_at_k(doc_score, k, total_relevant):
    
    if total_relevant == 0 or k <= 0:
        return 0.0

    top_k_scores = doc_score[:k]
    score_sum = 0.0

    for i, rel in enumerate(top_k_scores):
        if rel == 1:
            
            score_sum += precision_at_k(top_k_scores, i + 1)

    return float(score_sum / total_relevant)

# MAP
def mean_average_precision(all_ap_scores):
    
    if len(all_ap_scores) == 0:
        return 0.0
    return float(np.mean(all_ap_scores))

# RR
def reciprocal_rank(doc_score, k):
    
    top_k_scores = doc_score[:k]
    for position, rel in enumerate(top_k_scores):
        if rel == 1:
            return float(1.0 / (position + 1))
    return 0.0

# MRR
def mean_reciprocal_rank(all_rr_scores):
    
    if len(all_rr_scores) == 0:
        return 0.0
    return float(np.mean(all_rr_scores))

# DCG
def dcg_at_k(doc_score, k):
    
    top_k_scores = doc_score[:k]
    dcg = 0.0
    for i, rel in enumerate(top_k_scores):
        dcg += (2**rel - 1) / np.log2(i + 2)  # position index i is 0-based -> rank is i+1 -> log2(rank+1) = log2(i+2)
    return float(dcg)

# NDCG
def ndcg_at_k(doc_score, k, ideal_doc_score):
    
    dcg = dcg_at_k(doc_score, k)
    idcg = dcg_at_k(ideal_doc_score, k)

    if idcg == 0:
        return 0.0
    return float(dcg / idcg)


def evaluate_query(retrieved_relevance, total_relevant, k=10):
    
    ideal_relevance = sorted(retrieved_relevance, reverse=True)

    p_k = precision_at_k(retrieved_relevance, k)
    r_k = recall_at_k(retrieved_relevance, k, total_relevant)
    f1_k = f1_score_at_k(retrieved_relevance, k, total_relevant)
    ap_k = average_precision_at_k(retrieved_relevance, k, total_relevant)
    rr_k = reciprocal_rank(retrieved_relevance, k)
    ndcg_k = ndcg_at_k(retrieved_relevance, k, ideal_relevance)

    return {
        "P@K": round(p_k, 3),
        "R@K": round(r_k, 3),
        "F1@K": round(f1_k, 3),
        "AP@K": round(ap_k, 3),
        "RR@K": round(rr_k, 3),
        "NDCG@K": round(ndcg_k, 3),
    }


def evaluate_system(queries, labels_df, index, tf, idf, preprocessor, k=10):
    query_metrics = []
    ap_list = []
    rr_list = []

    for q_idx, query_text in enumerate(queries, start=1):
        ranked_results = search_tfidf(query_text, preprocessor, index, tf, idf)

        query_labels = labels_df[labels_df['query_id'] == q_idx]
        rel_map = dict(zip(query_labels['pid'], query_labels['labels']))
        total_relevant = int((query_labels['labels'] == 1).sum())

        retrieved_relevance = [rel_map.get(pid, 0) for score, pid in ranked_results]

        metrics = evaluate_query(retrieved_relevance, total_relevant, k=k)

        ap_list.append(metrics["AP@K"])
        rr_list.append(metrics["RR@K"])
        query_metrics.append((q_idx, query_text, metrics))

    map_score = round(mean_average_precision(ap_list), 3)
    mrr_score = round(mean_reciprocal_rank(rr_list), 3)

    return {
        "query_metrics": query_metrics,
        "MAP": map_score,
        "MRR": mrr_score
    }