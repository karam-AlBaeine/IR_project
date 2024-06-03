import numpy as np
#####################################################################################
# clculate Precision@k
def precision_at_k(ranked_indices, relevant_docs, k):
    if not relevant_docs:
        return 0.0
    return len(set(ranked_indices[:k]) & set(relevant_docs)) / k
#####################################################################################

#####################################################################################
# clculate Recall@k
def recall_at_k(ranked_indices, relevant_docs, k):
    if not relevant_docs:
        return 0.0
    return len(set(ranked_indices[:k]) & set(relevant_docs)) / len(relevant_docs)
#####################################################################################

#####################################################################################
# clculate avg precision
def average_precision(ranked_indices, relevant_docs):
    if not relevant_docs:
        return 0.0
    precisions = []
    num_relevant_docs = 0
    for k in range(1, len(ranked_indices) + 1):
        if ranked_indices[k - 1] in relevant_docs:
            num_relevant_docs += 1
            precisions.append(num_relevant_docs / k)
    return np.mean(precisions) if precisions else 0.0
#####################################################################################

#####################################################################################
# clculate MAP 
def mean_average_precision(ranked_lists, qrels):
    average_precisions = []
    for query_id, ranked_list in ranked_lists.items():
        if query_id in qrels:
            ap = average_precision(ranked_list, qrels[query_id])
            average_precisions.append(ap)
    return np.mean(average_precisions) if average_precisions else 0.0
#####################################################################################

#####################################################################################
# clculate RR
def reciprocal_rank(ranked_indices, relevant_docs):
    if not relevant_docs:
        return 0.0
    for i, doc_id in enumerate(ranked_indices):
        if doc_id in relevant_docs:
            return 1.0 / (i + 1)
    return 0.0
#####################################################################################

#####################################################################################
# clculate MRR
def mean_reciprocal_rank(ranked_lists, qrels):
    reciprocal_ranks = []
    for query_id, ranked_list in ranked_lists.items():
        if query_id in qrels:
            rr = reciprocal_rank(ranked_list, qrels[query_id])
            reciprocal_ranks.append(rr)
    return np.mean(reciprocal_ranks) if reciprocal_ranks else 0.0
#####################################################################################

#####################################################################################
# call funs
def evaluate_search_results(ranked_lists, qrels):
    map_score = mean_average_precision(ranked_lists, qrels)
    mrr_score = mean_reciprocal_rank(ranked_lists, qrels)
    pak_scores = [precision_at_k(ranked_lists[query_id], qrels[query_id], 10) for query_id in ranked_lists.keys()]
    rak_scores = [recall_at_k(ranked_lists[query_id], qrels[query_id], 10) for query_id in ranked_lists.keys()]
    pak_score = np.mean(pak_scores)
    rak_score = np.mean(rak_scores)
    return {"MAP": map_score, "MRR": mrr_score, "PAK": pak_score, "RAK": rak_score}
#####################################################################################