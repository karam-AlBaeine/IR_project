import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from collections import defaultdict
from Matching_And_Ranking import rank_documents
from Evaluation import evaluate_search_results
from Query_Processing import process_query
from Indexing import load_index
import time
############################################################################
# load queries 
def load_queries(queries_file):
    queries_df = pd.read_csv(queries_file)
    queries = {row['id_left']: row['text_left'] for _, row in queries_df.iterrows()}
    return queries
############################################################################

############################################################################
# load qrels 
qrels = defaultdict(set)
def load_qrels(qrels_file):
    qrels_df = pd.read_csv(qrels_file)
    
    for _, row in qrels_df.iterrows():
        if row['id_left'] not in qrels:
            qrels[row['id_left']] = set()
        qrels[row['id_left']].add(row['id_right'])
    return qrels
############################################################################

############################################################################
# evaluate TF-IDF to chose the best max
def evaluate_tfidf_max_df(data, queries, qrels, max_df_values, num_queries):
    best_max_df = None
    best_map = 0

    for max_df in max_df_values:
        vectorizer = TfidfVectorizer(max_df=max_df, min_df=5, ngram_range=(1, 2))
        tfidf_matrix = vectorizer.fit_transform(data['text'])
        print(tfidf_matrix.shape)
        print()
        total_map = 0
        total_mrr = 0
        total_precision_at_k_score = 0
        total_recall_at_k_score = 0


        start = time.time()
        for query_id, query in list(queries.items())[:num_queries]:
            relevant_docs = qrels.get(query_id)
            query_vector = process_query('wiki',query, vectorizer )

            ranked_indices, _ = rank_documents(query_vector, tfidf_matrix)

            ranked_doc_ids = data_orginal.iloc[ranked_indices[:10]]['id_right'].tolist()
            # Create a dictionary for the evaluation
            ranked_lists = {query_id: ranked_doc_ids}
            sub_qrels = {query_id: relevant_docs}
            results = evaluate_search_results(ranked_lists, sub_qrels)
            total_map += results["MAP"]
            total_mrr += results["MRR"]
            total_precision_at_k_score += results["PAK"]
            total_recall_at_k_score += results["RAK"]

        end = time.time()
        print('the time for evaluate all queries is : ', end - start)
        print()

        avg_map = total_map / num_queries
        avg_mrr = total_mrr / num_queries
        avg_recall = total_recall_at_k_score / num_queries
        avg_precision = total_precision_at_k_score / num_queries

        print(f"max_df : {max_df}Average MAP: {avg_map}, Average MRR: {avg_mrr}, Average ReCall: {avg_recall},Average precision: {avg_precision}")

        if avg_map > best_map:
            best_map = avg_map
            best_max_df = max_df

    return best_max_df
############################################################################

############################################################################
# run the code 
# if __name__ == '__main__':
#     data = pd.read_csv('cleaned_data_for_wiki_final.csv')
#     queries = load_queries('wiki_queries.csv')
#     qrels = load_qrels('wiki_qrels.csv')
#     data_orginal = pd.read_csv('wiki.csv', encoding='utf-8')
#     max_df_values = [0.155,0.1,0.5]
#     best_max_df = evaluate_tfidf_max_df(data, queries, qrels, max_df_values, num_queries=100)
#     print(f"Best max_df: {best_max_df}")
############################################################################



################################################################### WIKI CHOISE ###################################################################
# max_df : 0.005 Average MAP: 0.664272200176367, Average MRR: 0.6777777777777778, Average ReCall: 0.06039999999999992,Average precision: 0.6040000000000002
# max_df : 0.01 Average MAP: 0.6922045309481819, Average MRR: 0.7061851851851852, Average ReCall: 0.06313333333333325,Average precision: 0.6313333333333334
# max_df : 0.1 Average MAP: 0.7317977376753172, Average MRR: 0.7517407407407407, Average ReCall: 0.06766666666666657,Average precision: 0.6766666666666666
# max_df : 0.7 Average MAP: 0.7274428781389101, Average MRR: 0.7468412698412698, Average ReCall: 0.06706666666666658,Average precision: 0.6706666666666665
# max_df : 0.0001 Average MAP: 0.24319170340975893, Average MRR: 0.2513888888888889, Average ReCall: 0.021466666666666665,Average precision: 0.21466666666666664
# max_df : 0.25 Average MAP: 0.7276765642059295, Average MRR: 0.7468412698412698, Average ReCall: 0.06706666666666658,Average precision: 0.6706666666666665
# Best max_df: 0.1 ===> 150 doc

# max_df : 0.1A verage MAP: 0.7174664471529352, Average MRR: 0.7369134920634919, Average ReCall: 0.06546000000000028,Average precision: 0.6545999999999996
# max_df : 0.15 Average MAP: 0.7198319699546486, Average MRR: 0.7402349206349204, Average ReCall: 0.0656800000000003,Average precision: 0.6567999999999995
# max_df : 0.08 Average MAP: 0.7160583610481231, Average MRR: 0.7345968253968252, Average ReCall: 0.06524000000000027,Average precision: 0.6523999999999998
# max_df : 0.18 Average MAP: 0.7194666417863442, Average MRR: 0.7403126984126982, Average ReCall: 0.06562000000000029,Average precision: 0.6561999999999995
# max_df : 0.2 Average MAP: 0.719349743638196, Average MRR: 0.7403793650793647, Average ReCall: 0.06564000000000028,Average precision: 0.6563999999999994
# max_df : 0.13 Average MAP: 0.7194220587679516, Average MRR: 0.7400849206349204, Average ReCall: 0.0656800000000003,Average precision: 0.6567999999999996
# Best max_df: 0.15 ===> 500 doc


# max_df : 0.15 Average MAP: 0.6994387332181555, Average MRR: 0.7278347505668934, Average ReCall: 0.06375714285714255,Average precision: 0.6375714285714285
# max_df : 0.16 Average MAP: 0.6989317027273877, Average MRR: 0.7278466553287982, Average ReCall: 0.06362857142857113,Average precision: 0.6362857142857143
# max_df : 0.155 Average MAP: 0.6994922526814963, Average MRR: 0.7278942743764173, Average ReCall: 0.06376428571428541,Average precision: 0.6376428571428571
# max_df : 0.148 Average MAP: 0.6994579468559915, Average MRR: 0.7278248299319727, Average ReCall: 0.06375714285714254,Average precision: 0.6375714285714286
# max_df : 0.1525Average MAP: 0.6994922526814963, Average MRR: 0.7278942743764173, Average ReCall: 0.06376428571428541,Average precision: 0.6376428571428571
# Best max_df: 0.1525 ===> 1400 doc
# Best max_df: 0.155 ===> 1400 doc

# max_df : 0.155Average MAP: 0.7014215585969842, Average MRR: 0.7311575963718823, Average ReCall: 0.06374285714285688,Average precision: 0.6374285714285708 ----- 12 , without min , 1400

# _______________________________________________________________last Evaluation______________________________________________________________
#max_df : 0.155Average MAP: 0.875579475308642, Average MRR: 0.88, Average ReCall: 0.08579999999999988,Average precision: 0.858 ====>  ngrams = 1,4
#max_df : 0.155Average MAP: 0.751813592844545, Average MRR: 0.7741666666666666, Average ReCall: 0.06689999999999993,Average precision: 0.669   ====> ngrams = 1,2 

#Best max_df: 0.155 ===>100 doc , ngrams = 1,4
