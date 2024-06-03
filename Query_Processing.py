from Data_Processing_wiki import data_processing_wiki
from Data_Processing_Antique import data_processing_antique

#######################################################################################
# process the query using the dataset processer
def process_query(data,query, vectorizer):
    if data == 'wiki':
        processed_query = data_processing_wiki(query)
    else:
        processed_query = data_processing_antique(query)
    query_vector = vectorizer.transform([processed_query])
    return query_vector
#######################################################################################
