from scipy import sparse
from Data_Representaion import create_tfidf_representation
import pickle
#####################################################################
# save vectorizer & matrix
def save_index(tfidf_matrix, vectorizer, index_file, vectorizer_file):
    sparse.save_npz(index_file, tfidf_matrix)
    with open(vectorizer_file, 'wb') as f:
        pickle.dump(vectorizer, f)
#####################################################################

#####################################################################
# load vectorizer & matrix
def load_index(index_file, vectorizer_file):
    tfidf_matrix = sparse.load_npz(index_file)
    with open(vectorizer_file, 'rb') as f:
        vectorizer = pickle.load(f)
    return tfidf_matrix, vectorizer
#####################################################################


############################################### WIKI ###############################################

# import pandas as pd
# data_wiki = pd.read_csv('cleand_wiki.csv')
# data_wiki.fillna('defualt value')
# tfidf_matrix_wiki , vectorizer_wiki = create_tfidf_representation(data_wiki)
# save_index(tfidf_matrix_wiki, vectorizer_wiki, 'tfidf_wiki_N_grams.npz', 'victorizer_wiki_N_grams.pkl')


# import pandas as pd # >>>> for query refinment 
# data_wiki = pd.read_csv('wiki_queries.csv')
# data_wiki.fillna('defualt value')
# vectorizer_wiki = create_tfidf_representation(data_wiki)
# with open('wiki_for_query_refinment.pkl', 'wb') as f:
#         pickle.dump(vectorizer_wiki, f)

############################################### antique ###############################################

# import pandas as pd
# data_antique = pd.read_csv('antique.csv')
# data_antique.fillna('defualt value')
# tfidf_matrix_antique , vectorizer_antique = create_tfidf_representation(data_antique)
# save_index(tfidf_matrix_antique, vectorizer_antique, 'tfidf_antique.npz', 'victorizer_antique.pkl')


# import pandas as pd # >>>> for query refinment
# data_antique = pd.read_csv('antique_Queries.csv')
# data_antique.fillna('defualt value')
# vectorizer_antique = create_tfidf_representation(data_antique)
# with open('antique_for_query_refinment.pkl', 'wb') as f:
#         pickle.dump(vectorizer_antique, f)

