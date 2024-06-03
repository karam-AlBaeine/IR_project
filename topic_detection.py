import joblib
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.model_selection import GridSearchCV
from Indexing import load_index

def train_and_save_lda_model(tfidf_matrix_path, vectorizer_path, output_model_path, search_params, sample_size=None):

    # Load the TF-IDF matrix and vectorizer
    tfidf_matrix, vectorizer = load_index(tfidf_matrix_path, vectorizer_path)
    
    # Sample the data if sample_size is specified
    if sample_size:
        tfidf_matrix = tfidf_matrix[:sample_size]

    # Initialize LDA and GridSearchCV
    lda = LatentDirichletAllocation(random_state=42, max_iter=10, learning_method='batch')
    model = GridSearchCV(lda, param_grid=search_params, cv=3, n_jobs=-1, verbose=1)

    # Train the model
    model.fit(tfidf_matrix)

    # Get the best LDA model
    best_lda_model = model.best_estimator_
    print(best_lda_model.best_estimator_)
    # Save the best LDA model
    joblib.dump(best_lda_model, output_model_path)
    print(f"Model saved to {output_model_path}")

def get_topic_distribution(text, vectorizer, lda_model):
    # Transform the text to document-term matrix
    dtm = vectorizer.transform([text])
    
    # Get topic distribution
    topic_distribution = lda_model.transform(dtm)
    
    return list(enumerate(topic_distribution[0]))

# if __name__ == "__main__":
#     # Define paths
#     tfidf_matrix_path = 'tfidf_wiki.npz'
#     vectorizer_path = 'vectorizer_wiki.pkl'
#     output_model_path = 'lda_model_wiki.pkl'

#     # Define search parameters for GridSearchCV
#     search_params = {
#         'n_components': [20, 30, 40],  # Example common values for number of topics
#         'learning_decay': [0.5, 0.7, 0.9]
#     }

#     # Train and save LDA model with a sample size for faster training
#     train_and_save_lda_model(tfidf_matrix_path, vectorizer_path, output_model_path, search_params, sample_size=10000)
