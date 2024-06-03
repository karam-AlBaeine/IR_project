from flask import Flask, jsonify, request, render_template
import pandas as pd
import sqlite3
import pickle
from Indexing import load_index
from Query_Processing import process_query
from Matching_And_Ranking import rank_documents
from Query_Refinement import correct_spelling, suggest_queries
from flask_socketio import SocketIO, emit
from Personalization import get_personalized_recommendations
from Summarizing import summarize_text 
#from topic_detection import get_topic_distribution

app = Flask(__name__)
socketio = SocketIO(app)

###############################################
# load vectorizers for refinment queries
with open('wiki_for_query_refinment_new.pkl', 'rb') as f:
    vectorizer_wiki_for_refinment = pickle.load(f)

with open('antique_for_query_refinment_new.pkl', 'rb') as f:
    vectorizer_antique_for_refinment = pickle.load(f)
###############################################

###############################################
# load Datasets & vectorizers & matrix for both Datasets
data_wiki = pd.read_csv('wiki.csv')
tfidf_matrix_wiki, vectorizer_wiki = load_index('tfidf_wiki.npz', 'vectorizer_wiki.pkl')

data_antique = pd.read_csv('antique-data.csv')
tfidf_matrix_antique, vectorizer_antique = load_index('tfidf_antique.npz', 'vectorizer_antique.pkl')
###############################################

###############################################
# load LDA model for topic derection 
# with open('lda_model_wiki.pkl', 'rb') as f:
#     lda_model = pickle.load(f)
###############################################

###############################################
# Create DB 
def init_db():
    with sqlite3.connect('search_history.db') as conn:
        c = conn.cursor()
        c.execute('''CREATE TABLE IF NOT EXISTS searches
                     (id INTEGER PRIMARY KEY, user_id TEXT, query TEXT, dataset TEXT)''')
        conn.commit()

init_db()
###############################################

###############################################
# Render Home Page
@app.route('/')
def home():
    return render_template('index.html')
###############################################

###############################################
# Render ChatPot Page
@app.route('/chat')
def chat():
    return render_template('chat.html')
###############################################

###############################################
# The route for personalized_search " Get Results "
@app.route('/personalized_search', methods=['GET'])
def personalized_search():
    user_id = request.args.get('user_id')
    dataset = request.args.get('dataset')
    query = request.args.get('query')
    use_personalization = request.args.get('use_personalization') == 'true'

    if not query or not dataset or not user_id:
        return jsonify({"error": "User ID, query and dataset are required"}), 400

    corrected_query = correct_spelling(query)

    if dataset == 'wiki':
        if use_personalization:
            recommendations = get_personalized_recommendations(user_id, corrected_query, vectorizer_wiki, tfidf_matrix_wiki)
        else:
            query_vector = process_query(dataset, corrected_query, vectorizer_wiki)
            recommendations, _ = rank_documents(query_vector, tfidf_matrix_wiki)
        results = data_wiki.iloc[recommendations[:10]]
    elif dataset == 'antique':
        if use_personalization:
            recommendations = get_personalized_recommendations(user_id, corrected_query, vectorizer_antique, tfidf_matrix_antique)
        else:
            query_vector = process_query(dataset, corrected_query, vectorizer_antique)
            recommendations, _ = rank_documents(query_vector, tfidf_matrix_antique)
        results = data_wiki.iloc[recommendations]
    else:
        return jsonify({"error": "Invalid dataset"}), 400
    
    # save the search in history 
    with sqlite3.connect('search_history.db') as conn:
        c = conn.cursor()
        c.execute("INSERT INTO searches (user_id, query, dataset) VALUES (?, ?, ?)", (user_id, query, dataset))
        conn.commit()

    return render_template('results.html', query=query, corrected_query=corrected_query, results=results)
###############################################

###############################################
# The route for Suggest " get suggest queries "
@app.route('/suggest', methods=['GET'])
def suggest():
    dataset = request.args.get('dataset')
    query = request.args.get('query')
    if not query or not dataset:
        return jsonify([])

    suggestions = set()

    # get the suggest from DB
    with sqlite3.connect('search_history.db') as conn:
        c = conn.cursor()
        c.execute("SELECT query FROM searches WHERE query LIKE ?", ('%' + query + '%',))
        db_suggestions = c.fetchall()
        for s in db_suggestions:
            suggestions.add(s[0])
    
    # get the suggest from vectorizer
    if dataset == 'wiki':
        suggestions.update(suggest_queries(query, vectorizer_wiki_for_refinment))
    elif dataset == 'antique':
        suggestions.update(suggest_queries(query, vectorizer_antique_for_refinment))
    else:
        return jsonify({"error": "Invalid dataset"}), 400

    return jsonify(list(suggestions))
###############################################

###############################################
# THe route for correct queries
@app.route('/correct', methods=['GET'])
def correct():
    query = request.args.get('query')
    if not query:
        return jsonify({"error": "Query is required"}), 400

    corrected_query = correct_spelling(query)
    return jsonify({"corrected_query": corrected_query})
###############################################

###############################################
# conect to the chatpot & show msg
@socketio.on('connect')
def handle_connect():
    emit('response', {'message': 'Welcome to Yalla Search chatbot! How can I assist you today?'})
###############################################

###############################################
# chatpot conversation 
@socketio.on('user_message')
def handle_user_message(data):
    query = data.get('message')
    dataset = data.get('dataset')
    corrected_query = correct_spelling(query)
    #topic_distribution = get_topic_distribution(corrected_query, vectorizer_wiki, lda_model)
    if dataset == 'wiki':
        query_vector = process_query(dataset, corrected_query, vectorizer_wiki)
        ranked_indices, _ = rank_documents(query_vector, tfidf_matrix_wiki)
        results = data_wiki.iloc[ranked_indices[:1]].to_dict(orient='records')
        # results = summarize_text(results[0]['text_right'])
    elif dataset == 'antique':
        query_vector = process_query(dataset, corrected_query, vectorizer_antique)
        ranked_indices, _ = rank_documents(query_vector, tfidf_matrix_antique)
        results = data_antique.iloc[ranked_indices[:10]].to_dict(orient='records')
        # results = summarize_text(results)
    else:
        emit('response', {'error': 'Invalid dataset'})
        return

    emit('response', {'query': query, 'corrected_query': corrected_query, 'results': results})#, 'topic_distribution': topic_distribution })
###############################################


if __name__ == '__main__':
    app.run(debug=True)


