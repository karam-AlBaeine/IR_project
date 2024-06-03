import spacy
from spellchecker import SpellChecker

nlp = spacy.load('en_core_web_sm')
#####################################################################################################################################
# corect the query 
def correct_spelling(query):
    spell = SpellChecker()
    corrected_query = " ".join([spell.correction(word) if spell.correction(word) is not None else word for word in query.split()])
    return corrected_query
#####################################################################################################################################

#####################################################################################################################################
# calculate the similar query using the vectorizer
def suggest_queries(query, vectorizer, top_n=10):
    doc = nlp(query)
    suggestions = []
    for token in doc:
        if token.text in vectorizer.vocabulary_:
            index = vectorizer.vocabulary_[token.text]
            similar_terms = vectorizer.get_feature_names_out()[index:index+top_n]
            suggestions.extend(similar_terms)
    return suggestions
#####################################################################################################################################