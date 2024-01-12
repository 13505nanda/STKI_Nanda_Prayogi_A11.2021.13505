#LIBRARY
import streamlit as st
import pandas as pd
import nltk
import re
import contractions
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from gensim.models import FastText

#LOAD DATASET
true_df = pd.read_csv('dataset_stki.csv')

# Text preprocessing
stop_words = nltk.corpus.stopwords.words('english')

def normalize_document(doc):
    # Ensure document is a string
    doc = str(doc)
    # Convert to lowercase
    doc = doc.lower()
    # Remove punctuation and special characters
    doc = re.sub(r'[^a-zA-Z0-9\s]', '', doc, re.I|re.A)
    return doc

normalize_corpus = np.vectorize(normalize_document)
norm_corpus = normalize_corpus([item[0] if isinstance(item[0], str) else '' for item in list(true_df['description']) if isinstance(item[0], str)])
print(norm_corpus)

# TF-IDF vectorization
tf = TfidfVectorizer(ngram_range=(1, 2), min_df=2)
tfidf_matrix = tf.fit_transform(norm_corpus)

# Cosine similarity
doc_sim = cosine_similarity(tfidf_matrix)
doc_sim_df = pd.DataFrame(doc_sim)

# FastText model
tokenized_docs = [doc.split() for doc in norm_corpus]
ft_model = FastText(tokenized_docs, window=30, min_count=2, workers=4, sg=1)

# Averaged Word2Vec vectorizer
def averaged_word2vec_vectorizer(corpus, model, num_features):
    vocabulary = set(model.wv.index_to_key)
    
    def average_word_vectors(words, model, vocabulary, num_features):
        feature_vector = np.zeros((num_features,), dtype="float64")
        nwords = 0.
        
        for word in words:
            if word in vocabulary: 
                nwords = nwords + 1.
                feature_vector = np.add(feature_vector, model.wv[word])
        if nwords:
            feature_vector = np.divide(feature_vector, nwords)

        return feature_vector

    features = [average_word_vectors(tokenized_sentence, model, vocabulary, num_features)
                for tokenized_sentence in corpus]
    return np.array(features)

doc_vecs_ft = averaged_word2vec_vectorizer(tokenized_docs, ft_model, 100)
doc_sim_ft = cosine_similarity(doc_vecs_ft)
doc_sim_df_ft = pd.DataFrame(doc_sim_ft)

# Streamlit App
st.title('Book Recommendation System')

# Input Text
input_text = st.text_input('Input Book Description:', '')

if input_text:
    st.markdown('**Input Book Description:**')
    st.write(input_text)

    # TF-IDF Recommendations
    st.subheader('TF-IDF Recommendations:')
    input_tfidf = tf.transform([normalize_document(input_text)])
    input_sim_tfidf = cosine_similarity(input_tfidf, tfidf_matrix)
    input_sim_tfidf = input_sim_tfidf.flatten()
    recommended_books_tfidf = true_df.iloc[np.argsort(-input_sim_tfidf)[:5]]['title'].tolist()
    st.write(recommended_books_tfidf)

    # FastText Recommendations
    st.subheader('FastText Recommendations:')
    input_vec_ft = averaged_word2vec_vectorizer([normalize_document(input_text).split()], ft_model, 100)
    input_sim_ft = cosine_similarity(input_vec_ft, doc_vecs_ft)
    input_sim_ft = input_sim_ft.flatten()
    recommended_books_ft = true_df.iloc[np.argsort(-input_sim_ft)[:5]]['title'].tolist()
    st.write(recommended_books_ft)