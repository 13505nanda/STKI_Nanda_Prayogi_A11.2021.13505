import pandas as pd
import numpy as np
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Membaca file csv
authors_df = pd.read_csv('authors.csv')
dataset_df = pd.read_csv('dataset.csv')

# Membuat judul website
st.title('REKOMENDASI BUKU STKI')

# Membuat kolom untuk memasukkan judul buku
book_title = st.text_input('Masukkan judul buku', '')

# Membuat tombol untuk melakukan pencarian
if st.button('SEARCH'):
    # Kode untuk mencari buku akan dibuat di sini
    pass

def search_books(title):
    # Kode untuk mencari buku akan dibuat di sini
    pass

