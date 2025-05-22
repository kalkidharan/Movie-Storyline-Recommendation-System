import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re
import nltk
import spacy
import string
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

nltk.download('stopwords')
nlp = spacy.load("en_core_web_sm")
stop_words = set(nltk.corpus.stopwords.words('english'))

def clean_and_tokenize(text):
    text = text.lower()
    text = re.sub(rf"[{string.punctuation}\d]", "", text)

    doc = nlp(text)
    tokens = [token.lemma_ for token in doc if token.text not in stop_words and not token.is_space]

    return " ".join(tokens)

@st.cache_data
def load_and_prepare_data(csv_path):
    df = pd.read_csv(csv_path)
    df = df.dropna(subset=['Title', 'Storyline'])
    df['Cleaned_Storyline'] = df['Storyline'].apply(clean_and_tokenize)

    tfidf = TfidfVectorizer()
    tfidf_matrix = tfidf.fit_transform(df['Cleaned_Storyline'])

    cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

    return df, tfidf, tfidf_matrix, cosine_sim

def recommend_movies(user_input, df, tfidf, tfidf_matrix):
    input_cleaned = clean_and_tokenize(user_input)
    input_vec = tfidf.transform([input_cleaned])
    sim_scores = cosine_similarity(input_vec, tfidf_matrix).flatten()

    df_copy = df.copy()
    df_copy['Similarity Score'] = sim_scores
    top_recommendations = df_copy.sort_values(by='Similarity Score', ascending=False).head(5)

    return top_recommendations[['Title', 'Storyline', 'Similarity Score']]

st.set_page_config(page_title="Movie Storyline Recommender", layout="centered")
st.title("Movie Storyline Recommendation System")
st.markdown("Enter a short movie plot or storyline below, and get the top 5 most similar movies based on the storyline.")

csv_path = "imdb_movies_2024.csv"

df, tfidf, tfidf_matrix, cosine_sim = load_and_prepare_data(csv_path)

user_storyline = st.text_area("Enter a movie storyline:")

if st.button("Recommend Movies"):
    if user_storyline.strip() == "":
        st.warning("Please enter a storyline.")
    else:
        recommendations = recommend_movies(user_storyline, df, tfidf, tfidf_matrix)

        st.success("Top 5 Recommended Movies:")
        for idx, row in recommendations.iterrows():
            st.subheader(f"{row['Title']}")
            st.write(row['Storyline'])
            st.write(f"Similarity Score: `{row['Similarity Score']:.3f}`")
            st.markdown("---")

        st.subheader("Similarity Score Comparison")
        fig, ax = plt.subplots()
        ax.barh(recommendations['Title'], recommendations['Similarity Score'], color='skyblue')
        ax.set_xlabel("Similarity Score")
        ax.set_title("Top 5 Recommendations")
        ax.invert_yaxis()
        st.pyplot(fig)
