import streamlit as st
import pandas as pd
import nltk

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans


nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)
nltk.download('stopwords', quiet=True)

st.title("📰 News Article Topic Clustering")

file = st.file_uploader("Upload CSV file", type="csv")

if file:
    df = pd.read_csv(file)

    column_lookup = {str(col).strip().lower(): col for col in df.columns}

    text_column = None
    for candidate in ["content", "text", "article", "description", "body"]:
        if candidate in column_lookup:
            text_column = column_lookup[candidate]
            break

    if text_column is None:
        st.error(
            "No text column found. Please upload a CSV containing one of these columns: "
            "content, text, article, description, body"
        )
        st.stop()

    title_column = column_lookup.get("title")

    stop_words = set(stopwords.words("english"))

    def clean_text(text):
        words = word_tokenize(str(text).lower())
        words = [w for w in words if w.isalpha() and w not in stop_words]
        return " ".join(words)

    df["clean_text"] = df[text_column].apply(clean_text)

    k = st.slider("Number of Topics", 2, 5, 2)

    X = TfidfVectorizer().fit_transform(df["clean_text"])
    df["cluster"] = KMeans(n_clusters=k, random_state=42).fit_predict(X)

    st.subheader("Clustered News")
    if title_column:
        st.dataframe(df[[title_column, "cluster"]])
    else:
        st.dataframe(df[[text_column, "cluster"]])

    st.subheader("Top Keywords per Topic")
    terms = TfidfVectorizer().fit(df["clean_text"]).get_feature_names_out()
    centers = KMeans(n_clusters=k, random_state=42).fit(X).cluster_centers_

    for i in range(k):
        top_words = [terms[ind] for ind in centers[i].argsort()[-5:][::-1]]
        st.write(f"**Topic {i}:** {', '.join(top_words)}")
