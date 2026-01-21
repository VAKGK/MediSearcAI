from fastapi import FastAPI
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

app = FastAPI()

# --- PATH CONFIGURATION ---
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(CURRENT_DIR)
CSV_PATH = os.path.join(BASE_DIR, "2_Dataset", "xray_metadata.csv")

# Load data
df = pd.read_csv(CSV_PATH)
df['search_text'] = df['category'] + " " + df['source_url'].astype(str)
vectorizer = TfidfVectorizer(stop_words='english')
tfidf_matrix = vectorizer.fit_transform(df['search_text'])


@app.get("/")
def home():
    return {"message": "X-Ray Search API is Running"}


@app.get("/search/")
def search(query: str, limit: int = 5):
    """
    Search API: Returns top matching images as JSON.
    Example URL: http://127.0.0.1:8000/search/?query=pneumonia&limit=3
    """
    query_vec = vectorizer.transform([query])
    similarities = cosine_similarity(query_vec, tfidf_matrix).flatten()

    top_indices = similarities.argsort()[-limit:][::-1]

    results = []
    for idx in top_indices:
        if similarities[idx] > 0:
            row = df.iloc[idx]
            results.append({
                "image_name": row['image_name'],
                "category": row['category'],
                "score": float(similarities[idx]),
                "source_url": row['source_url']
            })

    return {"query": query, "results": results}