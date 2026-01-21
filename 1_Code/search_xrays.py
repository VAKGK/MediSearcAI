import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# --- PATH CONFIGURATION ---
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(CURRENT_DIR)
CSV_PATH = os.path.join(BASE_DIR, "2_Dataset", "xray_metadata.csv")

# Now load the dataframe using this path
df = pd.read_csv(CSV_PATH)

# 2. Preprocessing: Create a "Searchable Text" column
# We combine 'category' and 'source_url' because the URL often contains specific keywords like 'pneumonia' or 'arthritis'
# We replace special characters in URLs with spaces to make words separate
df['search_text'] = df['category'] + " " + df['source_url'].str.replace(r'[-_/:.]', ' ', regex=True)

# 3. Build the TF-IDF Model
vectorizer = TfidfVectorizer(stop_words='english')
tfidf_matrix = vectorizer.fit_transform(df['search_text'])


def search_images(query, top_k=5):
    """
    Takes a text query and returns the top K matching images.
    """
    # Convert the query to a vector
    query_vec = vectorizer.transform([query])

    # Calculate similarity between query and all images
    similarities = cosine_similarity(query_vec, tfidf_matrix).flatten()

    # Get the indices of the top K results (sorted by similarity score)
    # .argsort() returns indices of sorted values, we take the last K and reverse them
    top_indices = similarities.argsort()[-top_k:][::-1]

    # Check if we found any matches (if the top score is 0, no match found)
    if similarities[top_indices[0]] == 0:
        print(f"No relevant images found for query: '{query}'")
        return

    print(f"\n--- Top {top_k} Results for '{query}' ---")
    results = df.iloc[top_indices][['image_name', 'category', 'source_url']]

    # Add similarity score for visibility
    results['score'] = similarities[top_indices]

    for idx, row in results.iterrows():
        print(f"Image: {row['image_name']}")
        print(f"Category: {row['category']}")
        print(f"Match Score: {row['score']:.4f}")
        print(f"Source: {row['source_url']}")
        print("-" * 50)


# --- Interactive Part (for when you run it locally) ---
if __name__ == "__main__":
    while True:
        user_query = input("\nEnter search query (or 'exit'): ")
        if user_query.lower() == 'exit':
            break
        search_images(user_query)