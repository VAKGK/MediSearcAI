import streamlit as st
import pandas as pd
import os
import numpy as np
from PIL import Image
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import torch
import torch.nn as nn
from torchvision import models, transforms

# -----------------------------------------------------------------------------
# 1. PATH & DEVICE CONFIGURATION (LOGIC UNCHANGED)
# -----------------------------------------------------------------------------
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(CURRENT_DIR)

DATA_DIR = os.path.join(BASE_DIR, "2_Dataset")
DATASET_DIR = os.path.join(DATA_DIR, "dataset_root")
METADATA_FILE = os.path.join(DATA_DIR, "xray_metadata.csv")
FEATURE_FILE = os.path.join(DATA_DIR, "image_features.npy")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -----------------------------------------------------------------------------
# 2. UI CONFIGURATION & CUSTOM CSS (✨ UPDATED STYLES ✨)
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="MediSearch AI | X-Ray Database",
    page_icon="🩺",
    layout="wide",  # Ensures full width
    initial_sidebar_state="collapsed"
)

# Custom CSS
st.markdown("""
    <style>
    /* Main Background */
    .stApp {
        background-color: #f4f8fb;
    }

    /* --- UPDATED HEADER STYLING --- */
    .main-header {
        font-family: 'Helvetica Neue', sans-serif;
        color: #0d47a1; /* Medical Dark Blue */
        text-align: center;
        padding: 30px 20px; /* Increased padding */
        background: white;
        border-radius: 15px;
        box-shadow: 0px 4px 12px rgba(0,0,0,0.05);
        margin-bottom: 25px;
    }
    /* Make Title BIGGER */
    .main-header h1 {
        font-weight: 800;
        font-size: 3.5rem; /* Increased font size */
        margin-bottom: 10px;
    }
    /* Make Subtitle slightly bigger */
    .main-header p {
        font-size: 1.2rem;
        color: #666;
    }

    /* Tab Styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 10px;
        justify-content: center;
        margin-bottom: 20px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 55px;
        background-color: white;
        border-radius: 10px;
        box-shadow: 0px 2px 5px rgba(0,0,0,0.05);
        color: #444;
        font-weight: 600;
        font-size: 1.1rem;
        padding-left: 30px;
        padding-right: 30px;
    }
    .stTabs [data-baseweb="tab"]:hover {
        color: #0d47a1;
        background-color: #e3f2fd;
    }
    .stTabs [aria-selected="true"] {
        background-color: #0d47a1 !important;
        color: white !important;
        box-shadow: 0px 4px 10px rgba(13, 71, 161, 0.3);
    }

    /* Input Field Styling - Making them feel expanded */
    .stTextInput > div > div > input {
        border-radius: 12px;
        border: 2px solid #eee;
        padding: 15px; /* Larger padding for expanded feel */
        font-size: 18px; /* Larger text */
    }
    .stTextInput > div > div > input:focus {
        border-color: #0d47a1;
    }

    /* File Uploader Styling */
    [data-testid='stFileUploader'] {
        background-color: white;
        padding: 20px;
        border-radius: 12px;
        border: 2px dashed #eee;
    }

    /* Image Card Effect */
    div[data-testid="stImage"] {
        background: white;
        padding: 8px;
        border-radius: 12px;
        box-shadow: 0 2px 6px rgba(0,0,0,0.08);
        transition: all 0.2s ease-in-out;
    }
    div[data-testid="stImage"]:hover {
        transform: translateY(-5px);
        box-shadow: 0 8px 15px rgba(0,0,0,0.15);
    }

    /* Caption Styling */
    div[data-testid="stCaptionContainer"] {
        text-align: center;
        color: #555;
        font-weight: 500;
        margin-top: 5px;
        font-size: 0.9rem;
    }

    /* Headers inside tabs */
    h2, h3 {
        color: #333;
        font-weight: 700;
    }
    </style>
""", unsafe_allow_html=True)


# -----------------------------------------------------------------------------
# 3. DATA & MODEL LOADING (LOGIC UNCHANGED)
# -----------------------------------------------------------------------------
@st.cache_data
def load_data():
    if os.path.exists(METADATA_FILE):
        return pd.read_csv(METADATA_FILE)
    else:
        st.error(f"❌ Error: Metadata file not found at {METADATA_FILE}")
        return pd.DataFrame()


df = load_data()


@st.cache_resource
def load_features():
    if os.path.exists(FEATURE_FILE):
        return np.load(FEATURE_FILE, allow_pickle=True)
    return None


feature_bank = load_features()

if not df.empty:
    df['search_text'] = df['category'] + " " + df['source_url'].astype(str)
    vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_matrix = vectorizer.fit_transform(df['search_text'].fillna(''))


@st.cache_resource
def load_image_model():
    weights = models.ResNet50_Weights.IMAGENET1K_V1
    base_model = models.resnet50(weights=weights)
    model = nn.Sequential(*list(base_model.children())[:-1])
    model = model.to(DEVICE)
    model.eval()

    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    return model, preprocess


model, preprocess = load_image_model()

# -----------------------------------------------------------------------------
# 4. UI LAYOUT
# -----------------------------------------------------------------------------

# Fancy HTML Header (Updated classes respond to new CSS for larger size)
st.markdown('<div class="main-header"><h1>🩺 MediSearch AI</h1><p>Intelligent X-Ray Retrieval System</p></div>',
            unsafe_allow_html=True)

tab1, tab2 = st.tabs(["🔍 TEXT SEARCH Engine", "🖼️ IMAGE ANALYSIS Engine"])

# --- TAB 1: TEXT SEARCH ---
with tab1:
    st.markdown("### ⌨️ Search by Keywords")
    st.write("Enter keywords like 'pneumonia', 'fracture', or 'dental' to search metadata.")

    # REMOVED columns here so it expands to full width
    query = st.text_input("Search Query", placeholder="e.g., 'Lateral spine view'...", label_visibility="collapsed")

    if query and not df.empty:
        query_vec = vectorizer.transform([query])
        similarities = cosine_similarity(query_vec, tfidf_matrix).flatten()
        top_indices = similarities.argsort()[-10:][::-1]

        st.markdown(f"--- \n### 🔎 Results for: *'{query}'*")

        cols = st.columns(5)
        for i, idx in enumerate(top_indices):
            if similarities[idx] > 0:
                row = df.iloc[idx]
                image_path = os.path.join(DATASET_DIR, row['category'], row['image_name'])

                with cols[i % 5]:
                    if os.path.exists(image_path):
                        cat_display = row['category'].replace("_", " ").title()
                        st.image(Image.open(image_path), use_container_width=True)
                        st.caption(f"**{cat_display}**\nMatch: {similarities[idx]:.2f}")
    elif query:
        st.info("No matching records found in metadata.")

# --- TAB 2: IMAGE SEARCH ---
with tab2:
    st.markdown("### 📤 Visual Similarity Search")
    st.write("Upload a patient X-ray to find visually similar cases in the database.")

    # File uploader is naturally full width
    uploaded_file = st.file_uploader("Choose X-Ray Image", type=["jpg", "png", "jpeg"], label_visibility="collapsed")

    if uploaded_file is not None:
        st.markdown("---")
        # Using columns here is good to show input vs output side-by-side
        colA, colB = st.columns([1, 3])

        with colA:
            st.markdown("##### Your Input")
            st.image(uploaded_file, caption="Query Image", use_container_width=True)

        with colB:
            if feature_bank is None:
                st.error("⚠️ Database index missing. Please run `python image_search.py` first.")
            else:
                with st.spinner("Analyzing anatomical features with ResNet50..."):
                    img = Image.open(uploaded_file).convert('RGB')
                    img_tensor = preprocess(img).unsqueeze(0).to(DEVICE)

                    with torch.no_grad():
                        query_feat = model(img_tensor).cpu().numpy().flatten().reshape(1, -1)

                    dists = cosine_similarity(query_feat, feature_bank).flatten()
                    top_indices = dists.argsort()[-5:][::-1]

                st.markdown("##### ✅ Similar Database Matches")

                # Results Grid
                res_cols = st.columns(5)
                for i, idx in enumerate(top_indices):
                    row = df.iloc[idx]
                    image_path = os.path.join(DATASET_DIR, row['category'], row['image_name'])

                    with res_cols[i]:
                        if os.path.exists(image_path):
                            cat_display = row['category'].replace("_", " ").title()
                            st.image(Image.open(image_path), use_container_width=True)
                            st.caption(f"**{cat_display}**\nSim: {dists[idx]:.2f}")