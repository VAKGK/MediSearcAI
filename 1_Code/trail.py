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
# 2. UI CONFIGURATION & CUSTOM CSS (✨ NEW STYLING ✨)
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="MediSearch AI | X-Ray Database",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS for a "Medical Dashboard" Look
st.markdown("""
    <style>
    /* Main Background */
    .stApp {
        background-color: #f4f8fb;
    }

    /* Header Styling */
    .main-header {
        font-family: 'Helvetica Neue', sans-serif;
        font-weight: 700;
        color: #0d47a1; /* Medical Dark Blue */
        text-align: center;
        margin-bottom: 20px;
        padding: 20px;
        background: white;
        border-radius: 15px;
        box-shadow: 0px 4px 12px rgba(0,0,0,0.05);
    }

    /* Tab Styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 10px;
        justify-content: center;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        background-color: white;
        border-radius: 10px;
        box-shadow: 0px 2px 5px rgba(0,0,0,0.05);
        color: #444;
        font-weight: 600;
    }
    .stTabs [data-baseweb="tab"]:hover {
        color: #0d47a1;
        background-color: #e3f2fd;
    }
    .stTabs [aria-selected="true"] {
        background-color: #0d47a1 !important;
        color: white !important;
    }

    /* Input Field Styling */
    .stTextInput > div > div > input {
        border-radius: 10px;
        border: 1px solid #ddd;
        padding: 10px;
        font-size: 16px;
    }

    /* Image Card Effect */
    div[data-testid="stImage"] {
        background: white;
        padding: 10px;
        border-radius: 12px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        transition: transform 0.2s;
    }
    div[data-testid="stImage"]:hover {
        transform: scale(1.02);
        box-shadow: 0 8px 15px rgba(0,0,0,0.15);
    }

    /* Caption Styling */
    div[data-testid="stCaptionContainer"] {
        text-align: center;
        color: #555;
        font-weight: 500;
        margin-top: -5px;
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

# Fancy HTML Header
st.markdown('<div class="main-header"><h1>🩺 MediSearch AI</h1><p>Intelligent X-Ray Retrieval System</p></div>',
            unsafe_allow_html=True)

tab1, tab2 = st.tabs(["🔍 TEXT SEARCH", "🖼️ IMAGE ANALYSIS"])

# --- TAB 1: TEXT SEARCH ---
with tab1:
    col1, col2, col3 = st.columns([1, 6, 1])
    with col2:
        query = st.text_input("Type a condition (e.g., 'Viral Pneumonia', 'Fracture', 'Implant'):",
                              placeholder="Search medical database...")

    if query and not df.empty:
        query_vec = vectorizer.transform([query])
        similarities = cosine_similarity(query_vec, tfidf_matrix).flatten()
        top_indices = similarities.argsort()[-10:][::-1]

        st.markdown(f"### 🔎 Results for: *{query}*")

        cols = st.columns(5)
        for i, idx in enumerate(top_indices):
            if similarities[idx] > 0:
                row = df.iloc[idx]
                image_path = os.path.join(DATASET_DIR, row['category'], row['image_name'])

                with cols[i % 5]:
                    if os.path.exists(image_path):
                        # Clean up category name for display
                        cat_display = row['category'].replace("_", " ").title()
                        st.image(Image.open(image_path), use_container_width=True)
                        st.caption(f"**{cat_display}**\nMatch: {similarities[idx]:.2f}")

# --- TAB 2: IMAGE SEARCH ---
with tab2:
    st.markdown("### 📤 Upload Patient X-Ray")
    uploaded_file = st.file_uploader("Upload an image for visual similarity analysis", type=["jpg", "png", "jpeg"])

    if uploaded_file is not None:
        st.markdown("---")
        colA, colB = st.columns([1, 3])

        with colA:
            st.image(uploaded_file, caption="Query Image", width=200)

        with colB:
            if feature_bank is None:
                st.error("⚠️ Database index missing. Please run `python image_search.py` first.")
            else:
                with st.spinner("Analyzing anatomical features..."):
                    img = Image.open(uploaded_file).convert('RGB')
                    img_tensor = preprocess(img).unsqueeze(0).to(DEVICE)

                    with torch.no_grad():
                        query_feat = model(img_tensor).cpu().numpy().flatten().reshape(1, -1)

                    dists = cosine_similarity(query_feat, feature_bank).flatten()
                    top_indices = dists.argsort()[-5:][::-1]

                st.success("Analysis Complete. Found similar cases:")

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