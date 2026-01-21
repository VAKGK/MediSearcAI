import os
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from torchvision import models, transforms
from PIL import Image

# --- 1. PATH CONFIGURATION ---
# Get the folder where this script is (1_Code)
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
# Go up one level to (Final_Submission)
BASE_DIR = os.path.dirname(CURRENT_DIR)

# Define paths to the Data folder
DATA_DIR = os.path.join(BASE_DIR, "2_Dataset")
DATASET_DIR = os.path.join(DATA_DIR, "dataset_root")
METADATA_FILE = os.path.join(DATA_DIR, "xray_metadata.csv")
FEATURE_FILE = os.path.join(DATA_DIR, "image_features.npy")

# --- 2. DEVICE CONFIGURATION (Must be before print) ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"⚙️  Using device: {DEVICE}")

# --- 3. LOAD MODEL ---
print("⏳ Loading ResNet50 model...")
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

# --- 4. FEATURE EXTRACTION ---
if not os.path.exists(METADATA_FILE):
    print(f"❌ Error: Metadata file not found at {METADATA_FILE}")
    exit()

print(f"📂 Reading metadata from: {METADATA_FILE}")
df = pd.read_csv(METADATA_FILE)
features_list = []
valid_indices = []

print(f"🚀 Scanning {len(df)} images to build index...")

for idx, row in df.iterrows():
    # Construct full image path
    image_path = os.path.join(DATASET_DIR, row['category'], row['image_name'])

    try:
        if os.path.exists(image_path):
            img = Image.open(image_path).convert('RGB')
            img_tensor = preprocess(img).unsqueeze(0).to(DEVICE)

            with torch.no_grad():
                # Extract features
                feature = model(img_tensor).cpu().numpy().flatten()

            features_list.append(feature)
            valid_indices.append(idx)
        else:
            # Image file missing (skip it)
            pass
    except Exception as e:
        print(f"   Skipping bad image: {row['image_name']}")

    # Progress Update
    if idx > 0 and idx % 50 == 0:
        print(f"   Processed {idx} images...")

# --- 5. SAVE RESULTS ---
if len(features_list) > 0:
    final_features = np.array(features_list)
    np.save(FEATURE_FILE, final_features)
    print("-" * 50)
    print(f"✅ SUCCESS! Saved index to: {FEATURE_FILE}")
    print(f"📊 Total images indexed: {len(features_list)}")
    print("👉 Now you can run: streamlit run app.py")
else:
    print("❌ No images were processed. Check your dataset folder location.")