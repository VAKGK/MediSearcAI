import os
import requests
import pandas as pd
from duckduckgo_search import DDGS
from time import sleep

# --- CONFIGURATION ---
# Define categories and how many images you want per category
# We ask for 120 to ensure we have valid files (some downloads might fail)
SEARCH_QUERIES = {
    "chest": "chest x-ray pneuomonia medical",
    "dental": "panoramic dental x-ray",
    "spine": "lumbar spine x-ray lateral view",
    "fracture": "bone fracture x-ray hand",
    "knee": "knee joint x-ray arthritis"
}
IMAGES_PER_CATEGORY = 120 
OUTPUT_DIR = "dataset_root"
CSV_FILENAME = "xray_metadata.csv"

# --- SETUP ---
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

metadata = []
ddgs = DDGS()

print(f"Starting automation... Target: {len(SEARCH_QUERIES) * IMAGES_PER_CATEGORY} images.")

# --- MAIN LOOP ---
for category, query in SEARCH_QUERIES.items():
    print(f"\n--- Processing Category: {category} ---")
    
    # Create category folder
    category_path = os.path.join(OUTPUT_DIR, category)
    if not os.path.exists(category_path):
        os.makedirs(category_path)
    
    # 1. Search for images
    # This fetches image URLs from various public websites
    results = ddgs.images(
        keywords=query,
        region="wt-wt",
        safesearch="off",
        max_results=IMAGES_PER_CATEGORY
    )
    
    count = 0
    for result in results:
        image_url = result['image']
        source_url = result['url'] # The website the image came from
        
        try:
            # 2. Download the image
            response = requests.get(image_url, timeout=10)
            if response.status_code == 200:
                # Generate a filename
                filename = f"{category}_{count:03d}.jpg"
                file_path = os.path.join(category_path, filename)
                
                with open(file_path, 'wb') as f:
                    f.write(response.content)
                
                # 3. Log to Metadata
                metadata.append({
                    "image_name": filename,
                    "category": category,
                    "source_url": source_url,  # Satisfies "Source URL" requirement
                    "image_url": image_url
                })
                
                count += 1
                if count % 10 == 0:
                    print(f"  Downloaded {count} images...")
                    
        except Exception as e:
            # Skip if download fails (common with scraping)
            continue

print(f"\nDownloading complete. Total images downloaded: {len(metadata)}")

# --- SAVE METADATA ---
df = pd.DataFrame(metadata)
df.to_csv(CSV_FILENAME, index=False)
print(f"Metadata saved to {CSV_FILENAME}")