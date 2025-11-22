import os
# Disable telemetry, bugs are too distracting
os.environ["ANONYMIZED_TELEMETRY"] = "False"
os.environ["SCARF_NO_ANALYTICS"] = "true"

import logging
# Chroma + Posthog can be noisy; silence them so ingestion output stays clear.
logging.getLogger('chromadb').setLevel(logging.CRITICAL)
logging.getLogger('posthog').setLevel(logging.CRITICAL)

try:
    import chromadb.telemetry.product.posthog
    chromadb.telemetry.product.posthog.Posthog.capture = lambda *args, **kwargs: None
except ImportError:
    pass

import json
import chromadb
from config import DB_PATH, COLLECTION_NAME, get_embeddings

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(BASE_DIR)
# Allow overriding the dataset via env var so we can swap samples without code edits.
DATA_DIR = os.environ.get(
    "RAG_DATA_DIR",
    os.path.join(REPO_ROOT, "engagetest", "data")
)
DATA_DIR = os.path.abspath(DATA_DIR)

CITY_MAPPING = {
    "downtown dubai": "dubai",
    "dubai marina": "dubai",
    "al barsha": "dubai",
    "palm jumeirah": "dubai",
    "business bay": "dubai",
    "jumeirah lakes towers (jlt)": "dubai",
    "jumeirah beach residence (jbr)": "dubai",
    "abu dhabi": "abu dhabi",
    "sharjah": "sharjah",
    "ajman": "ajman"
}

def get_city(location_str):
    """Infer city from neighborhood."""
    loc_lower = location_str.lower()
    return CITY_MAPPING.get(loc_lower, "dubai")

def process_price(price_str):
    """
    Price ranges arrive as strings like 'AED 150 - 200'. We only need integer
    bounds for metadata filtering, so parse defensively and fall back to a wide
    range if the string is malformed.
    """
    try:
        clean = price_str.replace("AED", "").strip()
        parts = clean.split("-")
        return int(parts[0].strip()), int(parts[1].strip())
    except:
        return 0, 1000

def ingest_data():
    print("--- STARTING INGESTION (ALL LOWERCASE METADATA) ---")
    
    with open(os.path.join(DATA_DIR, "restaurant.json"), "r") as f:
        data = json.load(f)

    client = chromadb.PersistentClient(path=DB_PATH)
    embed_model = get_embeddings()
    
    try:
        client.delete_collection(COLLECTION_NAME)
    except:
        # Collection might not exist on first run, so ignore.
        pass
    
    collection = client.create_collection(name=COLLECTION_NAME)

    ids = []
    documents = []
    metadatas = []

    print(f"Enriching data for {len(data)} restaurants...")

    for item in data:
        min_p, max_p = process_price(item['price_range'])
        
        neighborhood = item['location']
        city = get_city(neighborhood)
        cuisine_clean = item['cuisine'].lower()
        
        location_clean = neighborhood.lower()
        
        text_content = f"{item['name']} in {neighborhood}, {city}. {item['cuisine']} cuisine. {item['description']} Amenities: {item['amenities']}"
        
        ids.append(str(item['id']))
        documents.append(text_content)
        metadatas.append({
            "name": item['name'],
            "location": location_clean, 
            "city": city,               
            "cuisine": cuisine_clean,   
            "price_min": min_p,
            "price_max": max_p,
            "amenities": item['amenities']
        })

    print("Generating embeddings...")
    embeddings = embed_model.embed_documents(documents) 

    collection.add(
        ids=ids,
        documents=documents,
        embeddings=embeddings,
        metadatas=metadatas
    )
    
    print(f"Successfully ingested {len(documents)} restaurants.")

if __name__ == "__main__":
    ingest_data()