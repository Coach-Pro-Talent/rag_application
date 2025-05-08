import json
import os
from langchain_huggingface import HuggingFaceEmbeddings
import numpy as np

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)

def load_json(path):
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)  
    return {}

def save_json(obj, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)

    # Pour rechercher une requête
def get_embedding_query(query):
    return np.array(embeddings.embed_query(query), dtype=np.float32)