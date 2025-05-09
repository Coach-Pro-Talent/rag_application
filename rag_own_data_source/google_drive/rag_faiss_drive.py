import io
import json
import os
import re
import numpy as np
import fitz
import faiss
from docx import Document
from google.oauth2.credentials import Credentials
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
import streamlit as st
from langchain.text_splitter import RecursiveCharacterTextSplitter
from .utils_function import load_json, save_json, embeddings


FOLDER_NAME = "Formateur"
SCOPES = ["https://www.googleapis.com/auth/drive"]
CLIENT_SECRET_FILE = "rag_own_data_source/google_drive/config/credentials.json"
TMP_DIR = "rag_own_data_source/google_drive/tmp"
EMBED_DIM = 384

os.makedirs(TMP_DIR, exist_ok=True)
token = st.secrets["token"]

TOKEN_FILE_PATH="rag_own_data_source/google_drive/token.json"
with open(TOKEN_FILE_PATH, "w") as f:
    json.dump(dict(token), f)

# Authentification
def authenticate():
    creds = None
    if os.path.exists("rag_own_data_source/google_drive/token.json"):
        creds = Credentials.from_authorized_user_file("rag_own_data_source/google_drive/token.json", SCOPES)

    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file(
                CLIENT_SECRET_FILE, SCOPES)
            creds = flow.run_local_server(port=0)
        with open("rag_own_data_source/google_drive/token.json", "w") as token:
            token.write(creds.to_json())

    return creds



# Outils
def sanitize_filename(name):
    return re.sub(r'[^\w\-. ]', '_', name)

def get_folder_id(name, service):
    query = f"name='{name}' and mimeType='application/vnd.google-apps.folder'"
    result = service.files().list(q=query, spaces='drive',
                                  corpora='allDrives',
                                  includeItemsFromAllDrives=True,
                                  supportsAllDrives=True).execute()
    return result["files"][0]["id"] if result["files"] else None

def list_files(folder_id, service):
    query = (
        f"'{folder_id}' in parents and trashed = false and ("
        f"mimeType = 'application/pdf' or "
        f"mimeType = 'application/vnd.openxmlformats-officedocument.wordprocessingml.document' or "
        f"mimeType = 'application/vnd.google-apps.document')"
    )
    result = service.files().list(
        q=query,
        spaces='drive',
        corpora='allDrives',
        includeItemsFromAllDrives=True,
        supportsAllDrives=True,
        fields="files(id, name, modifiedTime, mimeType)"
    ).execute()
    return result.get("files", [])

def download(file_id, filename, mime_type, service):
    if mime_type == "application/vnd.google-apps.document":
        filename += ".docx"
        request = service.files().export_media(
            fileId=file_id,
            mimeType='application/vnd.openxmlformats-officedocument.wordprocessingml.document'
        )
    elif mime_type == "application/pdf":
        filename += ".pdf"
        request = service.files().get_media(fileId=file_id)
    elif mime_type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
        filename += ".docx"
        request = service.files().get_media(fileId=file_id)
    else:
        print(f"[⚠️] MIME type non pris en charge pour le téléchargement : {mime_type}")
        return None

    with io.FileIO(filename, "wb") as fh:
        downloader = MediaIoBaseDownload(fh, request)
        done = False
        while not done:
            _, done = downloader.next_chunk()

    return filename

def extract_text(path):
    try:
        if path.endswith(".pdf"):
            return "\n".join(page.get_text() for page in fitz.open(path))
        elif path.endswith(".docx"):
            return "\n".join(p.text for p in Document(path).paragraphs if p.text.strip())
    except Exception as e:
        print(f"Erreur extraction : {e}")
    return ""


def start_index_documents():
    logs = []
    creds = authenticate()
    service = build("drive", "v3", credentials=creds)
    logs.append("🔎 Début de l'indexation FAISS depuis Google Drive...")

    folder_id = get_folder_id(FOLDER_NAME, service)
    if not folder_id:
        logs.append(f"[❌] Dossier '{FOLDER_NAME}' introuvable.")
        return
    logs.append(f"📁 ID du dossier : {folder_id}")

    files = list_files(folder_id, service)
    logs.append(f"📄 {len(files)} fichier(s) trouvé(s).")

    file_metadata = load_json("rag_own_data_source/google_drive/file_metadata.json") or {}
    docs_mapping = load_json("rag_own_data_source/google_drive/doc_mapping_id_to_text.json") or {}

    if os.path.exists("rag_own_data_source/google_drive/faiss_index.faiss"):
        logs.append("📦 Chargement de l'index FAISS existant...")
        index = faiss.read_index("rag_own_data_source/google_drive/faiss_index.faiss")
        idx_offset = len(docs_mapping)
    else:
        logs.append("🆕 Création d'un nouvel index FAISS...")
        index = faiss.IndexFlatL2(EMBED_DIM)
        idx_offset = 0
        file_metadata.clear()
        docs_mapping.clear()

    new_vectors = []
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500, chunk_overlap=100, separators=["\n\n", "\n", "."]
    )

    for f in files:
        fname, fid, mtime, mime_type = f["name"], f["id"], f["modifiedTime"], f["mimeType"]
        logs.append(f"\n🔍 Analyse : {fname}")
        if fname not in file_metadata or file_metadata[fname] != mtime:
            logs.append("⬇️ Téléchargement...")
            safe_name = sanitize_filename(fname)
            raw_path = os.path.join(TMP_DIR, safe_name)
            local_path = download(fid, raw_path, mime_type, service)
            if not local_path:
                continue

            logs.append("📤 Extraction du texte...")
            text = extract_text(local_path)
            if text.strip():
                logs.append("📐 Découpage et embedding...")
                chunks = text_splitter.split_text(text)
                vectors = embeddings.embed_documents(chunks)
                vectors = np.array(vectors, dtype=np.float32)

                for chunk, vector in zip(chunks, vectors):
                    new_vectors.append(vector)
                    docs_mapping[str(idx_offset)] = chunk
                    idx_offset += 1

                file_metadata[fname] = mtime
                logs.append(f"✅ Fichier indexé : {fname}")
            else:
                logs.append(f"[⚠️] Aucun texte détecté : {fname}")
        else:
            logs.append("🔁 Aucun changement détecté.")

    if new_vectors:
        logs.append("➕ Ajout des vecteurs dans FAISS...")
        all_vectors = np.vstack(new_vectors)
        index.add(all_vectors)
        faiss.write_index(index, "rag_own_data_source/google_drive/faiss_index.faiss")
        save_json(file_metadata, "file_metadata.json")
        save_json(docs_mapping, "doc_mapping_id_to_text.json")
        logs.append(f"✅ {len(new_vectors)} chunk(s) ajouté(s) à l'index.")
    else:
        logs.append("🟡 Aucun nouveau document à indexer.")

    logs.append("🏁 Fin de l'indexation.")
    return logs


