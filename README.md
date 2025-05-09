# 🚀 Coach Pro Talent - RAG intelligent avec Google Drive, FAISS & OpenAI

> Un assistant intelligent qui lit et comprend vos documents (PDF, DOCX, Google Docs) stockés sur **Google Drive**, pour répondre à vos questions avec **OpenAI**. Un vrai copilote personnel pour exploiter votre propre savoir.

---

## 🧠 Objectif

Ce projet résout un problème courant : accéder rapidement à des informations pertinentes contenues dans vos documents éparpillés. Grâce à un pipeline RAG (Retrieval-Augmented Generation), ce système :

- Télécharge vos fichiers depuis Google Drive
- Extrait et découpe leur contenu
- Vectorise les textes avec des embeddings OpenAI
- Indexe le tout dans **FAISS**
- Permet d'interroger cette base via une **interface Streamlit**
- Génère des réponses contextuelles grâce à **GPT-4o**

---

## 🔍 Fonctionnalités

- 🔐 Authentification Google Drive (OAuth2)
- 📁 Lecture de fichiers PDF, DOCX, Google Docs
- 🧠 Embedding + découpage intelligent des textes
- 📦 Indexation vectorielle avec FAISS (384 dimensions)
- ⚡ Mise à jour incrémentale (pas de re-indexation inutile)
- 🎙️ Synthèse vocale avec pyttsx3
- 🖥️ Interface simple en Streamlit avec mise à jour automatique

---
## 📁 Arborescence du projet `rag_own_data_source`

```
rag_own_data_source/
├── app.py                            # Point d'entrée principal de l'application
├── utils.py                          # Fonctions utilitaires globales
├── google_drive/
│   ├── config/
│   │   └── credentials.json          # Identifiants OAuth2 pour l'API Google Drive
│   ├── tmp/
│   │   └── .gitignore                # Ignore les fichiers temporaires
│   ├── doc_mapping_id_to_text.json  # Mapping entre les IDs de documents et leur contenu
│   ├── faiss_index.faiss            # Index vectoriel utilisé pour la recherche
│   ├── file_metadata.json           # Métadonnées des fichiers (ex: hash, nom)
│   ├── rag_faiss_drive.py           # Script d'indexation des fichiers Google Drive avec FAISS
│   ├── token.json                   # Jeton d'accès OAuth2
│   └── utils_function.py            # Fonctions utilitaires spécifiques à Google Drive
├── .env                              # Variables d'environnement (non versionné)
├── .env.example                      # Exemple de fichier .env
├── .gitignore                        # Fichiers et dossiers à ignorer par Git
├── requirements.txt                 # Liste des dépendances Python
├── README.md                         # Documentation du projet

```
