from dotenv import load_dotenv
import os
from pathlib import Path
import faiss
import sys
import subprocess

# Installation forcée dans l'environnement actuel
subprocess.check_call([sys.executable, "-m", "pip", "install", "pypdf"])
from langchain_openai import OpenAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.docstore.in_memory import InMemoryDocstore

# Spécifier explicitement le chemin du fichier .env
env_path = Path('.') / '.env'
load_dotenv(dotenv_path=env_path)

# Charger le PDF
print(" Chargement du PDF...")
loader = PyPDFLoader("~/work/herman-melville-moby-dick.pdf")
documents = loader.load()

print(f" Nombre de pages chargées: {len(documents)}")
print(f"\n Aperçu de la première page:")
print(documents[0].page_content[:300])
print(f"\n Métadonnées: {documents[0].metadata}")

# Découper les documents en chunks
print("\n Découpage des documents en chunks...")
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
    length_function=len,
)
chunks = text_splitter.split_documents(documents)
print(f"Nombre de chunks créés: {len(chunks)}")

# Créer les embeddings
print("\n Création des embeddings...")
embeddings = OpenAIEmbeddings(model="text-embedding-3-large")

# Créer l'index FAISS
print("\n Création de la base de données FAISS...")
embedding_dim = len(embeddings.embed_query("test"))  # 3072 pour text-embedding-3-large
index = faiss.IndexFlatL2(embedding_dim)

# Créer le vector store
vector_store = FAISS(
    embedding_function=embeddings,
    index=index,
    docstore=InMemoryDocstore(),
    index_to_docstore_id={}
)

# Ajouter les documents
print("\n Ajout des chunks dans le vector store...")
ids = vector_store.add_documents(documents=chunks)
print(f" {len(ids)} chunks ajoutés avec succès!")
print("Statistiques:")
print(f"  - Dimension des vecteurs: {embedding_dim}")
print(f"  - Nombre total de vecteurs: {vector_store.index.ntotal}")