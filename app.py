import streamlit as st
import os
from pathlib import Path
from dotenv import load_dotenv

# --- CONFIGURATION ---
load_dotenv()
st.set_page_config(page_title="Assistant Expert : Fiscalité Française", page_icon="⚖️", layout="wide")

# --- IMPORTS MODERNES (LCEL) ---
try:
    from langchain_openai import OpenAIEmbeddings, ChatOpenAI
    from langchain_community.document_loaders import PyPDFLoader
    from langchain_community.vectorstores import FAISS
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_core.runnables import RunnablePassthrough
    from langchain_core.output_parsers import StrOutputParser
except ImportError as e:
    st.error(f"❌ Erreur d'importation : {e}")
    st.stop()

# --- INTERFACE ---
st.title("⚖️ Assistant Expert : Fiscalité Française")
st.markdown("Analyse combinée : **CGI 2026**, **LPF 2026** et **Brochure Fiscale 2023**.")

# --- INITIALISATION DE LA BASE (MULTI-DOCS) ---
@st.cache_resource
def init_vector_db():
    current_dir = Path(__file__).parent
    pdf_files = list(current_dir.glob("*.pdf"))
    
    if not pdf_files:
        st.error("Aucun fichier PDF trouvé. Vérifiez que vos documents sont dans le dossier du projet.")
        return None

    all_docs = []
    
    with st.status("Indexation de la base documentaire fiscale...") as status:
        for pdf in pdf_files:
            st.write(f"📖 Lecture de : {pdf.name}...")
            try:
                loader = PyPDFLoader(str(pdf))
                all_docs.extend(loader.load())
            except Exception as e:
                st.warning(f"Impossible de lire {pdf.name}: {e}")
        
        # Découpage : on réduit un peu la taille des morceaux pour plus de précision
        splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
        chunks = splitter.split_documents(all_docs)
        
        # Création des vecteurs
        embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
        vectorstore = FAISS.from_documents(chunks, embeddings)
        status.update(label="Base de données prête !", state="complete")
        return vectorstore

vector_db = init_vector_db()

# --- INTERFACE DE DISCUSSION ---
if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Ex: Quelles sont les conditions d'exonération de la plus-value immobilière ?"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    if vector_db:
        with st.chat_message("assistant"):
            with st.spinner("Recherche croisée dans le CGI, le LPF et la Brochure..."):
                llm = ChatOpenAI(model="gpt-4o", temperature=0)
                
                template = """Vous êtes l'Assistant Expert : Fiscalité Française.
                Répondez à la question en utilisant les extraits fournis. 
                
                RÈGLES CRITIQUES :
                1. Identifiez la source de chaque information (ex: "Selon l'article 150 U du CGI..." ou "Le LPF précise que...").
                2. Si les documents se complètent, faites une synthèse (ex: règle de fond dans le CGI et procédure dans le LPF).
                3. Soyez extrêmement rigoureux sur les termes juridiques.
                4. Si l'information est absente des trois documents, dites-le.

                CONTEXTE :
                {context}

                QUESTION : 
                {question}

                RÉPONSE :"""
                
                rag_prompt = ChatPromptTemplate.from_template(template)

                def format_docs(docs):
                    # On affiche explicitement le nom du fichier source pour aider l'IA
                    return "\n\n".join(f"[Source: {doc.metadata.get('source')}]\n{doc.page_content}" for doc in docs)

                # On récupère les 10 meilleurs morceaux (k=10) car le CGI est très dense
                retriever = vector_db.as_retriever(search_kwargs={"k": 10})
                
                rag_chain = (
                    {"context": retriever | format_docs, "question": RunnablePassthrough()}
                    | rag_prompt
                    | llm
                    | StrOutputParser()
                )
                
                try:
                    response = rag_chain.invoke(prompt)
                    st.markdown(response)
                    st.session_state.messages.append({"role": "assistant", "content": response})
                except Exception as e:
                    st.error(f"Erreur de génération : {e}")