from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain_openai import ChatOpenAI  # Sostituisci OpenAI con ChatOpenAI
import os
from dotenv import load_dotenv

# Carica le variabili d'ambiente dal file .env
load_dotenv()

# Recupera la chiave API
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError("OPENAI_API_KEY non trovata nel file .env! Assicurati di averlo configurato correttamente.")

# Percorso del PDF basato sulla root del progetto
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
pdf_path = os.path.join(project_root, "data", "bilancio_societa_big.pdf")

# Verifica che il file esista
if not os.path.exists(pdf_path):
    raise FileNotFoundError(f"Il file {pdf_path} non esiste! Verifica il percorso.")

print("Carico il PDF...")
loader = PyPDFLoader(pdf_path)
documents = loader.load()

# Dividi il testo in chunk più piccoli
print("Divido il testo in chunk...")
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)
chunks = text_splitter.split_documents(documents)
print(f"Numero di chunk creati: {len(chunks)}")

# Crea un vettore di embedding con OpenAI
print("Creo gli embeddings...")
embeddings = OpenAIEmbeddings(openai_api_key=api_key)

# Crea un database vettoriale con FAISS
print("Indicizzo i chunk nel database vettoriale...")
vector_store = FAISS.from_documents(chunks, embeddings)

# Salva il database vettoriale
vector_store.save_local(os.path.join(project_root, "faiss_index"))

# Configura il modello OpenAI (usa ChatOpenAI invece di OpenAI)
llm = ChatOpenAI(
    api_key=api_key,
    model="gpt-4-turbo",  # Specifica il modello di chat
    temperature=0.7  # Opzionale: regola la creatività della risposta
)

# Crea la catena RAG
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=vector_store.as_retriever(search_kwargs={"k": 5}),
    return_source_documents=True
)

# Definisci la query per il value investing
query = """
Sei un esperto di value investing. Analizza il bilancio societario e forniscimi un parere su:
1. La solidità finanziaria (es. rapporto debito/capitale, liquidità).
2. La redditività (es. margine di profitto, ROE).
3. Eventuali segnali di sottovalutazione o sopravvalutazione rispetto al valore intrinseco.
Fornisci stime numeriche quando possibile basandoti sui dati disponibili.
"""

# Esegui l'analisi
print("Analizzo il bilancio...")
result = qa_chain({"query": query})

# Stampa il risultato
print("\nRisultato dell'analisi:\n")
print(result["result"])

# Stampa i documenti fonte usati
print("\nDocumenti utilizzati per l'analisi:")
for doc in result["source_documents"]:
    print(f"- {doc.page_content[:200]}... (Pagina {doc.metadata['page']})")