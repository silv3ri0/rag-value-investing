from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain_openai import ChatOpenAI
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
    chunk_size=1000,  # Tornato a 1000 per più granularità
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

# Configura il modello OpenAI
llm = ChatOpenAI(
    api_key=api_key,
    model="gpt-4-turbo",
    temperature=0.7
)

# Crea la catena RAG
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=vector_store.as_retriever(search_kwargs={"k": 20}),  # Aumentato a 20
    return_source_documents=True
)

# Definisci la query per il value investing (prompt con numeri espliciti)
query = """
Sei un esperto di value investing. Analizza il bilancio societario di Brunello Cucinelli S.p.A. e forniscimi un parere dettagliato basato sui dati numerici del PDF. Calcola:
1. La solidità finanziaria: rapporto debito/capitale (debiti totali €1.086.662 mila / patrimonio netto €449.202 mila) e current ratio (attività correnti €424.941 mila / passività correnti €446.938 mila) dalla "Situazione Patrimoniale e Finanziaria Consolidata" (pagina 53-54).
2. La redditività: margine di profitto netto (utile netto €66.077 mila / ricavi totali €620.662 mila) dal "Conto Economico" (pagina 31) e ROE (utile netto €66.077 mila / patrimonio netto €449.202 mila) dalla "Situazione Patrimoniale" (pagina 47).
3. Eventuali segnali di sottovalutazione: calcola book value (patrimonio netto €449.202 mila / numero di azioni 68.000.000) a pagina 47 e cerca P/E nelle note.
Conferma i numeri indicati, estrai ulteriori dettagli se disponibili, e indica le pagine esatte. Se i dati mancano, segnalalo.
"""

# Esegui l'analisi
print("Analizzo il bilancio...")
result = qa_chain({"query": query})

# Stampa il risultato
print("\nRisultato dell'analisi:\n")
print(result["result"])

# Stampa i documenti fonte usati (versione breve)
print("\nDocumenti utilizzati per l’analisi:")
for doc in result["source_documents"]:
    page_num = doc.metadata.get('page', 'N/A')
    print(f"- {doc.page_content[:200]}... (Pagina {page_num})")

# Debug: Stampa tutti i chunk recuperati per intero
print("\nTutti i chunk recuperati:")
for i, doc in enumerate(result["source_documents"]):
    page_num = doc.metadata.get('page', 'N/A')
    print(f"Chunk {i+1} (Pagina {page_num}):")
    print(f"{doc.page_content}")
    print("-" * 50)