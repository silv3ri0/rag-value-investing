from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain_openai import ChatOpenAI
from langchain_community.embeddings import HuggingFaceInferenceAPIEmbeddings
import os
from dotenv import load_dotenv
import yfinance as yf

def get_current_stock_price(ticker):
    try:
        stock = yf.Ticker(ticker)
        price = stock.history(period="1d")["Close"].iloc[-1]
        return price
    except Exception as e:
        print(f"Errore nel recupero del prezzo di {ticker}: {e}")
        return None

load_dotenv()
hf_api_key = os.getenv("HF_TOKEN")
openrouter_api_key = os.getenv("OPENROUTER_API_KEY")
if not hf_api_key or not openrouter_api_key:
    raise ValueError("Mancano HUGGINGFACE_API_KEY o OPENROUTER_API_KEY nel file .env!")

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
pdf_path = os.path.join(project_root, "data", "bilancio_societa_big.pdf")

if not os.path.exists(pdf_path):
    raise FileNotFoundError(f"Il file {pdf_path} non esiste!")

print("Carico il PDF...")
loader = PyPDFLoader(pdf_path)
documents = loader.load()

print("Divido il testo in chunk...")
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
chunks = text_splitter.split_documents(documents)
print(f"Numero di chunk creati: {len(chunks)}")

print("Creo gli embeddings con Sentence-Transformers...")
embeddings = HuggingFaceInferenceAPIEmbeddings(
    api_key=hf_api_key,
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

print("Indicizzo i chunk nel database vettoriale...")
vector_store = FAISS.from_documents(chunks, embeddings)
vector_store.save_local(os.path.join(project_root, "faiss_index"))

llm = ChatOpenAI(
    openai_api_key=openrouter_api_key,
    base_url="https://openrouter.ai/api/v1",
    model="deepseek/deepseek-chat:free",
    temperature=0.7
)

qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=vector_store.as_retriever(search_kwargs={"k": 40}),
    return_source_documents=True
)

ticker = "BC.MI"
current_price = get_current_stock_price(ticker)
if current_price is not None:
    print(f"Prezzo attuale di {ticker}: €{current_price:.2f}")
else:
    print("Impossibile recuperare il prezzo attuale. Procedo senza P/E.")

query = f"""
Sei un esperto di value investing. Analizza il bilancio societario di Brunello Cucinelli S.p.A. e forniscimi un parere dettagliato basato sui dati numerici del PDF. Calcola:
1. Solidità finanziaria:
   - Rapporto debito/capitale: debiti totali (passività correnti + passività non correnti, pagina 53-54) / patrimonio netto (pagina 47).
   - Current ratio: attività correnti (pagina 53-54, NON usare totale patrimonio netto + passività) / passività correnti (pagina 53-54).
2. Redditività:
   - Margine di profitto netto: utile netto totale del periodo (pagina 28, NON utile parziale) / ricavi totali (pagina 89).
   - ROE: utile netto totale del periodo (pagina 28) / patrimonio netto (pagina 47).
3. Segnali di sottovalutazione:
   - Book value: patrimonio netto (pagina 47) / numero di azioni (pagina 47, cerca '68.000.000').
   - P/E: prezzo azione €{current_price:.2f} / EPS (EPS = utile netto totale / numero di azioni).
Estrai: ricavi totali (pagina 89), utile netto totale (pagina 28), debiti totali (pagina 53-54), patrimonio netto (pagina 47), attività correnti (pagina 53-54), passività correnti (pagina 53-54), numero di azioni (pagina 47). Se un dato manca, indica la pagina necessaria. Usa SOLO i valori totali corretti, non parziali o sbagliati!
"""

print("Analizzo il bilancio...")
result = qa_chain({"query": query})

print("\nRisultato dell'analisi:\n")
print(result["result"])

print("\nDocumenti utilizzati per l’analisi:")
for doc in result["source_documents"]:
    page_num = doc.metadata.get('page', 'N/A')
    print(f"- {doc.page_content[:200]}... (Pagina {page_num})")

print("\nTutti i chunk recuperati:")
for i, doc in enumerate(result["source_documents"]):
    page_num = doc.metadata.get('page', 'N/A')
    print(f"Chunk {i+1} (Pagina {page_num}):")
    print(f"{doc.page_content}")
    print("-" * 50)