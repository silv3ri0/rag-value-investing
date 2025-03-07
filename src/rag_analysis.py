from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain_openai import ChatOpenAI
import os
from dotenv import load_dotenv
import yfinance as yf

# Funzione per recuperare il prezzo azione attuale
def get_current_stock_price(ticker):
    try:
        stock = yf.Ticker(ticker)
        price = stock.history(period="1d")["Close"].iloc[-1]  # Ultimo prezzo di chiusura
        return price
    except Exception as e:
        print(f"Errore nel recupero del prezzo di {ticker}: {e}")
        return None

# Carica le variabili d'ambiente dal file .env
load_dotenv()

# Recupera la chiave API
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError("OPENAI_API_KEY non trovata nel file .env! Assicurati di averlo configurato correttamente.")

# Percorso del PDF basato sulla root del progetto
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
pdf_path = os.path.join(project_root, "data", "bilancio_societa_big.pdf")
output_path = os.path.join(project_root, "output", "analisi_value_investing_chatgpt.md")  # Nome file modificato

# Verifica che il file esista
if not os.path.exists(pdf_path):
    raise FileNotFoundError(f"Il file {pdf_path} non esiste! Verifica il percorso.")

print("Carico il PDF...")
loader = PyPDFLoader(pdf_path)
documents = loader.load()

# Dividi il testo in chunk più piccoli
print("Divido il testo in chunk...")
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,  # Granularità per più precisione
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
    retriever=vector_store.as_retriever(search_kwargs={"k": 50}),
    return_source_documents=True
)

# Recupera il prezzo attuale di Brunello Cucinelli
ticker = "BC.MI"
current_price = get_current_stock_price(ticker)
if current_price is not None:
    print(f"Prezzo attuale di {ticker}: €{current_price:.2f}")
else:
    print("Impossibile recuperare il prezzo attuale. Procedo senza P/E.")

# Definisci la query dinamica senza numeri di pagina
query = f"""
Sei un esperto di value investing. Analizza il bilancio societario di Brunello Cucinelli S.p.A. e forniscimi un parere dettagliato basato sui dati numerici del PDF. Calcola:
1. La solidità finanziaria:
   - Rapporto debito/capitale: debiti totali (passività correnti + passività non correnti) / patrimonio netto, dalla "Situazione Patrimoniale e Finanziaria Consolidata". Usa SOLO 'Totale Passività Correnti' e 'Totale Passività Non Correnti'.
   - Current ratio: attività correnti / passività correnti, dalla stessa sezione. Usa SOLO 'Totale Attività Correnti' (564,201 migliaia di €) e 'Totale Passività Correnti'.
2. La redditività:
   - Margine di profitto netto: utile netto / ricavi totali, dal "Conto Economico". Usa SOLO 'Ricavi' (620,662 migliaia di €) e 'Risultato del periodo'.
   - ROE: utile netto / patrimonio netto, dalla "Situazione Patrimoniale". Usa SOLO 'Totale Patrimonio Netto'.
3. Segnali di sottovalutazione:
   - Book value: patrimonio netto / numero di azioni (in € per azione, converti migliaia di € dividendo per 1,000 prima di dividere per il numero di azioni).
   - P/E: prezzo azione €{current_price:.2f} / EPS (calcola EPS come utile netto / numero di azioni, in € per azione, converti migliaia di € dividendo per 1,000).
Estrai i numeri rilevanti (ricavi totali, utile netto, debiti totali, patrimonio netto, attività correnti, passività correnti, numero di azioni, EPS) dai chunk e indica da dove li hai presi (es. "Conto Economico" o "Situazione Patrimoniale"). Usa SOLO i valori totali specificati (es. 'Totale Attività Correnti', non somme parziali) e non fare stime se i dati sono presenti. Se un dato manca, segnalalo chiaramente.
"""

# Esegui l'analisi
print("Analizzo il bilancio...")
result = qa_chain({"query": query})

# Stampa il risultato
output = f"# Analisi Value Investing - Brunello Cucinelli S.p.A. ({ticker})\n\n"
output += f"**Prezzo attuale:** €{current_price:.2f}\n\n"
output += "## Risultati\n" + result["result"]
print(output)

# Salva l'output nel file specificato
os.makedirs(os.path.dirname(output_path), exist_ok=True)
with open(output_path, "w", encoding="utf-8") as f:
    f.write(output)
print(f"Risultati salvati in {output_path}")

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