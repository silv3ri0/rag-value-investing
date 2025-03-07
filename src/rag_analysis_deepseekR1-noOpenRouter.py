import os
import time
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.runnables import RunnableLambda
from openai import OpenAI
from dotenv import load_dotenv
import yfinance as yf

def get_current_stock_price(ticker):
    try:
        stock = yf.Ticker(ticker)
        price = stock.history(period="1d")["Close"].iloc[-1]
        return price
    except Exception as e:
        print(f"Errore nel recupero del prezzo di {ticker}: {e}")
        return 0

load_dotenv()
deepseek_api_key = os.getenv("DEEPSEEK_API_KEY")
if not deepseek_api_key:
    raise ValueError("DEEPSEEK_API_KEY non trovato nel file .env!")

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
pdf_path = os.path.join(project_root, "data", "bilancio_societa_big.pdf")
output_path = os.path.join(project_root, "output", "analisi_value_investing_deepseek.md")
ticker = "BC.MI"

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
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")

print("Indicizzo i chunk nel database vettoriale...")
vector_store = FAISS.from_documents(chunks, embeddings)
vector_store.save_local(os.path.join(project_root, "faiss_index"))

client = OpenAI(
    api_key=deepseek_api_key,
    base_url="https://api.deepseek.com/v1"
)

def call_deepseek(prompt, max_tokens=3000, **kwargs):
    prompt_str = str(prompt)
    print(f"Prompt inviato a DeepSeek:\n{prompt_str}\n{'-'*50}")
    if len(prompt_str) > 10000:
        prompt_str = prompt_str[:10000] + "\n[Troncato per limiti di lunghezza]"
    max_retries = 3
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model="deepseek-chat",
                messages=[{"role": "user", "content": prompt_str}],
                max_tokens=max_tokens,
                temperature=0.7,
                timeout=120
            )
            return response.choices[0].message.content
        except Exception as e:
            if attempt < max_retries - 1:
                print(f"Tentativo {attempt + 1} fallito: {e}. Riprovo...")
                time.sleep(5)
                continue
            return f"Errore API dopo {max_retries} tentativi: {e}"

llm = RunnableLambda(call_deepseek)

qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=vector_store.as_retriever(search_kwargs={"k": 20}),
    return_source_documents=True
)

current_price = get_current_stock_price(ticker)
print(f"Prezzo attuale di {ticker}: €{current_price:.2f}")

query = f"""
Perform a value investing analysis on the financial statement. EXTRACT the following numerical data from the text (in thousands of Euros, unless otherwise specified) and SHOW the extracted values:

1. Total debt = sum of 'Totale Passività Correnti' and 'Totale Passività Non Correnti' (search and add these values)
2. Equity = 'Totale Patrimonio Netto'
3. Current assets = 'Totale Attività Correnti'
4. Current liabilities = 'Totale Passività Correnti'
5. Net income = 'Risultato del periodo' or 'Utile netto'
6. Total revenue = 'Totale Ricavi' or 'Ricavi'
7. Number of shares = look for '68.000.000' or 'number of shares' (in units, not thousands)
8. Share price = {current_price:.2f} € (already provided)

CALCULATE the following ratios and SHOW the step-by-step calculations:
1. Debt-to-equity ratio = total debt / equity
2. Current ratio = current assets / current liabilities
3. Net profit margin = (net income / total revenue) * 100 (%)
4. ROE = (net income / equity) * 100 (%)
5. Book value = equity / number of shares (€ per share)
6. P/E = share price / (net income / number of shares)

IF a data point is not found, WRITE 'data not available' and skip the calculation. Provide your response in this exact format:
- **Extracted Data**:
  1. Total debt: [value] thousands of €
  2. Equity: [value] thousands of €
  3. Current assets: [value] thousands of €
  4. Current liabilities: [value] thousands of €
  5. Net income: [value] thousands of €
  6. Total revenue: [value] thousands of €
  7. Number of shares: [value]
  8. Share price: {current_price:.2f} €
- **Calculations**:
  1. Debt-to-equity ratio: [step-by-step]
  2. Current ratio: [step-by-step]
  3. Net profit margin: [step-by-step]
  4. ROE: [step-by-step]
  5. Book value: [step-by-step]
  6. P/E: [step-by-step]

Example response:
- **Extracted Data**:
  1. Total debt: 1,000,000 thousands of €
  2. Equity: 500,000 thousands of €
  3. Current assets: 600,000 thousands of €
  4. Current liabilities: 300,000 thousands of €
  5. Net income: 50,000 thousands of €
  6. Total revenue: 800,000 thousands of €
  7. Number of shares: 100,000,000
  8. Share price: 10.00 €
- **Calculations**:
  1. Debt-to-equity ratio: 1,000,000 / 500,000 = 2.00
  2. Current ratio: 600,000 / 300,000 = 2.00
  3. Net profit margin: (50,000 / 800,000) * 100 = 6.25%
  4. ROE: (50,000 / 500,000) * 100 = 10.00%
  5. Book value: 500,000 / 100,000 = 5.00 € per share
  6. P/E: 10.00 / (50,000 / 100,000) = 20.00
"""

print("Analizzo il bilancio...")
result = qa_chain({"query": query})

output = f"# Analisi Value Investing - Brunello Cucinelli S.p.A. ({ticker})\n\n"
output += f"**Prezzo attuale:** €{current_price:.2f}\n\n"
output += "## Risultati\n" + result["result"]

print(output)

os.makedirs(os.path.dirname(output_path), exist_ok=True)
with open(output_path, "w", encoding="utf-8") as f:
    f.write(output)
print(f"Risultati salvati in {output_path}")

print("\nDocumenti rilevanti (tutti i 20 chunk recuperati):")
for i, doc in enumerate(result["source_documents"]):
    page = doc.metadata.get('page', 'N/A')
    print(f"- Chunk {i+1} (Pagina {page}): {doc.page_content[:200]}...")