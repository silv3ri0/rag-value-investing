# RAG Value Investing

Un sistema RAG (Retrieval-Augmented Generation) per analizzare bilanci societari in formato PDF dal punto di vista del *value investing*, utilizzando LangChain e i modelli di OpenAI. Il progetto estrae dati da un PDF, li indicizza con FAISS e genera un’analisi finanziaria basata su metriche chiave come solidità finanziaria, redditività e sottovalutazione/sopravvalutazione.

Testato con successo su un PDF di circa 100 pagine.

## Requisiti

- **Python**: 3.10 o superiore.
- **Chiave API OpenAI**: Necessaria per embedding e generazione delle risposte.
- **File PDF**: Un bilancio societario in formato PDF (non scansionato, per garantire la lettura del testo).

## Installazione

Segui questi passi per configurare il progetto localmente:

1. **Clona il repository**:
   ```bash
   git clone https://github.com/silv3ri0/rag-value-investing.git
   cd rag-value-investing
   
## Crea un ambiente virtuale
python -m venv .venv
source .venv/bin/activate  # Su Windows: .venv\Scripts\activate

## Installa le dipendenze
pip install -r requirements.txt

Le librerie richieste sono elencate in requirements.txt:
* langchain
* langchain-community
* langchain-openai
* openai
* pypdf
* faiss-cpu
* python-dotenv
* tiktoken

## Configura la chiave API OpenAI
echo OPENAI_API_KEY=la_tua_chiave

## Aggiungi il PDF
* Posiziona il tuo bilancio societario in PDF nella cartella data/.
* Rinominalo bilancio_societa_big.pdf o modifica il percorso nel file src/rag_analysis.py (riga con pdf_path) per corrispondere al nome del tuo file.

## Esegui il programma
python src/rag_analysis.py

## Cosa fa?
* Carica il PDF e lo suddivide in chunk di testo.
* Crea un indice vettoriale con FAISS per la ricerca semantica.
* Usa il modello gpt-4-turbo di OpenAI per analizzare il bilancio e rispondere a una query predefinita sul value investing.
* Stampa i risultati, inclusi i chunk utilizzati per l’analisi.

## Output atteso:
* Un’analisi testuale con stime numeriche (se i dati sono presenti nel PDF) su:
* Solidità finanziaria (es. rapporto debito/capitale, liquidità).
* Redditività (es. margine di profitto, ROE).
* Sottovalutazione/sopravvalutazione.
* Viene salvata nella cartella output una versione per ogni run di un output con un analisi finanziaria dopo l'analisi del bilancio

## Spiegazione file:
* rag_analysis.py - usa chatgpt4
* rag_analysis_deepseekR1.py - usa deepseekR1 passando da openrouter
* rag_analysis_deepseekR1-noOpenRouter.py - usa deepseekR1 passando direttamente da loro
* rag_analysis_localModel.py - tentativo fallito perchè il mio notebook con 16Giga di Ram non ha retto Mistral in locale













