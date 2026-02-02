
## Summary
A production-ready **Arabic RAG Document Parser** designed to intelligently process, chunk, and index Arabic documents (PDF, DOCX, TXT) for retrieval-augmented generation. The system features advanced RTL text correction, diacritics preservation, and a dual-storage architecture (Vector + SQL) to ensure high-accuracy retrieval for Arabic content.

Key highlights:
- **Native Arabic Support:** specialized processing for RTL correction, diacritics, and normalization.
- **Intelligent Chunking:** Auto-selection between Fixed, Semantic, and structure-based strategies.
- **Dual-Engine PDF Parsing:** Combines `PyMuPDF` and `pdfplumber` for maximum extraction fidelity.
- **Production Architecture:** Modular design with `core/`, `storage/`, and `benchmarks/` packages.

## Contact Information
Name : abelrahman abuassi
phone : 0781277516
Email: abu2002assi@gmail.com 


## Features Implemented
- [x] Document parsing (PDF, DOCX, TXT) with encoding detection
- [x] Content analysis and chunking strategy selection (Auto-selector)
- [x] Fixed and dynamic (semantic) chunking strategies
- [x] Vector DB integration (ChromaDB with multilingual embeddings)
- [x] SQL DB integration (SQLite for granular metadata)
- [x] Arabic language support (RTL visual->logical correction)
- [x] Arabic diacritics support (Dual-version storage: retrieval vs search)
- [x] Benchmark suite (Performance & Quality metrics)
- [x] RAG integration ready (LangChain-compatible pipeline)

## Architecture
The system follows a modular "Clean Architecture" pattern:

```
rag2/
├── core/                       # Core Logic
│   ├── arabic_processor.py     # Arabic normalization, RTL fix, entities
│   ├── document_parser.py      # Multi-format extractors (Factory pattern)
│   ├── chunking_strategy.py    # Strategy pattern for chunking
│   └── embedding_manager.py    # Embedding generation wrapper
├── storage/                    # Data Persistence
│   ├── vector_store.py         # ChromaDB interface
│   └── metadata_store.py       # SQL Alchemy / SQLite interface
├── benchmarks/                 # Quality Assurance
│   └── benchmark_suite.py      # Precision/Recall & Latency tests
└── app.py                      # Streamlit User Interface
```

**Key Components:**
1.  **ArabicTextProcessor:** Centralizes all Arabic-specific logic (regex for cleaning, bidi algorithms, diacritic handling).
2.  **AutoChunker:** Analyzes document structure (header density, paragraph length) to automatically decide if "Semantic" or "Fixed" chunking is better.
3.  **Dual Storage:**
    *   **Vector Store:** Stores embeddings of text chunks for semantic search.
    *   **Metadata Store:** Relational DB for tracking file status, chunk indices, and benchmark runs.

## Technologies Used
- **Core:** Python 3.10+
- **Parsing:** `PyMuPDF` (fitz), `pdfplumber`, `python-docx`
- **NLP & Arabic:** `arabic-reshaper`, `python-bidi`, `regex`
- **Embeddings:** `sentence-transformers` (Model: `paraphrase-multilingual-MiniLM-L12-v2`)
- **Storage:** `ChromaDB` (Vector), `SQLAlchemy` (Metadata)
- **UI:** `Streamlit`, `Plotly`
- **Testing:** `pytest`, `psutil`

## Benchmark Results
The system includes a suite (`examples/run_benchmark.py`) that evaluates both performance and accuracy.

**Typical Performance (tested on standard hardware):**
*   **PDF Parsing Speed:** ~1.5s per page (using dual-engine approach)
*   **Chunking Latency:** <100ms for 10k words
*   **Embedding Generation:** ~50ms per chunk

**Quality Metrics:**
*   **RTL Correction Accuracy:** Near 100% on standard PDFs (vs Ground Truth)
*   **Diacritic Preservation:** 100% (Text is stored with diacritics for display/LLM context, normalized for search)

*Detailed benchmark results are generated in the `output/` directory upon running the suite.*

## How to Run

### 1. Setup Environment
```bash
git clone <repo_url>
cd rag2
python -m venv venv
# Activate venv (Windows: .\venv\Scripts\activate, Mac/Linux: source venv/bin/activate)
pip install -r requirements.txt
```

### 2. Run the UI (Demo)
```bash
streamlit run app.py
```

### 3. Run via CLI
```python
from main import ArabicRAGPipeline
pipeline = ArabicRAGPipeline()
pipeline.process_document('data/file_ar.pdf')
print(pipeline.query("ما هي خدمات إعادة التدوير؟"))
```

### 4. Run Benchmarks
```bash
python examples/run_benchmark.py
```

## Questions & Assumptions
- **Question 1:** Should we favor speed or accuracy for determining chunk boundaries?
    - **Assumption:** Favored accuracy. The `AutoChunker` analyzes the document first (small latency cost) to pick the best strategy, ensuring coherent chunks over raw speed.
- **Question 2:** How to handle mixed English/Arabic documents?
    - **Assumption:** Used a multilingual embedding model (`paraphrase-multilingual-MiniLM-L12-v2`) and ensured the `ArabicTextProcessor` only targets clear Arabic unicode ranges, leaving English text intact.

## Future Improvements
- [ ] **OCR Integration:** Add Tesseract support for scanned Arabic PDFs.
- [ ] **Advanced GraphRAG:** Implement knowledge graph extraction for entity relationships (e.g., Person <-> Organization).
- [ ] **Fine-tuned Embeddings:** Train a custom embedding model specifically on the target domain corpus.
