# Arabic RAG Document Parser

A high-performance, production-ready RAG (Retrieval-Augmented Generation) document parser optimized for Arabic text with diacritics (Harakat) and RTL (Right-to-Left) formatting support.

## 🌟 Features

### 📄 Multi-Format Document Parsing
- **PDF**: Dual-engine support (PyMuPDF & pdfplumber) for robust Arabic extraction
- **DOCX**: Full support for Word documents with tables
- **TXT**: Smart encoding detection (UTF-8, UTF-16, CP1256, ISO-8859-6)

### 🔤 Advanced Arabic Text Processing
- **RTL Correction**: Fixes reversed text from PDF extraction
- **Diacritics Preservation**: Maintains Harakat in original text
- **Dual-Version Creation**: 
  - Retrieval-friendly version (with diacritics)
  - Search-friendly version (normalized for better matching)
- **PDF Error Correction**: Fixes common Arabic PDF extraction issues
- **Named Entity Recognition**: Extracts Arabic organizations, locations, etc.

### ✂️ Intelligent Chunking Strategies
1. **Fixed-Size Chunking**: Standard slicing with configurable overlap
2. **Semantic Chunking**: Structure-aware segmentation based on:
   - Headers and sections
   - Paragraph boundaries
   - Semantic similarity (embedding-based)
3. **Auto-Selector**: Analyzes document structure to choose optimal strategy

### 💾 Dual Storage Architecture
- **Vector Database (ChromaDB)**: Stores embeddings for similarity search
  - Uses `paraphrase-multilingual-MiniLM-L12-v2` model
  - Optimized for Arabic and multilingual content
- **Metadata Database (SQLite)**: Tracks processing history
  - Document metadata (file info, processing timestamps)
  - Chunk metadata (positions, strategy used)
  - Benchmark results

### 📊 Comprehensive Benchmarking Suite
- **Performance Metrics**:
  - Processing time (parsing, chunking, embedding)
  - Memory usage (peak and average)
- **Quality Metrics**:
  - Retrieval accuracy (Hit Rate, MRR)
  - Arabic extraction accuracy (vs ground truth)
  - Word-level precision/recall/F1

## 🏗️ Architecture

```
rag2/
├── core/                      # Core processing modules
│   ├── arabic_processor.py    # Arabic text processing & normalization
│   ├── document_parser.py     # Multi-format document parsing
│   ├── chunking_strategy.py   # Intelligent chunking strategies
│   └── embedding_manager.py   # Embedding generation
├── storage/                   # Storage layer
│   ├── vector_store.py        # ChromaDB integration
│   └── metadata_store.py      # SQLite metadata tracking
├── benchmarks/                # Benchmarking suite
│   └── benchmark_suite.py     # Performance & quality metrics
├── examples/                  # Example scripts
│   ├── example_usage.py       # Basic usage example
│   └── run_benchmark.py       # Benchmark comparison
├── databases/                 # Database storage (auto-created)
├── data/                      # Input documents (auto-created)
├── output/                    # Benchmark results (auto-created)
├── config.py                  # Configuration settings
├── main.py                    # Main RAG pipeline
└── requirements.txt           # Dependencies
```

## 🚀 Quick Start

### Installation

1. **Clone or navigate to the project directory**

2. **Create and activate virtual environment**:
```bash
python -m venv venv
.\venv\Scripts\activate  # Windows
# source venv/bin/activate  # Linux/Mac
```

3. **Install dependencies**:
```bash
pip install -r requirements.txt
```

### Basic Usage

```python
from main import ArabicRAGPipeline

# Initialize pipeline
pipeline = ArabicRAGPipeline(
    chunking_strategy='auto',  # or 'fixed', 'semantic'
    device='cpu'               # or 'cuda' for GPU
)

# Process documents
results = pipeline.process_batch([
    'file_ar.pdf',
    'file.txt'
])

# Query the system
query_results = pipeline.query(
    "ما هي خدمات إعادة التدوير؟",
    n_results=5
)

# Display results
for doc in query_results['results']['documents'][0]:
    print(doc)
```

### 🖥️ Streamlit UI (Web Interface)

The project includes a full-featured web interface:

```bash
streamlit run app.py
```

This opens a dashboard where you can:
- Upload and process documents (PDF, DOCX, TXT)
- Choose chunking strategies visually
- Chat with your documents using the **Multi-Agent Orchestrator**
- View retrieval results and processing statistics

### 🤖 Multi-Agent Orchestration

The system features a Multi-Agent architecture combining **RAG** (Retrieval) and **Gemini** (Generation):

```python
from multi_agent import MultiAgentOrchestrator

# Initialize agents
orchestrator = MultiAgentOrchestrator()

# Ask questions related to your docs
result = orchestrator.ask("ما هي خدمات إعادة التدوير؟")
print(result['answer'])
```

**Setup for Multi-Agent:**
1. Ensure `GEMINI_API_KEY` is set in `.env`
2. Run the interactive chat: `python multi_agent.py`

### Run Examples

**Basic example**:
```bash
python examples/example_usage.py
```

**Benchmark comparison**:
```bash
python examples/run_benchmark.py
```

## 🔧 Configuration

Edit `config.py` to customize:

### Chunking Configuration
```python
# Fixed-size chunking
FIXED_CHUNK_SIZE = 512
FIXED_CHUNK_OVERLAP = 128

# Semantic chunking
SEMANTIC_MIN_CHUNK_SIZE = 200
SEMANTIC_MAX_CHUNK_SIZE = 1000
```

### Arabic Processing
```python
# Normalization options
NORMALIZE_ALEF = True
NORMALIZE_HAMZA = True
REMOVE_TATWEEL = True
```

### Embedding Model
```python
EMBEDDING_MODEL = "paraphrase-multilingual-MiniLM-L12-v2"
```

## 📊 Benchmarking

The system includes a comprehensive benchmarking suite:

```python
results = pipeline.run_benchmark(
    file_paths=['file_ar.pdf', 'file.txt'],
    test_queries=[
        "ما هي الخدمات المتوفرة؟",
        "معلومات عن إعادة التدوير",
    ],
    ground_truth_pdf='file_ar.pdf',
    ground_truth_txt='file.txt',
    run_name='my_benchmark'
)
```

**Metrics tracked**:
- Processing time (total and per-chunk)
- Memory usage (peak and average)
- Retrieval accuracy (hit rate, MRR)
- Arabic extraction quality (precision, recall, F1)

Results are saved to:
- `databases/metadata.db` (SQL database)
- `output/[run_name].json` (JSON file)

## 🎯 Key Components

### ArabicTextProcessor
Handles Arabic-specific text processing:
- RTL text correction
- Diacritics preservation
- PDF extraction error fixes
- Dual-version creation (retrieval + search)
- Named entity extraction

### DocumentParser
Multi-format parsing with Arabic support:
- Automatic format detection
- Encoding-aware text extraction
- Table extraction (DOCX)
- Fallback mechanisms

### Chunking Strategies
Three intelligent strategies:
1. **FixedSizeChunker**: Reliable baseline with overlap
2. **SemanticChunker**: Structure-aware or similarity-based
3. **AutoChunker**: Analyzes document to select best strategy

### Storage Layer
- **VectorStore**: ChromaDB for embedding storage
- **MetadataStore**: SQLite for metadata tracking

## 🔬 Arabic Text Processing Logic

The system incorporates advanced logic for Arabic text (adapted from `extract.py`):

### RTL Correction
```python
# Fixes visual RTL order to logical order
# Example: "ةملك" → "كلمة"
processor.fix_rtl_extraction(text)
```

### Common PDF Fixes
```python
# Fixes extraction issues like:
# "اال" → "ال"
# "األ" → "أل"
# "الإعادة" → "لإعادة"
processor.fix_common_pdf_errors(text)
```

### Normalization
```python
# Create search-friendly version
# Removes diacritics, normalizes Alef/Hamza variants
processor.normalize_for_search(text)
```

## 📈 Performance Considerations

### Memory Optimization
- Batch processing with configurable batch size
- Memory monitoring during benchmarking
- Efficient embedding generation

### Processing Speed
- Parallel processing support (configurable workers)
- Optimized chunking algorithms
- Fast vector similarity search

### Quality Assurance
- Dual-version text storage
- Multiple chunking strategies
- Comprehensive benchmarking

## 🧪 Testing with Ground Truth

The system supports testing against ground truth:

```python
# Compare PDF extraction against known-good text file
pipeline.benchmark_suite.benchmark_arabic_extraction(
    parser=pipeline.parser,
    pdf_file='file_ar.pdf',
    ground_truth_file='file.txt'
)
```

Metrics:
- Character-level accuracy
- Word-level precision/recall/F1
- Content preservation

## 🛠️ Advanced Usage

### Custom Chunking Strategy

```python
from core.chunking_strategy import SemanticChunker

# Create custom semantic chunker
chunker = SemanticChunker(
    min_chunk_size=300,
    max_chunk_size=800,
    use_embeddings=True  # Enable similarity-based chunking
)

pipeline = ArabicRAGPipeline(chunking_strategy=chunker)
```

### Filtering Queries

```python
# Query with metadata filter
results = pipeline.query(
    "Arabic query",
    n_results=5,
    filter_metadata={'is_arabic': True}
)
```

### Batch Processing with Custom Metadata

```python
results = pipeline.process_batch(
    file_paths=['doc1.pdf', 'doc2.docx'],
    custom_metadata={'source': 'research_papers', 'year': 2024}
)
```

## 📋 Requirements

- Python 3.8+
- PyMuPDF (fitz)
- pdfplumber
- python-docx
- ChromaDB
- sentence-transformers
- SQLAlchemy
- pandas, numpy
- psutil (for benchmarking)

See `requirements.txt` for complete list.

## 🤝 Contributing

This is a production-ready system designed for Arabic RAG applications. Key areas for contribution:
- Additional document formats
- More advanced NER for Arabic
- Fine-tuned embedding models
- Advanced semantic chunking

## 📝 License

MIT License

## 🙏 Acknowledgments

- Arabic text processing logic adapted and enhanced from specialized extraction utilities
- Multilingual embedding model from sentence-transformers
- Vector database powered by ChromaDB

## 📞 Support

For issues or questions:
1. Check the examples in `examples/`
2. Review configuration in `config.py`
3. Examine logs for detailed error messages

---

**Built with ❤️ for Arabic NLP and RAG applications**
