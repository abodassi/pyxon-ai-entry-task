# 🚀 Quick Start Guide

## Project Overview

You have successfully set up a **production-ready Arabic RAG Document Parser** with:

✅ **Multi-format parsing** (PDF, DOCX, TXT)  
✅ **Advanced Arabic text processing** (RTL, diacritics, normalization)  
✅ **Intelligent chunking** (Fixed, Semantic, Auto-selector)  
✅ **Vector storage** (ChromaDB with multilingual embeddings)  
✅ **Metadata tracking** (SQLite database)  
✅ **Comprehensive benchmarking** (Performance + Quality metrics)

## 📁 Project Structure

```
rag2/
├── core/                       ✅ Core processing modules
│   ├── arabic_processor.py     # Arabic text processing (from extract.py)
│   ├── document_parser.py      # Multi-format parsing
│   ├── chunking_strategy.py    # Intelligent chunking
│   └── embedding_manager.py    # Embedding generation
├── storage/                    ✅ Storage layer
│   ├── vector_store.py         # ChromaDB integration
│   └── metadata_store.py       # SQLite metadata
├── benchmarks/                 ✅ Benchmarking suite
│   └── benchmark_suite.py      # Performance metrics
├── examples/                   ✅ Example scripts
│   ├── example_usage.py        # Basic usage
│   └── run_benchmark.py        # Benchmark comparison
├── data/                       ✅ Your documents
│   ├── file_ar.pdf             # Test PDF
│   └── file.txt                # Ground truth text
├── databases/                  ✅ Auto-created storage
├── output/                     ✅ Benchmark results
├── config.py                   ✅ Configuration
├── main.py                     ✅ Main pipeline
├── setup.py                    ✅ Setup script
├── requirements.txt            ✅ Dependencies
└── README.md                   ✅ Full documentation
```

## 🏃 Next Steps

### 1. Install Dependencies (if not done)

```bash
pip install -r requirements.txt
```

This will install:
- PyMuPDF & pdfplumber (PDF parsing)
- python-docx (DOCX support)
- ChromaDB (vector storage)
- sentence-transformers (embeddings)
- SQLAlchemy (metadata)
- And more...

### 2. Run Basic Example

```bash
py examples\example_usage.py
```

This will:
- Process file_ar.pdf and file.txt
- Create chunks using auto-selected strategy
- Generate embeddings
- Store in ChromaDB + SQLite
- Run sample queries

### 3. Run Comprehensive Benchmark

```bash
py examples\run_benchmark.py
```

This will:
- Test all chunking strategies
- Measure processing time & memory
- Evaluate retrieval accuracy
- Compare PDF extraction vs ground truth
- Save detailed results to output/

## 💡 Quick Usage Examples

### Process a Single Document

```python
from main import ArabicRAGPipeline

pipeline = ArabicRAGPipeline(chunking_strategy='auto')
result = pipeline.process_document('data/file_ar.pdf')
print(f"Created {result['num_chunks']} chunks")
```

### Query the System

```python
results = pipeline.query("ما هي خدمات إعادة التدوير؟", n_results=5)

for doc in results['results']['documents'][0]:
    print(doc[:200])
```

### Run Benchmark

```python
benchmark = pipeline.run_benchmark(
    file_paths=['data/file_ar.pdf', 'data/file.txt'],
    test_queries=["ما هي الخدمات المتوفرة؟"],
    ground_truth_pdf='data/file_ar.pdf',
    ground_truth_txt='data/file.txt'
)
```

## 🔧 Configuration

Edit `config.py` to customize:

- **Chunking sizes**: `FIXED_CHUNK_SIZE = 512`
- **Overlap**: `FIXED_CHUNK_OVERLAP = 128`
- **Embedding model**: `EMBEDDING_MODEL = "paraphrase-multilingual-MiniLM-L12-v2"`
- **Database paths**: `CHROMA_DB_PATH`, `SQLITE_DB_PATH`

## 📊 Key Features Implemented

### From extract.py Integration
✅ RTL text correction  
✅ Common PDF error fixes (اال → ال, األ → أل, etc.)  
✅ Diacritics preservation  
✅ Arabic entity extraction  

### Intelligent Chunking
✅ Fixed-size with overlap  
✅ Semantic (structure-based)  
✅ Auto-selector (analyzes document)  

### Storage
✅ ChromaDB for vector embeddings  
✅ SQLite for metadata tracking  
✅ Dual-version text (retrieval + search)  

### Benchmarking
✅ Processing time & memory  
✅ Retrieval accuracy (Hit Rate, MRR)  
✅ Arabic extraction quality (F1 score)  

## 🎯 Testing Your Documents

1. **Add your documents** to `data/` directory
2. **Update file paths** in examples or use:

```python
pipeline = ArabicRAGPipeline()
pipeline.process_batch([
    'data/your_document.pdf',
    'data/another_document.docx'
])
```

3. **Query your content**:

```python
results = pipeline.query("your Arabic query here")
```

## 📈 Viewing Results

### Check Statistics

```python
stats = pipeline.get_stats()
print(f"Documents: {stats['metadata_store']['total_documents']}")
print(f"Chunks: {stats['metadata_store']['total_chunks']}")
```

### View Benchmark Results

Results are saved to:
- `output/[benchmark_name].json` - Detailed JSON results
- `databases/metadata.db` - SQLite database (queryable)

## 🛠️ Advanced Usage

### Custom Chunking Strategy

```python
from core.chunking_strategy import SemanticChunker

chunker = SemanticChunker(
    min_chunk_size=300,
    max_chunk_size=800,
    use_embeddings=True  # Similarity-based
)

pipeline = ArabicRAGPipeline(chunking_strategy=chunker)
```

### Filter Queries by Metadata

```python
results = pipeline.query(
    "query text",
    filter_metadata={'is_arabic': True, 'file_type': 'pdf'}
)
```

## 🧹 Clean Architecture Notes

The logic from `extract.py` has been **successfully integrated** into:

- **`core/arabic_processor.py`**: 
  - `fix_rtl_extraction()` - Line 39
  - `fix_common_pdf_errors()` - Line 73
  - `normalize_for_search()` - Line 109
  - `extract_arabic_entities()` - Line 155

The original `extract.py` can now be **safely deleted** as all functionality is preserved and enhanced in the modular architecture.

## 📞 Need Help?

1. Check `README.md` for full documentation
2. Review examples in `examples/`
3. Examine config in `config.py`
4. Check logs for detailed error messages

## 🎉 You're Ready!

The system is fully set up and ready to process Arabic documents for RAG applications. Start with the examples and customize as needed!

---

**Built with ❤️ for high-performance Arabic NLP**
