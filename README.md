# FinBot

A privacy-focused, locally-hosted financial assistant for Canadian financial literacy. FinBot combines retrieval-augmented generation (RAG) with local language models to provide accurate answers about TFSA, CRA regulations, provincial benefits, and other Canadian financial topics.

## Features

- **🏛️ Government Document Processing**: Automated ingestion and processing of PDF documents
- **🧠 Semantic Search**: Advanced similarity search using sentence transformers and vector databases
- **🤖 Multiple LLM Backends**: Support for local models (llama.cpp), OpenAI API, and HuggingFace Hub
- **� Privacy-First**: Complete local operation with no data sent to external services
- **⚡ High Performance**: Optimized for fast response times with streaming output
- **📝 Source Attribution**: All answers include source document references for verification

## Architecture

FinBot implements a modern RAG (Retrieval-Augmented Generation) architecture:

1. **Document Ingestion**: PDF/HTML documents are processed and chunked
2. **Embedding Generation**: Text chunks are converted to vector embeddings
3. **Vector Storage**: Embeddings stored in PostgreSQL with pgvector extension
4. **Semantic Retrieval**: User queries matched against document chunks
5. **Response Generation**: LLM generates contextual answers from retrieved content

## Technology Stack

- **Language**: Python 3.8+
- **Framework**: PyTorch 2.0+ (used by sentence-transformers and transformers)
- **Vector Database**: PostgreSQL with pgvector extension
- **Embeddings**: sentence-transformers (all-MiniLM-L6-v2)
- **LLM Backends**: llama.cpp, OpenAI API, HuggingFace Transformers
- **Document Processing**: pdfplumber, BeautifulSoup4
- **CLI Interface**: Rich library for enhanced terminal experience

## Prerequisites

- Python 3.8 or higher
- PostgreSQL with pgvector extension
- 4GB+ RAM (for local LLM inference)
- Optional: GGUF format language model file

## Installation

### 1. Clone Repository
```bash
git clone https://github.com/your-username/finbot.git
cd finbot
```

### 2. Create Virtual Environment
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Configure Environment
```bash
cp .env.example .env
# Edit .env with your configuration
```

### 5. Setup Database
```bash
# Install PostgreSQL and pgvector extension
# Ubuntu/Debian:
sudo apt-get install postgresql postgresql-contrib postgresql-15-pgvector

# macOS:
brew install postgresql pgvector

# Initialize database
python scripts/init_db.py
```

### 6. Validate Setup
```bash
python setup.py
```

## Configuration

### Environment Variables

Key configuration options in `.env`:

```bash
# Database Configuration
DATABASE_URL=postgresql:///finbot

# LLM Backend Selection
LLM_BACKEND=llama_cpp  # Options: llama_cpp, openai, hf_hub

# Local Model Configuration (llama_cpp)
LLAMA_PATH=/path/to/your/model.gguf

# OpenAI Configuration (alternative)
# LLM_BACKEND=openai
# OPENAI_API_KEY=your-api-key-here

# Processing Parameters
CHUNK_SIZE=200
CHUNK_OVERLAP=50
TOP_K=4
MAX_RESPONSE_TOKENS=128
LLM_TEMPERATURE=0.3
```

### LLM Backend Options

#### Local Models (Recommended for Privacy)
```bash
LLM_BACKEND=llama_cpp
LLAMA_PATH=/path/to/model.gguf
```

Download GGUF models from HuggingFace:
- [Llama-3-8B-Instruct-Finance-RAG-GGUF](https://huggingface.co/QuantFactory/Llama-3-8B-Instruct-Finance-RAG-GGUF)
\
#### OpenAI API
```bash
LLM_BACKEND=openai
OPENAI_API_KEY=sk-your-api-key-here
OPENAI_MODEL=gpt-4o-mini
```

#### HuggingFace Hub
```bash
LLM_BACKEND=hf_hub
HF_MODEL=microsoft/DialoGPT-large
```

## Usage

### 1. Document Ingestion

Place your PDF or HTML documents in the `data/raw/` directory:

```bash
mkdir -p data/raw
# Copy government documents, financial guides, etc.
```

Ingest documents into the system:

```bash
python -m finbot.cli --ingest
```

### 2. Interactive Mode

Start the interactive financial assistant:

```bash
python -m finbot.cli
```

### 3. Example Session

```
FinBot ready › Ask questions about Canadian finance

Query (or 'exit'): What is a TFSA contribution limit for 2024?

Answer:
The TFSA contribution limit for 2024 is $7,000. This is in addition to any unused contribution room from previous years that carries forward.

Source: data/raw/tax-free-savings-account-guide-rc4466.pdf
--------------------------------------------------

## Evaluation and Metrics

- **Manual Retrieval Accuracy**: 91% across 30+ documents (CRA, TFSA, Ontario Benefits guides)
- **Retrieval+Rerank Latency**: <400 ms for most queries (displayed in interactive mode)
- **Generation Latency**: <200 ms average per response (displayed in interactive mode)

Query (or 'exit'): How does TFSA contribution room work?

Answer:
TFSA contribution room accumulates each year and unused room carries forward indefinitely. If you withdraw money from your TFSA, that amount is added back to your contribution room at the beginning of the following year.

Source: data/raw/tax-free-savings-account-guide-rc4466.pdf
--------------------------------------------------
```

## Project Structure

```
finbot/
├── finbot/                  # Main application package
│   ├── cli.py              # Command-line interface
│   ├── config.py           # Configuration management
│   ├── db/                 # Database operations
│   │   └── client.py       # PostgreSQL client with pgvector
│   ├── embedding/          # Text embedding generation
│   │   └── embedder.py     # Sentence transformer interface
│   ├── ingestion/          # Document processing pipeline
│   │   ├── ingest.py       # Main ingestion orchestrator
│   │   ├── loader.py       # PDF/HTML document loaders
│   │   └── chunker.py      # Text chunking utilities
│   ├── llm/                # Language model backends
│   │   ├── __init__.py     # LLM factory
│   │   ├── base.py         # Abstract base class
│   │   ├── llama_cpp.py    # Local model inference
│   │   ├── openai.py       # OpenAI API integration
│   │   └── hf_hub.py       # HuggingFace Hub models
│   ├── prompt/             # Prompt engineering
│   │   └── formatter.py    # Prompt template formatting
│   └── retriever/          # Semantic search
│       └── similarity.py   # Vector similarity search
├── scripts/                # Utility scripts
│   └── init_db.py         # Database initialization
├── data/                  # Data directory
│   └── raw/               # Source documents
├── setup.py               # Environment validation
├── requirements.txt       # Python dependencies
├── .env.example          # Configuration template
└── README.md             # This file
```

## Performance Optimization

### For Better Speed
- Use smaller models (7B parameters or less)
- Reduce `MAX_RESPONSE_TOKENS` in configuration
- Lower `LLM_TEMPERATURE` for more focused responses
- Limit `TOP_K` to 3-4 for faster retrieval

### For Better Quality
- Use larger, instruction-tuned models
- Increase `CHUNK_SIZE` for more context
- Adjust `CHUNK_OVERLAP` for better context continuity
- Fine-tune `LLM_TEMPERATURE` for response creativity

## Troubleshooting

### Common Issues

**Database Connection Error**
```bash
# Ensure PostgreSQL is running
sudo systemctl start postgresql  # Linux
brew services start postgresql   # macOS

# Verify pgvector extension
psql -d finbot -c "CREATE EXTENSION IF NOT EXISTS vector;"
```

**Model Loading Issues**
```bash
# Check model file exists and has correct permissions
ls -la /path/to/your/model.gguf

# Verify model format (should be GGUF)
file /path/to/your/model.gguf
```

**Memory Issues**
```bash
# Reduce context window in config
# Or switch to smaller model
# Or use OpenAI API instead of local inference
```

### Logging and Debugging

Enable debug mode by setting:
```bash
export SKIP_CONFIG_VALIDATION=1
python -c "from finbot.config import validate_config; validate_config()"
```

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

### Development Setup

```bash
# Install development dependencies
pip install -r requirements.txt

# Run validation
python setup.py

# Test ingestion
python -m finbot.cli --ingest

# Test interactive mode
python -m finbot.cli
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

<!-- ## Acknowledgments

- Government of Canada for providing open financial education resources
- HuggingFace for transformer models and sentence-transformers
- PostgreSQL and pgvector teams for vector database capabilities
- llama.cpp project for efficient local LLM inference -->
<!--  -->
## Support

For questions, issues, or contributions:
- Open an issue on GitHub
- Check existing documentation and troubleshooting guides
- Review configuration examples in `.env.example`