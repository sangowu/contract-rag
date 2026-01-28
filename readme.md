# Contract Intelligence Analysis System (First draft - Under development..)

AI-powered contract analysis platform based on Retrieval-Augmented Generation (RAG), using SentenceTransformer + ChromaDB + Qwen3-8B tech stack.

## Core Features

- **Efficient Vector Retrieval**: SentenceTransformer embeddings + ChromaDB storage
- **Intelligent Q&A**: Qwen3-8B large language model for answer generation
- **Complete Evaluation System**: Retrieval performance + end-to-end generation quality assessment
- **Visualization Reports**: Automatically generate performance charts and analysis reports

## Tech Stack

- **Embedding Model**: SentenceTransformer (all-MiniLM-L6-v2)
- **Vector Database**: ChromaDB
- **Large Language Model**: Qwen3-8B (transformers)
- **Data Processing**: pandas, numpy
- **Visualization**: matplotlib
- **Evaluation Tools**: scikit-learn
- **Logging**: loguru

## Project Structure

```
├── config/          # Configuration files
├── data/            # Datasets and vector storage
├── model/           # Local model cache
├── results/         # Evaluation results and charts
├── scripts/         # Evaluation scripts
├── src/             # Core code
│   ├── rag/         # Retrieval module
│   ├── inference/   # Model inference
│   └── utils/       # Utility functions
└── test.py          # Result viewer
```

## Quick Start

> **Note**: This repository only contains code and scripts, **does not include datasets and pre-trained models**. You need to download separately:
> - CUAD dataset (for training and evaluation)
> - Qwen3-8B model files
> - SentenceTransformer embedding models

### Environment Requirements
- Python 3.8+
- GPU memory 16GB+ (Qwen3-8B)
- Storage space 50GB+

### Installation & Running

```bash
# Install dependencies
pip install -r requirements.txt

# Build vector index
python src/rag/embedding.py

# Run evaluation
python scripts/evaluate_script.py

# View results
./results/plots/vanilla_e2e
```

## Evaluation Results

Performance on CUAD dataset (End-to-End Evaluation):

### Overall E2E Metrics
- **Hit@k**: 0.3088
- **MRR@k**: 0.1680  
- **Recall@k**: 0.3088
- **F1 Score (mean)**: 0.6377
- **Exact Match (mean)**: 0.1832
- **Accuracy (mean)**: 0.8738

### Performance by Answer Type

| Answer Type | Samples | Hit@k | MRR@k | Recall@k | F1 | Exact Match | Accuracy |
|-------------|---------|-------|-------|----------|----|-------------|----------|
| BOOL | 16,320 | 0.2273 | 0.1131 | 0.2273 | 0.0000 | 0.0000 | 0.8738 |
| DATE | 1,153 | 0.6756 | 0.3973 | 0.6756 | 0.0000 | 0.0009 | 0.0000 |
| DURATION | 673 | 0.3566 | 0.2407 | 0.3566 | 0.0000 | 0.0416 | 0.0000 |
| LIST_ENTITY | 509 | 0.7917 | 0.2554 | 0.7917 | 0.7483 | 0.0000 | 0.0000 |
| LOCATION | 434 | 0.9954 | 0.9116 | 0.9954 | 0.0000 | 0.8871 | 0.0000 |
| TEXT | 510 | 0.9588 | 0.5892 | 0.9588 | 0.5274 | 0.0000 | 0.0000 |

## Configuration

The project uses a hierarchical configuration system:

- **`config/base.yaml`**: Base configuration shared across all environments
- **`config/{mode}.yaml`**: Environment-specific overrides (test/dev/prod)

Edit these files to adjust:
- Model parameters (LLM, Embedding, Reranker)
- Retrieval settings (Hybrid search, BM25, Vector DB)
- PDF parsing configuration
- Evaluation metrics
- API and logging settings

**Usage**:
```bash
# Test mode (loads base.yaml + test.yaml)
python scripts/run_api.py --mode test

# Dev mode (loads base.yaml + dev.yaml)
python scripts/run_api.py --mode dev

# Prod mode (loads base.yaml + prod.yaml)
python scripts/run_api.py --mode prod
```

## Usage Examples

```python
from src.rag.embedding import retrieve_top_k
from src.inference.llm_inference import llm_generate

# Retrieve relevant clauses
query = "Confidentiality clause content"
docs = retrieve_top_k(query, k=3)

# Generate answer
context = "\n".join([d['clause_text'] for d in docs])
answer = llm_generate(f"Question: {query}\nClauses: {context}")
```

## Contributing

Welcome to submit Issues and Pull Requests!

## Acknowledgments

- **Qwen Model**: Thanks to [Alibaba Cloud](https://github.com/QwenLM) for providing the open-source large language model
- **CUAD Dataset**: Thanks to [The Atticus Project](https://www.atticusprojectai.org/) for providing the contract understanding dataset

## License

MIT License
