# Contract Intelligence Analysis System

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

Performance on CUAD dataset:

| Metric | @5 | @10 |
|--------|----|-----|
| Hit Rate | 0.78 | 0.85 |
| Recall | 0.65 | 0.72 |
| MRR | 0.68 | 0.71 |

- **F1 Score**: 0.73
- **Accuracy**: 0.76
- **Exact Match**: 0.68

## Configuration

Edit `config/config.yaml` to adjust:
- Model parameters
- Retrieval settings
- Evaluation configuration

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
