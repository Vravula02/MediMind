# MediMind

### Biomedical Research Assistant using Retrieval-Augmented Generation, Semantic Search, and Knowledge Graphs

MediMind is an AI-powered biomedical research assistant that helps users explore biomedical literature through natural-language queries. It combines domain-specific embeddings, vector retrieval, cross-encoder reranking, knowledge-graph traversal, and large language models to retrieve relevant research context before generating responses.

---

## Overview

Traditional keyword-based search can struggle to capture the semantic relationships between biomedical concepts. MediMind uses a multi-stage retrieval pipeline to identify relevant research and provide contextually grounded answers.

| Capability            | Description                                                      |
| --------------------- | ---------------------------------------------------------------- |
| Semantic Search       | Retrieves literature based on meaning rather than exact keywords |
| Biomedical Embeddings | Uses a domain-specific embedding model for biomedical text       |
| Vector Search         | Uses FAISS for efficient similarity-based retrieval              |
| Reranking             | Refines retrieved results using a cross-encoder                  |
| Knowledge Graph       | Connects related biomedical entities and documents               |
| RAG                   | Grounds LLM responses using retrieved research context           |
| Context Validation    | Checks whether retrieved evidence is sufficient                  |
| Interactive UI        | Provides research exploration through Streamlit                  |

---

## System Architecture

The overall retrieval and generation flow is:

| Stage | Component             | Purpose                                      |
| ----- | --------------------- | -------------------------------------------- |
| 1     | Biomedical Literature | Source research documents and metadata       |
| 2     | Document Processing   | Clean, validate, and chunk documents         |
| 3     | MedEmbed              | Generate biomedical semantic embeddings      |
| 4     | FAISS                 | Retrieve relevant document chunks            |
| 5     | Cross-Encoder         | Rerank retrieved candidates                  |
| 6     | Knowledge Graph       | Explore related entities and documents       |
| 7     | RAG Pipeline          | Combine retrieved evidence and graph context |
| 8     | LLM                   | Generate the final research response         |
| 9     | Streamlit             | Provide an interactive user interface        |

### Retrieval Flow

```text
Biomedical Literature
        │
        ▼
Document Processing
        │
        ▼
Biomedical Embeddings
        │
        ▼
FAISS Vector Search
        │
        ▼
Cross-Encoder Reranking
        │
        ├──────────────► Knowledge Graph
        │                     │
        │                     ▼
        └──────────────► Relevant Context
                              │
                              ▼
                         RAG Pipeline
                              │
                              ▼
                             LLM
                              │
                              ▼
                        Streamlit UI
```

---

## How It Works

### 1. Document Processing

Biomedical research data is cleaned and divided into smaller chunks while preserving important publication metadata.

The processed documents retain information such as:

* PMID
* Title
* Authors
* Journal
* Publication date
* DOI
* PMC ID
* MeSH terms
* Publication types

The current document splitter uses:

| Parameter            |          Value |
| -------------------- | -------------: |
| Chunk size           | 300 characters |
| Chunk overlap        |  75 characters |
| Minimum chunk length |  50 characters |

---

### 2. Biomedical Embeddings

Each document chunk is converted into a semantic vector representation using:

`abhinand/MedEmbed-small-v0.1`

The same embedding model is used to represent user queries, allowing the system to retrieve documents based on semantic similarity.

---

### 3. FAISS Retrieval

The generated embeddings are stored in a FAISS vector index.

For a user query, MediMind:

1. Generates a query embedding.
2. Searches the FAISS index.
3. Retrieves the most relevant document candidates.
4. Passes those candidates to the reranking stage.

---

### 4. Cross-Encoder Reranking

Initial vector retrieval provides candidate documents, which are then refined using:

`jinaai/jina-reranker-v1-turbo-en`

The reranker evaluates the relationship between the query and retrieved documents to improve the relevance of the final context.

---

### 5. Knowledge Graph

MediMind complements vector retrieval with a biomedical knowledge graph.

The graph is built using:

* NetworkX
* Biomedical named-entity recognition
* Semantic relationships between documents and entities

Graph traversal allows the system to explore related biomedical concepts beyond the initial vector-search results.

---

### 6. Retrieval-Augmented Generation

Relevant documents and graph-derived context are combined into the RAG pipeline.

The system evaluates the retrieved context before generating an answer, helping the LLM base its response on the available research evidence.

---

### 7. Response Generation

The final context is provided to the configured large language model.

The current configuration uses:

`meta-llama/llama-3.3-70b-instruct`

The application is designed so that the model generates responses from the retrieved biomedical context rather than relying exclusively on its pretrained knowledge.

---

## Models

| Component              | Model                               |
| ---------------------- | ----------------------------------- |
| Embeddings             | `abhinand/MedEmbed-small-v0.1`      |
| Cross-Encoder Reranker | `jinaai/jina-reranker-v1-turbo-en`  |
| Biomedical NER         | `d4data/biomedical-ner-all`         |
| FlashRank              | `ms-marco-MiniLM-L-12-v2`           |
| Language Model         | `meta-llama/llama-3.3-70b-instruct` |

---

## Tech Stack

| Category                | Technologies                                   |
| ----------------------- | ---------------------------------------------- |
| Language                | Python                                         |
| LLM / NLP               | LangChain, Hugging Face, Sentence Transformers |
| Embeddings              | MedEmbed                                       |
| Vector Database         | FAISS                                          |
| Reranking               | Jina Cross-Encoder, FlashRank                  |
| Knowledge Graph         | NetworkX                                       |
| Biomedical NER          | Hugging Face Transformers                      |
| Frontend                | Streamlit                                      |
| Testing                 | pytest                                         |
| Environment / Packaging | uv                                             |
| Data                    | Biomedical JSON / PMC-oriented research data   |

---

## Project Structure

```text
MediMind/
│
├── app.py
├── streaming.py
├── testing.py
│
├── src/
│   └── medical_graph_rag/
│       ├── config.py
│       ├── document_processor.py
│       ├── rag_chain.py
│       └── nlp/
│           └── vectorstore.py
│
├── scripts/
├── tests/
├── artifacts/
│
├── .env.example
├── .gitignore
├── pyproject.toml
├── pytest.ini
├── uv.lock
└── LICENSE
```

---

## Getting Started

### Prerequisites

* Python 3.11+
* Git
* `uv`
* API credentials for the configured LLM provider
* Biomedical research data in the supported JSON format

### Installation

Clone the repository:

```bash
git clone https://github.com/Vravula02/MediMind.git
cd MediMind
```

Install the project dependencies:

```bash
uv sync
```

Activate the virtual environment.

**Git Bash:**

```bash
source .venv/Scripts/activate
```

**Windows PowerShell:**

```powershell
.venv\Scripts\Activate.ps1
```

---

## Configuration

Create a `.env` file in the project root and configure the credentials required by the selected LLM provider.

Example:

```env
OPENAI_API_KEY=your_api_key
OPENAI_BASE_URL=your_provider_base_url
```

> Do not commit API keys, credentials, or other secrets to the repository.

---

## Running the Application

Start the Streamlit application:

```bash
streamlit run app.py
```

The application provides an interactive interface for:

* Loading biomedical research data
* Entering natural-language research questions
* Retrieving relevant literature
* Reviewing retrieved content
* Exploring knowledge-graph relationships
* Generating context-aware research responses

---

## Data

MediMind supports biomedical research data containing document content and associated publication metadata.

The repository intentionally does **not** include large datasets or generated retrieval artifacts.

Examples of generated or local-only resources include:

```text
faiss_index/
demo_cache/
my_cache/
data/input/*.json
data/output/processed_pmc_data/
```

These resources are excluded from version control through `.gitignore`.

---

## Retrieval Configuration

Several retrieval parameters can be configured through the project configuration module.

| Parameter                     | Default |
| ----------------------------- | ------: |
| Initial retrieval `k`         |      20 |
| Reranker top-N                |       4 |
| Batch size                    |     500 |
| Maximum concurrent batches    |      10 |
| Similarity threshold          |     0.4 |
| Minimum graph traversal nodes |       8 |
| Maximum graph traversal nodes |      25 |
| Maximum LLM context           |    4000 |

---

## Testing

Run the test suite:

```bash
pytest
```

For verbose output:

```bash
pytest -v
```

---

## Design Principles

MediMind is built around four core ideas:

### Semantic Retrieval

Retrieve biomedical literature based on conceptual similarity rather than keyword matching alone.

### Retrieval Precision

Use multi-stage retrieval and reranking to improve the relevance of the context provided to the LLM.

### Graph-Aware Exploration

Use relationships between biomedical entities and documents to expand the retrieved context.

### Grounded Generation

Generate responses using retrieved research context to reduce reliance on unsupported model-generated information.

---

## Future Improvements

Potential areas for further development include:

* Hybrid keyword and semantic retrieval
* Improved biomedical entity and relationship extraction
* Citation-aware answer generation
* Retrieval evaluation benchmarks
* Larger biomedical datasets
* More advanced graph traversal strategies
* Additional LLM providers
* Scalable cloud deployment
* Persistent user sessions

---

## Disclaimer

MediMind is a research and educational project.

The system can produce incomplete, inaccurate, or outdated information. Its output should not be considered medical advice and should not replace qualified healthcare professionals, clinical guidelines, or peer-reviewed medical evaluation.

---

## License

This project is licensed under the MIT License.
