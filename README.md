\# MediMind — Biomedical Research Assistant



An AI-powered biomedical research assistant that combines \*\*semantic vector retrieval, reranking, knowledge graphs, and Retrieval-Augmented Generation (RAG)\*\* to answer questions using biomedical research literature.



MediMind processes biomedical documents, converts research text into dense vector representations, retrieves relevant literature with \*\*FAISS\*\*, improves retrieval precision through \*\*cross-encoder reranking\*\*, and provides retrieved context to an LLM through a LangChain-based RAG pipeline.



> \*\*Disclaimer:\*\* This project is intended for research and educational purposes. It is not a substitute for professional medical advice, diagnosis, or treatment.



\---



\## Overview



Medical and biomedical literature contains a large amount of information distributed across research papers and abstracts. Finding relevant evidence for a specific question can be difficult with traditional keyword search alone.



MediMind addresses this problem by combining:



\* Biomedical document processing

\* Domain-specific text embeddings

\* FAISS-based vector similarity search

\* Cross-encoder reranking

\* Knowledge graph construction and traversal

\* LangChain-based RAG

\* LLM-based answer generation and context checking

\* Interactive Streamlit interface



The system retrieves relevant biomedical literature, refines the retrieved results, and supplies the most relevant context to the language model before generating an answer.



\---



\## Architecture



```text

&#x20;                   ┌─────────────────────┐

&#x20;                   │   Biomedical Data   │

&#x20;                   │    PubMed / PMC     │

&#x20;                   └──────────┬──────────┘

&#x20;                              │

&#x20;                              ▼

&#x20;                   ┌─────────────────────┐

&#x20;                   │ Document Processing │

&#x20;                   │ Cleaning + Chunking │

&#x20;                   │ Metadata Extraction │

&#x20;                   └──────────┬──────────┘

&#x20;                              │

&#x20;                              ▼

&#x20;                   ┌─────────────────────┐

&#x20;                   │   MedEmbed Model    │

&#x20;                   │   Text Embeddings   │

&#x20;                   └──────────┬──────────┘

&#x20;                              │

&#x20;                              ▼

&#x20;                   ┌─────────────────────┐

&#x20;                   │       FAISS         │

&#x20;                   │ Vector Retrieval    │

&#x20;                   └──────────┬──────────┘

&#x20;                              │

&#x20;                   Candidate Documents

&#x20;                              │

&#x20;                              ▼

&#x20;                   ┌─────────────────────┐

&#x20;                   │  Cross-Encoder      │

&#x20;                   │     Reranking       │

&#x20;                   └──────────┬──────────┘

&#x20;                              │

&#x20;                        Top Documents

&#x20;                              │

&#x20;               ┌──────────────┴──────────────┐

&#x20;               │                             │

&#x20;               ▼                             ▼

&#x20;     ┌──────────────────┐          ┌──────────────────┐

&#x20;     │ Knowledge Graph  │          │ Retrieved Context│

&#x20;     │ Traversal        │          │ + Metadata       │

&#x20;     └────────┬─────────┘          └────────┬─────────┘

&#x20;              │                             │

&#x20;              └──────────────┬──────────────┘

&#x20;                             ▼

&#x20;                   ┌─────────────────────┐

&#x20;                   │   LangChain RAG     │

&#x20;                   │ Prompt + Context    │

&#x20;                   └──────────┬──────────┘

&#x20;                              │

&#x20;                              ▼

&#x20;                   ┌─────────────────────┐

&#x20;                   │        LLM          │

&#x20;                   │ Answer Generation   │

&#x20;                   └──────────┬──────────┘

&#x20;                              │

&#x20;                              ▼

&#x20;                   ┌─────────────────────┐

&#x20;                   │   Streamlit UI      │

&#x20;                   │ Answer + Sources /  │

&#x20;                   │ Graph Visualization │

&#x20;                   └─────────────────────┘

```



\---



\## How the System Works



\### 1. Biomedical Literature Processing



The system processes biomedical JSON data containing research abstracts and metadata.



Documents are:



\* Validated

\* Cleaned

\* Split into smaller chunks

\* Enriched with metadata



The processor preserves metadata including:



\* PMID

\* Title

\* Authors

\* Journal

\* Publication date

\* DOI

\* PMC ID

\* MeSH terms

\* Publication types

\* PubMed URLs

\* DOI URLs



The current implementation uses a `RecursiveCharacterTextSplitter` with a \*\*300-character chunk size and 75-character overlap\*\*.



\---



\### 2. Biomedical Embeddings



Each document chunk is converted into a dense vector representation using:



```text

abhinand/MedEmbed-small-v0.1

```



The same embedding model is used for user queries, allowing the system to compare the semantic representation of a question with biomedical literature.



This enables semantic retrieval where conceptually related documents can be retrieved even when the wording differs from the original query.



\---



\### 3. FAISS Vector Retrieval



The document embeddings are stored in a \*\*FAISS vector index\*\*.



For a user query:



```text

User Question

&#x20;     │

&#x20;     ▼

Query Embedding

&#x20;     │

&#x20;     ▼

FAISS Similarity Search

&#x20;     │

&#x20;     ▼

Candidate Documents

```



FAISS provides efficient similarity search over the embedded biomedical corpus.



The vector store also supports document hashing to identify previously indexed content and reduce duplicate indexing.



\---



\### 4. Cross-Encoder Reranking



Initial vector retrieval provides a fast way to identify candidate documents, but vector similarity alone may not always identify the most relevant documents for a specific query.



MediMind therefore applies a second-stage reranking process using:



```text

jinaai/jina-reranker-v1-turbo-en

```



The retrieval pipeline first obtains a larger candidate set from FAISS and then uses the cross-encoder to evaluate query-document relevance more precisely.



```text

Query

&#x20; │

&#x20; ▼

FAISS

&#x20; │

&#x20; ├── Candidate 1

&#x20; ├── Candidate 2

&#x20; ├── Candidate 3

&#x20; ├── ...

&#x20; └── Candidate 20

&#x20;         │

&#x20;         ▼

&#x20;  Cross-Encoder

&#x20;     Reranking

&#x20;         │

&#x20;         ▼

&#x20;   Top Relevant

&#x20;     Documents

```



The default configuration returns the top \*\*4\*\* documents after reranking.



\---



\## 5. Knowledge Graph



MediMind also incorporates a biomedical knowledge graph to represent relationships between concepts and research documents.



The graph is constructed using:



\* Biomedical named entity recognition

\* Document metadata

\* Semantic similarity

\* Shared biomedical entities



The system can traverse relevant graph nodes to expand the context available for answering a query.



Graph traversal maintains information such as:



\* Visited nodes

\* Traversal paths

\* Extracted entities

\* Related documents

\* Expanded context



The configured traversal limits allow the system to explore between \*\*8 and 25 nodes\*\* depending on the query and retrieved context.



\---



\## 6. Retrieval-Augmented Generation



After relevant biomedical content has been retrieved and refined, it is passed into the RAG pipeline.



```text

User Question

&#x20;     │

&#x20;     ▼

Semantic Retrieval

&#x20;     │

&#x20;     ▼

FAISS Candidates

&#x20;     │

&#x20;     ▼

Reranking

&#x20;     │

&#x20;     ▼

Relevant Biomedical Context

&#x20;     │

&#x20;     ▼

Prompt Construction

&#x20;     │

&#x20;     ▼

LLM

&#x20;     │

&#x20;     ▼

Research-Oriented Answer

```



Rather than relying exclusively on the LLM's pretrained knowledge, MediMind supplies retrieved biomedical literature as additional context.



This allows responses to be grounded in the available research corpus.



\---



\## 7. Prompting and Answer Checking



LangChain is used to construct the prompt and LLM pipeline.



The system also evaluates whether the retrieved context contains enough information to answer a user's question.



Conceptually:



```text

Retrieved Context

&#x20;      │

&#x20;      ▼

Context Sufficiency Check

&#x20;      │

&#x20;  ┌───┴────┐

&#x20;  │        │

Enough?   Not Enough?

&#x20;  │        │

&#x20;  ▼        ▼

Answer    Continue

```



This helps the system determine whether the retrieved information is sufficient before generating the final response.



\---



\## 8. LLM



The main configured language model is:



```text

meta-llama/llama-3.3-70b-instruct

```



The project accesses the model through an OpenRouter-compatible API configuration.



The LLM is used for:



\* Answer generation

\* Context evaluation

\* Context extraction

\* Final synthesis of retrieved biomedical information



\---



\## 9. Streamlit Application



MediMind includes an interactive Streamlit application.



The interface provides:



\* Pipeline initialization

\* API configuration

\* Document loading

\* Custom JSON document upload

\* Query input

\* Conversation history

\* Similarity threshold configuration

\* Retrieved content inspection

\* Knowledge graph traversal information

\* Graph visualization



The application also supports streaming visualization of graph traversal.



Run the application with:



```bash

streamlit run app.py

```



\---



\# Key Features



\* \*\*Biomedical semantic search\*\* using domain-specific embeddings

\* \*\*FAISS vector database\*\* for efficient similarity retrieval

\* \*\*Cross-encoder reranking\*\* for improved retrieval relevance

\* \*\*Knowledge graph traversal\*\* for context expansion

\* \*\*LangChain RAG pipeline\*\* for context-grounded generation

\* \*\*LLM context sufficiency checking\*\*

\* \*\*PubMed/PMC document processing\*\*

\* \*\*Metadata-aware document chunks\*\*

\* \*\*Interactive Streamlit interface\*\*

\* \*\*Custom biomedical JSON document upload\*\*

\* \*\*Configurable retrieval parameters\*\*

\* \*\*Semantic caching support\*\*

\* \*\*Graph traversal visualization\*\*

\* \*\*Batch document processing\*\*



\---



\# Models



| Model                               | Purpose                                  |

| ----------------------------------- | ---------------------------------------- |

| `abhinand/MedEmbed-small-v0.1`      | Biomedical document and query embeddings |

| `jinaai/jina-reranker-v1-turbo-en`  | Cross-encoder document reranking         |

| `ms-marco-MiniLM-L-12-v2`           | FlashRank reranking                      |

| `meta-llama/llama-3.3-70b-instruct` | Main answer-generation LLM               |

| `meta-llama/llama-3.1-8b-instruct`  | Prototype LLM configuration              |



\---



\# Technology Stack



\### AI / NLP



\* Python

\* LangChain

\* Hugging Face

\* MedEmbed

\* Cross-Encoder Reranking

\* FlashRank

\* LLM / RAG



\### Retrieval



\* FAISS

\* Dense Vector Embeddings

\* Contextual Compression

\* Similarity Filtering

\* Document Deduplication



\### Knowledge Graph



\* NetworkX

\* Biomedical Named Entity Recognition

\* Graph Traversal

\* Graph Visualization



\### Application



\* Streamlit

\* Async Python



\### Data



\* PubMed

\* PubMed Central (PMC)

\* JSON

\* Biomedical metadata



\### Development



\* pytest

\* pre-commit

\* Dependabot

\* uv



\---



\# Project Structure



```text

Medical-Research-Assistant/

│

├── .github/

│   ├── dependabot.yml

│   └── workflows/

│       ├── main.yml

│       └── python-lint.yml

│

├── artifacts/

│   └── kg\_demo\*.png

│

├── data/

│   ├── input/

│   └── output/

│

├── scripts/

│   └── build\_kg\_demo.py

│

├── src/

│   └── medical\_graph\_rag/

│       ├── core/

│       │   ├── config.py

│       │   ├── main.py

│       │   ├── common\_helpers.py

│       │   └── utils.py

│       │

│       ├── data\_processing/

│       │   ├── batch\_processor.py

│       │   ├── document\_processor.py

│       │   └── pubmed\_downloader.py

│       │

│       ├── knowledge\_graph/

│       │   ├── knowledge\_graph.py

│       │   └── graph\_viz.py

│       │

│       └── nlp/

│           ├── vectorstore.py

│           ├── rag\_chain.py

│           └── prompt\_caching.py

│

├── tests/

│   ├── batch\_test.py

│   ├── docprocessing\_test.py

│   ├── prompt\_test.py

│   └── vector\_test.py

│

├── app.py

├── streaming.py

├── testing.py

├── pyproject.toml

├── pytest.ini

├── .env.example

├── .gitignore

└── uv.lock

```



\---



\# Installation



\## Requirements



\* Python 3.11+

\* Git

\* API access to the configured LLM provider

\* Sufficient memory and storage for the embedding models and vector index



Clone the repository:



```bash

git clone <repository-url>

cd Medical-Research-Assistant

```



Install dependencies using `uv`:



```bash

pip install uv

uv sync

```



Alternatively, install the project using the dependencies defined in `pyproject.toml`.



\---



\# Environment Variables



Create a `.env` file based on `.env.example`.



Example:



```env

OPENAI\_API\_KEY=

OPENAI\_BASE\_URL=

```



The application uses the configured API provider to access the language model.



\*\*Never commit API keys or credentials to GitHub.\*\*



\---



\# Running the Application



Start the Streamlit application:



```bash

streamlit run app.py

```



Then:



1\. Initialize the pipeline.

2\. Load a processed biomedical dataset or upload a supported JSON file.

3\. Enter a biomedical research question.

4\. The system retrieves relevant literature.

5\. Retrieved documents are reranked.

6\. Relevant graph context can be traversed.

7\. The RAG pipeline supplies context to the LLM.

8\. The generated response is displayed in the Streamlit interface.



\---



\# Data Processing



MediMind supports biomedical document processing and preprocessed document loading.



Documents can contain fields such as:



```text

abstract

title

authors

PMID

DOI

PMC ID

```



The processing pipeline cleans and chunks the documents while preserving biomedical metadata.



Preprocessed documents can also be loaded directly to avoid repeating the preprocessing stage.



\---



\# Retrieval Pipeline



The primary retrieval path is:



```text

Biomedical Documents

&#x20;       │

&#x20;       ▼

Text Cleaning

&#x20;       │

&#x20;       ▼

Recursive Character Chunking

&#x20;       │

&#x20;       ▼

MedEmbed Embeddings

&#x20;       │

&#x20;       ▼

FAISS Index

&#x20;       │

&#x20;       ▼

Initial Candidate Retrieval

&#x20;       │

&#x20;       ▼

Cross-Encoder Reranking

&#x20;       │

&#x20;       ▼

Relevant Context

&#x20;       │

&#x20;       ▼

RAG / Knowledge Graph

&#x20;       │

&#x20;       ▼

LLM

```



The implementation also includes a contextual compression pipeline using embedding-based filtering, redundancy filtering, FlashRank reranking, and LLM-based extraction.



\---



\# Testing



The project includes pytest-based tests for:



\* Batch processing

\* Document processing

\* Prompt behavior

\* Vector retrieval



Run the test suite with:



```bash

pytest

```



\---



\# Future Improvements



Potential extensions include:



\* Improved retrieval evaluation using biomedical benchmark datasets

\* Citation-aware answer generation

\* Retrieval evaluation using metrics such as Recall@K and MRR

\* Hybrid lexical and semantic retrieval

\* Larger biomedical corpora

\* Improved source attribution

\* Distributed vector indexing

\* More advanced graph-based retrieval

\* Cloud deployment



\---



\# Disclaimer



MediMind is a research and educational project for exploring biomedical information retrieval and Retrieval-Augmented Generation.



The system may produce incomplete, incorrect, or outdated information. Generated responses should not be used as a substitute for professional medical advice, diagnosis, or treatment.



Important medical information should be verified against the original research literature and discussed with a qualified healthcare professional when appropriate.



\---



\# License



This project is licensed under the MIT License. See \[LICENSE](LICENSE) for details.



