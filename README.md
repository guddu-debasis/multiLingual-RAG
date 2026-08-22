# 🌐 Cross-Lingual Multilingual RAG Engine

A lightweight, production-ready **Cross-Lingual Retrieval-Augmented Generation (RAG)** pipeline built with **Node.js (JavaScript)** and **Python**.

This project allows users to ask questions in **Hindi or Hinglish** while retrieving relevant information directly from **English PDF documents**, without requiring an intermediate translation step.

The system uses **BAAI/bge-m3** for multilingual embeddings, **FAISS** for vector similarity search, and **Groq's Llama 3.3 70B** for final answer generation.

---

## 📌 Table of Contents

* [Architecture & Workflow](#-architecture--workflow)
* [Theoretical Foundations](#-theoretical-foundations)

  * [1. Cross-Lingual Semantic Embeddings](#1-cross-lingual-semantic-embeddings)
  * [2. Model Weights, ONNX & Transformers](#2-model-weights-onnx--transformers)
  * [3. Vector Indexing & Cosine Similarity](#3-vector-indexing--cosine-similarity)
  * [4. LLM Context Conditioning](#4-llm-context-conditioning)
* [Project Structure](#-project-structure)
* [Tech Stack](#-tech-stack)
* [Prerequisites & Installation](#-prerequisites--installation)
* [Step-by-Step Execution](#-step-by-step-execution)
* [Sample Queries](#-sample-queries)
* [Troubleshooting](#-troubleshooting)
* [License](#-license)

---

## 🏗 Architecture & Workflow

```text
                 ┌──────────────────────────┐
                 │      English PDF         │
                 │      document.pdf        │
                 └────────────┬─────────────┘
                              │
                              ▼
                 ┌──────────────────────────┐
                 │      PDF Text Extraction │
                 └────────────┬─────────────┘
                              │
                              ▼
                 ┌──────────────────────────┐
                 │      Text Chunking       │
                 │      Chunk: 600          │
                 │      Overlap: 100        │
                 └────────────┬─────────────┘
                              │
                              ▼
                 ┌──────────────────────────┐
                 │      BAAI/bge-m3         │
                 │  Multilingual Embeddings │
                 └────────────┬─────────────┘
                              │
                              ▼
                 ┌──────────────────────────┐
                 │      FAISS Index         │
                 │   1024-D Vector Space    │
                 └────────────┬─────────────┘
                              │
                              │
        ┌─────────────────────┘
        │
        │ User Query
        ▼
┌──────────────────────────────┐
│ Hindi / Hinglish Query       │
│                              │
│ "कंपनी का मुख्यालय कहाँ है?" │
└──────────────┬───────────────┘
               │
               ▼
┌──────────────────────────────┐
│ BAAI/bge-m3                  │
│ Query Embedding              │
└──────────────┬───────────────┘
               │
               ▼
┌──────────────────────────────┐
│ FAISS Similarity Search      │
│                              │
│ Top-K English Chunks         │
└──────────────┬───────────────┘
               │
               ▼
┌──────────────────────────────┐
│ Groq API                     │
│ Llama 3.3 70B                │
│                              │
│ Query + Retrieved Context    │
└──────────────┬───────────────┘
               │
               ▼
┌──────────────────────────────┐
│ Natural Hindi Response       │
└──────────────────────────────┘
```

### 🔄 Overall Pipeline

```text
English Documents
       ↓
PDF Extraction
       ↓
Chunking
       ↓
Multilingual Embeddings
       ↓
FAISS Vector Index
       ↓
Hindi/Hinglish Query
       ↓
Query Embedding
       ↓
Similarity Search
       ↓
Top-K English Context
       ↓
Groq LLM
       ↓
Hindi Answer
```

---

# 🧠 Theoretical Foundations

## 1. Cross-Lingual Semantic Embeddings

The main idea behind this project is **cross-lingual retrieval**.

Normally, an English query is matched against English documents. A Hindi query may not retrieve English documents effectively if the embedding model does not understand the relationship between the two languages.

A multilingual embedding model solves this problem by representing different languages in a **shared semantic vector space**.

This means that semantically similar sentences in different languages can have similar vector representations.

For example:

```text
Hindi:

"कंपनी का मुख्यालय कहाँ है?"

English:

"Where is the company headquarters located?"
```

Both sentences represent approximately the same meaning.

A multilingual embedding model attempts to map them close together:

```text
Hindi Query
     │
     ▼
[0.12, -0.43, 0.71, ...]
     
          ≈

English Document
     │
     ▼
[0.11, -0.41, 0.70, ...]
```

Therefore, the system can perform:

```text
Hindi Query
      ↓
Hindi Vector
      ↓
Similarity Search
      ↓
English Document Chunk
```

without translating the query into English first.

### Why BGE-M3?

`BAAI/bge-m3` is a multilingual embedding model designed to support many languages and multiple retrieval scenarios.

In this project it is used to generate:

1. Document embeddings
2. Query embeddings

Both embeddings are generated in the same vector space.

---

## 2. Model Weights, ONNX & Transformers

### What are Model Weights?

A neural network contains millions or billions of learned parameters.

These parameters are called **model weights**.

During inference, the model uses these weights to transform text into numerical representations.

For example:

```text
Input Text
    ↓
Tokenizer
    ↓
Neural Network
    ↓
Model Weights
    ↓
Embedding Vector
```

For BGE-M3, the resulting embedding has:

```text
1024 dimensions
```

So a sentence can become something like:

```text
[
  0.012,
  -0.321,
  0.182,
  ...
]
```

with 1024 numerical values.

---

## 3. Why ONNX?

Models are commonly trained and stored using frameworks such as:

* PyTorch
* TensorFlow

Node.js does not natively execute PyTorch model files such as:

```text
.pt
.bin
```

in the same way Python-based ML frameworks do.

**ONNX (Open Neural Network Exchange)** provides a portable representation of neural network computation graphs.

The general flow becomes:

```text
Original Model
     ↓
PyTorch
     ↓
ONNX Model
     ↓
ONNX Runtime
     ↓
Node.js
```

---

## 4. `@xenova/transformers`

`@xenova/transformers` allows transformer models to run locally in JavaScript environments.

Instead of sending every query to an external embedding API:

```text
Node.js
   ↓
External Embedding API
   ↓
Embedding
```

the application can perform:

```text
Node.js
   ↓
Local BGE-M3 Model
   ↓
Embedding
```

### Advantages

* No embedding API cost
* No network request for every embedding
* Data can remain local
* Works well for development and experimentation
* Reduces dependency on external embedding services

### Trade-off

Local inference requires:

* CPU/GPU resources
* Model storage
* Initial model download
* More memory compared with an API-only architecture

---

# 5. Vector Indexing & Similarity Search

After creating embeddings for every document chunk, we need a way to search them efficiently.

Suppose the document produces:

```text
Chunk 1 → Vector 1
Chunk 2 → Vector 2
Chunk 3 → Vector 3
...
Chunk 10000 → Vector 10000
```

Searching every vector manually would become expensive as the dataset grows.

FAISS is designed for efficient vector similarity search.

---

## Cosine Similarity

Cosine similarity measures how similar two vectors are based on their direction.

The formula is:

$$
\text{CosineSimilarity}(A,B)
============================

\frac{A \cdot B}
{|A||B|}
$$

If vectors are normalized to unit length:

$$
|A| = |B| = 1
$$

then:

$$
\text{CosineSimilarity}(A,B)
============================

A \cdot B
$$

Therefore, FAISS can use **Inner Product** search to perform cosine-style similarity search on normalized embeddings.

The dot product is:

$$
A \cdot B =
\sum_{i=1}^{1024} A_iB_i
$$

---

## Retrieval Example

Suppose the user asks:

```text
कंपनी का मुख्यालय कहाँ है?
```

The query becomes:

```text
Query
 ↓
BGE-M3
 ↓
1024-dimensional vector
```

FAISS compares this vector against the document vectors.

Example:

```text
Chunk 1 → Similarity: 0.91
Chunk 2 → Similarity: 0.82
Chunk 3 → Similarity: 0.37
Chunk 4 → Similarity: 0.88
Chunk 5 → Similarity: 0.21
```

If `topK = 3`:

```text
Chunk 1 → 0.91
Chunk 4 → 0.88
Chunk 2 → 0.82
```

These chunks are passed to the LLM.

---

# 6. LLM Context Conditioning

The vector database does **not** generate the final answer.

Its job is only:

```text
Find relevant information
```

The LLM then uses that information to generate the answer.

The pipeline is:

```text
User Query
     +
Retrieved Context
     ↓
LLM
     ↓
Final Answer
```

For example:

### Query

```text
कंपनी का मुख्यालय कहाँ है?
```

### Retrieved Context

```text
GlobalTech headquarters is located in Austin, Texas.
The company has an engineering center in Bengaluru, India.
```

### LLM

```text
Groq
+
Llama 3.3 70B
```

### Final Response

```text
ग्लोबलटेक का मुख्यालय ऑस्टिन, टेक्सास में स्थित है।
भारत में इसका इंजीनियरिंग सेंटर बेंगलुरु में है।
```

The LLM is therefore **grounded by the retrieved documents**.

---

# 📂 Project Structure

```text
multilingual-rag/
│
├── data/
│   └── document.pdf
│
├── faiss_index/
│   └── ...
│
├── .env
├── package.json
├── config.js
├── embeddings.js
├── ingest.js
├── ragEngine.js
└── app.js
```

### File Responsibilities

| File            | Responsibility                     |
| --------------- | ---------------------------------- |
| `config.js`     | Application configuration          |
| `embeddings.js` | BGE-M3 embedding generation        |
| `ingest.js`     | PDF loading, chunking and indexing |
| `ragEngine.js`  | Retrieval and LLM generation       |
| `app.js`        | CLI interface                      |
| `data/`         | Source PDF documents               |
| `faiss_index/`  | Persisted vector index             |
| `.env`          | API credentials                    |

---

# 🛠 Tech Stack

| Layer          | Technology             | Purpose                          |
| -------------- | ---------------------- | -------------------------------- |
| Language       | JavaScript / Node.js   | Application runtime              |
| Embeddings     | BAAI/bge-m3            | Multilingual semantic embeddings |
| Runtime        | `@xenova/transformers` | Local transformer inference      |
| Vector Store   | FAISS                  | Similarity search                |
| LLM            | Groq                   | Final answer generation          |
| Model          | Llama 3.3 70B          | Response generation              |
| PDF Processing | LangChain.js           | PDF loading                      |
| Text Splitting | LangChain.js           | Document chunking                |

---

# ⚙️ Prerequisites & Installation

## 1. Clone the Repository

```bash
git clone https://github.com/your-username/multilingual-rag.git

cd multilingual-rag
```

---

## 2. Install Dependencies

```bash
npm install
```

---

## 3. Configure Environment Variables

Create a `.env` file:

```env
GROQ_API_KEY=gsk_your_actual_groq_api_key_here
```

Never commit your `.env` file to GitHub.

Add this to `.gitignore`:

```text
.env
node_modules/
faiss_index/
```

---

## 4. Add Your PDF

Place your English PDF inside:

```text
data/
```

Example:

```text
data/
└── document.pdf
```

---

# 🚀 Step-by-Step Execution

## Step 1: Ingest the Documents

Run:

```bash
npm run ingest
```

or:

```bash
node ingest.js
```

The ingestion pipeline performs:

```text
PDF
 ↓
Text Extraction
 ↓
Chunking
 ↓
BGE-M3 Embeddings
 ↓
FAISS Index
 ↓
Save Index
```

Example output:

```text
Loading PDF files...
Loaded 2 pages.
Chunking text...
Created 8 chunks.

Generating BGE-M3 embeddings...
Building FAISS index...

FAISS index successfully saved.
```

---

# Step 2: Start the RAG Application

Run:

```bash
npm start
```

or:

```bash
node app.js
```

You can now ask questions in Hindi or Hinglish.

---

# 🧪 Sample Queries

## Example 1 — Hindi Query

### User

```text
कंपनी का मुख्यालय कहाँ है और भारत में उनका कार्यालय किस शहर में स्थित है?
```

### Retrieved Context

```text
GlobalTech headquarters is located in Austin, Texas (USA),
with a global engineering center in Bengaluru (India).
```

### Response

```text
ग्लोबलटेक का मुख्यालय ऑस्टिन, टेक्सास (USA) में स्थित है।
भारत में इसका इंजीनियरिंग सेंटर बेंगलुरु में स्थित है।
```

---

## Example 2 — Hinglish Query

### User

```text
Product pasand na aane par refund kitne dino me milta hai?
```

### Retrieved Context

```text
GlobalTech offers a comprehensive 30-day money-back guarantee.
Refunds are processed to the original payment method within
5 to 7 business days.
```

### Response

```text
यदि आप उत्पाद से संतुष्ट नहीं हैं, तो आप 30 दिनों के भीतर
रिफंड के लिए अनुरोध कर सकते हैं। रिफंड की प्रक्रिया मूल
भुगतान विधि पर 5 से 7 कार्यदिवसों के भीतर पूरी की जाती है।
```

---

# 🔍 Why No Translation Step?

A traditional multilingual RAG pipeline could look like:

```text
Hindi Query
     ↓
Translation Model
     ↓
English Query
     ↓
Embedding
     ↓
Vector Search
     ↓
English Documents
```

This project instead uses:

```text
Hindi Query
     ↓
Multilingual Embedding
     ↓
Vector Search
     ↓
English Documents
```

This reduces the number of inference steps and avoids introducing an additional translation dependency.

---

# ⚠️ Important Limitations

Cross-lingual retrieval is powerful, but it is not guaranteed to work perfectly.

Performance depends on:

* Embedding model quality
* Language
* Query complexity
* Document quality
* Chunk size
* Chunk overlap
* Top-K value
* Similarity threshold
* Domain-specific terminology

For production systems, retrieval should be evaluated using a proper multilingual test dataset.

Useful metrics include:

```text
Recall@K
Precision@K
MRR
nDCG
Faithfulness
Answer Relevancy
```

---

# 🔧 Troubleshooting

## 1. `getaddrinfo EAI_AGAIN huggingface.co`

### Cause

The application cannot resolve or connect to Hugging Face.

Possible causes include:

* DNS problems
* Network instability
* ISP issues
* Firewall restrictions

### Possible Fix

Try:

```bash
ipconfig /flushdns
```

You can also test another network or configure a reliable DNS resolver.

---

## 2. Model Not Found

If the embedding model cannot be downloaded, verify the model identifier and ensure that the model is available in the format expected by your runtime.

---

## 3. Groq Model Error

If Groq returns a model-not-found error, verify the currently available model ID in your Groq configuration.

For example:

```javascript
const modelName = "llama-3.3-70b-versatile";
```

Model availability can change over time, so always verify the model ID against the current Groq documentation.

---

## 4. Poor Retrieval Results

If Hindi queries are not retrieving the correct English chunks, investigate:

```text
Chunk Size
     ↓
Chunk Overlap
     ↓
Embedding Quality
     ↓
Query Quality
     ↓
Top-K
     ↓
Similarity Threshold
     ↓
Reranking
```

For a more advanced system, consider adding:

* Hybrid Search
* BM25
* Reranking
* Metadata Filtering
* Query Rewriting
* Parent-Document Retrieval
* Multilingual Evaluation

---

# 🚀 Future Improvements

This project can be extended into a more advanced production RAG system.

### Phase 1 — Better Retrieval

```text
Dense Retrieval
+
BM25
+
Hybrid Search
```

### Phase 2 — Reranking

```text
Query
 ↓
FAISS Top-20
 ↓
Reranker
 ↓
Top-5
 ↓
LLM
```

### Phase 3 — Production Architecture

```text
                    ┌───────────────┐
                    │    Client     │
                    └───────┬───────┘
                            ↓
                    ┌───────────────┐
                    │   API Server  │
                    └───────┬───────┘
                            ↓
                    ┌───────────────┐
                    │ Retrieval     │
                    │ Service       │
                    └───────┬───────┘
                            ↓
              ┌─────────────┴─────────────┐
              ↓                           ↓
        Vector Database             Reranker
              │                           │
              └─────────────┬─────────────┘
                            ↓
                         LLM
                            ↓
                    Hindi Response
```

Possible production additions include:

* Redis caching
* PostgreSQL
* Qdrant / Pinecone / Weaviate
* Docker
* Kubernetes
* FastAPI
* Node.js API layer
* Observability
* Rate limiting
* Authentication
* Evaluation pipelines
* CI/CD

---

# 📊 RAG Evaluation

A production RAG system should not be evaluated only by asking:

> "Does the answer look correct?"

Instead, evaluate individual stages.

## Retrieval Evaluation

Measure:

```text
Recall@K
Precision@K
MRR
nDCG
```

Example:

```text
Recall@5 = 0.92
```

This means the correct document was retrieved within the top 5 results for approximately 92% of evaluation queries.

---

## Generation Evaluation

Measure:

```text
Faithfulness
Answer Relevancy
Context Relevancy
```

The objective is:

```text
Correct Retrieval
        +
Grounded Generation
        =
Reliable RAG
```

---

# 🎯 Key Concepts Demonstrated

This project demonstrates practical understanding of:

* Retrieval-Augmented Generation
* Multilingual embeddings
* Cross-lingual retrieval
* Dense vector search
* FAISS
* Cosine similarity
* Inner product search
* ONNX inference
* Local embedding models
* PDF document processing
* Chunking
* LLM context conditioning
* Groq API
* Hindi/Hinglish NLP
* RAG evaluation
* Production RAG architecture

---

# 📄 License

MIT License

Free for educational and commercial use.
