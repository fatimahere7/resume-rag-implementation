# 🧠 RAG Resume Implementation — From Scratch to PDF-Based Semantic Search

A hands-on implementation of **Retrieval-Augmented Generation (RAG)** built from scratch, using my resume as the knowledge source.

Instead of jumping directly into LangChain, LlamaIndex, or another RAG framework, this project builds the core RAG pipeline step by step to understand **what actually happens behind the scenes**.

The project evolves through multiple versions, with each version solving a real limitation discovered in the previous one.

---

## 🚀 What is RAG?

**Retrieval-Augmented Generation (RAG)** combines information retrieval with a Large Language Model (LLM).

Instead of asking an LLM to answer purely from its trained knowledge, RAG:

```text
User Question
      ↓
   Retrieve
      ↓
Relevant Context
      ↓
    Augment
      ↓
Prompt + Context
      ↓
   Generate
      ↓
    Answer
```

For this project, the knowledge source is my resume.

For example:

> "What backend technologies does Fatima work with?"

The system first retrieves the relevant parts of the resume and then gives those parts to the LLM so that the answer is grounded in the actual resume content.

---

# 📈 Project Evolution

The project is intentionally built version by version.

Each version introduces a more realistic retrieval technique and solves a specific problem from the previous version.

```text
V1 → Keyword Retrieval
 ↓
V2 → Better Chunking
 ↓
V3 → Semantic Retrieval with Embeddings
 ↓
V4 → PDF Upload + Sliding Window Chunking
 ↓
Future → Advanced RAG
```

---

# 🟢 V1 — Keyword-Based RAG

### Goal

Prove that even a very simple retrieval mechanism can improve an LLM's answers.

The first version uses **keyword matching** as the retrieval mechanism.

### Pipeline

```text
resume.txt
    ↓
Load Resume
    ↓
Keyword Retrieval
    ↓
Relevant Lines
    ↓
Prompt Template
    ↓
LLM
    ↓
Answer
```

### How retrieval works

If the user asks:

> "What experience does Fatima have with Solidity?"

The system searches the resume for lines containing relevant keywords such as:

```text
Solidity
blockchain
smart contracts
```

The matching lines are then inserted into the prompt sent to the LLM.

### What V1 demonstrates

* Loading a knowledge source
* Basic retrieval
* Prompt augmentation
* LLM generation
* The fundamental RAG architecture

### Limitation

Keyword matching only understands **literal words**.

For example:

```text
Question:
"What backend frameworks does Fatima know?"

Resume:
"Node.js, Express.js, Django, Django REST Framework"
```

The question contains:

```text
backend frameworks
```

while the resume contains:

```text
Node.js
Express.js
Django
```

The concepts are related, but the words don't necessarily match.

That leads to **V2 and eventually V3**.

---

# 🟡 V2 — Chunk-Based RAG

V2 improves the retrieval process by introducing **chunking**.

### Problem with V1

Line-by-line keyword retrieval can lose important context.

For example:

```text
Backend Development

Node.js
Express.js
PostgreSQL
REST APIs
```

If only one line is retrieved, the answer may lose the relationship between the section heading and its content.

### Solution

Instead of treating every line independently, the resume is divided into meaningful **chunks**.

```text
Resume
  ↓
Chunking
  ↓
[Chunk 1]
[Chunk 2]
[Chunk 3]
[Chunk 4]
  ↓
Keyword Retrieval
  ↓
Relevant Chunks
  ↓
LLM
```

This keeps related information together.

### What V2 demonstrates

* Document chunking
* Context preservation
* Chunk-level retrieval
* Improved prompt context

### Remaining limitation

V2 still relies on **exact keyword matching**.

That means it can struggle when the question and resume use different words for the same concept.

---

# 🔵 V3 — Semantic RAG with Embeddings

V3 introduces **embeddings**.

This is where retrieval becomes much more meaningful.

### The problem

Consider:

```text
Question:
"What backend frameworks does Fatima use?"
```

and:

```text
Resume:
"Node.js, Express.js, Django, Django REST Framework"
```

There may be little or no exact word overlap.

Keyword search can fail.

### The solution: embeddings

Embeddings convert text into numerical vectors representing its meaning.

Conceptually:

```text
"backend frameworks"
        ↓
   [0.21, -0.43, 0.78, ...]


"Node.js, Express, Django"
        ↓
   [0.19, -0.40, 0.74, ...]
```

The vectors can then be compared using **cosine similarity**.

If two pieces of text have similar meanings, their vectors should be relatively close.

### V3 Pipeline

```text
resume.txt
    ↓
Chunk Resume
    ↓
Create Embeddings
    ↓
Store Chunk Vectors
    ↓
User Question
    ↓
Create Question Embedding
    ↓
Cosine Similarity
    ↓
Rank Chunks
    ↓
Top Relevant Chunks
    ↓
Prompt + Context
    ↓
LLM
    ↓
Answer
```

### Retrieval

For every chunk:

```text
similarity(question, chunk)
```

is calculated.

The chunks are then ranked:

```text
Chunk 3 → 0.89
Chunk 1 → 0.76
Chunk 5 → 0.61
Chunk 2 → 0.32
```

The highest-scoring chunks are provided to the LLM.

### What V3 demonstrates

* Text embeddings
* Semantic search
* Vector representations
* Cosine similarity
* Similarity-based ranking
* Semantic retrieval
* Context-aware generation

This solves the major weakness of keyword retrieval:

> **The question and the document don't need to use exactly the same words.**

---

# 🟣 V4 — PDF-Based RAG

V4 makes the system more realistic.

Real resumes are usually stored as **PDF files**, not perfectly formatted `.txt` files.

V4 therefore adds:

* PDF input
* PDF text extraction
* Automatic `.pdf` / `.txt` detection
* Sliding-window chunking
* Chunk overlap

---

## 📄 PDF Text Extraction Problem

V3 relied on:

```python
resume_text.split("\n\n")
```

This assumes the extracted text contains clean blank lines between sections.

But PDF extraction doesn't always preserve the original document structure.

A PDF might become:

```text
Fatima Mansoor Blockchain Developer Experience Tycoonz Solutions Backend Development Node.js Express...
```

instead of:

```text
Fatima Mansoor

Blockchain Developer

Experience

Tycoonz Solutions

Backend Development

Node.js
Express...
```

If V3 receives the first version, the entire resume could potentially become **one giant chunk**.

That destroys the quality of retrieval.

---

# 🔧 V4 Improvements

### 1. Automatic file detection

The system can work with:

```text
resume.pdf
```

or:

```text
resume.txt
```

For PDFs, text is extracted using `pypdf`.

---

### 2. Sliding-Window Chunking

Instead of depending on blank lines, V4 uses a fixed-size sliding window.

Conceptually:

```text
                 Chunk 1
        ┌──────────────────────┐
Text →  │ characters 0 → 1000  │
        └──────────────────────┘

                    Chunk 2
              ┌──────────────────────┐
              │ 800 → 1800           │
              └──────────────────────┘

                         Chunk 3
                    ┌──────────────────────┐
                    │ 1600 → 2600          │
                    └──────────────────────┘
```

Notice that chunks **overlap**.

This helps prevent important information from being lost at chunk boundaries.

---

# 🏗️ V4 Pipeline

```text
             Resume
                │
        ┌───────┴────────┐
        │                │
      PDF              TXT
        │                │
     pypdf              │
        │                │
        └───────┬────────┘
                ↓
           Extract Text
                ↓
        Sliding-Window
           Chunking
                ↓
           Embeddings
                ↓
        Semantic Retrieval
                ↓
       Top Relevant Chunks
                ↓
         Prompt Augmentation
                ↓
               LLM
                ↓
             Answer
```

---

# 🧩 Core RAG Architecture

Although each version improves retrieval, the fundamental RAG architecture stays the same:

```text
1. LOAD
   ↓
2. CHUNK
   ↓
3. RETRIEVE
   ↓
4. AUGMENT
   ↓
5. GENERATE
```

The major evolution is **how retrieval improves**:

| Version | Input     | Chunking       | Retrieval            |
| ------- | --------- | -------------- | -------------------- |
| V1      | TXT       | Basic          | Keyword matching     |
| V2      | TXT       | Chunk-based    | Keyword matching     |
| V3      | TXT       | Chunk-based    | Embedding similarity |
| V4      | PDF / TXT | Sliding window | Embedding similarity |

---

# 🛠️ Technologies

* Python
* Large Language Model API
* Text Embeddings
* Cosine Similarity
* `pypdf`
* Git / GitHub

The goal is to implement the RAG fundamentals **without relying on a high-level RAG framework**.

---

# 📁 Project Structure

The project is organized around the different stages of the RAG implementation.

```text
rag-project/
│
├── rag-demo/
│   ├── v1/
│   │   └── ...
│   │
│   ├── v2/
│   │   └── ...
│   │
│   ├── v3/
│   │   └── ...
│   │
│   ├── v4/
│   │   └── ...
│   │
│   └── ...
│
├── .gitignore
├── README.md
└── requirements.txt
```

> `venv/` is intentionally excluded from the repository. Dependencies are installed using `requirements.txt`.

---

# ▶️ Running the Project

Clone the repository:

```bash
git clone https://github.com/fatimahere7/resume-rag-implementation.git
cd resume-rag-implementation
```

Create a virtual environment:

```bash
python3 -m venv venv
```

Activate it:

### macOS / Linux

```bash
source venv/bin/activate
```

### Windows

```bash
venv\Scripts\activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Configure the required environment variables:

```text
OPENAI_API_KEY=your_api_key
```

Then run the desired RAG version.

---

# 🧪 What I Am Learning Through This Project

This project is not just about making a chatbot.

The goal is to understand the internal mechanics of RAG by implementing them progressively.

Through V1 → V4, the project explores:

* What retrieval actually means
* Why naive keyword search can fail
* Why chunking matters
* How context can be preserved
* What embeddings represent
* How semantic similarity works
* How cosine similarity can rank documents
* Why PDFs create additional retrieval challenges
* Why chunk overlap is useful
* How retrieved context is injected into an LLM prompt
* How retrieval quality affects generation quality

---

# 🔮 What's Next?

The project is still evolving.

Future versions will explore more advanced RAG concepts such as:

```text
V5
 ↓
Vector Database
 ↓
Persistent Embeddings
 ↓
Metadata Filtering
 ↓
Improved Retrieval

V6
 ↓
Hybrid Search
 ↓
Keyword + Semantic Retrieval

V7
 ↓
Reranking
 ↓
Better Context Selection

V8
 ↓
Advanced RAG
 ↓
Evaluation
 ↓
Production-oriented Pipeline
```

The goal is to move gradually from:

> **"I can make a RAG application."**

to:

> **"I understand how RAG works internally and can design the retrieval pipeline myself."**

---

# 🎯 Project Goal

This repository documents my journey of implementing **Retrieval-Augmented Generation from the ground up**, starting with simple keyword retrieval and progressively moving toward semantic, PDF-aware retrieval.

Rather than hiding the complexity behind a framework, each version exposes one more layer of the RAG pipeline.

**From simple retrieval → semantic search → realistic document ingestion → advanced RAG.**

---

## 👩‍💻 Author

**Fatima Mansoor**

Blockchain Developer | Backend Developer | RAG Learner

GitHub: [@fatimahere7](https://github.com/fatimahere7)
