"""
RAG Demo v3: Ask About My Resume (now with embeddings)
--------------------------------------------------------
v1 matched exact words. v2 fixed "headers lose their content" by chunking.
v3 fixes the real remaining problem: exact word matching still fails when
the question uses DIFFERENT words than the resume (e.g. "frameworks" vs
"Node.js"/"Express"/"Django"). Embeddings represent MEANING as vectors, so
"backend frameworks" and "Node.js, Express, Django" can be judged similar
even though they share zero literal words.

Pipeline (same shape as before, only retrieve() changed):
  1. Load resume.txt
  2. Chunk it (same as v2)
  3. RETRIEVE  -> embed the question + every chunk, rank by cosine similarity
  4. AUGMENT   -> stuff the top chunk(s) into a prompt template
  5. GENERATE  -> send the prompt to an LLM, print the answer
"""

RESUME_PATH = "resume.txt"

# Loaded once, lazily, the first time we need it - loading this model takes
# a few seconds, so we don't want to do it on every single question.
_embedding_model = None


def load_resume(path: str) -> str:
    with open(path, "r") as f:
        text = f.read()
    print(f"[LOAD] Read '{path}' - {len(text)} characters")
    return text


def get_embedding_model():
    global _embedding_model
    if _embedding_model is None:
        print("[MODEL] Loading embedding model 'all-MiniLM-L6-v2' (first call only)...")
        from sentence_transformers import SentenceTransformer
        # all-MiniLM-L6-v2: small, fast, good enough for this project.
        # Downloads once (~80MB) the first time you run this, then it's cached.
        _embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
        print("[MODEL] Model loaded and cached in memory.")
    return _embedding_model


def cosine_similarity(a, b) -> float:
    """
    Measures how similar two vectors are, from -1 (opposite) to 1 (identical
    direction). This is the standard way to compare embeddings.
    """
    import numpy as np
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))

  
def retrieve(question: str, resume_text: str, top_k: int = 2) -> str:
    """
    Version 3: embedding-based semantic retrieval.

    Instead of checking for shared WORDS, we:
      1. Convert the question into a vector (embedding)
      2. Convert every chunk into a vector
      3. Score each chunk by how similar its vector is to the question's
      4. Keep the top_k highest-scoring chunks

    This is why "What backend frameworks does she know?" can now correctly
    retrieve the Skills chunk, even though the word "frameworks" never
    appears anywhere in the resume - the model has learned that Node.js,
    Express, Django etc. are semantically related to "backend frameworks."
    """
    model = get_embedding_model()
    chunks = chunk_resume(resume_text)
    print(f"\n[CHUNK] Split resume into {len(chunks)} chunks")

    question_embedding = model.encode(question)
    print(f"[EMBED] Question -> vector of {len(question_embedding)} numbers")
    print(f"[EMBED] First 5 values: {question_embedding[:5]}")

    chunk_embeddings = model.encode(chunks)
    print(
        f"[EMBED] {len(chunks)} chunks -> {len(chunks)} vectors of {chunk_embeddings.shape[1]} numbers each")

    scored_chunks = [
        (chunk, cosine_similarity(question_embedding, chunk_emb))
        for chunk, chunk_emb in zip(chunks, chunk_embeddings)
    ]

    # Highest similarity first
    scored_chunks.sort(key=lambda pair: pair[1], reverse=True)

    # Handy for learning: see exactly which chunks won and by how much
    print("\n[SCORE] Similarity of question vs each chunk (1.0 = identical meaning):")
    for chunk, score in scored_chunks:
        preview = chunk.splitlines()[0]
        print(f"  {score:.3f}  {preview}")

    top_chunks = [chunk for chunk, score in scored_chunks[:top_k]]
    print(f"\n[RETRIEVE] Keeping top_k={top_k} chunk(s): "
          f"{[c.splitlines()[0] for c in top_chunks]}")

    return "\n\n".join(top_chunks)


def chunk_resume(resume_text: str) -> list[str]:
    """
    Split the resume into chunks on blank lines. Each chunk is a self-
    contained section, e.g.:

        Skills:
        Node.js
        Express
        Django

    This keeps headers glued to the content underneath them, which is
    exactly what line-by-line matching couldn't do.
    """
    raw_chunks = resume_text.split("\n\n")
    return [c.strip() for c in raw_chunks if c.strip()]


def augment(question: str, context: str) -> str:
    """
    Build the final prompt. This is the "A" in RAG - Augmented.
    We are augmenting the question with retrieved context.
    """
    prompt = f"""Answer using ONLY this information. If the answer isn't in the context, say you don't know.

Context:
{context}

Question:
{question}
"""
    print(
        f"\n[AUGMENT] Built prompt ({len(prompt)} characters) - this is exactly what the LLM will see:")
    print("  " + "-" * 55)
    for line in prompt.splitlines():
        print(f"  {line}")
    print("  " + "-" * 55)
    return prompt


OLLAMA_URL = "http://localhost:11434/api/chat"
OLLAMA_MODEL = "llama3.2"  # small (~2GB), free, runs fully on your machine


def generate(prompt: str) -> str:
    """
    Send the augmented prompt to a LOCAL LLM via Ollama. Completely free,
    no API key, no internet required after the model is downloaded once.

    Requires:
      1. Ollama installed and running (the Ollama app, or `ollama serve`)
      2. The model pulled once: `ollama pull llama3.2`

    If Ollama isn't running, we print what WOULD be sent, so you can still
    see the full pipeline working.
    """
    import requests
    import time

    print(
        f"\n[GENERATE] Sending prompt to Ollama at {OLLAMA_URL} (model={OLLAMA_MODEL})...")
    start = time.time()

    try:
        response = requests.post(
            OLLAMA_URL,
            json={
                "model": OLLAMA_MODEL,
                "messages": [{"role": "user", "content": prompt}],
                "stream": False,
            },
            timeout=60,
        )
        response.raise_for_status()
        elapsed = time.time() - start
        print(
            f"[GENERATE] Got HTTP {response.status_code} back in {elapsed:.1f}s")
        return response.json()["message"]["content"]

    except requests.exceptions.HTTPError as e:
        return (
            f"[ERROR - Ollama returned {response.status_code}]\n"
            f"This usually means the model '{OLLAMA_MODEL}' isn't downloaded yet.\n"
            f"Run this in a terminal: ollama pull {OLLAMA_MODEL}\n"
            f"Then rerun this script.\n"
            f"(Raw error: {e})"
        )

    except requests.exceptions.ConnectionError:
        return (
            "[DRY RUN - Ollama isn't running]\n"
            "This is the exact prompt that would be sent to the LLM:\n"
            "-----------------------------------------------------\n"
            f"{prompt}"
            "-----------------------------------------------------\n"
            "Start Ollama (`ollama serve`, or open the Ollama app) and "
            "rerun to get a real generated answer."
        )


def ask(question: str, resume_text: str) -> None:
    context = retrieve(question, resume_text)
    prompt = augment(question, context)
    answer = generate(prompt)

    print(f"\nQ: {question}")
    print(f"\n--- Retrieved context ---\n{context}")
    print(f"\n--- Answer ---\n{answer}\n")


if __name__ == "__main__":
    resume_text = load_resume(RESUME_PATH)

    print("Ask questions about the resume. Type 'exit' to quit.\n")
    while True:
        q = input("You: ")
        if q.strip().lower() in ("exit", "quit"):
            break
        ask(q, resume_text)
