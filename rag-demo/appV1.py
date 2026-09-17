"""
RAG Demo v1: Ask About My Resume
----------------------------------
The whole point of this version: prove that "retrieval" can be as dumb as
keyword matching, and it STILL makes the LLM's answer better and grounded.

Pipeline:
  1. Load resume.txt
  2. RETRIEVE  -> find the lines relevant to the question (keyword match)
  3. AUGMENT   -> stuff those lines into a prompt template
  4. GENERATE  -> send the prompt to an LLM, print the answer
"""

RESUME_PATH = "resume.txt"


def load_resume(path: str) -> str:
    with open(path, "r") as f:
          return f.read()
  

def retrieve(question: str, resume_text: str) -> str:
    """
    Dumbest possible retrieval: keyword matching.

    Idea: split the resume into lines. Split the question into words.
    Keep any resume line that shares a word with the question (case-insensitive).
    This is NOT smart (no synonyms, no meaning) - that's exactly the point
    of v1. We upgrade this function later; nothing else changes.
    """
    question_words = set(w.lower().strip("?,.:") for w in question.split())

    relevant_lines = []
    for line in resume_text.splitlines():
        line_words = set(w.lower().strip("?,.:") for w in line.split())
        if question_words & line_words:  # set intersection = shared words
            relevant_lines.append(line)

    if not relevant_lines:
        # Fallback: if keyword match finds nothing, don't return empty context
        return resume_text

    return "\n".join(relevant_lines)


def augment(question: str, context: str) -> str:
    """
    Build the final prompt. This is the "A" in RAG - Augmented.
    We are augmenting the question with retrieved context.
    """
    return f"""Answer using ONLY this information. If the answer isn't in the context, say you don't know.

Context:
{context}

Question:
{question}
"""


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