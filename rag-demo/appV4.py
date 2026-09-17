"""
RAG Demo v4: Ask About My Resume (now with PDF upload)
--------------------------------------------------------
v3 fixed word-mismatch with embeddings. v4 fixes the next real problem:
resumes usually live as PDFs, not clean .txt files - and PDF text
extraction often loses the blank-line breaks between sections that v3's
chunker depended on (`resume_text.split("\n\n")`). A PDF might extract as
one giant blob of text with no double newlines at all, which would make
v3 treat the ENTIRE resume as a single chunk - destroying retrieval.

Two changes from v3:
  1. LOAD    -> auto-detect .pdf vs .txt, extract PDF text with pypdf
  2. CHUNK   -> replace blank-line splitting with a sliding-window chunker
                (fixed character size + overlap) that works regardless of
                whether the source preserved paragraph breaks

Everything else (embed -> retrieve -> augment -> generate) is identical
to v3.
"""

# change to resume.txt if you want to test with plain text
RESUME_PATH = "resume2.pdf"

# Loaded once, lazily
_embedding_model = None


def load_resume(path: str) -> str:
    """
    v4: detects the file type from its extension and extracts text
    accordingly. PDFs need actual parsing (they're not plain text under
    the hood) - .txt files are read as before.
    """
    if path.lower().endswith(".pdf"):
        text = extract_text_from_pdf(path)
        print(f"[LOAD] Extracted '{path}' (PDF) - {len(text)} characters")
    else:
        with open(path, "r") as f:
            text = f.read()
        print(f"[LOAD] Read '{path}' (text) - {len(text)} characters")
    return clean_text(text)


def extract_text_from_pdf(path: str) -> str:
    """
    Pulls text out of every page of a PDF and joins it into one string.

    Uses pdfplumber instead of pypdf. pypdf's text extraction inserts a
    space wherever the gap between two glyphs exceeds a threshold - and
    some PDFs (depending on the font/encoding used when the resume was
    exported, e.g. from Canva or certain Word->PDF pipelines) trigger that
    threshold WITHIN words, producing garbage like "W eb T echnologies"
    instead of "Web Technologies". pdfplumber uses a different, generally
    more reliable text-reconstruction approach and doesn't have this problem
    on the same files.

    Note: PDF is a visual/layout format, not a text format - there's still
    no guarantee paragraph breaks survive extraction, and columns/tables
    can still extract in a strange reading order. If retrieval quality
    looks off, the first thing to check is what this function actually
    pulled out (print `text` and eyeball it) before assuming the embedding
    step is at fault.
    """
    import pdfplumber

    pages_text = []
    with pdfplumber.open(path) as pdf:
        for i, page in enumerate(pdf.pages):
            # x_tolerance controls how big a horizontal gap between
            # characters has to be before pdfplumber treats it as a word
            # boundary. The default (3) misses tightly-kerned fonts where
            # the visible space between words is narrower than usual,
            # producing jammed text like "AsabeginnerinthefieldofComputerScience".
            # Lowering it catches smaller gaps as real spaces. If a
            # different resume still comes out jammed, try tuning this
            # further (e.g. 1) or the reverse - too low can over-split.
            page_text = page.extract_text(x_tolerance=1) or ""
            pages_text.append(page_text)
            print(
                f"  [PDF] Page {i + 1}: extracted {len(page_text)} characters")

    return "\n\n".join(pages_text)


def get_embedding_model():
    global _embedding_model
    if _embedding_model is None:
        print("[MODEL] Loading embedding model 'all-MiniLM-L6-v2' (first call only)...")
        from sentence_transformers import SentenceTransformer
        _embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
        print("[MODEL] Model loaded and cached in memory.")
    return _embedding_model


def cosine_similarity(a, b) -> float:
    import numpy as np
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def clean_text(text: str) -> str:
    """
    Strips artifacts extraction sometimes leaves behind - most commonly
    `(cid:NNN)` placeholders, which appear when a PDF references a glyph
    (often an icon, like a LinkedIn or phone symbol) that the extractor
    can't map to a real Unicode character. These aren't meaningful text -
    they just add noise the embedding model has to see.
    """
    import re
    return re.sub(r"\(cid:\d+\)", "", text)


SECTION_HEADERS = {
    "professional experience", "experience", "work experience",
    "education", "skills", "technical skills", "projects",
    "personal projects", "objective", "summary", "professional summary",
    "certifications", "certificates", "achievements", "publications",
}


def chunk_by_section(text: str, max_chunk_size: int = 1200, overlap: int = 150) -> list[tuple[str, str]]:
    """
    Chunk by resume section headers (Skills, Education, Professional
    Experience...) instead of blindly slicing every N characters.

    Returns (section_name, chunk_text) pairs instead of bare strings, so
    retrieve() can tell which section a chunk came from - needed because a
    multi-job Experience section can span several chunks, and some
    questions ("overall experience") need ALL of them together, not just
    whichever one chunk scores highest.

    Sections that are too big for one chunk get sub-split by
    split_into_word_safe_chunks() below - never by raw character index,
    which is what caused "TYCOONZ SOLUTIONS" to get sliced into
    "UTIONS Nov 2025..." on the two-page resume.
    """
    lines = text.split("\n")
    raw_sections = []  # (header_name, section_text)
    current_header = "header"  # name/contact block before the first real header
    current_lines = []

    for line in lines:
        stripped = line.strip()
        header_key = stripped.lower().rstrip(":")
        is_header = len(stripped) < 40 and header_key in SECTION_HEADERS
        if is_header and current_lines:
            raw_sections.append(
                (current_header, "\n".join(current_lines).strip()))
            current_header = header_key
            current_lines = [line]
        else:
            current_lines.append(line)
    if current_lines:
        raw_sections.append((current_header, "\n".join(current_lines).strip()))

    chunks = []  # (header_name, chunk_text)
    for header_name, section in raw_sections:
        if not section:
            continue
        if len(section) <= max_chunk_size:
            chunks.append((header_name, section))
        else:
            for sub in split_into_word_safe_chunks(section, max_chunk_size, overlap):
                chunks.append((header_name, sub))

    return chunks


def split_into_word_safe_chunks(text: str, max_chunk_size: int = 1200, overlap: int = 150) -> list[str]:
    """
    Splits an oversized section into smaller chunks WITHOUT ever cutting a
    word in half - replaces the old raw-character sliding window, which
    sliced "SOLUTIONS" into "UTIONS" whenever a chunk boundary landed
    mid-word.

    Strategy: try blank-line paragraph boundaries first (each paragraph is
    usually one job entry / one bullet group, so this tends to keep a
    single job's title+dates+bullets together in one chunk). If a single
    paragraph is STILL too big on its own, pack it word-by-word instead of
    character-by-character, so a boundary can only ever fall between two
    whole words.
    """
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    chunks = []
    buffer = ""

    for para in paragraphs:
        candidate = f"{buffer}\n\n{para}".strip() if buffer else para
        if len(candidate) <= max_chunk_size:
            buffer = candidate
            continue

        if buffer:
            chunks.append(buffer)

        if len(para) <= max_chunk_size:
            buffer = para
        else:
            words = para.split()
            current, current_len = [], 0
            for word in words:
                if current_len + len(word) + 1 > max_chunk_size:
                    chunks.append(" ".join(current))
                    current, current_len = [word], len(word)
                else:
                    current.append(word)
                    current_len += len(word) + 1
            buffer = " ".join(current)

    if buffer:
        chunks.append(buffer)

    return chunks


# Questions matching any of these are treated as needing the FULL Experience
# section rather than just whatever scores highest. "How much/overall/total
# experience" is an aggregation question - the answer depends on ALL job
# entries together, not on retrieving the single most-similar chunk. Plain
# top-k similarity search has no way to know that on its own.
EXPERIENCE_INTENT_KEYWORDS = ["experience",
                              "worked", "years", "how long", "tenure"]


def retrieve(question: str, resume_text: str, top_k: int = 3) -> str:
    model = get_embedding_model()
    tagged_chunks = chunk_by_section(resume_text)
    print(f"\n[CHUNK] Split resume into {len(tagged_chunks)} chunks")

    header_chunk = next((c for s, c in tagged_chunks if s == "header"), None)

    # Special case: if the question is about experience/duration, don't
    # gamble on top-k picking the right job fragments - just hand over the
    # entire Experience section (every chunk tagged with a header containing
    # "experience"), however many chunks that takes. This is the resume
    # equivalent of "always include the header chunk" from before: some
    # questions need a whole section, not a ranked slice of it.
    if any(keyword in question.lower() for keyword in EXPERIENCE_INTENT_KEYWORDS):
        experience_chunks = [c for s, c in tagged_chunks if "experience" in s]
        print(f"[RETRIEVE] Question is about experience - including the full "
              f"Experience section ({len(experience_chunks)} chunk(s)), skipping top-k ranking")
        result_chunks = ([header_chunk] if header_chunk else []
                         ) + experience_chunks
        return "\n\n".join(result_chunks)

    rest = [c for s, c in tagged_chunks if c != header_chunk]

    question_embedding = model.encode(question)
    print(f"[EMBED] Question -> vector of {len(question_embedding)} numbers")

    rest_embeddings = model.encode(rest) if rest else []
    if rest:
        print(
            f"[EMBED] {len(rest)} chunks -> {len(rest)} vectors of {rest_embeddings.shape[1]} numbers each")

    scored_chunks = [
        (chunk, cosine_similarity(question_embedding, chunk_emb))
        for chunk, chunk_emb in zip(rest, rest_embeddings)
    ]
    scored_chunks.sort(key=lambda pair: pair[1], reverse=True)

    print("\n[SCORE] Similarity of question vs each chunk (1.0 = identical meaning):")
    print(
        f"  (always-included)  {header_chunk[:60].replace(chr(10), ' ') if header_chunk else '(none)'}...")
    for chunk, score in scored_chunks:
        preview = chunk[:60].replace("\n", " ")
        print(f"  {score:.3f}  {preview}...")

    remaining_slots = max(top_k - 1, 0)
    top_from_scores = [chunk for chunk,
                       score in scored_chunks[:remaining_slots]]

    result_chunks = ([header_chunk] if header_chunk else []) + top_from_scores
    print(f"\n[RETRIEVE] Keeping {len(result_chunks)} chunk(s) "
          f"(1 forced header + {len(top_from_scores)} by score)")

    return "\n\n".join(result_chunks)


def augment(question: str, context: str) -> str:
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
OLLAMA_MODEL = "llama3.2"


def generate(prompt: str) -> str:
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
