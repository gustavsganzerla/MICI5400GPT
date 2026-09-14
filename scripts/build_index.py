import os
import json
from pathlib import Path

import faiss
import numpy as np

from pypdf import PdfReader
from sentence_transformers import SentenceTransformer


DOCUMENT_DIR = Path("/app/data/documents")
INDEX_DIR = Path("/app/data/index")

EMBEDDING_MODEL = os.getenv(
    "EMBEDDING_MODEL",
    "sentence-transformers/all-MiniLM-L6-v2"
)

CHUNK_SIZE = 400
CHUNK_OVERLAP = 80


def read_txt(path):
    return path.read_text(
        encoding="utf-8",
        errors="ignore"
    )


def read_pdf(path):

    reader = PdfReader(path)

    text = []

    for page_number, page in enumerate(reader.pages):

        page_text = page.extract_text()

        if page_text:
            text.append(page_text)

    return "\n".join(text)


def load_document(path):

    suffix = path.suffix.lower()

    if suffix == ".txt":
        return read_txt(path)

    if suffix == ".pdf":
        return read_pdf(path)

    return None


def chunk_text(
    text,
    chunk_size=CHUNK_SIZE,
    overlap=CHUNK_OVERLAP
):

    words = text.split()

    chunks = []

    start = 0

    while start < len(words):

        end = start + chunk_size

        chunk = " ".join(
            words[start:end]
        )

        if chunk.strip():

            chunks.append(chunk)

        start += chunk_size - overlap

    return chunks


def main():

    INDEX_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    model = SentenceTransformer(
        EMBEDDING_MODEL
    )

    metadata = []

    texts = []

    files = list(
        DOCUMENT_DIR.glob("*")
    )

    print(
        f"Found {len(files)} files."
    )

    for path in files:

        if path.suffix.lower() not in [
            ".txt",
            ".pdf"
        ]:
            continue

        print(
            f"Processing: {path.name}"
        )

        text = load_document(path)

        if not text:
            continue

        chunks = chunk_text(text)

        print(
            f"  → {len(chunks)} chunks"
        )

        for chunk_id, chunk in enumerate(chunks):

            texts.append(chunk)

            metadata.append({
                "source": path.name,
                "chunk_id": chunk_id,
                "text": chunk
            })

    if not texts:

        raise RuntimeError(
            "No document text was found."
        )

    print(
        f"\nEmbedding {len(texts)} chunks..."
    )

    embeddings = model.encode(
        texts,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=True
    )

    embeddings = np.asarray(
        embeddings,
        dtype="float32"
    )

    dimension = embeddings.shape[1]

    index = faiss.IndexFlatIP(
        dimension
    )

    index.add(
        embeddings
    )

    faiss.write_index(
        index,
        str(
            INDEX_DIR /
            "documents.faiss"
        )
    )

    with open(
        INDEX_DIR / "metadata.json",
        "w",
        encoding="utf-8"
    ) as handle:

        json.dump(
            metadata,
            handle,
            ensure_ascii=False,
            indent=2
        )

    print("\nIndex complete.")

    print(
        f"Vectors: {index.ntotal}"
    )

    print(
        f"Dimension: {dimension}"
    )

    print(
        f"Metadata entries: {len(metadata)}"
    )


if __name__ == "__main__":
    main()