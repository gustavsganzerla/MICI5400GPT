import os
import json
import re
from pathlib import Path

import faiss
import numpy as np

from pypdf import PdfReader
from sentence_transformers import SentenceTransformer


# ============================================================
# CONFIGURATION
# ============================================================

DOCUMENT_DIR = Path(
    "/app/data/documents"
)

INDEX_DIR = Path(
    "/app/data/index"
)


EMBEDDING_MODEL = os.getenv(
    "EMBEDDING_MODEL",
    "sentence-transformers/all-MiniLM-L6-v2"
)


# Number of WORDS, not tokens.
#
# 100 is a good compromise:
# - small enough for specific factual retrieval
# - large enough to preserve context

CHUNK_SIZE = 120

CHUNK_OVERLAP = 30


# Avoid tiny fragments.
MIN_CHUNK_WORDS = 20


# ============================================================
# TEXT CLEANING
# ============================================================

def clean_text(text):
    """
    Basic cleanup of text extracted from PDFs.
    """

    if not text:
        return ""

    # Remove null characters
    text = text.replace(
        "\x00",
        " "
    )

    # Fix hyphenated words split across lines:
    #
    # trans-
    # mission
    #
    # becomes:
    #
    # transmission

    text = re.sub(
        r"(\w)-\s*\n\s*(\w)",
        r"\1\2",
        text
    )

    # Convert line breaks to spaces.
    #
    # This avoids arbitrary PDF line wrapping
    # becoming part of the embeddings.

    text = re.sub(
        r"\s*\n\s*",
        " ",
        text
    )

    # Collapse repeated whitespace

    text = re.sub(
        r"\s+",
        " ",
        text
    )

    return text.strip()


# ============================================================
# REFERENCES
# ============================================================

def find_references_page(pages):
    """
    Try to identify the page where the References section begins.

    We deliberately use a conservative heuristic.

    A page is considered a References page only when a standalone
    References/Bibliography heading appears near the beginning
    of the page.
    """

    pattern = re.compile(
        r"^\s*(references|bibliography)\s*$",
        re.IGNORECASE
    )


    for page_index, page_text in enumerate(pages):

        if not page_text:
            continue


        lines = [
            line.strip()
            for line in page_text.splitlines()
            if line.strip()
        ]


        # Only inspect the first ~15 lines.
        #
        # This prevents citations containing the word
        # "references" from triggering removal.

        for line in lines[:15]:

            if pattern.match(line):

                return page_index


    return None


# ============================================================
# PDF READING
# ============================================================

def read_pdf(path):
    """
    Extract text page-by-page.

    Returns a list of dictionaries:

    {
        "page": 1,
        "text": "..."
    }

    Page information is retained so retrieved chunks
    can later identify their source page.
    """

    reader = PdfReader(
        path
    )


    raw_pages = []


    for page in reader.pages:

        try:

            page_text = (
                page.extract_text()
                or ""
            )

        except Exception:

            page_text = ""


        raw_pages.append(
            page_text
        )


    # --------------------------------------------------------
    # Detect references
    # --------------------------------------------------------

    references_page = (
        find_references_page(
            raw_pages
        )
    )


    if references_page is not None:

        print(
            f"  References detected at "
            f"page {references_page + 1}"
        )


        raw_pages = raw_pages[
            :references_page
        ]


    else:

        print(
            "  References section not confidently detected"
        )


    # --------------------------------------------------------
    # Clean pages
    # --------------------------------------------------------

    pages = []


    for page_number, page_text in enumerate(
        raw_pages,
        start=1
    ):

        cleaned = clean_text(
            page_text
        )


        if not cleaned:

            continue


        pages.append({

            "page":
                page_number,

            "text":
                cleaned

        })


    return pages


# ============================================================
# TXT READING
# ============================================================

def read_txt(path):

    text = path.read_text(
        encoding="utf-8",
        errors="ignore"
    )


    return [{

        "page":
            None,

        "text":
            clean_text(text)

    }]


# ============================================================
# DOCUMENT LOADER
# ============================================================

def load_document(path):

    suffix = (
        path.suffix.lower()
    )


    if suffix == ".pdf":

        return read_pdf(
            path
        )


    if suffix == ".txt":

        return read_txt(
            path
        )


    return []


# ============================================================
# CHUNKING
# ============================================================

def chunk_text(
    text,
    chunk_size=CHUNK_SIZE,
    overlap=CHUNK_OVERLAP
):
    """
    Split text into overlapping word windows.

    Example:

    chunk_size = 100
    overlap = 20

    Chunk 1:
        words 0-99

    Chunk 2:
        words 80-179

    Chunk 3:
        words 160-259
    """

    words = (
        text.split()
    )


    if not words:

        return []


    chunks = []


    step = (
        chunk_size
        - overlap
    )


    start = 0


    while start < len(words):

        end = (
            start
            + chunk_size
        )


        chunk_words = (
            words[start:end]
        )


        # Don't index extremely small
        # trailing fragments.

        if (
            len(chunk_words)
            >= MIN_CHUNK_WORDS
        ):

            chunk = " ".join(
                chunk_words
            )


            chunks.append(
                chunk
            )


        start += step


    return chunks


# ============================================================
# BASIC CHUNK VALIDATION
# ============================================================

def valid_chunk(text):
    """
    Conservative filtering.

    The goal is NOT to decide whether the text is scientifically
    important.

    We only remove chunks that are clearly unusable.
    """

    if not text:

        return False


    words = (
        text.split()
    )


    if (
        len(words)
        < MIN_CHUNK_WORDS
    ):

        return False


    # --------------------------------------------------------
    # Alphabetic content
    # --------------------------------------------------------

    alphabetic_chars = sum(
        character.isalpha()
        for character in text
    )


    total_chars = max(
        len(text),
        1
    )


    alphabetic_ratio = (
        alphabetic_chars
        /
        total_chars
    )


    # Very low alphabetic content usually means:
    #
    # tables
    # formatting artefacts
    # numerical garbage

    if alphabetic_ratio < 0.40:

        return False


    return True


# ============================================================
# MAIN
# ============================================================

def main():

    print(
        "\n======================================"
    )

    print(
        "Building RAG index"
    )

    print(
        "======================================\n"
    )


    print(
        f"Embedding model: "
        f"{EMBEDDING_MODEL}"
    )


    print(
        f"Chunk size: "
        f"{CHUNK_SIZE} words"
    )


    print(
        f"Chunk overlap: "
        f"{CHUNK_OVERLAP} words"
    )


    print()


    # --------------------------------------------------------
    # Create output directory
    # --------------------------------------------------------

    INDEX_DIR.mkdir(
        parents=True,
        exist_ok=True
    )


    # --------------------------------------------------------
    # Load embedding model
    # --------------------------------------------------------

    print(
        "Loading embedding model..."
    )


    model = (
        SentenceTransformer(
            EMBEDDING_MODEL
        )
    )


    # --------------------------------------------------------
    # Containers
    # --------------------------------------------------------

    metadata = []

    texts = []


    # --------------------------------------------------------
    # Find documents
    # --------------------------------------------------------

    files = sorted(
        DOCUMENT_DIR.glob("*")
    )


    supported_files = [

        path

        for path in files

        if path.suffix.lower()
        in {
            ".pdf",
            ".txt"
        }

    ]


    print(
        f"\nFound "
        f"{len(supported_files)} "
        f"supported documents.\n"
    )


    # --------------------------------------------------------
    # Process documents
    # --------------------------------------------------------

    global_chunk_id = 0


    for path in supported_files:

        print(
            f"Processing: "
            f"{path.name}"
        )


        pages = (
            load_document(
                path
            )
        )


        if not pages:

            print(
                "  No usable text."
            )

            continue


        document_chunk_count = 0


        # ----------------------------------------------------
        # Chunk page-by-page
        # ----------------------------------------------------

        for page_data in pages:

            page_number = (
                page_data["page"]
            )


            page_text = (
                page_data["text"]
            )


            chunks = chunk_text(
                page_text
            )


            for page_chunk_id, chunk in enumerate(
                chunks
            ):

                if not valid_chunk(
                    chunk
                ):

                    continue


                texts.append(
                    chunk
                )


                metadata.append({

                    "source":
                        path.name,

                    "page":
                        page_number,

                    "chunk_id":
                        global_chunk_id,

                    "page_chunk_id":
                        page_chunk_id,

                    "text":
                        chunk

                })


                global_chunk_id += 1

                document_chunk_count += 1


        print(
            f"  → "
            f"{document_chunk_count} "
            f"chunks retained\n"
        )


    # --------------------------------------------------------
    # Validate
    # --------------------------------------------------------

    if not texts:

        raise RuntimeError(
            "No usable document text was found."
        )


    # --------------------------------------------------------
    # Generate embeddings
    # --------------------------------------------------------

    print(
        f"Embedding "
        f"{len(texts)} chunks...\n"
    )


    embeddings = (
        model.encode(

            texts,

            convert_to_numpy=True,

            normalize_embeddings=True,

            show_progress_bar=True

        )
    )


    embeddings = np.asarray(
        embeddings,
        dtype="float32"
    )


    # --------------------------------------------------------
    # FAISS
    # --------------------------------------------------------

    dimension = (
        embeddings.shape[1]
    )


    # Because embeddings are normalized:
    #
    # inner product == cosine similarity

    index = (
        faiss.IndexFlatIP(
            dimension
        )
    )


    index.add(
        embeddings
    )


    # --------------------------------------------------------
    # Save FAISS
    # --------------------------------------------------------

    faiss_path = (
        INDEX_DIR
        /
        "documents.faiss"
    )


    faiss.write_index(
        index,
        str(faiss_path)
    )


    # --------------------------------------------------------
    # Save metadata
    # --------------------------------------------------------

    metadata_path = (
        INDEX_DIR
        /
        "metadata.json"
    )


    with open(
        metadata_path,
        "w",
        encoding="utf-8"
    ) as handle:

        json.dump(

            metadata,

            handle,

            ensure_ascii=False,

            indent=2

        )


    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    print(
        "\n======================================"
    )

    print(
        "Index complete"
    )

    print(
        "======================================"
    )


    print(
        f"Documents: "
        f"{len(supported_files)}"
    )


    print(
        f"Vectors: "
        f"{index.ntotal}"
    )


    print(
        f"Dimension: "
        f"{dimension}"
    )


    print(
        f"Metadata entries: "
        f"{len(metadata)}"
    )


    print(
        f"FAISS index: "
        f"{faiss_path}"
    )


    print(
        f"Metadata: "
        f"{metadata_path}"
    )


    print(
        "======================================\n"
    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    main()