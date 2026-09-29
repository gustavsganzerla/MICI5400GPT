import os
import json
from pathlib import Path

import faiss
import numpy as np

from sentence_transformers import SentenceTransformer


INDEX_DIR = Path(
    "/app/data/index"
)

INDEX_PATH = (
    INDEX_DIR /
    "documents.faiss"
)

METADATA_PATH = (
    INDEX_DIR /
    "metadata.json"
)


EMBEDDING_MODEL = os.getenv(
    "EMBEDDING_MODEL",
    "sentence-transformers/all-MiniLM-L6-v2"
)

TOP_K = int(
    os.getenv(
        "RAG_TOP_K",
        "4"
    )
)


_embedding_model = None
_index = None
_metadata = None


def load_rag():

    global _embedding_model
    global _index
    global _metadata

    if _embedding_model is None:

        print(
            f"Loading embedding model: "
            f"{EMBEDDING_MODEL}"
        )

        _embedding_model = (
            SentenceTransformer(
                EMBEDDING_MODEL
            )
        )

    if _index is None:

        if not INDEX_PATH.exists():

            raise FileNotFoundError(
                "FAISS index does not exist. "
                "Run build_index.py first."
            )

        _index = faiss.read_index(
            str(INDEX_PATH)
        )

    if _metadata is None:

        with open(
            METADATA_PATH,
            "r",
            encoding="utf-8"
        ) as handle:

            _metadata = json.load(
                handle
            )


def retrieve(
    question,
    k=TOP_K
):

    load_rag()

    query_embedding = (
        _embedding_model.encode(
            [question],
            convert_to_numpy=True,
            normalize_embeddings=True
        )
    )

    query_embedding = np.asarray(
        query_embedding,
        dtype="float32"
    )

    scores, indices = (
        _index.search(
            query_embedding,
            k
        )
    )

    results = []

    for score, index_id in zip(
        scores[0],
        indices[0]
    ):

        if index_id < 0:
            continue

        item = dict(
            _metadata[index_id]
        )

        item["score"] = float(
            score
        )

        results.append(item)

    return results


def build_augmented_prompt(
    question,
    results
):

    evidence_parts = []

    for i, result in enumerate(
        results,
        start=1
    ):

        evidence_parts.append(
            f"""
[Source {i}]
Document: {result["source"]}
Score: {result['score']}
Page: {result.get("page", "unknown")}

{result["text"]}
""".strip()
        )


    evidence = "\n\n".join(
        evidence_parts
    )


    prompt = f"""
Answer the question using the evidence provided below.

If the answer is explicitly stated or can be directly supported by the evidence, answer the question and cite the relevant source as [Source 1], [Source 2], etc.

Only say "The provided evidence is insufficient to answer this question" if the evidence does not contain information that answers the question.

Keep the answer concise and complete, preferably under 150 words.

EVIDENCE:

{evidence}

QUESTION:
{question}

ANSWER:
""".strip()


    return prompt