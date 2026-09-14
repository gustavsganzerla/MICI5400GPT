import os
import requests

from flask import Flask, jsonify, render_template, request

from rag import retrieve, build_augmented_prompt

app = Flask(__name__)


OLLAMA_URL = os.getenv(
    "OLLAMA_URL",
    "http://ollama:11434"
)

OLLAMA_MODEL = os.getenv(
    "OLLAMA_MODEL",
    "qwen3:1.7b"
)


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/health")
def health():

    ollama_status = False

    try:
        response = requests.get(
            f"{OLLAMA_URL}/api/tags",
            timeout=5
        )

        ollama_status = response.ok

    except requests.RequestException:
        pass

    return jsonify({
        "flask": True,
        "ollama": ollama_status,
        "model": OLLAMA_MODEL
    })


@app.route("/ask", methods=["POST"])
def ask():

    data = request.get_json()

    if not data:
        return jsonify({
            "error":
            "No JSON body provided"
        }), 400


    question = (
        data
        .get("question", "")
        .strip()
    )

    use_rag = bool(
        data.get(
            "use_rag",
            False
        )
    )

    verbose_rag = bool(
        data.get(
            "verbose_rag",
            False
        )
    )


    if not question:

        return jsonify({
            "error":
            "Question is required"
        }), 400


    retrieved = []

    augmented_prompt = None


    #
    # BASELINE MODEL
    #

    if not use_rag:

        prompt = question


    #
    # RAG MODEL
    #

    else:

        try:

            retrieved = retrieve(
                question
            )

        except Exception as exc:

            return jsonify({
                "error":
                f"RAG retrieval failed: {exc}"
            }), 500


        augmented_prompt = (
            build_augmented_prompt(
                question,
                retrieved
            )
        )

        prompt = augmented_prompt


    payload = {

        "model":
            OLLAMA_MODEL,

        "prompt":
            prompt,

        "stream":
            False,

        # Qwen models that support thinking
        # can skip the additional reasoning pass.
        "think":
            False,

        "options": {

            "num_predict":
                250,

            "temperature":
                0.2,

            "num_ctx":
                4096
        }
    }


    try:

        response = requests.post(

            f"{OLLAMA_URL}/api/generate",

            json=payload,

            timeout=300
        )

        response.raise_for_status()

    except requests.RequestException as exc:

        return jsonify({
            "error":
                str(exc)
        }), 500


    result = response.json()

    answer = result.get(
        "response",
        ""
    )


    output = {

        "question":
            question,

        "answer":
            answer,

        "rag_used":
            use_rag
    }


    #
    # Sources can be shown for RAG answers
    #

    if use_rag:

        output["sources"] = [

            {
                "source":
                    item["source"],

                "chunk_id":
                    item["chunk_id"],

                "score":
                    item["score"]
            }

            for item in retrieved
        ]


    #
    # Verbose teaching mode
    #

    if (
        use_rag and
        verbose_rag
    ):

        output["rag_debug"] = {

            "retrieved_chunks":
                retrieved,

            "augmented_prompt":
                augmented_prompt
        }


    return jsonify(output)


if __name__ == "__main__":
    app.run(
        host="0.0.0.0",
        port=5000,
        debug=True
    )