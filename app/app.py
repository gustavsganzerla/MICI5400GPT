import os
import requests

from flask import Flask, jsonify, render_template, request


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
            "error": "No JSON body provided"
        }), 400

    question = data.get("question")

    if not question:
        return jsonify({
            "error": "Question is required"
        }), 400

    payload = {
        "model": OLLAMA_MODEL,
        "prompt": question,
        "stream": False,
        "options": {
            "num_predict": 300,
            "temperature": 0.2
        }
    }

    try:

        response = requests.post(
            f"{OLLAMA_URL}/api/generate",
            json=payload,
            timeout=180
        )

        response.raise_for_status()

    except requests.RequestException as exc:

        return jsonify({
            "error": str(exc)
        }), 500

    result = response.json()

    return jsonify({
        "question": question,
        "answer": result.get("response", "")
    })


if __name__ == "__main__":
    app.run(
        host="0.0.0.0",
        port=5000,
        debug=True
    )