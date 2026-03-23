
import os, tempfile, requests
import numpy as np
import sounddevice as sd
import soundfile as sf
from flask import Flask, render_template, request, jsonify
from transformers import pipeline

# --------------------------------------------------
# 1. Config – Speech-to-Text
# --------------------------------------------------
HF_API_TOKEN = os.environ.get("HF_TOKEN")
SPEECH_API_URL = "https://api-inference.huggingface.co/models/openai/whisper-large-v3"
headers = {"Authorization": f"Bearer {HF_API_TOKEN}"}

def speech_to_text_hf(audio_bytes: bytes) -> str:
    headers_ct = headers.copy()
    headers_ct["Content-Type"] = "audio/wav"
    r = requests.post(SPEECH_API_URL, headers=headers_ct, data=audio_bytes, timeout=60)
    if r.status_code == 200:
        return r.json().get("text", "").strip()
    return ""

# --------------------------------------------------
# 2. Answering Model – Open & Fact-Friendly
# --------------------------------------------------
# flan-t5-large is fully public, great for Q&A
qa_pipeline = pipeline("text-generation", model="google/flan-t5-large")

def get_answer(question: str) -> str:
    prompt = f"Answer factually and concisely: {question}"
    out = qa_pipeline(prompt, max_new_tokens=150)
    return out[0]["generated_text"].strip()

# --------------------------------------------------
# 3. Flask Web App
# --------------------------------------------------
app = Flask(__name__)

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/ask", methods=["POST"])
def ask():
    if "audio" not in request.files:
        return jsonify({"error": "No audio file."}), 400
    audio = request.files["audio"]
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        audio.save(tmp.name)
        with open(tmp.name, "rb") as f:
            audio_bytes = f.read()
    os.unlink(tmp.name)

    text = speech_to_text_hf(audio_bytes)
    if not text:
        return jsonify({"error": "Speech recognition failed."}), 500

    answer = get_answer(text)
    return jsonify({"question": text, "answer": answer})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)