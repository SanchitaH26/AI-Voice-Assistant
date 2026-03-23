import os, tempfile, requests, time
import numpy as np
import sounddevice as sd
import soundfile as sf
from flask import Flask, render_template, request, jsonify
from transformers import T5ForConditionalGeneration, T5Tokenizer
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# --------------------------------------------------
# 1. Config – Speech-to-Text
# FIXED: Updated to new HF router URL
# Old URL (deprecated): https://api-inference.huggingface.co
# New URL: https://router.huggingface.co
# --------------------------------------------------
HF_API_TOKEN = os.environ.get("HF_TOKEN")
SPEECH_API_URL = "https://router.huggingface.co/hf-inference/models/openai/whisper-large-v3/v1/audio/transcriptions"
headers = {"Authorization": f"Bearer {HF_API_TOKEN}"}

def speech_to_text_hf(audio_bytes: bytes) -> str:
    try:
        files = {
            "file": ("audio.wav", audio_bytes, "audio/wav"),
        }
        data = {
            "model": "openai/whisper-large-v3"
        }
        r = requests.post(
            SPEECH_API_URL,
            headers=headers,
            files=files,
            data=data,
            timeout=60
        )
        print(f"[Whisper API] Status: {r.status_code}, Response: {r.text[:300]}")

        if r.status_code == 200:
            return r.json().get("text", "").strip()

        elif r.status_code == 503:
            try:
                wait = r.json().get("estimated_time", 20)
            except Exception:
                wait = 20
            print(f"[Whisper API] Model loading, waiting {wait}s before retry...")
            time.sleep(wait)
            r = requests.post(
                SPEECH_API_URL,
                headers=headers,
                files=files,
                data=data,
                timeout=60
            )
            print(f"[Whisper API] Retry Status: {r.status_code}, Response: {r.text[:300]}")
            if r.status_code == 200:
                return r.json().get("text", "").strip()

        elif r.status_code == 401:
            print("[Whisper API] Unauthorized — check your HF_TOKEN in .env")

        elif r.status_code == 429:
            print("[Whisper API] Rate limit exceeded — wait before retrying")

    except Exception as e:
        print(f"[Whisper API] Exception: {e}")

    return ""

# --------------------------------------------------
# 2. Answering Model – FLAN-T5 Large
# FIXED: Load model directly instead of using pipeline()
# because newer transformers dropped text2text-generation from pipeline
# --------------------------------------------------
print("[App] Loading FLAN-T5 model (this may take a moment on first run)...")
flan_tokenizer = T5Tokenizer.from_pretrained("google/flan-t5-large")
flan_model = T5ForConditionalGeneration.from_pretrained("google/flan-t5-large")
print("[App] Model loaded successfully.")

def get_answer(question: str) -> str:
    prompt = f"Answer factually and concisely: {question}"
    inputs = flan_tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
    outputs = flan_model.generate(**inputs, max_new_tokens=150)
    return flan_tokenizer.decode(outputs[0], skip_special_tokens=True).strip()

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
        return jsonify({"error": "No audio file provided."}), 400

    audio = request.files["audio"]

    # Save uploaded audio to a temp file and read bytes
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        audio.save(tmp.name)
        with open(tmp.name, "rb") as f:
            audio_bytes = f.read()
    os.unlink(tmp.name)

    # Step 1: Transcribe audio
    text = speech_to_text_hf(audio_bytes)
    if not text:
        return jsonify({"error": "Speech recognition failed. Check terminal logs for details."}), 500

    print(f"[App] Transcribed text: {text}")

    # Step 2: Generate answer
    try:
        answer = get_answer(text)
        print(f"[App] Answer: {answer}")
    except Exception as e:
        print(f"[App] Error generating answer: {e}")
        return jsonify({"error": "Answer generation failed."}), 500

    return jsonify({"question": text, "answer": answer})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)