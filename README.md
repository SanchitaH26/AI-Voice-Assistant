# 🎙️ AI Voice Chat Assistant

A Flask-based web application that accepts voice input, transcribes it using OpenAI's Whisper model via the Hugging Face Inference API, and generates a factual answer using Google's FLAN-T5 language model.

---

## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Tech Stack](#tech-stack)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Configuration](#configuration)
- [Running the App](#running-the-app)
- [API Endpoints](#api-endpoints)
- [How It Works](#how-it-works)
- [Disclaimer](#disclaimer)

---

## Overview

The user records or uploads a `.wav` audio file through the web interface. The audio is sent to the Hugging Face Inference API for transcription using Whisper Large V3. The transcribed text is then passed to a locally loaded FLAN-T5 model which generates a concise, factual answer — all returned to the browser as JSON.

---

## Features

- 🎤 Voice-to-text transcription via **Whisper Large V3** (Hugging Face API)
- 🤖 Factual Q&A powered by **Google FLAN-T5 Large** (local inference)
- 🌐 Lightweight **Flask** web server
- 🔐 Secure token loading via environment variables (no hardcoded secrets)
- 📦 Simple REST API endpoint for audio upload and response

---

## Tech Stack

| Component | Technology |
|---|---|
| Web Framework | Flask |
| Speech-to-Text | Whisper Large V3 (Hugging Face Inference API) |
| Question Answering | google/flan-t5-large (local via Transformers) |
| Audio Handling | sounddevice, soundfile |
| HTTP Requests | requests |
| ML Framework | Hugging Face Transformers |

---

## Project Structure

```
AI-Voice-Assistant/
│
├── app.py                  # Main Flask application
├── templates/
│   └── index.html          # Frontend UI
├── .env                    # Environment variables (never commit this)
├── .gitignore              # Should include .env
└── README.md               # This file
```

---

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/SanchitaH26/AI-Voice-Assistant.git
cd AI-Voice-Assistant
```

### 2. Create and activate a virtual environment

```bash
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # macOS/Linux
```

### 3. Install dependencies

```bash
pip install flask transformers requests numpy sounddevice soundfile torch python-dotenv
```

### requirements.txt (reference)

```
flask
transformers
requests
numpy
sounddevice
soundfile
torch
python-dotenv
```

---

## Configuration

### Setting up your Hugging Face API Token

The app requires a Hugging Face User Access Token to call the Whisper API.

**Step 1 — Get your token:**
Go to [huggingface.co → Settings → Access Tokens](https://huggingface.co/settings/tokens) and create a token with **read** access.

**Step 2 — Create a `.env` file** in the project root:

```
HF_TOKEN=hf_your_token_here
```

**Step 3 — Make sure `.env` is in your `.gitignore`:**

```
.env
```

> ⚠️ **Never hardcode your token directly in the source code or commit your `.env` file to GitHub.** GitHub's secret scanning will block the push and your token will be compromised.

**Step 4 — Load the token in your app** (already handled in `app.py`):

```python
import os
HF_API_TOKEN = os.environ.get("HF_TOKEN")
```

To auto-load from `.env` during development, add this at the top of `app.py`:

```python
from dotenv import load_dotenv
load_dotenv()
```

---

## Running the App

```bash
python voice_chat_assistant.py
```

The app will start at `http://localhost:5000`.

> **Note:** The first run will download `google/flan-t5-large` (~3 GB). Subsequent runs load from cache.

---

## API Endpoints

### `GET /`
Serves the main web interface (`index.html`).

---

### `POST /ask`
Accepts a `.wav` audio file, transcribes it, and returns a generated answer.

**Request:**
- Content-Type: `multipart/form-data`
- Field: `audio` — a `.wav` audio file

**Success Response (`200`):**
```json
{
  "question": "What is the capital of France?",
  "answer": "Paris"
}
```

**Error Responses:**
```json
{ "error": "No audio file." }             // 400 - missing file
{ "error": "Speech recognition failed." } // 500 - Whisper API failed
```

---

## How It Works

```
User speaks / uploads audio
        ↓
  .wav file sent to /ask
        ↓
  Whisper Large V3 (HF API)
  transcribes audio → text
        ↓
  FLAN-T5 Large (local)
  generates answer from text
        ↓
  JSON response returned
  { question, answer }
```

### Speech-to-Text
Audio bytes are posted directly to the Hugging Face Inference API endpoint for `openai/whisper-large-v3` with the `audio/wav` content type. The API returns the transcribed text.

### Question Answering
The transcribed question is wrapped in a prompt — `"Answer factually and concisely: {question}"` — and passed to the locally running FLAN-T5 Large model via the Hugging Face `pipeline`, which returns a generated text response.

---

## Disclaimer

> This app uses the Hugging Face **free** Inference API for Whisper, which may have rate limits and cold-start delays. For production use, consider hosting Whisper locally or upgrading to a paid Inference Endpoint.
