# Vera AI Composer - Magicpin Challenge Submission

![Python](https://img.shields.io/badge/Python-3.13-blue.svg)
![FastAPI](https://img.shields.io/badge/FastAPI-0.110.0-green.svg)
![LLM](https://img.shields.io/badge/LLM-Llama_3.3_(Groq)-orange.svg)

This is my submission for the Magicpin AI Challenge. The goal was to build a robust, high-speed AI backend for Vera that handles merchant engagement without hallucinations or timing out.

Vera is designed with a **Hybrid Cloud/Local architecture** for maximum reliability and speed.

---

## 🛠️ How I Solved the Challenge Constraints

### 1. Hybrid Cloud/Local Architecture (Uptime Guarantee)
Vera uses a multi-provider strategy for maximum reliability:
* **Primary Engine:** Llama 3.3 70B (via Groq Cloud) for lightning-fast, high-reasoning responses (~1s per batch).
* **Backup Engine:** Automated local fallback to Llama 3.2 3B (Ollama) if the cloud API is unreachable. This ensures the bot never stays silent even if the internet flickers.

### 2. Concurrent Trigger Processing (Bypassing Timeouts)
The judge simulator is strict about 15-45s timeouts.
* **Fix:** I wrapped the `POST /tick` logic in a `ThreadPoolExecutor`. Vera processes all active merchant triggers in parallel, bringing batch generation time down from 60+ seconds to **under 2 seconds**.

### 3. Prompt Optimization (Avoiding Context Bloat)
LLMs struggle when you dump raw JSON payloads into the prompt.
* **Fix:** Vera intercepts the `POST /context` payload and distills it into a concise, human-readable string map. This reduces token overhead by over 60%, speeding up response times and improving accuracy scores.

### 4. Deterministic State Machine (Passing Intent Tests)
I set the `temperature` to `0.0` and built a strict logic filter for the `POST /reply` endpoint. When a merchant commits (e.g., "let's do it"), the bot drops the advisory talk and switches to **ACTION mode**, correctly closing the conversation turn.

---

## 🚀 Setup & Prerequisites

### 1. Requirements
*   **Python 3.13+**
*   **Groq API Key** (Free from [console.groq.com](https://console.groq.com/))
*   **Ollama** (Optional fallback) with `llama3.2` model.

### 2. Installation
```bash
# Clone and install
pip install -r requirements.txt

# Create .env file with your keys
# LLM_PROVIDER=groq
# LLM_API_KEY=your_groq_key_here
# LLM_MODEL=llama-3.3-70b-versatile
```

### 3. Running the Bot
```bash
python bot.py
```

---

## 📡 API Endpoints

* `GET /healthz` - Liveness probe.
* `GET /metadata` - Team & Model info.
* `POST /context` - Ingests merchant/category data.
* `POST /tick` - Concurrently processes active triggers.
* `POST /reply` - Handles active turns with intent enforcement.

---

## 🧪 Evaluation Results

Averaging **38-42/50 (~76% - 84%)** on the full judge evaluation.
- [x] Zero timeouts (due to Async/Groq).
- [x] Correct intent transitions.
- [x] Proper hostile/auto-reply detection.
- [x] Natural Hindi-English (Hinglish) code-mixing.
