# Vera AI Composer - Magicpin Challenge Submission

![Python](https://img.shields.io/badge/Python-3.13-blue.svg)
![FastAPI](https://img.shields.io/badge/FastAPI-0.110.0-green.svg)
![LLM](https://img.shields.io/badge/LLM-Llama_3.2_(Ollama)-orange.svg)

This is my submission for the Magicpin AI Challenge. The goal of this project was to build a reliable FastAPI backend for Vera that doesn't hallucinate or timeout under load. It manages merchant engagement, handles triggers (like review drops or profile views), and shoots out personalized messages.

The main focus here was passing the strict automated judge constraints without relying on heavy cloud GPUs. Everything runs locally and fast.

---

## 🛠️ How I Solved the Challenge Constraints

The judge simulator is strict. Here is how I got the app to pass the timeout and intent checks:

### 1. Concurrent Trigger Processing (Bypassing Timeouts)
A standard sequential loop over the `POST /tick` endpoint fails the 15-45s hackathon constraints because LLM generation is slow. 
* **Fix:** I wrapped the trigger logic in a `ThreadPoolExecutor` so it processes all 5 active triggers concurrently instead of one by one. This brought batch generation time down from 60+ seconds to ~20 seconds, safely under the limit.

### 2. Prompt Optimization (Avoiding massive JSON dumps)
LLMs struggle when you just dump raw JSON payloads into the prompt—it bloats the context window and makes inference super slow.
* **Fix:** Instead of injecting the raw dicts, Vera intercepts the `POST /context` payload and distills it into a concise, human-readable string map. Less tokens = much faster response times and better specificity scores.

### 3. Deterministic State Machine (Fixing the "Intent" test)
LLMs often get stuck asking qualifying questions (e.g., *"Would you like to proceed?"*) even after a merchant commits.
* **Fix:** I set the Ollama `temperature` to `0.0` to make it deterministic. Then, I updated the `POST /reply` endpoint with strict rules. When a user indicates intent (e.g., *"let's do it"*), the bot drops the advisory talk and switches to **ACTION mode** (e.g., "Done.", "Next step:"), passing the judge's intent test.

### 4. Hindi/English Code-Mixing
I added logic to check the `languages` array in the merchant schema. If `"hi"` is present, the prompt forces a natural Hinglish code-mix, which fits real-world Indian business communication better than pure English.

---

## 🚀 Getting Started

### Prerequisites
1. **Python 3.13+**
2. **Ollama** running locally with the `llama3.2` model.
   ```bash
   ollama run llama3.2
   ```

### Setup
1. Clone the repo.
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

### Running it
Start the FastAPI server (runs on `http://localhost:8000`).
```bash
python bot.py
```

---

## 📡 API Endpoints

* `GET /healthz` - Liveness probe.
* `GET /metadata` - Returns team info and model config.
* `POST /context` - Ingests merchant/category data.
* `POST /tick` - Concurrently processes active triggers and returns generated messages.
* `POST /reply` - Handles active conversation turns, enforcing intent transition, hostile opt-outs, and auto-reply detection.

---

## 🧪 Evaluation Results

Tested locally with `judge_simulator.py`. Averaging 38-42/50 (~76% - 84%) on the full eval mode, and correctly handling all edge cases:
- Passes the Warmup check.
- Passes the Auto-reply detection.
- Passes Hostile user opt-outs.
- Passes Intent transitions.
- Zero timeouts.
