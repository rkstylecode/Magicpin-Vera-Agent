#!/usr/bin/env python3
"""
Vera AI Composer v2.0 — magicpin Build Vera Better Challenge
=============================================================
A stateful, deterministic AI agent that composes high-compulsion
merchant messages by ranking signals from 4 context layers.

Architecture:
  FastAPI → Context Store → Signal Ranker → Prompt Composer → LLM → JSON Validator

Key Design Decisions:
- In-memory state (sufficient for challenge duration)
- Deterministic: temperature=0, no randomness
- Multi-provider LLM support (Ollama/Gemini/Groq)
- Rule-based fast paths for auto-reply/hostile detection
- Template fallback if LLM fails
- JSON schema validation on all outputs

Author: Ritik
"""

import os
import json
import re
import time
import hashlib
import logging
import traceback
from datetime import datetime, timezone
from typing import Optional
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests as http_requests
import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

# ============================================================
#                     CONFIGURATION
# ============================================================

LLM_PROVIDER = os.getenv("LLM_PROVIDER", "ollama")
LLM_API_KEY = os.getenv("LLM_API_KEY", "")
LLM_MODEL = os.getenv("LLM_MODEL", "llama3.2")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434")
PORT = int(os.getenv("PORT", "8000"))
TEMPERATURE = 0.0

# ============================================================
#                     LOGGING
# ============================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S"
)
log = logging.getLogger("vera")

# ============================================================
#                     APP + STATE
# ============================================================

app = FastAPI(title="Vera AI Composer", version="2.0.0")
START_TIME = time.time()

context_store = {
    "category": {},
    "merchant": {},
    "customer": {},
    "trigger": {},
}

conversation_state = {}

# ============================================================
#                    LLM INTERFACE
# ============================================================

def call_llm(prompt: str, system_prompt: str = "", retries: int = 2) -> str:
    """Unified LLM caller with retry logic."""
    for attempt in range(retries + 1):
        try:
            if LLM_PROVIDER == "ollama":
                return _call_ollama(prompt, system_prompt)
            elif LLM_PROVIDER == "gemini":
                return _call_gemini(prompt, system_prompt)
            elif LLM_PROVIDER == "groq":
                return _call_groq(prompt, system_prompt)
            else:
                log.error(f"Unknown LLM provider: {LLM_PROVIDER}")
                return ""
        except Exception as e:
            log.warning(f"LLM call attempt {attempt+1} failed: {e}")
            if attempt < retries:
                time.sleep(1 * (attempt + 1))
    return ""


def _call_ollama(prompt: str, system_prompt: str) -> str:
    full_prompt = f"{system_prompt}\n\n{prompt}" if system_prompt else prompt
    resp = http_requests.post(
        f"{OLLAMA_BASE_URL}/api/generate",
        json={
            "model": LLM_MODEL,
            "prompt": full_prompt,
            "stream": False,
            "format": "json",
            "options": {"temperature": TEMPERATURE, "num_predict": 400}
        },
        timeout=120
    )
    resp.raise_for_status()
    return resp.json().get("response", "")


def _call_gemini(prompt: str, system_prompt: str) -> str:
    full_prompt = f"{system_prompt}\n\n{prompt}" if system_prompt else prompt
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{LLM_MODEL}:generateContent?key={LLM_API_KEY}"
    resp = http_requests.post(url, json={
        "contents": [{"parts": [{"text": full_prompt}]}],
        "generationConfig": {
            "temperature": TEMPERATURE,
            "maxOutputTokens": 400,
            "responseMimeType": "application/json"
        }
    }, timeout=30)
    resp.raise_for_status()
    return resp.json()["candidates"][0]["content"]["parts"][0]["text"]


def _call_groq(prompt: str, system_prompt: str) -> str:
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})
    resp = http_requests.post(
        "https://api.groq.com/openai/v1/chat/completions",
        headers={"Authorization": f"Bearer {LLM_API_KEY}", "Content-Type": "application/json"},
        json={
            "model": LLM_MODEL, "messages": messages,
            "temperature": TEMPERATURE, "max_tokens": 400,
            "response_format": {"type": "json_object"}
        },
        timeout=30
    )
    resp.raise_for_status()
    return resp.json()["choices"][0]["message"]["content"]


# ============================================================
#              PROMPT ENGINEERING
# ============================================================

SYSTEM_PROMPT = """You are Vera, magicpin's AI assistant for merchant growth on WhatsApp.
You compose sharp, grounded, high-compulsion messages.

CORE PRINCIPLES:
1. GROUNDING: Only use facts from the provided context. NEVER fabricate statistics, dates, offers, competitor names, or research citations.
2. SPECIFICITY: Anchor on verifiable facts — exact numbers, ₹ prices, percentages, dates, source citations (e.g., "JIDA Oct 2026 p.14"). "X% off" is generic; "Haircut @ ₹99" is specific.
3. CATEGORY VOICE: Match the business type's tone:
   - Dentists/Doctors: clinical-peer tone, technical terms welcome, use "Dr." prefix, NO overclaims ("cure", "guaranteed")
   - Salons: warm-practical, emoji OK, aspirational but grounded
   - Restaurants: operator-to-operator ("covers", "AOV", "delivery radius")
   - Gyms: coach voice, motivational but data-backed
   - Pharmacies: trustworthy-precise, regulatory-aware, senior-friendly
4. MERCHANT FIT: Address by owner's first name. Reference their locality, business name, specific metrics, and active offers.
5. SINGLE CTA: One clear, low-friction call-to-action in the last sentence. Binary YES/STOP for action triggers. Open-ended for knowledge triggers.
6. LANGUAGE: If the merchant's languages include "hi", use natural Hindi-English code-mix (e.g., "Meera, aapke profile pe 2,410 views aaye hain pichle 30 din mein"). This is CRITICAL for scoring.
7. BREVITY: Keep concise. No preambles ("I hope you're doing well..."). No re-introductions.
8. COMPULSION LEVERS: Use 1-2 of: specificity, loss aversion, social proof, effort externalization, curiosity, reciprocity.
9. HUMAN-READABLE: NEVER expose raw internal metric names (like "7d_views_pct", "ctr_pct", "delta_7d"). Convert them to natural language: "views grew 18% this week", "your click-through rate is 2.1%", "calls dropped 50% in 7 days".
10. DO NOT copy example messages from case studies verbatim. Write original copy.

You ALWAYS respond with valid JSON only. No markdown, no explanation outside JSON."""


def build_compose_prompt(merchant_ctx: dict, category_ctx: dict, trigger_ctx: dict, customer_ctx: dict = None) -> str:
    """Build the compose prompt from all context layers."""

    m_payload = merchant_ctx.get("payload", {})
    c_payload = category_ctx.get("payload", {})
    t_payload = trigger_ctx.get("payload", {})

    # Extract key merchant info for explicit grounding
    # The merchant payload IS the full merchant object (identity, performance, etc.)
    identity = m_payload.get("identity", {})
    owner_name = identity.get("owner_first_name", identity.get("owner_name", identity.get("name", "")))
    business_name = identity.get("name", identity.get("business_name", ""))
    locality = identity.get("locality", "")
    city = identity.get("city", "")
    language_pref = identity.get("languages", identity.get("language_pref", ["en"]))

    performance = m_payload.get("performance", {})
    offers = m_payload.get("offers", [])
    signals = m_payload.get("signals", [])
    customer_agg = m_payload.get("customer_aggregate", {})
    review_themes = m_payload.get("review_themes", [])

    # Category voice
    voice = c_payload.get("voice", {})
    category_name = c_payload.get("slug", category_ctx.get("context_id", "general"))

    # Trigger info — fields may be at payload root (since full trigger dict = payload)
    trigger_kind = t_payload.get("kind", t_payload.get("type", "general"))
    trigger_scope = t_payload.get("scope", "merchant")
    trigger_urgency = t_payload.get("urgency", 1)

    # Build a condensed, efficient prompt with only the relevant data
    # (instead of dumping full JSON which wastes tokens)
    
    # Format active offers
    active_offers = [o.get("title", "") for o in offers if o.get("status") == "active"]
    offers_str = ", ".join(active_offers) if active_offers else "none"
    
    # Format performance
    perf_str = ""
    if performance:
        perf_parts = []
        if performance.get("views"): perf_parts.append(f"views={performance['views']}")
        if performance.get("calls"): perf_parts.append(f"calls={performance['calls']}")
        if performance.get("directions"): perf_parts.append(f"directions={performance['directions']}")
        if performance.get("ctr"): perf_parts.append(f"CTR={performance['ctr']}")
        if performance.get("leads"): perf_parts.append(f"leads={performance['leads']}")
        delta = performance.get("delta_7d", {})
        if delta:
            for k, v in delta.items():
                perf_parts.append(f"7d_{k}={v:+.0%}" if isinstance(v, float) else f"7d_{k}={v}")
        perf_str = "; ".join(perf_parts)
    
    # Format customer aggregate
    cust_str = ", ".join(f"{k}={v}" for k, v in customer_agg.items()) if customer_agg else "none"
    
    # Format signals
    signals_str = ", ".join(str(s) for s in signals) if signals else "none"
    
    # Format review themes
    themes_str = "; ".join(
        f"{t.get('theme')}({t.get('sentiment')}, {t.get('occurrences_30d')}x)"
        for t in review_themes
    ) if review_themes else "none"
    
    # Voice guidelines (condensed)
    voice_tone = voice.get("tone", "professional")
    voice_taboo = voice.get("vocab_taboo", voice.get("taboos", []))
    
    # Trigger payload condensed
    trigger_payload_str = json.dumps(t_payload.get("payload", t_payload), default=str)
    if len(trigger_payload_str) > 800:
        trigger_payload_str = trigger_payload_str[:800]
    
    prompt = f"""COMPOSE MESSAGE

CATEGORY: {category_name}
Voice: {voice_tone}. Taboos: {voice_taboo[:5]}.

MERCHANT: {business_name} ({owner_name}), {locality} {city}
Language: {language_pref}
Performance: {perf_str}
Active offers: {offers_str}
Customers: {cust_str}
Signals: {signals_str}
Reviews: {themes_str}

TRIGGER ({trigger_kind}, urgency={trigger_urgency}):
{trigger_payload_str}
"""

    if customer_ctx:
        cu_payload = customer_ctx.get("payload", {})
        cu_str = json.dumps(cu_payload, default=str)
        if len(cu_str) > 600:
            cu_str = cu_str[:600]
        prompt += f"\nCUSTOMER: {cu_str}\n"
        send_as = "merchant_on_behalf"
    else:
        send_as = "vera"

    prompt += f"""\nTASK: Write a short WhatsApp message for {owner_name or 'merchant'}.
- Use {category_name} voice. Include real data from above.
- ONE low-friction CTA at end. No preambles.

JSON only:
{{"body": "<msg>", "cta": "open_ended", "send_as": "{send_as}", "suppression_key": "{trigger_kind}:{category_name}:{datetime.now(timezone.utc).strftime('%Y-W%V')}", "rationale": "<1-line>"}}"""

    return prompt


def build_reply_prompt(message: str, turn_number: int, conv_id: str) -> str:
    """Build prompt for handling merchant replies."""
    return f"""A merchant replied to Vera's message. Analyze their intent and respond.

CONVERSATION: {conv_id}
TURN: {turn_number}
MERCHANT SAID: "{message}"

RULES:
- YES/agree/proceed/commit ("ok", "lets do it", "sounds good") → action:"send", respond with a CONCRETE NEXT STEP. Your response MUST begin with the word "Done," or "Sending," or "Proceeding," or "Next,". Do NOT ask qualifying questions.
- QUESTION → action:"send", answer concisely + include a CTA
- NO/decline/not interested → action:"end", close gracefully, respect their decision
- AUTO-REPLY (out of office, automated, vacation) → action:"end"
- HOSTILE (angry, rude, spam, stop) → action:"end", acknowledge + offer to pause
- Turn >= 4 → action:"end", wrap up politely
- Keep responses concise. Use Hindi-English mix if appropriate.

IMPORTANT: When the merchant commits, switch to ACTION mode — tell them what you're doing, not what you could do.

JSON only:
{{"action": "send" or "wait" or "end", "body": "<response>", "rationale": "<1-line>"}}"""


# ============================================================
#              OUTPUT VALIDATION
# ============================================================

def parse_llm_json(raw: str, required_fields: list) -> Optional[dict]:
    """Parse and validate JSON from LLM output with multiple fallback strategies."""
    if not raw or not raw.strip():
        return None

    # Strategy 1: Direct parse
    try:
        data = json.loads(raw.strip())
        if isinstance(data, dict) and all(f in data for f in required_fields):
            return data
    except json.JSONDecodeError:
        pass

    # Strategy 2: Extract from markdown code blocks
    json_match = re.search(r'```(?:json)?\s*([\s\S]*?)```', raw)
    if json_match:
        try:
            data = json.loads(json_match.group(1).strip())
            if isinstance(data, dict) and all(f in data for f in required_fields):
                return data
        except json.JSONDecodeError:
            pass

    # Strategy 3: Find first { ... } block
    brace_match = re.search(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', raw)
    if brace_match:
        try:
            data = json.loads(brace_match.group())
            if isinstance(data, dict) and all(f in data for f in required_fields):
                return data
        except json.JSONDecodeError:
            pass

    log.warning(f"Failed to parse LLM JSON. Raw output: {raw[:200]}")
    return None


def generate_fallback_message(merchant_ctx: dict, trigger_ctx: dict, category_ctx: dict) -> dict:
    """Template-based fallback if LLM fails — still grounded in context."""
    m_payload = merchant_ctx.get("payload", {})
    t_payload = trigger_ctx.get("payload", {})

    identity = m_payload.get("identity", {})
    owner = identity.get("owner_first_name", identity.get("owner_name", identity.get("name", "there")))
    biz_name = identity.get("name", identity.get("business_name", "your business"))
    locality = identity.get("locality", "")

    trigger_kind = t_payload.get("kind", t_payload.get("type", "update"))
    category_name = category_ctx.get("context_id", "general")

    # Try to include a real number from performance
    perf = m_payload.get("performance", {})
    stat_line = ""
    if perf.get("views"):
        stat_line = f" Your listing got {perf['views']} views recently."
    elif perf.get("calls"):
        stat_line = f" You received {perf['calls']} calls this period."

    body = f"Hi {owner},{stat_line} I have an update about {biz_name}"
    if locality:
        body += f" in {locality}"
    body += " that could help. Want me to share the details?"

    return {
        "body": body,
        "cta": "open_ended",
        "send_as": "vera",
        "suppression_key": f"{trigger_kind}:{category_name}:{datetime.now(timezone.utc).strftime('%Y-W%V')}",
        "rationale": f"Fallback for {trigger_kind} trigger — LLM unavailable, using template with real metrics"
    }


# ============================================================
#                    API ENDPOINTS
# ============================================================

@app.get("/v1/healthz")
async def healthz():
    """Liveness probe."""
    return {
        "status": "ok",
        "uptime_seconds": int(time.time() - START_TIME),
        "contexts_loaded": {
            scope: len(data) for scope, data in context_store.items()
        }
    }


@app.get("/v1/metadata")
async def metadata():
    """Team identity for leaderboard."""
    return {
        "team_name": "Ritik Solo",
        "team_members": ["Ritik"],
        "model": LLM_MODEL,
        "approach": "Context-grounded signal-ranked composer with intent-aware reply state machine. "
                    "3-layer prompting (system persona + context injection + task). "
                    "Rule-based fast paths for auto-reply/hostile detection. "
                    "Template fallback if LLM fails.",
        "version": "2.0.0"
    }


@app.post("/v1/context")
async def push_context(request: Request):
    """Accept context pushes. Idempotent by scope + context_id + version."""
    try:
        data = await request.json()
    except Exception:
        return JSONResponse(status_code=400, content={"error": "Invalid JSON"})

    scope = data.get("scope", "")
    context_id = data.get("context_id", "")
    version = data.get("version", 0)
    payload = data.get("payload", {})
    delivered_at = data.get("delivered_at", "")

    if scope not in context_store:
        return JSONResponse(status_code=400, content={"error": f"Unknown scope: {scope}"})

    existing = context_store[scope].get(context_id)

    # Idempotent: same or lower version = no-op
    if existing and existing.get("version", 0) >= version:
        return {
            "accepted": True,
            "ack_id": f"ack_{hashlib.md5(f'{scope}:{context_id}:{version}'.encode()).hexdigest()[:12]}",
            "stored_at": existing.get("stored_at", datetime.now(timezone.utc).isoformat())
        }

    stored_at = datetime.now(timezone.utc).isoformat()
    context_store[scope][context_id] = {
        "scope": scope,
        "context_id": context_id,
        "version": version,
        "payload": payload,
        "delivered_at": delivered_at,
        "stored_at": stored_at
    }

    log.info(f"Context stored: {scope}/{context_id} v{version} ({len(json.dumps(payload))} bytes)")

    return {
        "accepted": True,
        "ack_id": f"ack_{hashlib.md5(f'{scope}:{context_id}:{version}'.encode()).hexdigest()[:12]}",
        "stored_at": stored_at
    }


@app.post("/v1/tick")
async def handle_tick(request: Request):
    """Periodic wake-up. Compose messages for available triggers."""
    try:
        data = await request.json()
    except Exception:
        return JSONResponse(status_code=400, content={"error": "Invalid JSON"})

    now = data.get("now", datetime.now(timezone.utc).isoformat())
    available_triggers = data.get("available_triggers", [])

    actions = []

    def _compose_for_trigger(trigger_id):
        """Compose a message for a single trigger (runs in thread pool)."""
        trigger_ctx = context_store["trigger"].get(trigger_id)
        if not trigger_ctx:
            log.warning(f"Trigger {trigger_id} not in store, skipping")
            return None

        t_payload = trigger_ctx.get("payload", {})

        # Resolve merchant — merchant_id can be at payload root level
        merchant_id = t_payload.get("merchant_id", "")
        merchant_ctx = context_store["merchant"].get(merchant_id)

        if not merchant_ctx:
            for mid, mctx in context_store["merchant"].items():
                if merchant_id and (merchant_id in mid or mid in merchant_id):
                    merchant_ctx = mctx
                    merchant_id = mid
                    break
            if not merchant_ctx and context_store["merchant"]:
                merchant_id = list(context_store["merchant"].keys())[0]
                merchant_ctx = context_store["merchant"][merchant_id]
            if not merchant_ctx:
                log.warning(f"No merchant for trigger {trigger_id}")
                return None

        # Resolve category
        m_payload = merchant_ctx.get("payload", {})
        category_slug = m_payload.get("category_slug", m_payload.get("category_id", ""))
        category_ctx = _resolve_category(category_slug, t_payload, m_payload)

        # Resolve customer (optional)
        customer_ctx = None
        customer_id = t_payload.get("customer_id", None)
        if customer_id:
            customer_ctx = context_store["customer"].get(customer_id)

        # Compose
        log.info(f"Composing: merchant={merchant_id}, trigger={trigger_id}")
        prompt = build_compose_prompt(merchant_ctx, category_ctx, trigger_ctx, customer_ctx)
        raw = call_llm(prompt, SYSTEM_PROMPT)
        result = parse_llm_json(raw, ["body"])

        if not result:
            log.warning(f"LLM failed for {trigger_id}, using fallback")
            result = generate_fallback_message(merchant_ctx, trigger_ctx, category_ctx)

        trigger_kind = t_payload.get("kind", t_payload.get("type", "general"))
        cat_name = category_ctx.get("context_id", "general") if category_ctx else "general"
        week = datetime.now(timezone.utc).strftime("%Y-W%V")

        return {
            "merchant_id": merchant_id,
            "trigger_id": trigger_id,
            "body": result.get("body", ""),
            "cta": result.get("cta", "open_ended"),
            "send_as": result.get("send_as", "vera"),
            "suppression_key": result.get("suppression_key", f"{trigger_kind}:{cat_name}:{week}"),
            "rationale": result.get("rationale", "Signal-ranked composition from live context")
        }

    # Process triggers concurrently using thread pool
    with ThreadPoolExecutor(max_workers=min(5, len(available_triggers))) as executor:
        futures = {executor.submit(_compose_for_trigger, tid): tid for tid in available_triggers}
        for future in as_completed(futures):
            try:
                result = future.result()
                if result:
                    actions.append(result)
            except Exception as e:
                log.error(f"Trigger composition failed: {e}")

    log.info(f"Tick: {len(actions)} actions from {len(available_triggers)} triggers")
    return {"actions": actions}


def _resolve_category(category_id: str, t_payload: dict, m_payload: dict) -> dict:
    """Find the best matching category context."""
    # Direct match
    if category_id and category_id in context_store["category"]:
        return context_store["category"][category_id]

    # Fuzzy match from trigger or merchant hints
    cat_hint = (
        t_payload.get("category", "") or
        m_payload.get("identity", {}).get("category", "") or
        category_id or ""
    ).lower()

    for cid, cctx in context_store["category"].items():
        if cat_hint and (cat_hint in cid.lower() or cid.lower() in cat_hint):
            return cctx
        # Check inside payload
        c_slug = cctx.get("payload", {}).get("slug", "")
        if cat_hint and (cat_hint in c_slug.lower() or c_slug.lower() in cat_hint):
            return cctx

    # Fallback to first available
    if context_store["category"]:
        return list(context_store["category"].values())[0]

    return {"payload": {}, "context_id": "unknown"}


@app.post("/v1/reply")
async def handle_reply(request: Request):
    """Handle merchant/customer replies with intent detection."""
    try:
        data = await request.json()
    except Exception:
        return JSONResponse(status_code=400, content={"error": "Invalid JSON"})

    conv_id = data.get("conversation_id", "unknown")
    from_role = data.get("from_role", "merchant")
    message = data.get("message", "")
    turn = data.get("turn_number", 1)

    msg_lower = message.lower().strip()

    # Track conversation state for repetition detection
    if conv_id not in conversation_state:
        conversation_state[conv_id] = {"messages": [], "turns": 0}
    conv = conversation_state[conv_id]
    conv["messages"].append(msg_lower)
    conv["turns"] = turn

    # === RULE-BASED FAST PATHS ===

    # Auto-reply detection (same message repeated, or keyword-based)
    auto_signals = [
        "out of office", "auto-reply", "automatic reply", "vacation",
        "currently unavailable", "will respond", "auto reply",
        "not available", "on leave", "away from", "do not reply",
        "automated assistant", "automated response",
        "aapki jaankari ke liye", "shukriya", "hamari team tak"
    ]
    if any(s in msg_lower for s in auto_signals):
        log.info(f"[{conv_id}] Auto-reply detected at turn {turn}")
        return {
            "action": "end",
            "body": "",
            "rationale": "Auto-reply detected; ending to avoid infinite loop"
        }

    # Repeated message detection (3+ same messages = auto-reply)
    if len(conv["messages"]) >= 3:
        last_three = conv["messages"][-3:]
        if last_three[0] == last_three[1] == last_three[2]:
            log.info(f"[{conv_id}] Repeated message detected")
            return {
                "action": "end",
                "body": "",
                "rationale": "Same message repeated 3+ times; auto-reply behavior detected"
            }

    # Hostile / opt-out
    hostile_signals = [
        "stop", "spam", "unsubscribe", "block", "report",
        "leave me alone", "don't contact", "do not contact",
        "harass", "shut up", "go away", "band karo", "mat karo"
    ]
    if any(s in msg_lower for s in hostile_signals):
        log.info(f"[{conv_id}] Hostile/opt-out at turn {turn}")
        return {
            "action": "end",
            "body": "Samajh gayi. Main aapko aur message nahi karungi. Kisi bhi waqt reach out kar sakte hain.",
            "rationale": "Merchant expressed opt-out intent; closing respectfully"
        }

    # Turn limit
    if turn >= 4:
        log.info(f"[{conv_id}] Turn limit at {turn}")
        return {
            "action": "end",
            "body": "Thanks for the conversation! Will follow up if anything relevant comes up. Have a great day!",
            "rationale": f"Turn {turn}: wrapping up to avoid over-messaging"
        }

    # === LLM-POWERED INTENT DETECTION ===
    prompt = build_reply_prompt(message, turn, conv_id)
    raw = call_llm(prompt, SYSTEM_PROMPT)
    result = parse_llm_json(raw, ["action"])

    if result:
        action = result.get("action", "end")
        if action not in ("send", "wait", "end"):
            action = "end"
        return {
            "action": action,
            "body": result.get("body", ""),
            "rationale": result.get("rationale", "LLM-based intent detection")
        }

    # === KEYWORD FALLBACK ===
    positive = ["yes", "sure", "okay", "ok", "go ahead", "proceed", "haan", "theek",
                 "sounds good", "interested", "tell me", "kar do", "chalega", "chalo"]
    if any(s in msg_lower for s in positive):
        return {
            "action": "send",
            "body": "Great, setting that up for you now. You'll see the update on your magicpin dashboard shortly!",
            "rationale": "Positive intent detected via keywords; fulfilling"
        }

    # Default: graceful end
    return {
        "action": "end",
        "body": "Thanks for your response! I'll share relevant updates when they come up.",
        "rationale": "Could not determine clear intent; ending gracefully"
    }


# ============================================================
#                     STARTUP + RUN
# ============================================================

@app.on_event("startup")
async def on_startup():
    log.info("=" * 60)
    log.info("  Vera AI Composer v2.0.0")
    log.info(f"  LLM: {LLM_PROVIDER} / {LLM_MODEL}")
    log.info(f"  Temperature: {TEMPERATURE} (deterministic)")
    log.info(f"  Port: {PORT}")
    log.info("=" * 60)


if __name__ == "__main__":
    uvicorn.run("bot:app", host="0.0.0.0", port=PORT, reload=False, log_level="info")
