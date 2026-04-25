"""LLM provider abstraction for grounded patient-facing explanations."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any
from urllib import error, request


@dataclass(frozen=True)
class LLMResult:
    text: str
    provider: str
    model: str


def generate_patient_answer(
    query: str,
    context: dict[str, Any],
    config: dict[str, Any],
    history: list[dict[str, str]] | None = None,
) -> LLMResult:
    llm_cfg = config.get("llm", {})
    provider = str(llm_cfg.get("provider", "ollama")).lower()
    prompt = _build_prompt(query, context, int(llm_cfg.get("max_context_chars", 12000)), history or [])
    if provider == "gemini":
        return _call_gemini(prompt, llm_cfg)
    return _call_ollama(prompt, llm_cfg)


def _build_prompt(
    query: str,
    context: dict[str, Any],
    max_context_chars: int,
    history: list[dict[str, str]],
) -> str:
    context_text = json.dumps(context, indent=2, sort_keys=True, default=str)
    if len(context_text) > max_context_chars:
        context_text = context_text[:max_context_chars] + "\n...TRUNCATED..."
    history_text = "\n".join(
        f"{item.get('role', 'user').upper()}: {item.get('content', '')}"
        for item in history
    ) or "No previous conversation."
    system_prompt = str(
        context.get("llm_system_prompt")
        or "You are OralCare-AI's grounded decision-support assistant. Use only supplied context and keep answers concise."
    )
    if "rag_retrieved_documents" in str(context):
        system_prompt += " The context includes 'rag_retrieved_documents' which are excerpts from the patient's past clinical PDFs. Base your answers on these retrieved records when relevant."
    system_prompt += (
        " Answer the current patient question directly. Do not paste a generic risk summary unless the user asked about risk."
        " If the user asks who their doctor is, use doctor_details. If they ask about daily intraoral AI check, explain image upload, date-wise storage, and risk trend tracking."
        " If they ask for suggestions, provide at most three practical suggestions using risk_guidance and XAI context."
        " Keep a casual, reassuring healthcare tone and avoid sounding like a report."
    )
    return f"""{system_prompt}

Recent conversation:
{history_text}

Patient question:
{query}

Patient/model context:
{context_text}
"""


def _call_ollama(prompt: str, llm_cfg: dict[str, Any]) -> LLMResult:
    ollama_cfg = llm_cfg.get("ollama", {})
    base_url = str(ollama_cfg.get("base_url", "http://localhost:11434")).rstrip("/")
    model = str(ollama_cfg.get("model", "llama3.1:8b"))
    timeout = float(ollama_cfg.get("timeout_seconds", 20))
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": 0.2,
            "top_p": 0.8,
            "num_predict": 450,
        },
    }
    response = _post_json(f"{base_url}/api/generate", payload, timeout)
    text = str(response.get("response", "")).strip()
    if not text:
        raise RuntimeError("Ollama returned an empty response")
    return LLMResult(text=text, provider="ollama", model=model)


def _call_gemini(prompt: str, llm_cfg: dict[str, Any]) -> LLMResult:
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY is not set")
    gemini_cfg = llm_cfg.get("gemini", {})
    model = str(gemini_cfg.get("model", "gemini-3-flash-preview"))
    timeout = float(gemini_cfg.get("timeout_seconds", 30))
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
    payload = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {
            "temperature": 0.2,
            "topP": 0.8,
            "maxOutputTokens": 650,
        },
    }
    req = request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json", "x-goog-api-key": api_key},
        method="POST",
    )
    try:
        with request.urlopen(req, timeout=timeout) as response:
            data = json.loads(response.read().decode("utf-8"))
    except error.URLError as exc:
        raise RuntimeError(f"Gemini request failed: {exc}") from exc

    candidates = data.get("candidates", [])
    parts = candidates[0].get("content", {}).get("parts", []) if candidates else []
    text = "".join(str(part.get("text", "")) for part in parts).strip()
    if not text:
        raise RuntimeError("Gemini returned an empty response")
    return LLMResult(text=text, provider="gemini", model=model)


def _post_json(url: str, payload: dict[str, Any], timeout: float) -> dict[str, Any]:
    req = request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with request.urlopen(req, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8"))
    except error.URLError as exc:
        raise RuntimeError(f"LLM request failed: {exc}") from exc
