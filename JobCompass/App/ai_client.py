# # App/ai_client.py
# import os
# import logging
# import requests
# from typing import Optional

# log = logging.getLogger(__name__)

# # ---------------- LM STUDIO CONFIG ----------------
# LMSTUDIO_BASE_URL = os.getenv("LMSTUDIO_BASE_URL", "http://localhost:1234/v1").rstrip("/")
# LMSTUDIO_MODEL = os.getenv("LMSTUDIO_MODEL", "qwen/qwen3-8b")

# HEADERS = {
#     "Content-Type": "application/json"
# }

# # ---------------- LM STUDIO CALL ----------------
# def _call_lmstudio(prompt: str,
#                    system: Optional[str] = None,
#                    max_tokens: int = 512,
#                    temperature: float = 0.2):
#     """
#     Calls LM Studio local server (OpenAI-compatible).
#     """

#     url = f"{LMSTUDIO_BASE_URL}/chat/completions"

#     messages = [
#         {
#         "role": "system",
#         "content": (
#             "You are a professional AI assistant. "
#             "Do NOT reveal your chain-of-thought, reasoning steps, or internal analysis. "
#             "Do NOT use <think> tags. "
#             "Provide only the final answer or the requested structured output."
#             )
#         }
#     ]

#     if system:
#         messages.append({"role": "system", "content": system})

#     messages.append({"role": "user", "content": prompt})

#     payload = {
#         "model": LMSTUDIO_MODEL,
#         "messages": messages,
#         "temperature": temperature,
#         "max_tokens": max_tokens,
#         "stream": False
#     }

#     try:
#         r = requests.post(url, headers=HEADERS, json=payload, timeout=120)
#         r.raise_for_status()
#         data = r.json()

#         text = (
#             data.get("choices", [{}])[0]
#                 .get("message", {})
#                 .get("content", "")
#         )

#         return {
#             "text": text,
#             "raw": data
#         }

#     except Exception as e:
#         log.exception("LM Studio call failed")
#         return {
#             "text": "",
#             "error": str(e)
#         }

# # ---------------- UNIFIED ENTRY POINT ----------------
# def call_llm(prompt: str,
#              system: Optional[str] = None,
#              max_tokens: int = 512,
#              temperature: float = 0.2):
#     return _call_lmstudio(
#         prompt=prompt,
#         system=system,
#         max_tokens=max_tokens,
#         temperature=temperature
#     )

# # ---------------- BACKWARD COMPATIBILITY ----------------
# # utils.py and views.py already call call_mixtral()
# def call_mixtral(prompt: str,
#                  system: Optional[str] = None,
#                  temperature: float = 0.2,
#                  max_tokens: int = 512):
#     return call_llm(
#         prompt=prompt,
#         system=system,
#         temperature=temperature,
#         max_tokens=max_tokens
#     )

import os
import logging
from groq import Groq

log = logging.getLogger(__name__)

# Initialize Groq client once
_client = Groq(api_key=os.getenv("GROQ_API_KEY"))


def call_llm(
    system: str,
    prompt: str,
    model: str = "llama-3.1-8b-instant",
    temperature: float = 0.2,
    max_tokens: int = 800,
) -> dict:
    """
    Unified LLM call interface for Groq.
    Returns: {"text": "..."} to match utils.py expectations.
    """

    if not _client:
        raise RuntimeError("Groq client not initialized. Check GROQ_API_KEY.")

    try:
        response = _client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": prompt},
            ],
            temperature=temperature,
            max_tokens=max_tokens,
        )

        text = response.choices[0].message.content
        return {"text": text}

    except Exception as e:
        log.exception("Groq LLM call failed")
        # Fail fast with clear signal
        raise RuntimeError(f"Groq LLM error: {str(e)}")
