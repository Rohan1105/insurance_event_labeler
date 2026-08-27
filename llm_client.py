"""Minimal Groq chat-completions client (stdlib only, no SDK).

Groq exposes an OpenAI-compatible /chat/completions endpoint, so this is a single
POST with a JSON body. Kept in its own module so the classifier can be tested with a
stub, and so swapping to Gemini/Ollama means editing one file.

Endpoint and request shape per Groq's API docs: https://console.groq.com/docs/api-reference
"""

from __future__ import annotations

import json
import os
import re
import time
import urllib.error
import urllib.request
from typing import Optional

API_URL = "https://api.groq.com/openai/v1/chat/completions"

# Chosen from the models this account can actually reach (GET /openai/v1/models).
# gpt-oss-120b is a reasoning model: it spends completion tokens thinking before it
# answers, which is why max_tokens below is 800 rather than the ~50 the JSON needs.
# The reasoning arrives in a separate `reasoning` field, so `content` stays clean JSON.
DEFAULT_MODEL = "openai/gpt-oss-120b"

# Groq sits behind Cloudflare, which rejects the default "Python-urllib/3.x" agent with
# HTTP 403 / Cloudflare error 1010. Any non-default UA is accepted.
USER_AGENT = "leadenhall-takehome/1.0"

# The rubric, compressed to what the model needs to make the call.
#
# DELIBERATELY FROZEN at the version of RUBRIC.md that produced the evaluated run.
# RUBRIC.md has since gained two rules this prompt does NOT carry:
#   1. Precedence: an explicit "no incident occurred" + stated resolution beats the
#      precautionary-response rule (added after this prompt returned YES on snippet 21).
#   2. Corroboration: sourcing is judged on the event, not the numbers -- an uncorroborated
#      single-source claim the best-placed party declines to confirm is NO, not YES
#      (added after the annotator revised snippet 09).
# Neither is copied down here on purpose. Editing the prompt after seeing which items it got
# wrong would be tuning the classifier on the only 40 examples available, and would buy a
# score that means nothing on the next dataset. Both propagate on the next run against
# held-out data. See EVALUATION.md Cases 2 and 3.
SYSTEM_PROMPT = """You triage world news for an insurer's property & casualty event feed.

Answer one question: does this snippet describe a SPECIFIC physical event that has already
begun or just happened, with damage, injury, or an active emergency/precautionary response?

YES if: fire, flood, storm, hail, earthquake, explosion, collapse, derailment, grounding,
industrial or infrastructure failure -- AND there is damage, injury, people missing, or an
active response (evacuation, rescue, salvage, emergency shutdown, tow, emergency landing).

NO if the subject is money, words, or the future rather than a physical event: markets,
corporate news, funding, labour disputes, regulation, litigation about past events, academic
research, forecasts/warnings/seasonal outlooks, sport. Insurance vocabulary alone
(reinsurance pricing, premiums, insurtech) is NOT an event.

Specific rules for hard cases:
- Unconfirmed or single-source reports of a real event type: YES, confidence LOW.
- Forecasts, warnings, or outlooks where nothing has happened yet: NO, confidence HIGH.
- Precautionary response with damage unconfirmed (pipeline shut on a sensor anomaly, dam
  evacuated over cracks, ferry towed, aircraft emergency landing): YES, confidence LOW.
- Explicitly routine or "no incidents reported": NO, confidence HIGH.
- Cyber or IT-caused interruption with no physical damage (ransomware, data breach,
  software fault): NO, confidence LOW.
- Court judgments or settlements over historical events: NO, confidence HIGH.
- Long-onset conditions (drought): YES only if an active consequence is already in force
  (e.g. restrictions imposed), confidence LOW; otherwise NO.
- Event confirmed but loss not yet quantified: YES, confidence HIGH.

Confidence is about how certain the LABEL is, not how severe the event is.

Reply with JSON only, no prose, no code fence:
{"label": "YES"|"NO", "confidence": "HIGH"|"LOW", "rationale": "one short sentence"}"""


class GroqClient:
    def __init__(
        self,
        api_key: str,
        model: str = DEFAULT_MODEL,
        timeout: int = 60,
        max_retries: int = 5,
    ):
        self.api_key = api_key
        self.model = model
        self.timeout = timeout
        self.max_retries = max_retries
        self.last_error: Optional[str] = None  # why the most recent call failed

    @staticmethod
    def _retry_delay(exc: Optional[urllib.error.HTTPError], attempt: int) -> float:
        """Seconds to wait before retrying.

        Prefer the server's own `retry-after` header -- for a TPM cap it knows exactly
        when the window resets, so guessing is strictly worse. Fall back to exponential
        backoff (2, 4, 8, 16s) when there is no header.
        """
        if exc is not None:
            header = exc.headers.get("retry-after")
            if header:
                try:
                    return min(float(header) + 0.5, 60.0)
                except ValueError:
                    pass
        return min(2.0 * (2 ** attempt), 30.0)

    def classify(self, text: str) -> Optional[tuple[str, str, str]]:
        """Return (label, confidence, rationale), or None if the call/parse fails.

        Failures are recorded on `self.last_error` rather than raised, so one bad row
        cannot kill a batch -- but the caller can still report the cause instead of
        silently degrading, which is what a bare `return None` would have done.
        """
        self.last_error = None
        body = json.dumps(
            {
                "model": self.model,
                "temperature": 0,  # determinism matters more than variety for triage
                "max_tokens": 800,
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": text},
                ],
            }
        ).encode("utf-8")

        request = urllib.request.Request(
            API_URL,
            data=body,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
                "User-Agent": USER_AGENT,
            },
            method="POST",
        )

        content = None
        for attempt in range(self.max_retries + 1):
            try:
                with urllib.request.urlopen(request, timeout=self.timeout) as response:
                    payload = json.loads(response.read().decode("utf-8"))
                content = payload["choices"][0]["message"]["content"]
                break
            except urllib.error.HTTPError as exc:
                detail = exc.read().decode("utf-8", "replace")[:200]
                self.last_error = f"HTTP {exc.code}: {detail}"
                # 429 = free-tier tokens-per-minute cap; Groq tells us how long to wait.
                # 5xx is transient. Anything else (401, 404) will not fix itself.
                if exc.code == 429 or 500 <= exc.code < 600:
                    if attempt == self.max_retries:
                        return None
                    time.sleep(self._retry_delay(exc, attempt))
                    continue
                return None
            except (urllib.error.URLError, TimeoutError) as exc:
                self.last_error = f"network: {exc}"
                if attempt == self.max_retries:
                    return None
                time.sleep(self._retry_delay(None, attempt))
                continue
            except (KeyError, IndexError, json.JSONDecodeError) as exc:
                self.last_error = f"unexpected response shape: {type(exc).__name__} {exc}"
                return None

        if content is None:
            return None

        verdict = _parse_verdict(content)
        if verdict is None:
            self.last_error = f"unparseable verdict: {content[:200]!r}"
        return verdict


def _parse_verdict(content: str) -> Optional[tuple[str, str, str]]:
    """Pull the JSON object out of the model's reply and validate the enum fields.

    Models occasionally wrap JSON in a code fence or add a sentence around it, so we
    take the first {...} block rather than trusting the whole string to parse.
    """
    match = re.search(r"\{.*\}", content, re.DOTALL)
    if not match:
        return None
    try:
        obj = json.loads(match.group(0))
    except json.JSONDecodeError:
        return None

    label = str(obj.get("label", "")).strip().upper()
    confidence = str(obj.get("confidence", "")).strip().upper()
    rationale = str(obj.get("rationale", "")).strip()
    if label not in ("YES", "NO") or confidence not in ("HIGH", "LOW"):
        return None
    return label, confidence, rationale


def load_dotenv(path: str = ".env") -> None:
    """Read KEY=VALUE lines from `path` into os.environ.

    Twelve lines instead of a python-dotenv dependency. Real environment variables
    win over the file, which is the convention people expect: exporting a key in the
    shell should override a stale .env.
    """
    if not os.path.exists(path):
        return
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            key = key.strip()
            value = value.strip().strip("'\"")  # tolerate quoted values
            if key and key not in os.environ:
                os.environ[key] = value


def from_env() -> Optional[GroqClient]:
    """Build a client from GROQ_API_KEY / GROQ_MODEL, or None if no key is set."""
    load_dotenv()
    key = os.environ.get("GROQ_API_KEY")
    if not key:
        return None
    return GroqClient(key, os.environ.get("GROQ_MODEL", DEFAULT_MODEL))
