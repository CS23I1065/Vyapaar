"""
Gemini provider adapter.

Design constraints, and why they exist:

- thinking_budget defaults to 0 on every call. Thinking tokens bill as
  OUTPUT tokens (see pricing.py) — leaving thinking on by accident is the
  single biggest way this project's eval matrix could blow its $5 cap.

- A hard client-side timeout backstops every call via a thread pool, not
  just the SDK's own http_options.timeout. Some models are known to
  accept the connection and then never respond at all (HTTP 000 after
  60s+) — not a slow response, no response. An HTTP-layer timeout is the
  right tool for a slow server; a thread-pool .result(timeout=...) is
  the right tool for a server that never replies, because it bounds
  wall-clock time regardless of what's happening inside the SDK call.

- Known-hanging models are refused outright (BANNED_MODELS), independent
  of the timeout backstop, so a stale .env can't silently reintroduce a
  hang during a live demo.

- The LLM proposes; it never authorizes. Nothing in this module writes
  to a Decision. It only ever returns text/typed JSON for a caller to
  treat as untrusted, provenance-tagged input.
"""

from __future__ import annotations

import concurrent.futures
import logging
import os
import time
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

from google import genai
from google.genai import types as genai_types

from .pricing import BANNED_MODELS, estimate_cost_usd

logger = logging.getLogger(__name__)

HARD_TIMEOUT_S = 30.0
MAX_RETRIES = 4
BACKOFF_BASE_S = 1.5


class ProviderError(RuntimeError):
    """Base error for llm provider failures."""


class ProviderTimeout(ProviderError):
    """The call did not return within the hard timeout."""


class ProviderRateLimited(ProviderError):
    """Exhausted retries against a 429 / RESOURCE_EXHAUSTED response."""


class ProviderBannedModel(ProviderError):
    """Caller asked for a model in BANNED_MODELS."""


@dataclass(frozen=True)
class Usage:
    input_tokens: int
    output_tokens: int
    thinking_tokens: int
    cost_usd: float


@dataclass(frozen=True)
class Completion:
    text: str
    usage: Usage
    model: str
    raw: Any = field(repr=False, default=None, compare=False)


@runtime_checkable
class SpendGuard(Protocol):
    """Implemented by eval/budget.py. Matched structurally, not by inheritance,
    so llm/ has no import dependency on eval/."""

    def check(self, projected_cost_usd: float) -> None:
        """Raise BudgetExceeded (or subclass) if this call would breach the cap."""
        ...

    def record(self, actual_cost_usd: float) -> None: ...


class GeminiProvider:
    def __init__(
        self,
        *,
        api_key: str | None = None,
        spend_guard: SpendGuard | None = None,
        timeout_s: float = HARD_TIMEOUT_S,
        max_retries: int = MAX_RETRIES,
    ) -> None:
        api_key = api_key or os.environ.get("GEMINI_API_KEY")
        if not api_key:
            raise ProviderError(
                "GEMINI_API_KEY not set (checked constructor arg and environment)."
            )
        # Belt-and-suspenders: ask the SDK's own transport to time out too,
        # in addition to the thread-pool backstop below. Neither alone was
        # trusted after the 3.7-flash hang, so both are in place.
        self._client = genai.Client(
            api_key=api_key,
            http_options=genai_types.HttpOptions(timeout=int(timeout_s * 1000)),
        )
        self._spend_guard = spend_guard
        self._timeout_s = timeout_s
        self._max_retries = max_retries
        self._executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=4, thread_name_prefix="gemini-call"
        )

    def close(self) -> None:
        self._executor.shutdown(wait=False, cancel_futures=True)

    def __enter__(self) -> "GeminiProvider":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def complete(
        self,
        *,
        model: str,
        system: str,
        messages: list[dict],  # [{"role": "user"|"model", "text": "..."}]
        temperature: float = 0.0,
        max_output_tokens: int = 2048,
        response_schema: dict | None = None,
        thinking_budget: int = 0,
        call_site: str = "unspecified",
    ) -> Completion:
        if model in BANNED_MODELS:
            raise ProviderBannedModel(
                f"{model!r} is in llm.pricing.BANNED_MODELS — verified to hang "
                f"(HTTP 000, no response) on 2026-08-24. Re-verify with a "
                f"bounded `curl --max-time 60` before removing the ban."
            )

        contents = [
            genai_types.Content(role=m["role"], parts=[genai_types.Part(text=m["text"])])
            for m in messages
        ]

        config_kwargs: dict[str, Any] = dict(
            system_instruction=system,
            temperature=temperature,
            max_output_tokens=max_output_tokens,
            thinking_config=genai_types.ThinkingConfig(thinking_budget=thinking_budget),
        )
        if response_schema is not None:
            config_kwargs["response_mime_type"] = "application/json"
            config_kwargs["response_schema"] = response_schema
        config = genai_types.GenerateContentConfig(**config_kwargs)

        if self._spend_guard is not None:
            # Conservative pre-flight estimate (chars/4 as a token proxy, and
            # max_output_tokens as the worst case) so a run aborts BEFORE
            # spending the money that would breach the cap, not after.
            est_input_tokens = (sum(len(m["text"]) for m in messages) + len(system)) // 4
            projected = estimate_cost_usd(model, est_input_tokens, max_output_tokens)
            self._spend_guard.check(projected)

        response = self._call_with_retry(model=model, contents=contents, config=config)

        text = response.text or ""
        um = response.usage_metadata
        input_tokens = getattr(um, "prompt_token_count", None) or 0
        output_tokens = getattr(um, "candidates_token_count", None) or 0
        thinking_tokens = getattr(um, "thoughts_token_count", None) or 0
        # Thinking tokens bill as output — fold them in before pricing.
        cost = estimate_cost_usd(model, input_tokens, output_tokens + thinking_tokens)
        usage = Usage(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            thinking_tokens=thinking_tokens,
            cost_usd=cost,
        )

        if self._spend_guard is not None:
            self._spend_guard.record(cost)

        return Completion(text=text, usage=usage, model=model, raw=response)

    def _call_with_retry(
        self, *, model: str, contents: list, config: "genai_types.GenerateContentConfig"
    ):
        last_err: Exception | None = None
        for attempt in range(self._max_retries):
            future = self._executor.submit(
                self._client.models.generate_content,
                model=model,
                contents=contents,
                config=config,
            )
            try:
                return future.result(timeout=self._timeout_s)
            except concurrent.futures.TimeoutError as e:
                future.cancel()
                raise ProviderTimeout(
                    f"{model} did not respond within {self._timeout_s}s. "
                    f"(gemini-3.7-flash and gemini-flash-latest are known to "
                    f"do this — check BANNED_MODELS in llm/pricing.py.)"
                ) from e
            except Exception as e:
                msg = str(e)
                is_rate_limit = "429" in msg or "RESOURCE_EXHAUSTED" in msg
                if is_rate_limit and attempt < self._max_retries - 1:
                    sleep_s = BACKOFF_BASE_S * (2**attempt)
                    logger.warning(
                        "Gemini rate-limited (attempt %d/%d), backing off %.1fs",
                        attempt + 1,
                        self._max_retries,
                        sleep_s,
                    )
                    time.sleep(sleep_s)
                    last_err = e
                    continue
                if is_rate_limit:
                    raise ProviderRateLimited(msg) from e
                raise ProviderError(msg) from e
        raise ProviderRateLimited(str(last_err))
