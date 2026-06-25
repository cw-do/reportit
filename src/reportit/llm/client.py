"""LLMClient — OpenRouter (OpenAI-compatible) with caching, JSON, and tool-calling.

Built on the eqsanstools-cli llm_handler pattern: construct an OpenAI client
pointed at OpenRouter, iterate [model, fallback_model] on failure. Adds:
  - chat()            : plain completion
  - chat_json()       : structured JSON (json_object response_format + repair)
  - chat_with_tools() : the agentic loop — model calls read-only probe tools
                        repeatedly until it calls a finalize tool or hits a cap.

All calls are cached by md5(model + messages [+ tools]) so reruns are free and
deterministic. temperature=0 throughout.
"""

from __future__ import annotations

import json
import logging
from typing import Any, Callable, Optional

from ..cache import Cache
from ..cache.store import md5
from ..config import LLMSettings

logger = logging.getLogger(__name__)


class LLMError(RuntimeError):
    pass


class LLMClient:
    def __init__(self, settings: LLMSettings, cache: Optional[Cache] = None):
        self.settings = settings
        self.cache = cache
        self._client = None

    @property
    def client(self):
        if self._client is None:
            from openai import OpenAI

            self._client = OpenAI(
                base_url=self.settings.base_url,
                api_key=self.settings.api_key,
                timeout=180.0,
            )
        return self._client

    @property
    def models(self) -> list[str]:
        out = [self.settings.model]
        if self.settings.fallback_model and self.settings.fallback_model not in out:
            out.append(self.settings.fallback_model)
        return out

    def _models_with_fallback(self, model: Optional[str]) -> list[str]:
        """Preferred model first (if given), then the default chain as fallback."""
        if not model:
            return self.models
        out = [model]
        for m in self.models:
            if m not in out:
                out.append(m)
        return out

    # ------------------------------------------------------------------ #
    # plain chat
    # ------------------------------------------------------------------ #
    def chat(self, system: str, user: str, *, max_tokens: int = 4000,
             model: Optional[str] = None, cache_key: Optional[str] = None) -> str:
        messages = [{"role": "system", "content": system},
                    {"role": "user", "content": user}]
        key = cache_key or ("chat:" + (model or "") + json.dumps(messages, sort_keys=True))
        if self.cache:
            hit = self.cache.get(key)
            if hit is not None:
                return hit.get("text", "")

        models = self._models_with_fallback(model)
        last_err = None
        for model in models:
            try:
                resp = self.client.chat.completions.create(
                    model=model, messages=messages, temperature=0, max_tokens=max_tokens,
                )
                text = (resp.choices[0].message.content or "").strip()
                if self.cache:
                    self.cache.set(key, {"text": text, "model": model})
                return text
            except Exception as e:  # noqa: BLE001
                logger.warning("chat failed on %s: %s", model, e)
                last_err = e
        raise LLMError(f"All models failed: {last_err}")

    # ------------------------------------------------------------------ #
    # structured JSON
    # ------------------------------------------------------------------ #
    def chat_json(self, system: str, user: str, *, max_tokens: int = 8000,
                  model: Optional[str] = None, cache_key: Optional[str] = None) -> dict:
        sys_p = system + "\n\nRespond with a single valid JSON object and nothing else."
        messages = [{"role": "system", "content": sys_p},
                    {"role": "user", "content": user}]
        key = cache_key or ("json:" + (model or "") + json.dumps(messages, sort_keys=True))
        if self.cache:
            hit = self.cache.get(key)
            if hit is not None:
                return hit

        models = self._models_with_fallback(model)
        last_err = None
        for mdl in models:
            conv = list(messages)
            for attempt in range(3):  # retry the SAME model on malformed JSON
                use_fmt = attempt == 0  # drop response_format on retries (some models choke)
                try:
                    kwargs = dict(model=mdl, messages=conv, temperature=0,
                                  max_tokens=max_tokens)
                    if use_fmt:
                        kwargs["response_format"] = {"type": "json_object"}
                    resp = self.client.chat.completions.create(**kwargs)
                except Exception as e:  # noqa: BLE001 — API error: move to next model
                    logger.warning("chat_json API error on %s: %s", mdl, e)
                    last_err = e
                    break
                text = (resp.choices[0].message.content or "").strip()
                try:
                    data = _loads_lenient(text)
                    if self.cache:
                        self.cache.set(key, data)
                    return data
                except Exception as e:  # noqa: BLE001 — malformed: nudge & retry
                    last_err = e
                    logger.warning("invalid JSON from %s (attempt %d/3): %s",
                                   mdl, attempt + 1, e)
                    conv = conv + [
                        {"role": "assistant", "content": text[:4000]},
                        {"role": "user", "content":
                            f"Your previous reply was not valid JSON ({e}). Reply again "
                            "with ONLY a single valid JSON object — no prose, no code "
                            "fences, no comments, no trailing commas."},
                    ]
        raise LLMError(f"All models failed (json): {last_err}")

    # ------------------------------------------------------------------ #
    # multimodal: inspect an image (e.g. a fit-vs-data plot)
    # ------------------------------------------------------------------ #
    def chat_vision(self, system: str, user: str, image_path, *,
                    model: Optional[str] = None, max_tokens: int = 1500,
                    cache_key: Optional[str] = None) -> str:
        import base64
        from pathlib import Path as _P

        img = _P(image_path)
        try:
            b64 = base64.b64encode(img.read_bytes()).decode()
        except Exception as e:  # noqa: BLE001
            logger.warning("vision: cannot read image %s: %s", image_path, e)
            return ""
        mdl = model or self.settings.vision_model
        key = cache_key or ("vision:" + mdl + ":" + md5(b64) + ":" + user[:200])
        if self.cache:
            hit = self.cache.get(key)
            if hit is not None:
                return hit.get("text", "")
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": [
                {"type": "text", "text": user},
                {"type": "image_url",
                 "image_url": {"url": f"data:image/png;base64,{b64}"}},
            ]},
        ]
        for m in (mdl, self.settings.model):
            try:
                resp = self.client.chat.completions.create(
                    model=m, messages=messages, temperature=0, max_tokens=max_tokens)
                text = (resp.choices[0].message.content or "").strip()
                if self.cache:
                    self.cache.set(key, {"text": text, "model": m})
                return text
            except Exception as e:  # noqa: BLE001
                logger.warning("vision call failed on %s: %s", m, e)
        return ""

    # ------------------------------------------------------------------ #
    # agentic tool-calling loop
    # ------------------------------------------------------------------ #
    def chat_with_tools(
        self,
        system: str,
        user: str,
        tools: list[dict],
        dispatch: Callable[[str, dict], Any],
        *,
        finalize_tool: str,
        max_steps: int = 30,
        max_tokens: int = 4000,
        on_step: Optional[Callable[[int, str, dict], None]] = None,
        cache_key: Optional[str] = None,
    ) -> dict:
        """Run the model in a tool-calling loop until it calls `finalize_tool`.

        `dispatch(name, args)` executes a (read-only) probe tool and returns a
        JSON-serializable result. The arguments passed to `finalize_tool` are
        returned as the final structured result.
        """
        base_key = cache_key or ("tools:" + json.dumps(
            {"s": system, "u": user, "t": [t["function"]["name"] for t in tools]},
            sort_keys=True))
        if self.cache:
            hit = self.cache.get(base_key)
            if hit is not None:
                logger.info("strategy loop served from cache")
                return hit

        messages: list[dict] = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]

        model = self.models[0]
        nudged = False
        for step in range(1, max_steps + 1):
            # Escalating nudge: once we've used ~2/3 of the budget, tell the model
            # to wrap up so it finalizes instead of exploring forever.
            if not nudged and step >= max(4, int(max_steps * 0.66)):
                messages.append({
                    "role": "user",
                    "content": (f"You have used {step - 1} of {max_steps} investigation "
                                f"steps. Wrap up within the next 1-2 tool calls and then "
                                f"call `{finalize_tool}` with your best complete strategy."),
                })
                nudged = True

            try:
                resp = self.client.chat.completions.create(
                    model=model, messages=messages, tools=tools,
                    tool_choice="auto", temperature=0, max_tokens=max_tokens,
                )
            except Exception as e:  # noqa: BLE001
                logger.warning("tool loop failed on %s: %s; trying fallback", model, e)
                if model != self.models[-1]:
                    model = self.models[-1]
                    continue
                raise LLMError(f"tool loop failed: {e}") from e

            msg = resp.choices[0].message
            tool_calls = msg.tool_calls or []

            if not tool_calls:
                # Model answered without finalizing — nudge it.
                messages.append({"role": "assistant", "content": msg.content or ""})
                messages.append({
                    "role": "user",
                    "content": f"You must call the `{finalize_tool}` tool to finish.",
                })
                continue

            # record the assistant turn (with its tool calls)
            messages.append({
                "role": "assistant",
                "content": msg.content or "",
                "tool_calls": [
                    {"id": tc.id, "type": "function",
                     "function": {"name": tc.function.name, "arguments": tc.function.arguments}}
                    for tc in tool_calls
                ],
            })

            finalized = None
            for tc in tool_calls:
                name = tc.function.name
                try:
                    args = json.loads(tc.function.arguments or "{}")
                except json.JSONDecodeError:
                    args = {}
                if on_step:
                    on_step(step, name, args)

                if name == finalize_tool:
                    finalized = args
                    result = {"ok": True, "note": "strategy recorded"}
                else:
                    try:
                        result = dispatch(name, args)
                    except Exception as e:  # noqa: BLE001
                        result = {"error": str(e)}
                messages.append({
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": _truncate(json.dumps(result, default=str), 12000),
                })

            if finalized is not None:
                if self.cache:
                    self.cache.set(base_key, finalized)
                return finalized

        # Hit the step cap without finalizing — force a finalize, robustly.
        logger.warning("strategy loop hit max_steps=%d without finalize", max_steps)
        messages.append({"role": "user",
                         "content": (f"Stop exploring now. Call `{finalize_tool}` with your "
                                     f"best complete strategy based on what you have learned.")})
        for attempt_model in (model, self.models[-1]):
            try:
                resp = self.client.chat.completions.create(
                    model=attempt_model, messages=messages, tools=tools,
                    tool_choice={"type": "function", "function": {"name": finalize_tool}},
                    temperature=0, max_tokens=max_tokens,
                )
                fmsg = resp.choices[0].message
                tcs = fmsg.tool_calls or []
                if tcs:
                    finalized = json.loads(tcs[0].function.arguments or "{}")
                elif fmsg.content:
                    finalized = _loads_lenient(fmsg.content)
                else:
                    continue
                if self.cache:
                    self.cache.set(base_key, finalized)
                return finalized
            except Exception as e:  # noqa: BLE001
                logger.warning("forced finalize failed on %s: %s", attempt_model, e)
        raise LLMError("strategy loop never finalized")


def _truncate(s: str, n: int) -> str:
    return s if len(s) <= n else s[:n] + f"\n...[truncated {len(s) - n} chars]"


def _loads_lenient(text: str) -> dict:
    """Parse JSON, tolerating code fences, leading prose, and trailing commas."""
    import re

    text = text.strip()
    if text.startswith("```"):
        text = text.strip("`")
        if text.lower().startswith("json"):
            text = text[4:]
    candidates = [text]
    start, end = text.find("{"), text.rfind("}")
    if start != -1 and end != -1 and end > start:
        candidates.append(text[start:end + 1])
    last_err: Exception = ValueError("empty")
    for cand in candidates:
        for variant in (cand, re.sub(r",\s*([}\]])", r"\1", cand)):  # strip trailing commas
            try:
                return json.loads(variant)
            except json.JSONDecodeError as e:
                last_err = e
    raise last_err
