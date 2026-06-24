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

    # ------------------------------------------------------------------ #
    # plain chat
    # ------------------------------------------------------------------ #
    def chat(self, system: str, user: str, *, max_tokens: int = 4000,
             cache_key: Optional[str] = None) -> str:
        messages = [{"role": "system", "content": system},
                    {"role": "user", "content": user}]
        key = cache_key or ("chat:" + json.dumps(messages, sort_keys=True))
        if self.cache:
            hit = self.cache.get(key)
            if hit is not None:
                return hit.get("text", "")

        last_err = None
        for model in self.models:
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
                  cache_key: Optional[str] = None) -> dict:
        sys_p = system + "\n\nRespond with a single valid JSON object and nothing else."
        messages = [{"role": "system", "content": sys_p},
                    {"role": "user", "content": user}]
        key = cache_key or ("json:" + json.dumps(messages, sort_keys=True))
        if self.cache:
            hit = self.cache.get(key)
            if hit is not None:
                return hit

        last_err = None
        for model in self.models:
            try:
                resp = self.client.chat.completions.create(
                    model=model, messages=messages, temperature=0, max_tokens=max_tokens,
                    response_format={"type": "json_object"},
                )
                text = (resp.choices[0].message.content or "").strip()
                data = _loads_lenient(text)
                if self.cache:
                    self.cache.set(key, data)
                return data
            except Exception as e:  # noqa: BLE001
                logger.warning("chat_json failed on %s: %s", model, e)
                last_err = e
        raise LLMError(f"All models failed (json): {last_err}")

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
    """Parse JSON, tolerating code fences / leading prose."""
    text = text.strip()
    if text.startswith("```"):
        text = text.strip("`")
        if text.lower().startswith("json"):
            text = text[4:]
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1 and end > start:
            return json.loads(text[start:end + 1])
        raise
