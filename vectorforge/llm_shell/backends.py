"""Model-agnostic LLM backend registry.

Every LLM integration (Anthropic, OpenAI-compatible, local Ollama, ...) is a concrete LLMBackend
subclass registered in the BACKENDS list. ``resolve_backend()`` returns the first available backend
(key present), so adding a new provider = one entry in BACKENDS, no code changes in ops.py.

The registry is ordered by PRIORITY -- callers get the first available backend unless they pass an
explicit ``preference`` name.  OpenAI-compatible is the universal protocol: Prime Intellect, OpenAI,
Together, Groq, Fireworks, OpenRouter, vLLM, and Ollama all speak it, so one class covers them all.

Security: keys are resolved via ``resolve_key`` (env var -> key file -> None); no key is ever logged.
"""
import json
import os
from typing import Any, List, Optional, Sequence, Tuple

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


# ============================================================================== key resolution (unified)
def resolve_key(env_var: str, key_file_suffix: Optional[str] = None,
                key_prefix: Optional[str] = None,
                key_file_env_var: Optional[str] = None) -> Optional[str]:
    """Resolve an API key: env var -> key file -> None.

    ``key_file_suffix`` (e.g. ".anthropic_key") is a gitignored file at the repo root.
    ``key_prefix`` (e.g. "sk-") is an optional guard: only lines starting with that prefix
    are accepted from a key file, so a placeholder file resolves to None (deterministic fallback).
    ``key_file_env_var`` (e.g. "VF_PI_KEY_FILE") is the env var that overrides the key file path
    for this specific provider. Each provider uses its own env var so they don't collide.
    """
    val = os.environ.get(env_var)
    if val:
        return val
    if key_file_suffix:
        # per-provider env var override (e.g. VF_PI_KEY_FILE for Prime Intellect)
        path = os.path.join(_REPO_ROOT, key_file_suffix)
        if key_file_env_var:
            path = os.environ.get(key_file_env_var, path)
        try:
            with open(path) as fh:
                for line in fh:
                    s = line.strip()
                    if not s or s.startswith("#") or "PASTE_YOUR" in s:
                        continue
                    if key_prefix and not s.startswith(key_prefix):
                        continue
                    return s
        except (OSError, IOError):
            pass
    return None


# ============================================================================== abstract backend
class LLMBackend:
    """Abstract LLM backend. Subclasses must implement ``available`` and ``call``."""
    name: str = "abstract"
    priority: int = 100         # lower = preferred

    def available(self) -> bool:
        """True when the backend has valid credentials and can accept requests."""
        raise NotImplementedError

    def call(self, req, timeout: float) -> Tuple[Any, dict]:
        """Execute the LLM request. Returns ``(raw_output, usage_dict)``.

        ``raw_output`` is a dict (structured / tool-call mode) or str (text mode).
        ``usage_dict`` has ``input_tokens`` and ``output_tokens`` (ints or None).
        Raises on API/network errors (callers catch and fall back).
        """
        raise NotImplementedError

    def describe(self) -> dict:
        """Human-readable capabilities summary for preflight reports."""
        return {"name": self.name, "available": self.available()}


# ============================================================================== Anthropic backend
class AnthropicBackend(LLMBackend):
    """Native Anthropic Messages API (Claude). Supports tool use + adaptive thinking."""
    name = "anthropic"
    priority = 10

    def __init__(self, *, api_key: Optional[str] = None):
        self._explicit_key = api_key

    def _resolve(self) -> Optional[str]:
        if self._explicit_key:
            return self._explicit_key
        return resolve_key("ANTHROPIC_API_KEY", ".anthropic_key", key_prefix="sk-",
                           key_file_env_var="VF_KEY_FILE")

    def available(self) -> bool:
        return bool(self._resolve())

    def call(self, req, timeout: float) -> Tuple[Any, dict]:
        import anthropic
        api_key = self._resolve()
        if not api_key:
            raise RuntimeError("no Anthropic API key")
        client = anthropic.Anthropic(api_key=api_key, timeout=timeout)
        kwargs = dict(model=req.model, max_tokens=req.max_tokens,
                      system=req.system,
                      messages=[{"role": "user", "content": _render_user(req)}])
        if req.schema is not None:
            kwargs["tools"] = [{"name": req.tool_name,
                                "description": "Emit the structured result. Call exactly once.",
                                "input_schema": req.schema}]
            if req.force_tool:
                kwargs["tool_choice"] = {"type": "tool", "name": req.tool_name}
            else:
                kwargs["tool_choice"] = {"type": "auto"}
                kwargs["thinking"] = {"type": "adaptive"}
        else:
            kwargs["thinking"] = {"type": "adaptive"}
        with client.messages.stream(**kwargs) as stream:
            message = stream.get_final_message()
        usage = _anthropic_usage(message)
        if req.schema is not None:
            raw = _anthropic_tool_input(message, req.tool_name)
            if raw is None:
                raise ValueError("model did not call the tool")
            return raw, usage
        return _anthropic_text(message), usage

    def describe(self) -> dict:
        return {"name": self.name, "available": self.available(),
                "model": "claude-opus-4-8", "protocol": "anthropic-messages"}


# ============================================================================== OpenAI-compatible backend
class OpenAICompatBackend(LLMBackend):
    """Any OpenAI-compatible endpoint (Prime Intellect, OpenAI, Together, Groq, Fireworks,
    OpenRouter, vLLM, Ollama, ...). Config-driven: adding a provider = one entry in BACKENDS."""
    priority = 50

    def __init__(self, name: str, base_url: str, api_key_env: Optional[str],
                 default_model: str, *,
                 tool_models: Sequence[str] = (),
                 key_file_suffix: Optional[str] = None,
                 key_file_env_var: Optional[str] = None,
                 api_key: Optional[str] = None):
        self.name = name
        self.base_url = base_url
        self.api_key_env = api_key_env
        self.default_model = default_model
        self.tool_models = tuple(tool_models)
        self.key_file_suffix = key_file_suffix
        self.key_file_env_var = key_file_env_var
        self._explicit_key = api_key

    def _resolve(self) -> Optional[str]:
        if self._explicit_key:
            return self._explicit_key
        if self.api_key_env:
            return resolve_key(self.api_key_env, self.key_file_suffix,
                               key_file_env_var=self.key_file_env_var)
        return "no-key-needed"

    def available(self) -> bool:
        return bool(self._resolve())

    def _pick_model(self, req) -> str:
        if req.schema is not None and self.tool_models:
            return self.tool_models[0]
        return self.default_model

    def call(self, req, timeout: float) -> Tuple[Any, dict]:
        import openai
        api_key = self._resolve()
        if not api_key:
            raise RuntimeError(f"no API key for {self.name}")
        client = openai.OpenAI(api_key=api_key, base_url=self.base_url,
                               timeout=timeout)
        model = self._pick_model(req)
        messages = [{"role": "system", "content": req.system},
                    {"role": "user", "content": _render_user(req)}]
        kwargs = dict(model=model, messages=messages, max_tokens=req.max_tokens)

        if req.schema is not None:
            return self._call_structured(client, kwargs, req, messages)
        resp = client.chat.completions.create(**kwargs)
        usage = _openai_usage(resp)
        return (resp.choices[0].message.content or "").strip(), usage

    def _call_structured(self, client, kwargs, req, messages):
        """Structured output: try tool calling first, fall back to JSON-in-prompt."""
        kwargs["tools"] = [{
            "type": "function",
            "function": {
                "name": req.tool_name,
                "description": "Emit the structured result. Call exactly once.",
                "parameters": req.schema,
            }
        }]
        if req.force_tool:
            kwargs["tool_choice"] = {"type": "function",
                                     "function": {"name": req.tool_name}}
        else:
            kwargs["tool_choice"] = "auto"
        try:
            resp = client.chat.completions.create(**kwargs)
            usage = _openai_usage(resp)
            msg = resp.choices[0].message
            if msg.tool_calls:
                raw = json.loads(msg.tool_calls[0].function.arguments)
                return raw, usage
            raise ValueError("model did not call the tool")
        except (ValueError, KeyError, json.JSONDecodeError):
            raise
        except Exception:
            pass
        # Fallback: inline JSON schema into prompt, ask for raw JSON
        del kwargs["tools"]
        if "tool_choice" in kwargs:
            del kwargs["tool_choice"]
        schema_str = json.dumps(req.schema, indent=2)
        messages[-1]["content"] += (
            f"\n\nRespond with ONLY a valid JSON object matching this schema "
            f"(no markdown, no explanation):\n```json\n{schema_str}\n```")
        resp = client.chat.completions.create(**kwargs)
        usage = _openai_usage(resp)
        text = (resp.choices[0].message.content or "").strip()
        if text.startswith("```"):
            text = text.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
        return json.loads(text), usage

    def describe(self) -> dict:
        return {"name": self.name, "available": self.available(),
                "base_url": self.base_url, "default_model": self.default_model,
                "tool_models": list(self.tool_models), "protocol": "openai-compat"}


# ============================================================================== helpers (Anthropic SDK)
def _anthropic_usage(message):
    u = getattr(message, "usage", None)
    if u is None:
        return {}
    return {"input_tokens": getattr(u, "input_tokens", None),
            "output_tokens": getattr(u, "output_tokens", None)}


def _anthropic_tool_input(message, tool_name):
    for b in getattr(message, "content", []) or []:
        if (getattr(b, "type", None) == "tool_use"
                and getattr(b, "name", "") == tool_name):
            return getattr(b, "input", None)
    return None


def _anthropic_text(message):
    return "".join(
        getattr(b, "text", "") for b in (message.content or [])
        if getattr(b, "type", None) == "text").strip()


# ============================================================================== helpers (OpenAI SDK)
def _openai_usage(resp):
    return {"input_tokens": resp.usage.prompt_tokens if resp.usage else None,
            "output_tokens": resp.usage.completion_tokens if resp.usage else None}


def _render_user(req):
    """Build the user turn (imported from ops at call time to stay in sync)."""
    from .ops import render_user
    return render_user(req)


# ============================================================================== the registry
# Ordered by priority (lower = preferred). The first available backend wins.
# Adding a new provider = one entry here. No code changes in ops.py.
BACKENDS: List[LLMBackend] = [
    AnthropicBackend(),
    OpenAICompatBackend(
        "prime-intellect",
        base_url="https://api.pinference.ai/api/v1",
        api_key_env="PRIME_INTELLECT_API_KEY",
        default_model="meta-llama/llama-3.3-70b-instruct",
        tool_models=["anthropic/claude-opus-4.8"],
        key_file_suffix=".prime_intellect_key",
        key_file_env_var="VF_PI_KEY_FILE",
    ),
    # --- additional providers: auto-activate when their env var / key file is present ---
    OpenAICompatBackend(
        "openai",
        base_url="https://api.openai.com/v1",
        api_key_env="OPENAI_API_KEY",
        default_model="gpt-4o",
        tool_models=["gpt-4o"],
        key_file_suffix=".openai_key",
    ),
    OpenAICompatBackend(
        "together",
        base_url="https://api.together.xyz/v1",
        api_key_env="TOGETHER_API_KEY",
        default_model="meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo",
        tool_models=["meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo"],
        key_file_suffix=".together_key",
    ),
    OpenAICompatBackend(
        "groq",
        base_url="https://api.groq.com/openai/v1",
        api_key_env="GROQ_API_KEY",
        default_model="llama-3.3-70b-versatile",
        tool_models=["llama-3.3-70b-versatile"],
        key_file_suffix=".groq_key",
    ),
    OpenAICompatBackend(
        "fireworks",
        base_url="https://api.fireworks.ai/inference/v1",
        api_key_env="FIREWORKS_API_KEY",
        default_model="accounts/fireworks/models/llama-v3p1-70b-instruct",
        tool_models=["accounts/fireworks/models/llama-v3p1-70b-instruct"],
        key_file_suffix=".fireworks_key",
    ),
    OpenAICompatBackend(
        "openrouter",
        base_url="https://openrouter.ai/api/v1",
        api_key_env="OPENROUTER_API_KEY",
        default_model="meta-llama/llama-3.3-70b-instruct",
        tool_models=["meta-llama/llama-3.3-70b-instruct"],
        key_file_suffix=".openrouter_key",
    ),
    OpenAICompatBackend(
        "ollama-local",
        base_url="http://localhost:11434/v1",
        api_key_env="OLLAMA_HOST",   # presence of OLLAMA_HOST activates this backend
        default_model="llama3",
        tool_models=[],
        key_file_suffix=None,
    ),
]


def resolve_backend(preference: Optional[str] = None,
                    api_key: Optional[str] = None) -> Optional[LLMBackend]:
    """Return the first available backend (or a named one if ``preference`` is given).

    ``api_key`` is an explicit key passed by the caller (e.g. the Anthropic key from
    the old resolve_api_key path). If set, it overrides the first backend that can use it.
    """
    if api_key:
        # Explicit key — try Anthropic first (legacy path), then any backend
        ab = AnthropicBackend(api_key=api_key)
        if ab.available():
            return ab
    if preference:
        for b in BACKENDS:
            if b.name == preference and b.available():
                return b
    for b in BACKENDS:
        if b.available():
            return b
    return None


def list_backends() -> List[dict]:
    """Preflight summary of all registered backends."""
    return [b.describe() for b in BACKENDS]
