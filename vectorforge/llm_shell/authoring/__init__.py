"""A8 -- build-time authoring of data connectors behind a frozen safety + conformance harness.

An LLM authors CODE (a `load(payload) -> list[dict]` connector) from a spec + a QUARANTINED sample
payload. Because the output is executable, the bound is three layers, all frozen:
  1. a STRICT AST static gate (import allowlist + banned-name/dunder denylist) -- the real defense,
  2. a RESTRICTED sandbox exec (safe builtins only, SIGALRM timeout) -- defense in depth,
  3. a CONFORMANCE + ACCEPTANCE suite the generated connector must pass.
Only code that clears all three is surfaced `ready_for_review`. It is NEVER auto-merged and NEVER run
on the live request path -- A8 is build/CI-time, converting human-CODING cost into human-REVIEW cost.

Honest limit (NOT overclaimed): in-process restricted exec is not a hardened security boundary. The
AST allowlist is the primary control (only pure data-transformation code passes); for fully-untrusted
authoring at scale, run the sandbox in a separate process/container with seccomp. Documented, not hidden.
"""
from .contract import ConnectorSpec, ConformanceCase
from .static_gate import static_check, StaticReport
from .sandbox import run_entrypoint
from .conformance import run_conformance, ConformanceReport
from .author import author_connector, AuthoringResult

__all__ = ["ConnectorSpec", "ConformanceCase", "static_check", "StaticReport", "run_entrypoint",
           "run_conformance", "ConformanceReport", "author_connector", "AuthoringResult"]
