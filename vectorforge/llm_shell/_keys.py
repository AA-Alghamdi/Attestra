"""Resolve the Anthropic API key without ever committing it.

Resolution order (first hit wins):
  1. an explicit `api_key=` argument,
  2. the ANTHROPIC_API_KEY environment variable,
  3. a local, GIT-IGNORED key file: `<repo>/.anthropic_key` (or $VF_KEY_FILE).

The key file holds ONLY the key on its first non-comment line. It is in .gitignore and must never be
committed. Returns None when no key is found (callers then use the deterministic fallback).
"""
import os

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _key_file_path():
    # Resolved at CALL time (not import) so tests can redirect it via the VF_KEY_FILE env var.
    return os.environ.get("VF_KEY_FILE", os.path.join(_REPO_ROOT, ".anthropic_key"))


def _from_file(path):
    try:
        with open(path, "r") as fh:
            for line in fh:
                s = line.strip()
                # Accept only a real-looking key; the placeholder / comments are ignored, so an
                # un-edited key file correctly resolves to "no key" (deterministic fallback).
                if s and not s.startswith("#") and s.startswith("sk-"):
                    return s
    except (OSError, IOError):
        return None
    return None


def resolve_api_key(explicit=None):
    if explicit:
        return explicit
    env = os.environ.get("ANTHROPIC_API_KEY")
    if env:
        return env
    return _from_file(_key_file_path())
