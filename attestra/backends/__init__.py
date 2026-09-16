"""LLM backend registry: re-exports from vectorforge.llm_shell.backends."""
try:
    from vectorforge.llm_shell.backends import get_backend, list_backends, Backend
except ImportError:
    def get_backend(name: str):
        raise ImportError(f"Backend {name} requires vectorforge.llm_shell.backends")
    def list_backends():
        return []
    class Backend:  # type: ignore[no-redef]
        pass
