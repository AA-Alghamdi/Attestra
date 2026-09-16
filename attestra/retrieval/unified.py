"""Unified retriever: re-exports from the existing vfplatform.hf_retrieval.

This is a compatibility layer. The actual retrieval logic lives in
vfplatform/hf_retrieval.py (which is well-implemented and working).
"""
from __future__ import annotations

try:
    from vfplatform.hf_retrieval import (
        UnifiedRetriever,
        HFRetriever,
        ArxivRetriever,
        RetrievedKnowledge,
    )
except ImportError:
    # Fallback: define stubs
    class RetrievedKnowledge:  # type: ignore[no-redef]
        def __init__(self, **kw):
            for k, v in kw.items():
                setattr(self, k, v)

    class UnifiedRetriever:  # type: ignore[no-redef]
        def __init__(self, **kw):
            pass
        def search(self, *a, **kw):
            return []
        def format_for_llm(self, results):
            return "(retrieval unavailable)"
        def search_for_task(self, *a, **kw):
            return []

    class HFRetriever:  # type: ignore[no-redef]
        def __init__(self, **kw):
            pass

    class ArxivRetriever:  # type: ignore[no-redef]
        def __init__(self, **kw):
            pass

__all__ = ["UnifiedRetriever", "HFRetriever", "ArxivRetriever", "RetrievedKnowledge"]
