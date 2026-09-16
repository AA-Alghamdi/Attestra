"""Proposal sources for the unified research cycle.

Each source implements the same interface: given the current research state,
propose one or more candidate solutions. Sources:
  - catalog: deterministic model catalog (sklearn families + tuned configs)
  - llm: LLM-generated Python code (novel architectures)
  - retrieval: informed by HuggingFace/arXiv search
  - mutation: mutate/combine existing successful solutions
"""
