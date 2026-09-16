"""frontier.core: THE CORE of the autoresearcher.

The generative modeling engine + execution/training substrate that actually solves ML
problems, as opposed to the platform layer around it. Built against the frozen Phase-0
contract (see ../CONTRACT.md) and the design specs in ./design/. Additive: nothing in
Phase 0 / vectorforge / vfplatform / attestra is edited.

Components (see ./design/ and ./README.md):
  - authoring.py      : the generative model-authoring engine (LLM writes arbitrary model
                        code; firewalled; recipe floor as fallback). [design 01]
  - execution.py      : one backend-agnostic Executor (SklearnBackend now, TorchBackend
                        gated); GPU/pod as a substrate swap. [design 02]
  - neural.py / render.py : typed NeuralSpec -> code; sklearn stand-in locally, torch gated;
                        certify-gated promotion. [design 03]
  - sandbox_policy.py : layered sandbox tiers; stamps only the guarantees that held. [design 05]
  - orchestrator.py   : CoreOrchestrator wiring the end-to-end CORE loop by composition,
                        reusing the frozen certify path for the single sealed peek. [design 04]

Public surface (the symbols an integrator wires into the spine):
  CoreOrchestrator, CoreConfig, CoreResult     -- the end-to-end loop (supersedes ResearchEngine.run)
  CoreAuthoringProposer, AuthoringEngine        -- generative model authoring (LLM; [] offline)
  Executor, LocalSubprocessExecutor, RemotePodExecutor,
  BackendSpec, SklearnBackend, TorchBackend,
  SKLEARN_SPEC, TORCH_SPEC, run_program         -- the backend-agnostic execution substrate
  NeuralSpec, NASProposer, LLMArchitectProposer -- the neural core
  SandboxPolicy, Tier, probe_host               -- layered sandbox tiers

Importing this package is side-effect-free and offline-safe: no torch, no LLM client, and no
network are touched at import time (verified by the execution leaf test).
"""

# THE CORE end-to-end loop (composition over the frozen ResearchEngine).
from frontier.core.orchestrator import (
    CoreOrchestrator,
    CoreConfig,
    CoreResult,
)

# The generative model-authoring engine (LLM-backed; degrades to [] offline).
from frontier.core.authoring import (
    CoreAuthoringProposer,
    AuthoringEngine,
    AuthoringConfig,
    Firewall,
)

# The backend-agnostic execution substrate (sklearn now, torch/pod gated).
from frontier.core.execution import (
    Executor,
    LocalSubprocessExecutor,
    RemotePodExecutor,
    BackendSpec,
    SklearnBackend,
    TorchBackend,
    SKLEARN_SPEC,
    TORCH_SPEC,
    run_program,
)

# The neural core: typed spec -> code, NAS + LLM-architect proposers.
from frontier.core.neural import (
    NeuralSpec,
    NASProposer,
    LLMArchitectProposer,
)

# Layered sandbox policy: tiers, host probe, guarantee stamping.
from frontier.core.sandbox_policy import (
    SandboxPolicy,
    Tier,
    probe_host,
)

__all__ = [
    # orchestrator
    "CoreOrchestrator",
    "CoreConfig",
    "CoreResult",
    # authoring
    "CoreAuthoringProposer",
    "AuthoringEngine",
    "AuthoringConfig",
    "Firewall",
    # execution
    "Executor",
    "LocalSubprocessExecutor",
    "RemotePodExecutor",
    "BackendSpec",
    "SklearnBackend",
    "TorchBackend",
    "SKLEARN_SPEC",
    "TORCH_SPEC",
    "run_program",
    # neural
    "NeuralSpec",
    "NASProposer",
    "LLMArchitectProposer",
    # sandbox policy
    "SandboxPolicy",
    "Tier",
    "probe_host",
]
