# THE CORE (`frontier.core`)

THE CORE is the part of the autoresearcher that actually authors and trains models, as
opposed to the platform fabric (proposers, ranking, budget, knowledge, oracles) that surrounds
it. It is the generative modeling engine plus the execution/training substrate it runs on. It is
built **additively** against the frozen Phase-0 contract (`../CONTRACT.md`): no Phase-0 file
(`program.py`, `task.py`, `certify.py`, `sandbox.py`, `proposers.py`, `engine.py`, `__init__.py`,
`demo.py`, `tests/test_spine.py`) is edited. Design specs live in `./design/` (01–05).

Interpreter for everything below: `/Users/abdullahalghamdi/jax-env-311/bin/python`.

## The five modules + the orchestrator

| Module | Public symbols | What it is | LLM? |
| --- | --- | --- | --- |
| `authoring.py` [design 01] | `CoreAuthoringProposer`, `AuthoringEngine`, `Firewall`, `AuthoringConfig` | The generative model-authoring engine: the LLM writes arbitrary model code; a `Firewall` (AST allow-list + symbol-registry rename) sanitizes it; a recipe floor is the fallback. | LLM (degrades to `[]` offline) |
| `execution.py` [design 02] | `Executor`, `LocalSubprocessExecutor`, `RemotePodExecutor`, `BackendSpec`, `SklearnBackend`, `TorchBackend`, `SKLEARN_SPEC`, `TORCH_SPEC`, `run_program` | One backend-agnostic `Executor`. `SklearnBackend` is the default and is **bit-for-bit identical** to the Phase-0 sandbox path; `TorchBackend` is gated (declines honestly when torch is absent); GPU is a substrate swap via `RemotePodExecutor`. | no |
| `neural.py` / `render.py` [design 03] | `NeuralSpec`, `NASProposer`, `LLMArchitectProposer` | Typed `NeuralSpec` → code. `NASProposer` mutates specs (widen/deepen/regularize) deterministically; `LLMArchitectProposer` lets the LLM propose specs. A sklearn stand-in runs locally; torch runs on the pod; promotion is certify-gated. | NAS=no, Architect=LLM |
| `sandbox_policy.py` [design 05] | `SandboxPolicy`, `Tier`, `probe_host` | Layered isolation tiers (`LOCAL` → `LINUX_UID` → `CONTAINER` → `POD_GPU`). `resolve()` degrades **downward only** to what the host can actually provide; the run stamps only the guarantees that held. | no |
| `orchestrator.py` [design 04] | `CoreOrchestrator`, `CoreConfig`, `CoreResult` | Wires the whole CORE loop by composition. Owns the outer control flow and the single sealed peek; reuses the frozen `certify.py` as the only promoter. | optional |

Everything is re-exported from `frontier.core` (see `__init__.py`); `import frontier.core` is
side-effect-free and offline-safe (no torch / no LLM / no network at import time).

## Running the demo

```bash
cd <repo-root>
/Users/abdullahalghamdi/jax-env-311/bin/python frontier/core/demo_core.py
```

`demo_core.py` runs `CoreOrchestrator` end-to-end on `sklearn.datasets.load_breast_cancer`,
fully **OFFLINE** (`llm_client=None`). It exercises the whole loop:

```
router -> harness self-test -> Task -> knowledge warm-start
  -> [authoring(offline=[]) + features + neural-template + seed/mutation] proposals
  -> LinUCB rank -> budget-admitted ASHA portfolio -> diagnosis feed-forward
  -> certify the winner ONCE on the sealed split -> oracles.verify_before_promote ANDed in
  -> report
```

It prints the frozen certificate (observed / lower-bound / theta / peeks / certified), the oracle
verdict, and the per-backend provenance. `theta` is the operator's promotion bar passed in by the
caller (here 0.90), **not** a number reverse-engineered from any reference solution.

## local (sklearn) vs pod (torch)

THE CORE separates the *what to train* (a `Program` / `NeuralSpec`) from the *where to train it*
(a `BackendSpec` + an `Executor`). One contract, two substrates:

- **Local - `SKLEARN_SPEC` on `LocalSubprocessExecutor` (the default).** The classical path. It is
  bit-for-bit identical to the frozen Phase-0 `sandbox.run_program`: same npz bundle, same verbatim
  runner, same CLI, same rlimits/timeout/killpg, same OK/ERR parser, same `RunResult`. This is the
  default test path and the only path the demo uses. No torch, no GPU, no network.
- **Pod - `TORCH_SPEC` on `RemotePodExecutor`.** The neural path. The same predictions-only
  `RunResult` contract crosses to a remote GPU pod. `probe_backend` checks that torch is reachable
  on the target; if it is not, the path **declines honestly** with a typed reason
  (`[backend_unavailable]` / `missing modules: ['torch']`) instead of faking a result. Torch is
  never imported in the parent process - the firewall scan confirms `torch not in sys.modules`
  after `import frontier.core`.

So neural archs are authored and ranked locally; only the actual torch fit is dispatched to the
pod, and only when the pod really has torch + a usable GPU.

## Firewall / sealed / honest-decline guarantees

1. **Predictions-only firewall.** Untrusted candidate code never returns a number this parent did
   not compute. It writes `preds.npy` only; the parent scores. This holds at all four crossings:
   sklearn exec (Phase-0 sandbox / `core.execution` child), torch exec (pod child, same contract),
   VAL scoring (`certify.score_val`, in-parent), SEALED scoring (`certify.certify_on_sealed`,
   in-parent). `RunResult` has no `score` field by construction.
2. **One sealed peek, owned by the orchestrator.** `CoreOrchestrator` is the sole holder of the
   `Splits`. Proposers, rankers, portfolio, budget, diagnosis, and knowledge only ever see
   train/val rows. The single `certify_on_sealed` call lives in `_certify_winner`, runs once on the
   winner, and the loop asserts `sealed_peeks == 1` and `certificate["peeks"] == 1`. The oracle's
   reproducibility check uses its **own** fresh split with its own peek budget, never the winner's.
3. **Sandbox tiers stamp only what held.** `SandboxPolicy.resolve()` degrades downward to what the
   host actually provides (e.g. `setpriv`/seccomp unavailable on macOS, GPU pod with no usable GPU)
   and the certificate/audit log records the *enforced* guarantees, never the *requested* ones.
   Untrusted code on the weakest `LOCAL` tier warns (non-strict) or refuses (strict).
4. **Honest decline, never silent fake.** Every LLM path has a documented offline fallback
   (authoring → `[]`, repair → deterministic-then-give-up, neural → templates) and the result
   surfaces `llm_active` so the report states which paths were live. An unreachable `theta`, a
   refuted oracle verdict, or an unavailable backend all produce a typed honest decline
   (`result.certified = False` with `result.decline_reason`), carrying the un-promoted certificate
   for the report rather than promoting a result that did not actually clear the bar.
