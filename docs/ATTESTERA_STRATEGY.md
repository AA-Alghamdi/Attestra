# Attestera strategy (synthesized from the session's comments)

## 0. Thesis: two equal non-negotiables
Attestera is an autoresearcher that must **generalize to any/all ML problems** AND produce **rigorous,
defensible, replayable certifications**. These are co-equal. The single invariant that lets us be aggressive
on one without sacrificing the other:

> The frozen deterministic certifier is the ONLY thing that can promote. The LLM lives only at the soft edges
> (intake, propose-next-move, narrate, synthesize data). Delete the LLM and the certificates still come out.

That separation is what licenses bold generalization (the LLM may propose anything) while keeping rigor
(only the frozen core promotes, on a sealed test, with no threshold relaxation).

## 1. Where we are (honest state)
Built + verified this session:
- Recursive LLM-guided cycle: diagnose -> LLM-propose moves -> VoI-rank -> execute -> measure -> update,
  many rounds (39 to 47 in tests), big model zoo, time/round budget, maximize-mode + best_effort.
- Frozen core intact: Clopper-Pearson + bootstrap + Bonferroni, sealed one-peek test, leakage audit, peek
  ledger, VoI casebase. Certify-or-honest-stop. Fingerprints unchanged across all swarm edits.
- Attestera-rebranded UI: live 2D recursive cycle, Stop button, GPU options, no em dashes.
- Public deploy: lab.abdullahalghamdi.com behind Caddy (HTTPS + basic auth) + per-day spend cap.

In flight: select-then-bound fix (certify-mode multiplicity regression).

Open gaps: only 4 modalities (tabular/text/timeseries/ranking) -> the big generalization gap; RunPod worker
health (config/image); soft-edge resilience; cross-experiment rigor; portfolio scaling.

## 2. Sections to ADD, by axis

### A. GENERALIZATION (the biggest gap vs "any/all problems"; promote from "honest decline" to a build track)
- A1. Modality-adapter framework: a clean Harness contract (featurizer + model catalog + target encode +
  certifier-compatible metric) so a new modality is a plug-in, not a fork.
- A2. New modalities in priority order, each shipping ONLY when its frozen certifier + leakage audit are
  defensible: vision (image classification) -> deep sequence/NLP -> graph/recsys -> generative/RL (hardest).
- A3. General problem-typing intake: deterministic profiler + LLM classify the uploaded problem into an
  adapter, else honestly decline with "exactly what is missing to support it."
- A4. Dataset admissibility/inspector gate (adopt from ml-intern): schema + samples + format verdict +
  leakage/contamination pre-check before any run, for arbitrary uploaded data.
- A5. "Deep research" proposer: the LLM reasons about the problem type and proposes appropriate,
  modality-specific candidate families (bounded by the adapter catalog, verified + clamped), not a fixed list.

### B. RIGOR (keep and extend the moat)
- B1. select-then-bound (in flight): correct selection multiplicity so trying more models never makes a
  genuinely-good model fail to certify, while a no-signal dataset still honest-stops.
- B2. Pre-registration commitment: content-address the spec (metric, theta, alpha, forbidden fields,
  selection rule); post-hoc bending is impossible; a relaxed theta becomes a new recorded commitment.
  Mechanizes the no-anchoring / no-threshold-relaxation rule.
- B3. Cross-experiment multiplicity (online-FDR, LORD/SAFFRON) over the sequence of promotions. Needed once
  we run many experiments/goals; this is the answer to "the full cycle spanning multiple experiments."
- B4. Negative-result / STOPPED-FUTILE certificates: honest-stop becomes a first-class recorded negative
  (tighter alpha), so "what does not work" is a real, auditable output.
- B5. Research track (dedicated, nemesis-gated, decided together): WAL effect-log fold(state,effects) +
  Jepsen-style deterministic simulation testing + e-process/DP anytime-valid + N-version binomial. High
  value for the paper, high risk to the trust core; never bolted on between features.

### C. ADAPTABILITY & SCALING
- C1. Portfolio scheduler: a layer above the single-goal loop that spans many experiments/approaches, fans
  out, kills-and-reclaims, and selects the best ("direct a research policy across many experiments").
- C2. Cost-aware VoI allocation: a real cost model (priced GPU flavors) feeding compute allocation.
- C3. GPU scaling via RunPod: fix worker health (config/image) + a real capability/health probe so GPU is
  used when truly available and falls back honestly otherwise. Parallel fan-out already built.
- C4. Content-addressed cross-goal memoization + warm-start: never recompute a bit-identical unit; warm-start
  the VoI casebase across goals.

### D. SOFT-EDGE RESILIENCE (adopt-now from ml-intern; pure leverage, no trust-core risk)
- D1. Transient-vs-persistent error taxonomy + differentiated backoff for fan-out across flaky endpoints
  (directly fixes RunPod robustness).
- D2. Capability/health probe before committing a run (honest "GPU unavailable").
- D3. Doom-loop / stall detector keyed on the result-hash (not just name+args).
- D4. Approval + budget-reservation refinements (we already have the spend cap + checkpoint).

### E. UX & OPS
- Done: Attestera rebrand, live recursive 2D cycle, Stop, GPU options, no em dashes, public HTTPS + auth +
  spend cap.
- Add: honest GPU-status messaging (from D2), examples/help surface, per-run cost display, a clear
  "what it can and cannot do yet" modality-coverage panel.

### F. SELF-IMPROVEMENT (research track, deferred)
- Certified-outcomes flywheel: a calibrator + a corpus surrogate, gated ONLY on frozen-core promotion events,
  never feeding the trusted core. Reject ml-intern's trace-SFT (it launders contaminated trajectories).

## 3. Updated architecture (layered stack over one append-only content-addressed ledger)
The frozen certifier is the only promoter; the LLM only at the soft edges.

    /goal (NL: metric, threshold, latency, cost, time budget)
      -> INTAKE            profiler + LLM problem-typing + light adversarial operationalization check
      -> PRE-REGISTRATION  content-addressed spec (metric, theta, alpha, forbidden fields, selection rule)
      -> MODALITY ADAPTER  featurizer + model catalog + metric + certifier hook (plug-in per modality)
      -> EXPERIMENT DESIGN VoI + power, cost-aware
      -> ORCHESTRATION     portfolio fan-out across candidates/modalities/GPUs; error-taxonomy + health-probe
      -> RECURSIVE CYCLE   diagnose -> LLM-propose -> VoI-rank -> checkpoint -> execute -> measure -> update
      -> CERTIFICATION     FROZEN: select-then-bound val gate -> ONE sealed peek -> CP/bootstrap
                                   -> cross-experiment FDR over the promotion sequence
      -> LEDGER            content-addressed; positive AND negative certificates
      -> [research track]  SELF-IMPROVEMENT (calibrator + corpus surrogate, promotion-gated)

Two standing invariants, enforced every phase:
1. The frozen certifier is the only promoter; deleting the LLM leaves certificates byte-stable.
2. Generalization expands what the LLM may PROPOSE and which adapters exist; it never expands what may PROMOTE.

## 4. Updated build order (phased, tests-first, delicate)
- Phase 0 (now): select-then-bound fix + hardened audit (UI test + bad-model-fails gate) -> verify -> redeploy.
- Phase 1: soft-edge resilience (error taxonomy + health-probe + doom-loop) + fix the RunPod worker image.
  Lowest risk, fixes current pain.
- Phase 2: rigor adds: pre-registration -> cross-experiment FDR -> negative certificates.
- Phase 3: GENERALIZATION: modality-adapter framework + first new modality (vision) + problem-typing intake
  + dataset admissibility gate. The biggest lever for "any/all problems."
- Phase 4: scaling: portfolio scheduler + cost-aware VoI + cross-goal memoization.
- Phase 5 (dedicated research track): WAL fold + DST nemesis + e-process/DP + flywheel.

## 5. Updated workflow (how we build each phase) -- the hardened method
Every phase runs as a swarm workflow:
1. Disjoint-file parallel implement to a pinned contract.
2. Parallel multi-dimensional audit that NOW ALWAYS includes:
   (a) frozen-core fingerprint check (science.py / sealed.py unchanged),
   (b) the FULL test suite INCLUDING webapp/test_client.py (the gap that let the regression through),
   (c) a behavioral regression gate: good models certify AND no-signal data still honest-stops AND no
       previously-supported modality regresses,
   (d) prove-on-real-data with expectations treated as BLOCKING (not advisory).
3. Repair until clean.
4. MY independent verification (fingerprints + full suite + UI test + a real run).
5. Rebuild the deploy bundle -> redeploy.

Tests-first. The frozen core stays untouched unless a phase explicitly scopes it, and then only with property
tests + a fingerprint diff + the bad-model-fails guard, and only after we decide together.
