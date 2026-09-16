"""B2 -- DOES AN LLM-AUTHORED NOVEL METHOD BEAT A STRONG BASELINE GIVEN THE REPRESENTATION?

This is the gate B1 set: B1 proved (5/5 FDR) that on a headroom vision arena the win is captured by
REPRESENTATION (frozen resnet18 embeddings), and that the search CYCLE adds nothing over a tuned GBM on those
embeddings (0/5). The only untested lever left to EXCEED a strong baseline given a fixed representation is
NOVEL-METHOD AUTHORING: a full-strength LLM writes a sklearn-compatible estimator, the FROZEN three-stage
admission gate (static AST -> isolated exec -> scientific self-test) admits or rejects it on SAFETY +
BUILDABILITY + CONTRACT only, and -- exactly like a zoo family -- the method is selected on validation and
bounded ONCE on the IDENTICAL sealed test B1 used. Quality is decided ONLY by the frozen lower bound; the LLM
never promotes.

Arms, all on the SAME sealed test (n per task identical to B1), paired one-sided exact McNemar + BH-FDR(0.1):
  A2 emb-strong : tuned GBM + random search over the catalog, on frozen resnet18 embeddings (the comparator).
  D  authored   : the best-on-VALIDATION LLM-authored method (select-then-bound, checks=1), on the SAME
                  embeddings, scored on the SAME sealed rows.
The frontier claim is D > A2 with an FDR-surviving lift. EXPECTED (per B1/tabular) ~0; a negative result is a
first-class deliverable and we say so. The frozen certifier is untouched (read-only clopper_pearson_lower).

SECURITY: authoring (the only step that needs the key) runs the admission self-test in a SPAWNED, rlimited,
env-scrubbed child. Authored fit/predict on REAL embeddings runs in-process AFTER the key is removed from the
environment (defense-in-depth on top of the static gate that already denies os/sys/file/net/import-tricks and
numpy C-extension I/O). Run only on a throwaway host.
"""
import hashlib
import json
import os
import sys
import time
import warnings

import numpy as np

warnings.filterwarnings("ignore")   # authored methods may use soon-deprecated sklearn kwargs; not our concern

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))   # so we can reuse the B1 helpers verbatim

import benchmark_vision_transfer as B1                           # noqa: E402  (shared arena construction)
from vectorforge import science                                  # noqa: E402
from vfplatform.loop import _split                               # noqa: E402
from vfplatform.battery import mcnemar_pvalue, benjamini_hochberg  # noqa: E402
from vfplatform.featurizers import ImageFeaturizer, ResnetBackbone  # noqa: E402
from vfplatform import authoring as A                            # noqa: E402

ALPHA = B1.ALPHA
SUITE = B1.SUITE
PER_CLASS = int(os.environ.get("ATTESTRA_PER_CLASS", "500"))
N_AUTHOR = int(os.environ.get("ATTESTRA_N_AUTHOR", "8"))         # how many novel methods to author
AUTHOR_MODEL = os.environ.get("ATTESTRA_AUTHOR_MODEL", "claude-opus-4-8")


def _scrub_secrets():
    """Remove every key/token/secret from the parent environment BEFORE any authored code is fit/predicted
    in-process. The authoring API calls are already done by the time this runs."""
    for k in list(os.environ.keys()):
        up = k.upper()
        if "KEY" in up or "TOKEN" in up or "SECRET" in up or "ANTHROPIC" in up:
            os.environ.pop(k, None)


def _env_context():
    """Accurate, trusted environment facts so the author targets the INSTALLED API (not a removed/renamed
    one). This is plain library-version information, not a quality hint and not task data."""
    import sklearn
    import numpy
    return {
        "environment": {"python": "3.12", "numpy": numpy.__version__, "sklearn": sklearn.__version__},
        "api_notes": [
            "Target the EXACT installed sklearn version above; do NOT pass removed/renamed kwargs "
            "(e.g. recent sklearn LogisticRegression has no 'multi_class'; OneVsRest handles multiclass).",
            "The estimator is fit on ~700 rows of 512-dim L2-normalized CNN-embedding features for a "
            "BINARY task; ensure every matrix multiply / broadcast has consistent shapes.",
            "Prefer current stable sklearn APIs; thread the seed into every random_state.",
        ],
    }


def author_library(n, *, model, api_key, max_revise=2):
    """Author up to `n` DISTINCT novel 512-d -> 2-class classifiers. Each rides the frozen admit() gate; only
    admitted, de-duplicated (by code_digest) methods are kept. On a rejection the author is shown its OWN
    failure reason and may REVISE (up to max_revise times) -- the frozen gate still decides admission. To
    steer variety, the author is told which inductive biases were already admitted. Returns (kept, log)."""
    spec = A.EstimatorSpec(role="classifier", n_features=512, n_classes=2, family_prefix="authored")
    env = _env_context()
    kept, seen, log = [], set(), []
    slot = 0
    while len(kept) < n and slot < n:
        slot += 1
        prior_attempts = []           # (code, rejection_reason) fed back so the author can fix its own bugs
        admitted_here = False
        for rev in range(max_revise + 1):
            extra = dict(env)
            extra["already_admitted_biases"] = [{"family": e.family, "rationale": e.rationale} for e in kept]
            extra["goal"] = ("Author a method with a GENUINELY DIFFERENT inductive bias from the "
                             "already_admitted_biases and from a plain forest/boosting/linear baseline.")
            if prior_attempts:
                extra["your_previous_rejected_attempts"] = prior_attempts[-2:]
                extra["fix_instruction"] = ("Your previous code was REJECTED by the frozen admission gate for "
                                            "the reason shown. Emit a corrected, self-contained method.")
            res = A.author_estimator(spec, api_key=api_key, use_llm=True, cache_path=None,
                                     model=model, timeout=240.0, extra_context=extra)
            if res.authored and res.estimator is not None:
                est = res.estimator
                if est.code_digest in seen:
                    log.append({"slot": slot, "rev": rev, "authored": True, "duplicate": True,
                                "family": est.family})
                    break
                seen.add(est.code_digest)
                kept.append(est)
                admitted_here = True
                log.append({"slot": slot, "rev": rev, "authored": True, "family": est.family,
                            "rationale": est.rationale, "code_digest": est.code_digest, "usage": res.usage})
                print(f"    authored #{len(kept)} (slot {slot}, rev {rev}): {est.family} -- {est.rationale[:80]}")
                break
            reason = res.reason
            log.append({"slot": slot, "rev": rev, "authored": False, "reason": reason, "usage": res.usage})
            print(f"    slot {slot} rev {rev} REJECTED: {reason[:110]}")
            prior_attempts.append({"reason": reason[:400]})
        if not admitted_here:
            print(f"    slot {slot}: no admitted method after {max_revise + 1} attempts")
    return kept, log


def _fit_score(est_factory, Xtr, ytr, Xeval, yeval):
    """Build (seed=0), fit on TRAIN, return (accuracy_on_eval, per-row correct list). Never raises -> a method
    that blows up on real data scores 0 honestly (it just loses selection)."""
    try:
        builder = est_factory.catalog_entry().builder
        est = builder({}, 0)
        est.fit(Xtr, ytr)
        pred = np.asarray(est.predict(Xeval))
        correct = (pred == yeval).astype(int).tolist()
        return float(np.mean(correct)), correct
    except Exception as ex:  # noqa: BLE001
        return -1.0, None


def run_task(imgs, labels, cls_a, cls_b, backbone, authored, per_class, seed=0):
    """Reconstructs B1's IDENTICAL embeddings / sealed test / train-val split, recomputes the emb-strong
    comparator, then selects the best-on-val authored method and bounds it ONCE on the same sealed rows."""
    name = f"{cls_a}_vs_{cls_b}"
    task_imgs, y = B1._task_data(imgs, labels, cls_a, cls_b, per_class, seed)
    n = len(y); rids = np.arange(n)
    emb = ImageFeaturizer(backbone=backbone, backbone_dim=backbone.feat_dim).transform(task_imgs)
    emb_recs = B1._records(emb, y, rids)

    # outer sealed test = identical fresh stratified 30% as B1 (same seed+7), held out before any fit.
    rng = np.random.RandomState(seed + 7)
    test_ids = []
    for c in (0, 1):
        ci = rids[y == c]; rng.shuffle(ci); test_ids += list(ci[: int(0.3 * len(ci))])
    test_ids = set(int(i) for i in test_ids)
    trainval_ids = [int(i) for i in rids if int(i) not in test_ids]

    tv_emb_recs = [emb_recs[i] for i in trainval_ids]
    tr2, va2, te2 = _split(tv_emb_recs, "target", "text", seed, False)
    tr_pool_ids = [r["rid"] for r in (tr2 + te2)]
    val_ids = [r["rid"] for r in va2]
    test_ids_l = sorted(test_ids)

    def _slice(ids, M):
        ids = list(ids); return M[ids], y[ids]

    Xtr, ytr = _slice(tr_pool_ids, emb)
    Xva, yva = _slice(val_ids, emb)
    Xte, yte = _slice(test_ids_l, emb)

    # A2 emb-strong: the SAME comparator B1 used.
    a2 = B1._random_search_best(np.random.RandomState(seed), Xtr, ytr, Xva, yva)
    a2_c = B1._correct(a2, Xte, yte)

    # D authored: free selection on VALIDATION, then bound the single winner ONCE on the sealed test.
    val_scores = []
    for est in authored:
        acc_va, _ = _fit_score(est, Xtr, ytr, Xva, yva)
        val_scores.append(acc_va)
    if not authored or max(val_scores) < 0:
        return {"name": name, "n_test": len(yte), "no_authored": True,
                "acc_emb_strong": round(float(np.mean(a2_c)), 4), "lb_emb_strong": B1._lb(a2_c)}
    win_i = int(np.argmax(val_scores))
    winner = authored[win_i]
    acc_te, d_c = _fit_score(winner, Xtr, ytr, Xte, yte)     # refit on TRAIN, score on the SEALED rows

    return {"name": name, "n_test": len(yte),
            "authored_winner": winner.family, "authored_rationale": winner.rationale,
            "authored_val_acc": round(float(val_scores[win_i]), 4),
            "n_authored": len(authored),
            "acc_emb_strong": round(float(np.mean(a2_c)), 4), "lb_emb_strong": B1._lb(a2_c),
            "acc_authored": round(float(np.mean(d_c)), 4), "lb_authored": B1._lb(d_c),
            "lift_authored_vs_embstrong": round(float(np.mean(d_c) - np.mean(a2_c)), 4),
            "p_authored_gt_embstrong": mcnemar_pvalue(d_c, a2_c),
            "p_embstrong_gt_authored": mcnemar_pvalue(a2_c, d_c)}


def main():
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        print("[B2] no ANTHROPIC_API_KEY in env -- cannot author. Aborting (honest: nothing to measure).")
        sys.exit(2)
    print(f"[B2] novel-method authoring arena | per_class={PER_CLASS} | n_author={N_AUTHOR} | model={AUTHOR_MODEL}")
    imgs, labels = B1._load_cifar()
    backbone = ResnetBackbone(arch="resnet18", resize=112, batch_size=64, pretrained=True)

    print("  authoring novel methods (LLM -> frozen admission gate)...")
    authored, author_log = author_library(N_AUTHOR, model=AUTHOR_MODEL, api_key=api_key)
    n_attempts = len(author_log)
    n_admitted = len(authored)
    print(f"  -> {n_admitted} distinct admitted method(s) from {n_attempts} attempt(s).")

    _scrub_secrets()          # the key is no longer needed; remove it before running authored code in-process

    rows = []
    for a, b in SUITE:
        t0 = time.time()
        try:
            r = run_task(imgs, labels, a, b, backbone, authored, PER_CLASS)
        except Exception as e:  # noqa: BLE001
            r = {"name": f"{a}_vs_{b}", "error": str(e)[:200]}
        r["secs"] = round(time.time() - t0, 1)
        rows.append(r)
        if "error" in r:
            print(f"  {r['name']:18} ERROR {r['error']}")
        elif r.get("no_authored"):
            print(f"  {r['name']:18} NO ADMITTED METHOD; emb_strong={r['acc_emb_strong']}(lb{r['lb_emb_strong']})")
        else:
            print(f"  {r['name']:18} emb_strong={r['acc_emb_strong']}(lb{r['lb_emb_strong']}) "
                  f"authored={r['acc_authored']}(lb{r['lb_authored']}) "
                  f"D>embGBM {r['lift_authored_vs_embstrong']:+.3f}"
                  f"(p{r['p_authored_gt_embstrong']:.1e}) [{r['authored_winner']}] [{r['secs']}s]")

    ok = [r for r in rows if "p_authored_gt_embstrong" in r]
    summary = {}
    for key, liftk, label in [("p_authored_gt_embstrong", "lift_authored_vs_embstrong",
                               "AUTHORED vs EMB-STRONG")]:
        ps = [r[key] for r in ok]
        rej = set(benjamini_hochberg(ps, alpha=0.1)) if ps else set()
        wins = [ok[i]["name"] for i in range(len(ok)) if i in rej and ok[i][liftk] > 0]
        summary[label] = {"survivors": wins, "n": len(ok)}
        print(f"\n=== {label}: BH-FDR(0.1) survivors with positive lift: {len(wins)}/{len(ok)} -> {wins}")

    print("\nHONEST READING:")
    print("  AUTHORED > EMB-STRONG survivors -> an LLM-authored method beats a STRONG baseline given the")
    print("                                     representation (the frontier claim). If 0, authoring -- like")
    print("                                     search -- does not beat a tuned GBM on these embeddings, and")
    print("                                     the lever that moves the metric remains the REPRESENTATION.")

    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    fh = {f: hashlib.sha256(open(os.path.join(root, f), "rb").read()).hexdigest()[:8]
          for f in ("vectorforge/science.py", "vfplatform/sealed.py")}
    out = {"per_class": PER_CLASS, "n_author_requested": N_AUTHOR, "model": AUTHOR_MODEL,
           "n_admitted": n_admitted, "n_attempts": n_attempts, "author_log": author_log,
           "rows": rows, "summary": summary, "frozen_hashes": fh}
    dst = os.path.join(root, "docs", "BENCHMARK_AUTHORING_RESULT.json")
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    with open(dst, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nwrote {dst}")
    print(f"frozen hashes: {fh}")


if __name__ == "__main__":
    main()
