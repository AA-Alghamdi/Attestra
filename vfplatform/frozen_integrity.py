"""Read-only integrity check for the FROZEN certifier core.

WHY THIS EXISTS (the load-bearing reason). This is a certify-or-honest-fail system: every certificate
is produced ONLY by the frozen core (vectorforge/science.py + vfplatform/sealed.py). The audit's
"the frozen core is untouched" claim is only as strong as the baseline it compares against. The prior
audit harness shipped the baseline as the literal string "undefined" for both files, so the read-only
hash check could not actually PROVE the core was unmodified -- "undefined" != any real hash, so the
comparison was vacuous. This module pins the REAL sha256 constants (in frozen_manifest.json) and turns
the check into a real one: it fails on ANY difference from the pinned values.

This module does NOT touch the frozen files. It only reads them and hashes them. It is the in-repo
counterpart of the external audit's read-only hash gate, so the same pinned baseline is enforced both
by `python test_vfplatform.py` and by the audit harness (which reads MANIFEST_PATH / the constants
below). If either frozen file is legitimately re-frozen, recompute and re-pin in frozen_manifest.json
in the SAME commit and document why; do not loosen this check to make it pass.
"""
import hashlib
import json
import os

# Repo root = parent of this package directory.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MANIFEST_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "frozen_manifest.json")

# The REAL pinned baseline, mirrored as module constants so an audit harness can import them directly
# without parsing JSON. These MUST match frozen_manifest.json (verify_frozen_core cross-checks both).
FROZEN_SHA256 = {
    "vectorforge/science.py": "b564fba248ea009495fcfc2cb7e2e14a1d1285c2f46165cd9401dfbcd461270d",
    "vfplatform/sealed.py": "30ad62450c53cc4a52ed55e65593352c146549d24e756dcb38532569007c8661",
}


class FrozenCoreTampered(Exception):
    """Raised when a frozen-core file's live sha256 differs from its pinned baseline."""


def _sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def load_pinned():
    """Return the pinned {relpath: sha256} baseline from frozen_manifest.json, cross-checked against the
    in-module FROZEN_SHA256 constants. A mismatch between the two pinning sites is itself an error: it
    means someone edited one place and not the other, which would silently weaken the gate."""
    with open(MANIFEST_PATH, "r") as fh:
        manifest = json.load(fh)
    pinned = manifest.get("files", {})
    if pinned != FROZEN_SHA256:
        raise FrozenCoreTampered(
            "frozen baseline disagrees between frozen_manifest.json and FROZEN_SHA256 constants; "
            f"manifest={pinned} constants={FROZEN_SHA256}. Re-pin both together.")
    # Guard against a "undefined"/placeholder baseline ever sneaking back in.
    for rel, want in pinned.items():
        if not (isinstance(want, str) and len(want) == 64 and all(c in "0123456789abcdef" for c in want)):
            raise FrozenCoreTampered(
                f"pinned baseline for {rel} is not a real sha256 hex digest: {want!r}. "
                "A non-hex baseline (e.g. 'undefined') makes the read-only check vacuous; refusing.")
    return pinned


def compute_live():
    """Return the live {relpath: sha256} of the frozen-core files."""
    return {rel: _sha256_file(os.path.join(_REPO_ROOT, rel)) for rel in FROZEN_SHA256}


def verify_frozen_core(*, raise_on_fail=True):
    """Verify each frozen-core file's live sha256 against its REAL pinned baseline.

    Returns (ok: bool, report: dict[relpath -> {"expected", "actual", "match"}]). When raise_on_fail
    is True (the audit/test default) it raises FrozenCoreTampered on ANY mismatch -- no silent pass."""
    pinned = load_pinned()
    live = compute_live()
    report = {}
    ok = True
    for rel, expected in pinned.items():
        actual = live.get(rel)
        match = (actual == expected)
        ok = ok and match
        report[rel] = {"expected": expected, "actual": actual, "match": match}
    if not ok and raise_on_fail:
        bad = {r: v for r, v in report.items() if not v["match"]}
        raise FrozenCoreTampered(
            "FROZEN CORE MODIFIED -- live sha256 differs from pinned baseline. The certificate root of "
            f"trust is no longer the reviewed core. Offending files: {bad}")
    return ok, report


if __name__ == "__main__":
    import sys
    ok, report = verify_frozen_core(raise_on_fail=False)
    for rel, v in report.items():
        flag = "OK " if v["match"] else "BAD"
        print(f"  [{flag}] {rel}  expected={v['expected']}  actual={v['actual']}")
    print("frozen core intact" if ok else "FROZEN CORE TAMPERED")
    sys.exit(0 if ok else 1)
