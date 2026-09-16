"""Provenance join (build-order item 3): enforce that the CERTIFIED artifact IS the leaderboard winner.

The swarm flagged that "certified == deployed" is fiction unless verifiable. The loop certifies the
estimator of `leaderboard.best()` by construction; this module makes that an explicit, re-derivable check
so a swap between selection and certification is caught. Also re-derives the winner from the leaderboard
independently and confirms the certificate was produced from it.
"""


class ProvenanceError(Exception):
    pass


def verify_provenance(result):
    """Return (ok, reasons). ok iff the certificate's winner_run == the independently re-derived
    validation winner, the sealed digest is present, and the peek count is exactly 1."""
    reasons = []
    cert = result.certificate
    if cert is None:
        return (result.decision in ("blocked", "no_model"),
                [f"no certificate (decision={result.decision})"])
    rederived = result.leaderboard.best()
    if rederived is None:
        reasons.append("leaderboard has no finished winner to re-derive")
    elif cert.get("winner_run") != rederived.run_id:
        reasons.append(f"certified winner_run {cert.get('winner_run')} != re-derived best "
                       f"{rederived.run_id} (selection/certification mismatch)")
    if not cert.get("sealed_digest", "").startswith("sha256:"):
        reasons.append("certificate is not bound to a content-addressed sealed test")
    if cert.get("peeks") != 1:
        reasons.append(f"sealed test peeked {cert.get('peeks')} times (expected exactly 1)")
    return (len(reasons) == 0, reasons)


def assert_provenance(result):
    ok, reasons = verify_provenance(result)
    if not ok:
        raise ProvenanceError("provenance check failed: " + "; ".join(reasons))
    return True
