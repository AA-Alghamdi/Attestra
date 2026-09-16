"""Run the conformance + acceptance suite against a (statically-gated, sandboxed) connector."""
from dataclasses import dataclass, field

from .static_gate import static_check, StaticReport
from .sandbox import run_entrypoint
from .contract import ConnectorSpec


@dataclass
class ConformanceReport:
    static: StaticReport
    passed: bool
    cases: list = field(default_factory=list)   # [{name, ok, reason}]

    def as_dict(self):
        return {"static": self.static.as_dict(), "passed": self.passed, "cases": list(self.cases)}


def run_conformance(code: str, spec: ConnectorSpec, *, timeout=3.0) -> ConformanceReport:
    """Static gate FIRST (never execute ungated code), then every standard + spec conformance case."""
    static = static_check(code, entrypoint=spec.entrypoint)
    if not static.ok:
        return ConformanceReport(static=static, passed=False,
                                 cases=[{"name": "static_gate", "ok": False,
                                         "reason": "; ".join(static.violations)}])

    cases = []
    all_ok = True
    for case in (spec.standard_checks() + list(spec.conformance)):
        ok, result, err = run_entrypoint(code, case.payload, entrypoint=spec.entrypoint, timeout=timeout)
        if not ok:
            cases.append({"name": case.name, "ok": False, "reason": err})
            all_ok = False
            continue
        try:
            cok, creason = case.check(result)
        except Exception as ex:  # noqa: BLE001  a buggy check must not crash the harness
            cok, creason = False, f"check raised: {str(ex)[:120]}"
        cases.append({"name": case.name, "ok": bool(cok), "reason": creason})
        all_ok = all_ok and bool(cok)

    return ConformanceReport(static=static, passed=all_ok, cases=cases)
