"""The connector contract A8 authors against: what a generated `load` must satisfy."""
from dataclasses import dataclass, field
from typing import Callable, Optional


@dataclass
class ConformanceCase:
    """One acceptance test: feed `payload` to the connector; `check(result) -> (ok, reason)`."""
    name: str
    payload: str
    check: Callable[[object], tuple]


@dataclass
class ConnectorSpec:
    """Description of the connector to author. `format_description` + `required_keys` + sample drive the
    prompt; `conformance` is the frozen acceptance suite the generated code must pass."""
    name: str
    format_description: str
    required_keys: list                      # keys every returned record must have
    sample_payload: str                      # a representative input (QUARANTINED in the prompt)
    conformance: list = field(default_factory=list)   # [ConformanceCase]
    entrypoint: str = "load"

    def standard_checks(self):
        """Baseline conformance every connector must pass, derived from the spec (frozen)."""
        req = list(self.required_keys)

        def _is_record_list(result):
            if not isinstance(result, list):
                return False, f"expected a list, got {type(result).__name__}"
            if not result:
                return False, "returned an empty list"
            for i, r in enumerate(result):
                if not isinstance(r, dict):
                    return False, f"row {i} is {type(r).__name__}, not a dict"
                missing = [k for k in req if k not in r]
                if missing:
                    return False, f"row {i} missing required keys {missing}"
            return True, "ok"

        return [ConformanceCase(name="returns_record_list_with_required_keys",
                                payload=self.sample_payload, check=_is_record_list)]
