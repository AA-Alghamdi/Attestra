"""Human-checkpoint gate (build-order step 5).

The loop must clear a Checkpoint before any move that SPENDS (a paid/GPU provider) or DEPLOYS. Free
local-CPU moves auto-pass; anything that hits a billed provider raises CheckpointRequired that the caller
(CLI/human) must approve. This is the (4) CHECKPOINT node of the /goal loop.
"""


class CheckpointRequired(Exception):
    """Raised when a move needs human approval (spend / scale / deploy) and none was given."""


class Checkpoint:
    def __init__(self, *, approve_spend=False, on_request=None):
        self.approve_spend = approve_spend
        self.on_request = on_request          # optional callback(move_name, provider) -> bool

    def clear(self, move_name, provider):
        caps = provider.capabilities()
        # FREE only if it is a non-gated CPU provider with zero cost. Anything else -- GPU device, a gated
        # provider, OR an unknown/None cost (which must NOT be read as zero) -- is PAID and requires
        # approval. (Spend-gate hardening from the RunPod architecture swarm: a configured GPU endpoint
        # reports cost None / gated False and previously slipped through as "free".)
        cost = caps.get("cost_per_hour_usd")
        free = (caps.get("device") == "cpu" and (cost is not None and cost <= 0.0)
                and not caps.get("gated", False))
        if free:
            return True
        if self.approve_spend:
            return True
        if self.on_request and self.on_request(move_name, provider):
            return True
        raise CheckpointRequired(
            f"move {move_name!r} would run on a paid/gated provider ({provider.name}); human approval "
            f"required before spend. Pass approve_spend=True or provide an on_request approver.")
