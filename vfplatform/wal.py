"""Typed, append-only, content-addressed, HASH-CHAINED effect log (a RECORDING layer).

WHAT THIS IS
------------
A write-ahead-style audit log of EFFECTS the platform performs (provider calls, sealed peeks requested,
brackets opened, candidates graded, ...). Each entry is:
  * typed:            carries an explicit effect_type string,
  * append-only:      we only ever add lines; existing lines are never rewritten,
  * content-addressed: each entry's id is the digest of its canonical content,
  * hash-chained:     each entry stores prev_hash, so the whole log is a tamper-evident chain --
                      altering, reordering, or dropping any entry breaks the chain at that point.

WHAT THIS IS NOT
----------------
This RECORDS effects for auditability. It is NOT the fold-based certifier replacement (that remains
future, gated work) and it changes NO certificate. The frozen Clopper-Pearson promotion gate in
vectorforge/science.py is untouched and remains the sole promotion gate. Recording an effect here has
zero influence on any score, bound, or promotion decision. This module is ADDITIVE and STANDALONE: it
imports `digest` from vectorforge.science READ-ONLY (for content addressing) and nothing else.

DETERMINISM
-----------
No clock, no RNG. Sequence numbers and any timestamp are CALLER-SUPPLIED arguments. The same payload (and
same chain position) always produces the same entry hash, so content addressing is reproducible.

FILE FORMAT
-----------
One JSON object per line (JSONL). Each line:
    {"seq": int, "effect_type": str, "payload": {...}, "prev_hash": str|null, "hash": str}
where hash = digest(("wal-v1", seq, effect_type, canonical_payload, prev_hash)) via science.digest.
The genesis entry has prev_hash = null. Appends are atomic (write a temp file with the new content, then
os.replace) so a crash cannot leave a half-written line.
"""

import json
import os

from vectorforge.science import digest  # READ-ONLY import for content addressing; nothing is mutated.

GENESIS_PREV = None


def _entry_hash(seq, effect_type, payload, prev_hash):
    """Content address of an entry. Deterministic: depends only on its content + chain position.

    The payload is canonicalized with sort_keys so dict insertion order is irrelevant; the fields are
    passed in a FIXED order so the same logical entry always maps to the same hash.
    """
    canonical_payload = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return digest(("wal-v1", int(seq), str(effect_type), canonical_payload, prev_hash))


class WAL:
    """Append-only, content-addressed, hash-chained effect log. Tolerant of a missing file."""

    def __init__(self, path):
        self.path = str(path)

    # -- reading -----------------------------------------------------------------------------------
    def _read_lines(self):
        """Decoded entries in log order. May raise on a corrupt file -> internal callers guard it."""
        if not os.path.exists(self.path):
            return []
        entries = []
        with open(self.path, "r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                entries.append(json.loads(line))
        return entries

    def all(self):
        """All entries as a list of dicts, in log order. Missing file -> []. A corrupt file raises;
        for tamper-tolerant inspection use verify_chain (which never raises)."""
        return self._read_lines()

    # -- appending ---------------------------------------------------------------------------------
    def append(self, effect_type, payload, *, seq=None):
        """Append one typed effect; return the entry's content hash.

        seq: optional caller-supplied sequence number. If None, defaults to the current log length
        (0-based, monotone). Determinism holds either way -- the hash is a pure function of
        (seq, effect_type, payload, prev_hash).
        """
        if not isinstance(effect_type, str) or not effect_type:
            raise ValueError("effect_type must be a non-empty str")
        if not isinstance(payload, dict):
            raise ValueError("payload must be a dict")
        try:
            existing = self._read_lines()
        except (json.JSONDecodeError, ValueError, OSError):
            raise ValueError(
                f"WAL at {self.path!r} is unreadable/corrupt; refusing to append onto a broken chain."
            )
        prev_hash = existing[-1]["hash"] if existing else GENESIS_PREV
        if seq is None:
            seq = len(existing)
        h = _entry_hash(seq, effect_type, payload, prev_hash)
        entry = {
            "seq": int(seq),
            "effect_type": str(effect_type),
            "payload": payload,
            "prev_hash": prev_hash,
            "hash": h,
        }
        line = json.dumps(entry, sort_keys=True, separators=(",", ":"))

        # Atomic append: rebuild full content (existing lines verbatim + new line) into a temp file, then
        # os.replace. Existing entries' content is never altered (append-only is preserved).
        directory = os.path.dirname(os.path.abspath(self.path)) or "."
        os.makedirs(directory, exist_ok=True)
        tmp = self.path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as fh:
            if os.path.exists(self.path):
                with open(self.path, "r", encoding="utf-8") as src:
                    for ln in src:
                        if ln.strip():
                            fh.write(ln if ln.endswith("\n") else ln + "\n")
            fh.write(line + "\n")
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, self.path)
        return h

    # -- verification ------------------------------------------------------------------------------
    def verify_chain(self):
        """Recompute the chain and detect any tamper / reorder / drop.

        Returns {"ok": bool, "broken_at": int|None}. broken_at is the 0-based index of the FIRST entry
        whose stored hash or prev_hash linkage fails to recompute. A missing file is a valid empty chain
        -> ok=True. A corrupt/unparseable file -> ok=False with broken_at as far as it could be read.
        """
        if not os.path.exists(self.path):
            return {"ok": True, "broken_at": None}

        entries = []
        idx = 0
        try:
            with open(self.path, "r", encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    entries.append(json.loads(line))
                    idx += 1
        except (json.JSONDecodeError, ValueError, OSError):
            return {"ok": False, "broken_at": idx}

        required = {"seq", "effect_type", "payload", "prev_hash", "hash"}
        prev = GENESIS_PREV
        for i, e in enumerate(entries):
            if not isinstance(e, dict) or not required.issubset(e.keys()):
                return {"ok": False, "broken_at": i}
            if e["prev_hash"] != prev:                      # linkage: reorder/drop detection
                return {"ok": False, "broken_at": i}
            recomputed = _entry_hash(e["seq"], e["effect_type"], e["payload"], e["prev_hash"])
            if recomputed != e["hash"]:                     # integrity: tamper detection
                return {"ok": False, "broken_at": i}
            prev = e["hash"]
        return {"ok": True, "broken_at": None}


# ============================================================================ self-test stub
if __name__ == "__main__":
    import tempfile
    d = tempfile.mkdtemp()
    w = WAL(os.path.join(d, "wal.jsonl"))
    h1 = w.append("bracket_open", {"bracket": 0, "n": 16})
    h2 = w.append("grade", {"cand": "a", "score": 0.87})
    print("hashes:", h1, h2)
    print("verify:", w.verify_chain())
    print("len:", len(w.all()))
