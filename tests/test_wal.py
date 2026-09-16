"""Tests for vfplatform/wal.py -- the typed, append-only, content-addressed, hash-chained effect log.

The WAL is a RECORDING layer; it changes no certificate and gates no promotion. These tests verify:
  * append returns a content hash, the chain verifies clean,
  * a TAMPERED entry is detected by verify_chain,
  * a REORDERED chain is detected,
  * a DROPPED entry is detected,
  * a CORRUPT file -> verify_chain ok=False (never raises),
  * content addressing is deterministic (same payload + position -> same hash),
  * a missing file is a valid empty chain and append onto it works.

Run with: /Users/abdullahalghamdi/jax-env-311/bin/python tests/test_wal.py
"""
import json
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from vfplatform.wal import WAL, _entry_hash  # noqa: E402


def _tmp_path():
    d = tempfile.mkdtemp()
    return os.path.join(d, "wal.jsonl")


def test_append_returns_hash_and_chain_verifies():
    w = WAL(_tmp_path())
    h1 = w.append("bracket_open", {"bracket": 0, "n": 16})
    h2 = w.append("grade", {"cand": "a", "score": 0.87})
    h3 = w.append("grade", {"cand": "b", "score": 0.42})
    assert isinstance(h1, str) and h1.startswith("sha256:")
    assert h1 != h2 != h3
    entries = w.all()
    assert len(entries) == 3
    assert entries[0]["prev_hash"] is None        # genesis
    assert entries[1]["prev_hash"] == h1          # linkage
    assert entries[2]["prev_hash"] == h2
    v = w.verify_chain()
    assert v == {"ok": True, "broken_at": None}, v


def test_missing_file_is_valid_empty_chain():
    w = WAL(_tmp_path())  # nothing written yet
    assert w.all() == []
    assert w.verify_chain() == {"ok": True, "broken_at": None}
    h = w.append("first", {"k": 1})
    assert h.startswith("sha256:")
    assert w.verify_chain()["ok"] is True


def test_content_addressing_is_deterministic():
    # Same payload at the same chain position -> identical hash, regardless of dict insertion order.
    p1 = {"a": 1, "b": [2, 3], "c": {"d": 4}}
    p2 = {"c": {"d": 4}, "b": [2, 3], "a": 1}     # reordered keys, same content
    h1 = _entry_hash(0, "e", p1, None)
    h2 = _entry_hash(0, "e", p2, None)
    assert h1 == h2, (h1, h2)
    # different position -> different hash
    assert _entry_hash(1, "e", p1, None) != h1
    # two independent WALs with identical writes produce identical hashes
    wa, wb = WAL(_tmp_path()), WAL(_tmp_path())
    assert wa.append("t", {"x": 1}) == wb.append("t", {"x": 1})


def test_tamper_detected():
    path = _tmp_path()
    w = WAL(path)
    w.append("a", {"v": 1})
    w.append("b", {"v": 2})
    w.append("c", {"v": 3})
    assert w.verify_chain()["ok"] is True
    # tamper with the payload of entry 1, leaving its stored hash unchanged
    lines = [json.loads(l) for l in open(path) if l.strip()]
    lines[1]["payload"] = {"v": 999}
    with open(path, "w") as fh:
        for e in lines:
            fh.write(json.dumps(e, sort_keys=True, separators=(",", ":")) + "\n")
    v = w.verify_chain()
    assert v["ok"] is False and v["broken_at"] == 1, v


def test_reorder_detected():
    path = _tmp_path()
    w = WAL(path)
    w.append("a", {"v": 1})
    w.append("b", {"v": 2})
    w.append("c", {"v": 3})
    lines = [json.loads(l) for l in open(path) if l.strip()]
    # swap entries 1 and 2: their prev_hash linkage no longer matches the chain
    lines[1], lines[2] = lines[2], lines[1]
    with open(path, "w") as fh:
        for e in lines:
            fh.write(json.dumps(e, sort_keys=True, separators=(",", ":")) + "\n")
    v = w.verify_chain()
    assert v["ok"] is False, v
    assert v["broken_at"] == 1, v  # first position where linkage breaks


def test_drop_detected():
    path = _tmp_path()
    w = WAL(path)
    w.append("a", {"v": 1})
    w.append("b", {"v": 2})
    w.append("c", {"v": 3})
    lines = [json.loads(l) for l in open(path) if l.strip()]
    del lines[1]  # drop the middle entry
    with open(path, "w") as fh:
        for e in lines:
            fh.write(json.dumps(e, sort_keys=True, separators=(",", ":")) + "\n")
    v = w.verify_chain()
    assert v["ok"] is False, v
    # entry that was index 2 is now at index 1 and its prev_hash points at the dropped entry
    assert v["broken_at"] == 1, v


def test_corrupt_file_verify_false_no_raise():
    path = _tmp_path()
    w = WAL(path)
    w.append("a", {"v": 1})
    with open(path, "a") as fh:
        fh.write("this is not json\n")
    v = w.verify_chain()  # must NOT raise
    assert v["ok"] is False, v
    assert v["broken_at"] == 1, v  # genesis ok, second line corrupt


def test_append_onto_corrupt_refuses():
    path = _tmp_path()
    w = WAL(path)
    w.append("a", {"v": 1})
    with open(path, "a") as fh:
        fh.write("garbage\n")
    try:
        w.append("b", {"v": 2})
        raise AssertionError("append onto a corrupt chain should refuse")
    except ValueError:
        pass


def test_seq_argument_is_honored_and_deterministic():
    w = WAL(_tmp_path())
    h = w.append("a", {"v": 1}, seq=42)
    assert w.all()[0]["seq"] == 42
    # hash matches the explicit recompute with that seq
    assert h == _entry_hash(42, "a", {"v": 1}, None)


# --------------------------------------------------------------------------- runner
TESTS = [
    test_append_returns_hash_and_chain_verifies,
    test_missing_file_is_valid_empty_chain,
    test_content_addressing_is_deterministic,
    test_tamper_detected,
    test_reorder_detected,
    test_drop_detected,
    test_corrupt_file_verify_false_no_raise,
    test_append_onto_corrupt_refuses,
    test_seq_argument_is_honored_and_deterministic,
]


def run(tests):
    p = f = 0
    for t in tests:
        try:
            t(); print(f"  PASS  {t.__name__}"); p += 1
        except AssertionError as e:
            print(f"  FAIL  {t.__name__}: {e}"); f += 1
        except Exception as e:  # noqa: BLE001
            print(f"  ERROR {t.__name__}: {e}"); f += 1
    print(f"  ---- {p} passed, {f} failed ----")
    return p, f


def main():
    _, fails = run(TESTS)
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
