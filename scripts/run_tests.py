"""Reproducible full-suite test runner.

Running `pytest tests/` as a single monolithic process accumulates resident memory
across ~80 heavy ML test files and OOMs on a 32 GB box, which is why the suite could
previously only be reported per-file. This runner executes each test file in its own
fresh subprocess so memory is reclaimed between files, applies a per-file wall-clock
timeout, and aggregates a trustworthy pass/fail/skip count in one command:

    python scripts/run_tests.py                 # all of tests/
    python scripts/run_tests.py frontier/tests  # a different suite
    python scripts/run_tests.py --timeout 300 -j 2

Exit code is non-zero if any file has a failure/error or times out, so it is CI-usable.
"""
import argparse
import concurrent.futures
import re
import subprocess
import sys
import time
from pathlib import Path

_SUMMARY = re.compile(r"(\d+) (passed|failed|skipped|error|errors|xfailed|xpassed)")


def _parse_counts(text: str) -> dict:
    counts: dict = {}
    # last summary line wins
    for line in text.splitlines():
        if " passed" in line or " failed" in line or " error" in line or " skipped" in line:
            found = dict((k2, int(n)) for n, k2 in _SUMMARY.findall(line))
            if found:
                counts = found
    return counts


def _run_file(path: Path, timeout: int) -> dict:
    start = time.time()
    try:
        proc = subprocess.run(
            [sys.executable, "-m", "pytest", str(path), "-q", "-p", "no:cacheprovider"],
            capture_output=True, text=True, timeout=timeout,
        )
        out = proc.stdout + proc.stderr
        counts = _parse_counts(out)
        status = "ok" if proc.returncode == 0 else "fail"
        # pytest returns 5 when no tests collected; treat as ok-empty
        if proc.returncode == 5:
            status = "empty"
        return {"file": path.name, "status": status, "counts": counts,
                "secs": time.time() - start, "rc": proc.returncode, "tail": out[-2000:]}
    except subprocess.TimeoutExpired:
        return {"file": path.name, "status": "timeout", "counts": {},
                "secs": time.time() - start, "rc": -1, "tail": ""}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("suite", nargs="?", default="tests", help="directory of test files")
    ap.add_argument("--timeout", type=int, default=900, help="per-file timeout (s); the heaviest E2E file takes ~626s")
    ap.add_argument("-j", "--jobs", type=int, default=1, help="parallel files (keep low; tests are heavy)")
    args = ap.parse_args()

    root = Path(__file__).resolve().parent.parent
    suite_dir = (root / args.suite) if not Path(args.suite).is_absolute() else Path(args.suite)
    files = sorted(suite_dir.glob("test_*.py"))
    if not files:
        print(f"no test_*.py files under {suite_dir}")
        return 1

    print(f"running {len(files)} files from {suite_dir} (timeout={args.timeout}s, jobs={args.jobs})\n", flush=True)

    def _report(r: dict) -> None:
        summary = " ".join(f"{v} {k}" for k, v in sorted(r["counts"].items())) or r["status"]
        flag = "" if r["status"] in ("ok", "empty") else f"  <<< {r['status'].upper()}"
        print(f"  {r['file']:<48} {summary:<32} {r['secs']:6.1f}s{flag}", flush=True)

    results = []
    if args.jobs > 1:
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.jobs) as ex:
            futs = {ex.submit(_run_file, f, args.timeout): f for f in files}
            for fut in concurrent.futures.as_completed(futs):
                r = fut.result()
                _report(r)
                results.append(r)
    else:
        for f in files:
            r = _run_file(f, args.timeout)
            _report(r)
            results.append(r)

    results.sort(key=lambda r: r["file"])
    totals: dict = {}
    bad = []
    for r in results:
        for k, v in r["counts"].items():
            totals[k] = totals.get(k, 0) + v
        if r["status"] not in ("ok", "empty"):
            bad.append(r)

    print("\n" + "=" * 70)
    print("TOTALS: " + ", ".join(f"{v} {k}" for k, v in sorted(totals.items())))
    if bad:
        print(f"\n{len(bad)} file(s) not green:")
        for r in bad:
            print(f"\n----- {r['file']} ({r['status']}, rc={r['rc']}) -----")
            print(r["tail"])
    print("=" * 70)
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
