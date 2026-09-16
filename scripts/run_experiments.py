"""ONE COMMAND to run a declarative experiment suite -- CPU now, GPU the day a RunPod key is connected.

  # read-only GPU readiness (NO spend, NO runs): what is runnable today and the exact reason if not
  python scripts/run_experiments.py --preflight

  # run the bundled suite on CPU (the /goal front door + the in-process worker for the GPU loop lane)
  python scripts/run_experiments.py --lane cpu

  # run it preferring GPU; honestly falls back to CPU per-lane if no endpoint/pod is connected
  RUNPOD_API_KEY=... RUNPOD_ENDPOINT_ID=... python scripts/run_experiments.py --lane gpu

  # a single experiment from the manifest
  python scripts/run_experiments.py --only wine_goal

  # CROSS-EXPERIMENT MEMORY: the loop lane warm-starts VoI from prior experiments' realized gains
  python scripts/run_experiments.py --lane cpu --memory .vf_campaign_memory

The frozen certifier (vectorforge/science.py + vfplatform/sealed.py) is asserted byte-identical before AND
after the suite; the report records the hashes. See docs/EXPERIMENTS.md for the manifest schema + GPU setup.
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from vfplatform import experiments as E   # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_MANIFEST = os.path.join(ROOT, "experiments", "default_suite.json")
DEFAULT_OUT = os.path.join(ROOT, "docs", "EXPERIMENTS_RESULT.json")


def main():
    ap = argparse.ArgumentParser(description="Declarative experiment-suite driver (CPU now / GPU-ready).")
    ap.add_argument("--manifest", default=DEFAULT_MANIFEST, help="suite JSON (see docs/EXPERIMENTS.md)")
    ap.add_argument("--lane", choices=["auto", "cpu", "gpu"], default="auto",
                    help="cpu=force CPU; gpu=prefer GPU; auto=prefer GPU, honest CPU fallback")
    ap.add_argument("--only", default=None, help="run only the named experiment from the manifest")
    ap.add_argument("--out", default=DEFAULT_OUT, help="where to write the suite report JSON")
    ap.add_argument("--memory", default=None,
                    help="opt-in cross-experiment memory dir: the loop lane warm-starts VoI from -- and "
                         "appends realized gains back to -- a per-(kind,task_type) case-base here, so a "
                         "campaign learns which families pay off across experiments (default: cold start)")
    ap.add_argument("--preflight", action="store_true",
                    help="print the read-only GPU readiness report and exit (NO runs, NO spend)")
    args = ap.parse_args()

    if args.preflight:
        report = E.gpu_preflight()
        print(json.dumps(report, indent=2))
        print(f"\nGPU ready today: {report['ready_for_gpu']}  "
              f"(torch={report['torch']['available']} cuda={report['torch']['cuda']})")
        if not report["ready_for_gpu"]:
            sl = report["providers"]["runpod-gpu"]["capabilities"]
            print(f"runpod-gpu gated: {sl.get('reason', 'n/a')}")
            print("CPU smoke is runnable now: `python scripts/run_experiments.py --lane cpu`")
        return 0

    suite_name, specs = E.load_suite(args.manifest)
    print(f"== suite '{suite_name}' from {args.manifest}: {len(specs)} experiments, lane={args.lane} ==\n")

    def _on_event(ev):
        if ev.get("event") == "experiment_start":
            print(f"-> [{ev['lane']:4}] {ev['name']} ...", flush=True)
        elif ev.get("event") == "experiment_done":
            r = ev["result"]
            if not r.get("ok"):
                print(f"   ERROR: {r.get('error')}")
            elif r["lane"] == "goal":
                verdict = ("declined" if r["declined"] else "refused" if r["refused"]
                           else "solved" if r["solved"] else "not-solved")
                print(f"   {verdict}  champion={r.get('champion')} head={r.get('champion_head')}  "
                      f"theta={r.get('theta_floor')} sealed_lb={r.get('pooled_sealed_lb')} "
                      f"({r.get('elapsed_s')}s)")
            else:
                print(f"   decision={r.get('decision')} certified={r.get('certified')} "
                      f"family={r.get('champion_family')} device={r.get('device')} ({r.get('elapsed_s')}s)")

    report = E.run_suite(specs, suite_name=suite_name, lane=args.lane, only=args.only,
                         out_path=args.out, memory_dir=args.memory, on_event=_on_event)

    print("\n" + E.leaderboard_markdown(report))
    print(f"\nwrote {args.out}")
    print(f"frozen certifier byte-identical: {report['frozen_ok']}  {report['frozen_hashes']}")
    print(f"ALL EXPERIMENTS OK: {report['all_ok']}")
    return 0 if (report["all_ok"] and report["frozen_ok"]) else 1


if __name__ == "__main__":
    sys.exit(main())
