"""Attestra CLI: autonomous ML research from the command line.

Usage:
  python -m attestra run --goal "Classify digits with 95% accuracy" --data digits
  python -m attestra run --goal "Predict housing prices" --data path/to/data.csv --target price
  python -m attestra run --goal "Best classifier for this dataset" --data path/to/data.csv --target label --time 600
  python -m attestra status              # show experiment registry summary
  python -m attestra history             # show recent experiments
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path


def main(argv=None):
    parser = argparse.ArgumentParser(
        prog="attestra",
        description="Attestra: Autonomous ML Research Executor",
    )
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # run
    run_parser = subparsers.add_parser("run", help="Run autonomous ML research")
    run_parser.add_argument("--goal", required=True, help="Research goal (free text)")
    run_parser.add_argument("--data", required=True,
                            help="Data source: built-in name (iris/digits/wine/breast_cancer/"
                                 "diabetes/california_housing/covtype), 'openml:<name|id>', "
                                 "an http(s):// URL to a CSV/TSV, or a local .csv/.tsv/.npz path")
    run_parser.add_argument("--target", default=None, help="Target column name (for CSV)")
    run_parser.add_argument("--metric", default=None, help="Metric: accuracy, balanced_accuracy, macro_f1, r2, neg_rmse, neg_mae")
    run_parser.add_argument("--threshold", type=float, default=None, help="Certification threshold")
    run_parser.add_argument("--time", type=float, default=300, help="Time budget in seconds")
    run_parser.add_argument("--rounds", type=int, default=15, help="Max research rounds")
    run_parser.add_argument("--no-llm", action="store_true", help="Disable LLM proposals")
    run_parser.add_argument("--api-key", default=None, help="LLM API key")
    run_parser.add_argument("--seed", type=int, default=42, help="Random seed")
    run_parser.add_argument("--quiet", action="store_true", help="Minimal output")
    run_parser.add_argument("--output", default=None, help="Output JSON path")
    run_parser.add_argument("--engine", default="auto",
                            choices=["auto", "frontier", "generative", "catalog"],
                            help="Engine selection: auto (default), frontier, generative, catalog")
    run_parser.add_argument("--gpu", action="store_true",
                            help="Enable GPU execution (auto-detects local GPU or Prime Intellect remote pods)")
    run_parser.add_argument("--gpu-status", action="store_true",
                            help="Show GPU status and exit")
    run_parser.add_argument("--strategy-loop", action="store_true",
                            help="Enable strategy loop (Loop 2): retry with escalation until certified or budget exhausted")
    run_parser.add_argument("--max-attempts", type=int, default=8,
                            help="Max strategy loop attempts (requires --strategy-loop)")

    # status
    subparsers.add_parser("status", help="Show experiment registry summary")

    # history
    hist_parser = subparsers.add_parser("history", help="Show recent experiments")
    hist_parser.add_argument("-n", type=int, default=10, help="Number of recent experiments")

    args = parser.parse_args(argv)

    if args.command == "run":
        return _cmd_run(args)
    elif args.command == "status":
        return _cmd_status(args)
    elif args.command == "history":
        return _cmd_history(args)
    else:
        parser.print_help()
        return 1


def _cmd_run(args):
    import numpy as np
    from ..orchestration.orchestrator import OrchestrateConfig, orchestrate

    # GPU status check
    if getattr(args, 'gpu_status', False):
        from ..execution.gpu_backend import check_gpu_status
        status = check_gpu_status()
        print("Attestra GPU Status")
        print("=" * 50)
        print(f"  API key set:       {status.api_key_set}")
        print(f"  Local GPU:         {status.local_device}" +
              (f" ({status.local_info})" if status.local_info else ""))
        print(f"  Deployed pods:     {len(status.active_pods)} active (reused, not re-provisioned)")
        for p in status.active_pods:
            print(f"      - {p.get('gpuName', '?')} x{p.get('gpuCount', '?')} "
                  f"[{p.get('id', '?')}] {p.get('sshConnection', '')}")
        print(f"  Remote GPU offers: {len(status.remote_offers)} available" +
              (f" ({status.remote_provider})" if status.remote_provider else ""))
        print(f"  LLM inference:     {status.inference_available}" +
              (f" ({len(status.inference_models)} models)" if status.inference_models else ""))
        if status.error:
            print(f"  Error:             {status.error}")
        print()
        if status.inference_models:
            print("Available inference models (top 20):")
            for m in status.inference_models[:20]:
                print(f"    {m}")
            if len(status.inference_models) > 20:
                print(f"    ... and {len(status.inference_models) - 20} more")
        return 0

    # Load data
    X, y, feature_names = _load_data(args.data, args.target)
    if X is None:
        print(f"Error: Could not load data from '{args.data}'", file=sys.stderr)
        return 1

    # Setup API key
    api_key = args.api_key or os.environ.get("PRIME_INTELLECT_API_KEY")
    if not api_key and not args.no_llm:
        print("[attestra] No API key found. Running without LLM proposals. "
              "Set PRIME_INTELLECT_API_KEY or use --api-key.", file=sys.stderr)

    llm_call = None
    if api_key and not args.no_llm:
        from ..orchestration.orchestrator import _build_llm_call
        llm_call = _build_llm_call(api_key)

    print(f"[attestra] Goal: {args.goal}")
    print(f"[attestra] Data: {X.shape[0]} samples, {X.shape[1]} features")
    print(f"[attestra] Budget: {args.time}s, {args.rounds} rounds")
    print(f"[attestra] LLM: {'enabled' if llm_call else 'disabled'}")
    print()

    config = OrchestrateConfig(
        goal=args.goal,
        X=X,
        y=y,
        metric=args.metric,
        threshold=args.threshold,
        max_rounds=args.rounds,
        time_budget_s=args.time,
        llm_call=llm_call,
        api_key=api_key if not args.no_llm else None,
        use_llm=bool(llm_call) and not args.no_llm,
        gpu=getattr(args, 'gpu', False),
        feature_names=feature_names,
        seed=args.seed,
        verbose=not args.quiet,
        engine=args.engine,
    )

    if args.strategy_loop:
        from ..orchestration.strategy_loop import run_strategy_loop, StrategyLoopConfig
        loop_config = StrategyLoopConfig(
            total_budget_s=args.time * 3,
            max_attempts=args.max_attempts,
        )
        loop_result = run_strategy_loop(config, loop_config=loop_config)
        # Adapt loop result to standard output format
        print()
        print("=" * 60)
        print(f"DECISION: {loop_result.decision.upper()}")
        print(f"Best score: {loop_result.best_score:.4f} ({loop_result.best_technique})")
        print(f"Attempts: {loop_result.n_attempts} strategies tried: {loop_result.strategies_tried}")
        print(f"Time: {loop_result.total_elapsed_s:.1f}s total")
        if loop_result.certificate:
            print(f"Certificate: {json.dumps(loop_result.certificate, indent=2, default=str)[:200]}")
        if loop_result.diagnoses:
            print(f"Diagnoses: {[d.failure_mode for d in loop_result.diagnoses]}")
        print("=" * 60)

        if args.output:
            out = {
                "decision": loop_result.decision,
                "best_score": loop_result.best_score,
                "best_technique": loop_result.best_technique,
                "certificate": loop_result.certificate,
                "n_attempts": loop_result.n_attempts,
                "strategies_tried": loop_result.strategies_tried,
                "total_elapsed_s": loop_result.total_elapsed_s,
                "escalation_history": loop_result.escalation_history,
            }
            Path(args.output).write_text(json.dumps(out, indent=2, default=str))
            print(f"Results saved to {args.output}")

        return 0 if loop_result.decision == "certified" else 1

    result = orchestrate(config)

    # Output
    print()
    print("=" * 60)
    print(f"DECISION: {result.decision.upper()}")
    print(f"Best score: {result.best_score:.4f} ({result.best_technique})")
    print(f"Proposals: {result.n_proposals} total, {result.n_successful} successful, {result.n_failed} failed")
    print(f"Time: {result.elapsed_s:.1f}s")
    if result.certificate:
        print(f"Certificate: lower_bound={result.certificate.get('lower_bound', 'N/A')}")
    if result.failure_report:
        print(f"Failure: {json.dumps(result.failure_report, indent=2, default=str)}")
    print("=" * 60)

    if args.output:
        out = {
            "decision": result.decision,
            "best_score": result.best_score,
            "best_technique": result.best_technique,
            "certificate": result.certificate,
            "n_proposals": result.n_proposals,
            "elapsed_s": result.elapsed_s,
            "profile": result.profile.to_dict() if result.profile else None,
        }
        Path(args.output).write_text(json.dumps(out, indent=2, default=str))
        print(f"Results saved to {args.output}")

    return 0 if result.decision == "certified" else 1


def _cmd_status(args):
    from ..ledger.registry import ExperimentRegistry
    from ..improvement.strategy_learner import StrategyLearner

    registry = ExperimentRegistry()
    learner = StrategyLearner()

    reg_summary = registry.summary()
    strat_summary = learner.summary()

    print("Attestra Experiment Registry")
    print("=" * 40)
    print(f"Total experiments: {reg_summary['total']}")
    print(f"Positive (certified): {reg_summary['positive']}")
    print(f"Negative: {reg_summary['negative']}")
    print(f"Unique datasets: {reg_summary['unique_datasets']}")
    print(f"Unique techniques: {reg_summary['unique_techniques']}")
    print()
    print("Strategy Learner")
    print("=" * 40)
    print(f"Total outcomes: {strat_summary['total_outcomes']}")
    print(f"Unique strategies: {strat_summary['unique_strategies']}")
    print(f"Overall success rate: {strat_summary['overall_success_rate']:.1%}")
    return 0


def _cmd_history(args):
    from ..ledger.registry import ExperimentRegistry
    registry = ExperimentRegistry()
    records = registry.all_records()
    records.sort(key=lambda r: r.created_at, reverse=True)

    for r in records[:args.n]:
        ts = time.strftime("%Y-%m-%d %H:%M", time.localtime(r.created_at))
        status = "OK" if r.is_positive else "FAIL"
        print(f"[{ts}] {status:4s} | {r.metric}={r.best_score:.4f} | "
              f"{r.best_technique[:30]:30s} | {r.task_type} | {r.n_samples}x{r.n_features} | "
              f"{r.elapsed_s:.0f}s")
    if not records:
        print("No experiments recorded yet.")
    return 0


_BUILTIN_DATASETS = (
    "iris", "digits", "wine", "breast_cancer",
    "diabetes", "california_housing", "covtype",
)


def _load_data(source: str, target: str = None):
    """Load data from many sources without the user writing any Python.

    Supported `source` forms:
      - built-in name:      iris | digits | wine | breast_cancer | diabetes |
                            california_housing | covtype
      - OpenML:             openml:<name>  OR  openml:<numeric_id>
                            (uses sklearn.datasets.fetch_openml, no extra deps)
      - HTTP(S) URL:        https://.../data.csv  (downloaded, then parsed)
      - local file:         *.csv / *.tsv / *.npz
    """
    src = source.strip()
    low = src.lower()

    if low in _BUILTIN_DATASETS:
        return _load_sklearn(low)

    if low.startswith("openml:"):
        return _load_openml(src.split(":", 1)[1].strip(), target)

    if low.startswith("http://") or low.startswith("https://"):
        return _load_url(src, target)

    if os.path.exists(src):
        ext = os.path.splitext(src)[1].lower()
        if ext == ".npz":
            return _load_npz(src)
        return _load_csv(src, target, sep="\t" if ext == ".tsv" else ",")

    print(f"Unknown data source: {source}")
    print(f"Available built-ins: {', '.join(_BUILTIN_DATASETS)}")
    print("Or use:  openml:<name|id>  |  https://.../file.csv  |  path/to/file.csv|.tsv|.npz")
    return None, None, None


def _load_url(url: str, target: str = None):
    """Download a remote CSV/TSV to a temp file and parse it."""
    import tempfile
    import urllib.request
    ext = os.path.splitext(url.split("?", 1)[0])[1].lower() or ".csv"
    sep = "\t" if ext == ".tsv" else ","
    try:
        with urllib.request.urlopen(url, timeout=60) as resp:
            raw = resp.read()
        with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as tmp:
            tmp.write(raw)
            tmp_path = tmp.name
        try:
            return _load_csv(tmp_path, target, sep=sep)
        finally:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
    except Exception as e:
        print(f"Failed to download {url}: {e}")
        return None, None, None


def _load_openml(name_or_id: str, target: str = None):
    """Load an OpenML dataset via sklearn (as_frame=False -> no pandas needed)."""
    from sklearn.datasets import fetch_openml
    try:
        kwargs = dict(as_frame=False, parser="liac-arff")
        if name_or_id.isdigit():
            data = fetch_openml(data_id=int(name_or_id), **kwargs)
        else:
            data = fetch_openml(name=name_or_id, version="active", **kwargs)
    except Exception as e:
        print(f"Failed to fetch OpenML dataset '{name_or_id}': {e}")
        return None, None, None
    feature_names = list(data.feature_names) if getattr(data, "feature_names", None) else None
    return data.data, data.target, feature_names


def _load_npz(path: str):
    """Load an .npz with arrays X and y (feature_names optional)."""
    import numpy as np
    with np.load(path, allow_pickle=True) as d:
        if "X" not in d or "y" not in d:
            print(f".npz must contain arrays 'X' and 'y'; found {list(d.keys())}")
            return None, None, None
        fn = list(d["feature_names"]) if "feature_names" in d else None
        return d["X"], d["y"], fn


def _load_sklearn(name: str):
    import numpy as np
    from sklearn.datasets import (load_iris, load_digits, load_wine,
                                   load_breast_cancer, load_diabetes,
                                   fetch_california_housing)

    loaders = {
        "iris": load_iris,
        "digits": load_digits,
        "wine": load_wine,
        "breast_cancer": load_breast_cancer,
        "diabetes": load_diabetes,
        "california_housing": fetch_california_housing,
    }

    if name == "covtype":
        from sklearn.datasets import fetch_covtype
        data = fetch_covtype()
        # Subsample for speed
        rng = np.random.default_rng(42)
        idx = rng.choice(len(data.data), 10000, replace=False)
        return data.data[idx], data.target[idx], None

    loader = loaders[name]
    data = loader()
    feature_names = list(data.feature_names) if hasattr(data, 'feature_names') else None
    return data.data, data.target, feature_names


def _load_csv(path: str, target: str = None, sep: str = ","):
    """Load a delimited file. Uses pandas if available, else a stdlib fallback."""
    try:
        import pandas as pd
        df = pd.read_csv(path, sep=sep)
        if target is None:
            target = df.columns[-1]
        if target not in df.columns:
            print(f"Target column '{target}' not found. Columns: {list(df.columns)}")
            return None, None, None
        feature_names = [c for c in df.columns if c != target]
        X = df[feature_names].values
        y = df[target].values
        return X, y, feature_names
    except ImportError:
        return _load_csv_stdlib(path, target, sep)


def _load_csv_stdlib(path: str, target: str = None, sep: str = ","):
    """pandas-free delimited loader (csv + numpy). Encodes non-numeric columns."""
    import csv
    import numpy as np

    with open(path, newline="") as f:
        rows = list(csv.reader(f, delimiter=sep))
    if not rows:
        print(f"Empty file: {path}")
        return None, None, None

    header = rows[0]
    has_header = any(not _looks_numeric(c) for c in header)
    if has_header:
        columns = header
        data_rows = rows[1:]
    else:
        columns = [f"col_{i}" for i in range(len(header))]
        data_rows = rows

    if target is None:
        target = columns[-1]
    if target not in columns:
        print(f"Target column '{target}' not found. Columns: {columns}")
        return None, None, None

    t_idx = columns.index(target)
    feat_idx = [i for i in range(len(columns)) if i != t_idx]
    feature_names = [columns[i] for i in feat_idx]

    def _col(j):
        vals = [r[j] for r in data_rows if len(r) > j]
        if all(_looks_numeric(v) for v in vals):
            return np.array([float(v) for v in vals], dtype=float)
        # label-encode categorical/string columns deterministically
        uniq = {v: k for k, v in enumerate(sorted(set(vals)))}
        return np.array([uniq[v] for v in vals], dtype=float)

    X = np.column_stack([_col(j) for j in feat_idx]) if feat_idx else np.empty((len(data_rows), 0))
    y = _col(t_idx)
    return X, y, feature_names


def _looks_numeric(v: str) -> bool:
    try:
        float(v)
        return True
    except (TypeError, ValueError):
        return False


if __name__ == "__main__":
    sys.exit(main())
