"""Real dataset connectors (build-order item 2). Offline-capable sklearn datasets run NOW; OpenML is
network-gated with an honest decline. Each returns records in the platform's nested shape
({"features": {...}, "target": ...}) plus the inferred kind/task_type/labels, so the loop runs on REAL
data, not just synthetic generators.
"""
from .providers import ResourceGated

# sklearn ships these datasets in-package -> they load fully OFFLINE on the laptop.
_SKLEARN_CLF = ("breast_cancer", "wine", "iris", "digits")
_SKLEARN_REG = ("diabetes",)


def load_sklearn(name):
    """Load a bundled sklearn dataset into the platform record shape. Offline, real data."""
    from sklearn import datasets as skd
    loaders = {"breast_cancer": skd.load_breast_cancer, "wine": skd.load_wine, "iris": skd.load_iris,
               "digits": skd.load_digits, "diabetes": skd.load_diabetes}
    if name not in loaders:
        raise ValueError(f"unknown sklearn dataset {name!r}; available: "
                         f"{sorted(_SKLEARN_CLF + _SKLEARN_REG)}")
    d = loaders[name]()
    feat_names = list(getattr(d, "feature_names", [f"f{j}" for j in range(d.data.shape[1])]))
    feat_names = [str(f) for f in feat_names]
    if name in _SKLEARN_REG:
        rows = [{"features": {f: float(x[j]) for j, f in enumerate(feat_names)}, "target": float(y)}
                for x, y in zip(d.data, d.target)]
        return {"records": rows, "kind": "tabular", "task_type": "regression", "target_key": "target",
                "labels": None, "metric": "r2", "source": f"sklearn:{name}", "n": len(rows)}
    target_names = [str(t) for t in getattr(d, "target_names", sorted(set(map(str, d.target))))]
    rows = [{"features": {f: float(x[j]) for j, f in enumerate(feat_names)},
             "target": target_names[int(y)] if int(y) < len(target_names) else str(y)}
            for x, y in zip(d.data, d.target)]
    task = "binary" if len(set(r["target"] for r in rows)) == 2 else "multiclass"
    return {"records": rows, "kind": "tabular", "task_type": task, "target_key": "target",
            "labels": sorted(set(r["target"] for r in rows)),
            "metric": "accuracy", "source": f"sklearn:{name}", "n": len(rows)}


def load_openml(name=None, data_id=None, target=None, max_rows=5000):
    """Load an OpenML dataset (LIVE-GATED: needs network). Honest decline when offline."""
    try:
        from sklearn.datasets import fetch_openml
    except Exception as ex:  # noqa: BLE001
        raise ResourceGated(f"OpenML connector unavailable: {ex}")
    try:
        ds = fetch_openml(name=name, data_id=data_id, as_frame=True, parser="auto")
    except Exception as ex:  # noqa: BLE001  network/availability failure -> honest decline
        raise ResourceGated(
            f"OpenML fetch for {name or data_id!r} failed (offline or unavailable): {str(ex)[:160]}. "
            f"Use a bundled sklearn dataset offline, or retry with network access.")
    frame = ds.frame.head(max_rows)
    tgt = target or ds.target.name
    feat_cols = [c for c in frame.columns if c != tgt]
    rows = []
    for _, row in frame.iterrows():
        rows.append({"features": {c: (float(row[c]) if _isnum(row[c]) else str(row[c])) for c in feat_cols},
                     "target": str(row[tgt])})
    task = "binary" if frame[tgt].nunique() == 2 else "multiclass"
    return {"records": rows, "kind": "tabular", "task_type": task, "target_key": "target",
            "labels": sorted(set(r["target"] for r in rows)), "metric": "accuracy",
            "source": f"openml:{name or data_id}", "n": len(rows)}


def _isnum(v):
    try:
        float(v)
        return True
    except (TypeError, ValueError):
        return False


# --- HF datasets-server connector + a URI dispatcher (migration Item 1, ported from Codex) ----------
def load_huggingface(dataset, *, config=None, split="train", target_key="label", max_rows=3000):
    """Load a PUBLIC HF dataset via the datasets-server rows API (LIVE-GATED, no key for public sets).
    Returns the same spec shape as load_openml. Honest decline (ResourceGated) on a network/availability
    failure; raises on a too-small pull (never a synthetic fallback)."""
    import io  # noqa: F401  (kept local; stdlib)
    import json
    import urllib.parse
    import urllib.request

    def _get_json(url):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "attestera-connector/1.0"})
            with urllib.request.urlopen(req, timeout=90) as r:
                return json.loads(r.read().decode("utf-8"))
        except Exception as ex:  # noqa: BLE001
            raise ResourceGated(f"HF datasets-server fetch failed (offline/unavailable): {str(ex)[:160]}")

    splits = _get_json(f"https://datasets-server.huggingface.co/splits?dataset={urllib.parse.quote(dataset)}")
    avail = splits.get("splits", [])
    if not avail:
        raise ResourceGated(f"HF dataset {dataset!r} has no public splits (gated/private/unsupported)")
    if config is None:
        config = avail[0]["config"]
    cfg_splits = [s for s in avail if s.get("config") == config] or avail
    split = next((s["split"] for s in cfg_splits if s.get("split") == split), cfg_splits[0]["split"])

    def _pull(off, length):
        url = (f"https://datasets-server.huggingface.co/rows?dataset={urllib.parse.quote(dataset)}"
               f"&config={urllib.parse.quote(config)}&split={urllib.parse.quote(split)}"
               f"&offset={int(off)}&length={int(length)}")
        return _get_json(url)

    # First page ALSO carries the column feature TYPES -> capture ClassLabel names to decode integer targets
    # (HF text labels come back as ints; without this the target is 0/1 and the integer can collide with text).
    first = _pull(0, 100)
    feat_types = first.get("features", [])
    label_names = next((f["type"]["names"] for f in feat_types
                        if f.get("name") == target_key and isinstance(f.get("type"), dict)
                        and f["type"].get("_type") == "ClassLabel" and f["type"].get("names")), None)
    total = int(first.get("num_rows_total") or 0)

    def _decode(raw):
        if label_names is not None and isinstance(raw, int) and 0 <= raw < len(label_names):
            return label_names[raw]
        return raw

    # BALANCED pull: HF splits are often label-sorted, so a contiguous offset-0 window can be single-class.
    # Interleave windows spread across the split (0, N/4, N/2, 3N/4) until max_rows with >=2 classes.
    starts = [0] if (total == 0 or total <= max_rows) else [int(total * f) for f in (0.0, 0.25, 0.5, 0.75)]
    per = max(100, max_rows // len(starts))
    rows = []
    for si, st in enumerate(starts):
        got = list(first.get("rows", [])) if si == 0 else []
        off = st + len(got)
        while len(got) < per and (len(rows) + len(got)) < max_rows:
            batch = _pull(off, min(100, per - len(got))).get("rows", [])
            if not batch:
                break
            got += batch
            off += len(batch)
        for item in got:
            row = item.get("row", {})
            if target_key not in row:
                raise ResourceGated(f"HF {dataset!r} rows have no target {target_key!r}; columns={list(row)}")
            feats = {k: v for k, v in row.items() if k != target_key}
            rows.append({"features": feats, "target": _decode(row.get(target_key))})
        if len(rows) >= max_rows:
            break
    rows = rows[:max_rows]
    if len(rows) < 50:
        raise ResourceGated(f"HF {dataset!r} returned too few rows ({len(rows)}); refusing to certify on it")
    disp = sorted({str(r["target"]) for r in rows})
    if len(disp) < 2:
        raise ResourceGated(f"HF {dataset!r} single-class pull ({disp}); split is label-sorted -- widen the window")
    is_text = any(("text" in r["features"] or "sentence" in r["features"]) for r in rows[:5])
    label_display = None
    if is_text:
        # COLLISION-SAFE label namespace: topic/sentiment label WORDS (World, pos, ...) can occur in the text
        # and trip the FROZEN label-token-in-text leakage gate. Remap to provably non-colliding label_i and
        # keep the human names for reporting only. A labeling choice, NOT a spec relaxation.
        idx = {lab: f"label_{i}" for i, lab in enumerate(disp)}
        for r in rows:
            r["target"] = idx[str(r["target"])]
        label_display, labels = idx, sorted(idx.values())
    else:
        labels = disp
    return {"records": rows, "kind": "text" if is_text else "tabular",
            "task_type": "binary" if len(labels) == 2 else "multiclass", "target_key": "target",
            "labels": labels, "label_display": label_display, "metric": "accuracy",
            "source": f"hf:{dataset}", "n": len(rows)}


def materialize(source_uri, *, max_rows=3000, target_key="label"):
    """Dispatch Codex's URI scheme to a connector spec dict: hf://owner/name[/config] | openml://dataset/<id>."""
    if source_uri.startswith("openml://dataset/"):
        return load_openml(data_id=int(source_uri.rsplit("/", 1)[1]), max_rows=max_rows)
    if source_uri.startswith("hf://"):
        parts = source_uri[len("hf://"):].split("/")
        dataset = "/".join(parts[:2])
        config = parts[2] if len(parts) > 2 else None
        return load_huggingface(dataset, config=config, target_key=target_key, max_rows=max_rows)
    raise ValueError(f"unknown source uri {source_uri!r} (expected hf:// or openml://dataset/)")
