"""LITERATURE-GROUNDED DISCOVERY -- retrieve published methods/models from the open research surface
(arXiv, Papers-with-Code, GitHub, the Hugging Face Hub) and turn them into typed recipe ingredients
(backbone ids + technique MOTIFS) with PROVENANCE.

WHY THIS EXISTS (the deepest anti-menu)
---------------------------------------
A frontier ML researcher's edge is not a better idea in a vacuum: it is *reading the literature* and
adapting what already works for the problem in front of them. Until now Attestra's "literature library"
was a HARD-CODED list of motifs (`recipe_generator.literature_library`) -- which is itself a menu a human
wrote. This module replaces that hand-list with LIVE retrieval:

  * it queries the open research surface for the problem's keywords,
  * extracts (a) concrete model ids to add to the OPEN backbone pool and (b) technique MOTIFS (partial
    recipe genomes) distilled from the retrieved methods,
  * and records a PROVENANCE row for every ingredient (which paper / repo / model surfaced it),

so the certificate can prove the certified champion's decisive ingredient traces to something the system
*read in the wild*, not to a list it was handed. The retrieved ingredients feed the SAME generator ->
validity cascade -> frozen Tier-3 certifier. Nothing here evaluates, and nothing here promotes.

HONESTY / ROBUSTNESS
--------------------
* OFFLINE-GRACEFUL by construction. Every connector returns [] on any error (no network, no token, rate
  limit, malformed response) and the scout falls back to a small bundled corpus, so the test suite is
  hermetic and a live run never crashes on a flaky API.
* The LLM (when ANTHROPIC_API_KEY is present) READS the retrieved abstracts/READMEs and proposes the
  motifs; a deterministic keyword extractor is the fallback. Either way, every proposed motif is VALIDATED
  against the typed recipe axes (illegal keys/values are dropped) before it can reach the generator -- a
  hallucinated technique can at worst be ignored, never corrupt a recipe.
* Family-mention -> representative-model canonicalization (e.g. a paper that says "DINOv2" -> a concrete
  timm id) is a small, transparent map; the OPEN part is *which* family/model the literature surfaces for
  this problem and the real Hub ids returned verbatim by the HF connector.

CONTRACT: stdlib (urllib/xml/json) + vfplatform.regeneration (Motif/Genome) + vfplatform.recipe (the typed
axes). No torch / sklearn / frozen-science import -- a malformed finding can at worst waste a pilot.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import ssl
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from dataclasses import asdict, dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

from .recipe import (ADAPTATIONS, AGGREGATIONS, AUGMENTATIONS, DATA_STRATEGIES, HEADS, OPTIMIZERS,
                     SCHEDULES)
from .regeneration import Genome, Motif

_UA = "Attestra/0.1 (autoresearcher; literature-grounded discovery)"
_TIMEOUT = 15
_CTX = ssl.create_default_context()


# ======================================================================= findings
@dataclass
class Finding:
    """One retrieved item from the research surface. `text` is the title+abstract/description the extractor
    reads; `model_ids` are concrete Hub/timm ids the source names directly (HF connector). `score` is a
    source-native popularity proxy (downloads / stars) used only to rank, never to certify."""
    source: str                       # 'arxiv' | 'paperswithcode' | 'github' | 'huggingface' | 'corpus'
    ident: str                        # arxiv id / repo full_name / hub model id
    title: str
    url: str
    text: str = ""
    score: float = 0.0
    model_ids: List[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


# ======================================================================= http (offline-graceful)
def _http_get(url: str, *, headers: Optional[dict] = None, timeout: int = _TIMEOUT) -> Optional[bytes]:
    """GET bytes or None on ANY error. Discovery must never crash a run on a flaky/absent API."""
    h = {"User-Agent": _UA}
    if headers:
        h.update(headers)
    try:
        req = urllib.request.Request(url, headers=h)
        with urllib.request.urlopen(req, timeout=timeout, context=_CTX) as r:
            return r.read()
    except Exception:
        return None


def _get_json(url: str, *, headers: Optional[dict] = None) -> Optional[object]:
    raw = _http_get(url, headers=headers)
    if raw is None:
        return None
    try:
        return json.loads(raw.decode("utf-8", "replace"))
    except Exception:
        return None


# ======================================================================= connectors
def fetch_arxiv(query: str, *, limit: int = 8) -> List[Finding]:
    """arXiv Atom API. Reads title + abstract; no model ids (papers name techniques, not Hub ids)."""
    q = urllib.parse.quote(query)
    url = (f"http://export.arxiv.org/api/query?search_query=all:{q}"
           f"&start=0&max_results={int(limit)}&sortBy=relevance&sortOrder=descending")
    raw = _http_get(url)
    if raw is None:
        return []
    try:
        root = ET.fromstring(raw)
    except Exception:
        return []
    ns = {"a": "http://www.w3.org/2005/Atom"}
    out: List[Finding] = []
    for e in root.findall("a:entry", ns):
        title = (e.findtext("a:title", default="", namespaces=ns) or "").strip()
        summ = (e.findtext("a:summary", default="", namespaces=ns) or "").strip()
        ident = (e.findtext("a:id", default="", namespaces=ns) or "").strip()
        if not title:
            continue
        out.append(Finding(source="arxiv", ident=ident.split("/")[-1], title=title, url=ident,
                           text=f"{title}. {summ}"))
    return out


def fetch_paperswithcode(query: str, *, limit: int = 8) -> List[Finding]:
    """Papers-with-Code papers API (best-effort: the endpoint sometimes serves HTML -> []) ."""
    q = urllib.parse.quote(query)
    url = f"https://paperswithcode.com/api/v1/papers/?q={q}&items_per_page={int(limit)}"
    data = _get_json(url)
    if not isinstance(data, dict):
        return []
    out: List[Finding] = []
    for r in (data.get("results") or [])[:limit]:
        if not isinstance(r, dict):
            continue
        title = (r.get("title") or "").strip()
        abstract = (r.get("abstract") or "").strip()
        ident = str(r.get("id") or r.get("arxiv_id") or title)
        url_p = r.get("url_abs") or (f"https://paperswithcode.com/paper/{ident}")
        if not title:
            continue
        out.append(Finding(source="paperswithcode", ident=ident, title=title, url=url_p,
                           text=f"{title}. {abstract}"))
    return out


def fetch_github(query: str, *, limit: int = 8) -> List[Finding]:
    """GitHub repository search, ranked by stars. README text is not fetched (one call); the description
    + repo name carry the technique keywords. Token (GITHUB_TOKEN) used if present to lift rate limits."""
    q = urllib.parse.quote(query)
    url = f"https://api.github.com/search/repositories?q={q}&sort=stars&order=desc&per_page={int(limit)}"
    headers = {"Accept": "application/vnd.github+json"}
    tok = os.environ.get("GITHUB_TOKEN") or os.environ.get("GH_TOKEN")
    if tok:
        headers["Authorization"] = f"Bearer {tok}"
    data = _get_json(url, headers=headers)
    if not isinstance(data, dict):
        return []
    out: List[Finding] = []
    for it in (data.get("items") or [])[:limit]:
        if not isinstance(it, dict):
            continue
        name = it.get("full_name") or ""
        desc = it.get("description") or ""
        topics = " ".join(it.get("topics") or [])
        out.append(Finding(source="github", ident=name, title=name,
                           url=it.get("html_url") or f"https://github.com/{name}",
                           text=f"{name}. {desc} {topics}", score=float(it.get("stargazers_count") or 0)))
    return out


# Hub pipeline tags that indicate a usable vision/representation backbone.
_HF_OK_TAGS = {"image-classification", "image-feature-extraction", "feature-extraction", "zero-shot-image-classification"}


def fetch_huggingface(query: str, *, limit: int = 8) -> List[Finding]:
    """Hugging Face Hub model search, ranked by downloads. These return CONCRETE Hub ids -- the open,
    verbatim part of the discovered pool. Prefer the huggingface_hub client (honours HF_TOKEN); fall back
    to the public JSON API."""
    out: List[Finding] = []
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")
    try:
        from huggingface_hub import list_models
        for info in list_models(search=query, sort="downloads", direction=-1, limit=limit, token=token):
            mid = getattr(info, "id", None) or getattr(info, "modelId", None)
            if not mid:
                continue
            tag = getattr(info, "pipeline_tag", None) or ""
            dl = float(getattr(info, "downloads", 0) or 0)
            out.append(Finding(source="huggingface", ident=mid, title=mid,
                               url=f"https://huggingface.co/{mid}", text=f"{mid} {tag}", score=dl,
                               model_ids=[mid]))
        if out:
            return out
    except Exception:
        pass
    q = urllib.parse.quote(query)
    data = _get_json(f"https://huggingface.co/api/models?search={q}&limit={int(limit)}&sort=downloads")
    if not isinstance(data, list):
        return []
    for m in data[:limit]:
        if not isinstance(m, dict):
            continue
        mid = m.get("id") or m.get("modelId")
        if not mid:
            continue
        tag = m.get("pipeline_tag") or ""
        out.append(Finding(source="huggingface", ident=mid, title=mid,
                           url=f"https://huggingface.co/{mid}", text=f"{mid} {tag}",
                           score=float(m.get("downloads") or 0), model_ids=[mid]))
    return out


_CONNECTORS = {
    "arxiv": fetch_arxiv,
    "paperswithcode": fetch_paperswithcode,
    "github": fetch_github,
    "huggingface": fetch_huggingface,
}

_STOPWORDS = {"the", "a", "an", "and", "or", "of", "for", "with", "under", "on", "in", "to", "by",
              "using", "via", "from", "into", "over", "at", "is", "are", "be", "this", "that"}


def keywords(problem: str, *, k: int = 5) -> List[str]:
    """Salient query keywords from a free-text problem (stopwords removed, order-preserving, de-duplicated).
    arXiv/GitHub take the whole phrase; the HF Hub `search` matches model-id tokens, so it is queried with
    these keywords individually -- general, not a hard-coded family list."""
    toks = re.findall(r"[a-zA-Z][a-zA-Z0-9\-]{2,}", problem.lower())
    out: List[str] = []
    for t in toks:
        if t not in _STOPWORDS and t not in out:
            out.append(t)
    return out[:k]


def _fetch_huggingface_expanded(problem: str, *, limit: int) -> List[Finding]:
    """HF Hub is keyword-indexed: query the full phrase AND each salient keyword, merge (dedup by id,
    keep highest score)."""
    merged: Dict[str, Finding] = {}
    queries = [problem] + keywords(problem)
    per = max(3, limit // 2)
    for q in queries:
        for f in fetch_huggingface(q, limit=per):
            cur = merged.get(f.ident)
            if cur is None or f.score > cur.score:
                merged[f.ident] = f
    return sorted(merged.values(), key=lambda f: f.score, reverse=True)[:limit]


# ======================================================================= bundled offline corpus
# A tiny, fixed corpus used when no source returns anything (offline / hermetic tests). It is deliberately
# NOT a recipe menu: it is a handful of representative published findings whose TECHNIQUE keywords the same
# extractor reads, so the offline path exercises the identical retrieval->extraction->provenance machinery.
_OFFLINE_CORPUS: List[dict] = [
    {"source": "corpus", "ident": "2304.07193", "title": "DINOv2: Learning Robust Visual Features without Supervision",
     "url": "https://arxiv.org/abs/2304.07193",
     "text": "DINOv2 self-supervised vision transformers produce robust visual features; a linear probe on "
             "frozen features is a strong transfer baseline across domains."},
    {"source": "corpus", "ident": "2106.09685", "title": "LoRA: Low-Rank Adaptation of Large Models",
     "url": "https://arxiv.org/abs/2106.09685",
     "text": "Low-rank adaptation (LoRA) freezes the backbone and learns low-rank updates; parameter-efficient "
             "fine-tuning competitive with full fine-tuning at a fraction of the cost."},
    {"source": "corpus", "ident": "2203.05482", "title": "Model soups: averaging weights of multiple fine-tuned models",
     "url": "https://arxiv.org/abs/2203.05482",
     "text": "Averaging the weights of multiple fine-tuned models (a model soup) improves accuracy and "
             "robustness to distribution shift without extra inference cost."},
    {"source": "corpus", "ident": "wilds-camelyon17", "title": "WILDS Camelyon17 tumor classification under hospital shift",
     "url": "https://wilds.stanford.edu",
     "text": "Histopathology tumor classification with a distribution shift across hospitals; class rebalancing "
             "and strong augmentation such as RandAugment help under shift."},
    {"source": "corpus", "ident": "siglip", "title": "SigLIP sigmoid loss for language-image pretraining",
     "url": "https://huggingface.co/timm",
     "text": "SigLIP image encoders give strong transfer features; visual prompt tuning (VPT) adapts a frozen "
             "ViT with a few learned prompt tokens."},
]


def _offline_corpus() -> List[Finding]:
    return [Finding(**d) for d in _OFFLINE_CORPUS]


# ======================================================================= caching
def _default_cache_dir() -> str:
    root = os.environ.get("ATTESTRA_LIT_CACHE") or os.path.join(os.path.expanduser("~"), "wilds_data",
                                                                 "_litcache")
    return root


def _cache_key(source: str, query: str, limit: int) -> str:
    return hashlib.sha256(f"{source}|{query}|{limit}".encode()).hexdigest()[:20]


def _cache_read(cache_dir: str, key: str) -> Optional[List[Finding]]:
    path = os.path.join(cache_dir, f"{key}.json")
    try:
        with open(path, "r") as f:
            rows = json.load(f)
        return [Finding(**r) for r in rows]
    except Exception:
        return None


def _cache_write(cache_dir: str, key: str, findings: Sequence[Finding]) -> None:
    try:
        os.makedirs(cache_dir, exist_ok=True)
        with open(os.path.join(cache_dir, f"{key}.json"), "w") as f:
            json.dump([x.to_dict() for x in findings], f)
    except Exception:
        pass


# ======================================================================= retrieval
def retrieve(query: str, *, sources: Sequence[str] = ("arxiv", "github", "huggingface", "paperswithcode"),
             limit: int = 8, online: bool = True, use_cache: bool = True,
             cache_dir: Optional[str] = None) -> List[Finding]:
    """Retrieve findings for `query` across `sources`. When `online`, hits the live APIs (with on-disk
    caching); otherwise reads cache only. If everything is empty (offline, no cache, all errors), falls back
    to the bundled corpus so downstream is never starved. Deterministic given the same cache/network."""
    cache_dir = cache_dir or _default_cache_dir()
    all_findings: List[Finding] = []
    for src in sources:
        conn = _CONNECTORS.get(src)
        if conn is None:
            continue
        key = _cache_key(src, query, limit)
        cached = _cache_read(cache_dir, key) if use_cache else None
        if cached is not None:
            all_findings.extend(cached)
            continue
        if not online:
            rows = []
        elif src == "huggingface":
            rows = _fetch_huggingface_expanded(query, limit=limit)
        else:
            rows = conn(query, limit=limit)
        if rows:
            if use_cache:
                _cache_write(cache_dir, key, rows)
            all_findings.extend(rows)
    if not all_findings:
        return _offline_corpus()
    return all_findings


# ======================================================================= extraction: backbones
# Family mention -> a representative, loadable timm id. The OPEN signal is WHICH family the literature
# surfaces for the problem; this map only canonicalizes a family word into one concrete id the runner can
# load. Real Hub ids returned by the HF connector are used verbatim (not through this map).
_ARCH_CANON: Dict[str, str] = {
    "dinov2": "vit_base_patch14_reg4_dinov2.lvd142m",
    "siglip": "vit_base_patch16_siglip_224.webli",
    "clip": "vit_base_patch16_clip_224.openai",
    "eva02": "eva02_base_patch14_224.mim_in22k",
    "convnext": "convnext_base.fb_in22k",
    "swin": "swin_base_patch4_window7_224.ms_in22k",
    "beit": "beit_base_patch16_224.in22k_ft_in22k",
    "deit": "deit3_base_patch16_224.fb_in22k_ft_in1k",
    "efficientnet": "efficientnet_b0.ra_in1k",
    "resnet": "resnet50.a1_in1k",
    "vit": "vit_base_patch16_224.augreg2_in21k_ft_in1k",
}
# longest / most-specific families first so 'eva02' wins over 'vit', 'dinov2' over 'vit', etc.
_ARCH_ORDER = ["dinov2", "siglip", "eva02", "convnext", "swin", "beit", "deit", "efficientnet",
               "clip", "resnet", "vit"]


def extract_backbones(findings: Sequence[Finding]) -> List[Tuple[str, Finding]]:
    """Return (model_id, source_finding) pairs. Concrete Hub ids (HF connector) are used verbatim; family
    mentions in paper/repo text are canonicalized to one representative id. De-duplicated, preserving the
    first (highest-ranked) source for provenance."""
    out: List[Tuple[str, Finding]] = []
    seen: set = set()

    def add(mid: str, f: Finding) -> None:
        if mid and mid not in seen:
            seen.add(mid)
            out.append((mid, f))

    for f in findings:
        for mid in f.model_ids:               # verbatim Hub ids
            add(mid, f)
    for f in findings:
        low = f.text.lower()
        for fam in _ARCH_ORDER:
            if fam in low:
                add(_ARCH_CANON[fam], f)
    return out


# ======================================================================= extraction: motifs
# keyword(s) -> a partial recipe genome (a technique motif). These genes are DISTILLED from the technique
# the keyword names; they are validated against the typed axes before use, so an off-axis suggestion is
# dropped. Each rule, when matched, yields a Motif tagged with the matching finding(s) as provenance.
_MOTIF_RULES: List[Tuple[Tuple[str, ...], str, Genome]] = [
    (("linear probe", "linear probing", "frozen feature"), "lit:linear_probe",
     {"adaptation": "linear_probe", "head": "gbm"}),
    (("low-rank adaptation", "low rank adaptation", "lora"), "lit:lora",
     {"adaptation": "lora", "schedule": "cosine", "llrd": True}),
    (("visual prompt", "prompt tuning", "vpt"), "lit:vpt",
     {"adaptation": "vpt", "schedule": "cosine"}),
    (("adapter",), "lit:adapter", {"adaptation": "adapter"}),
    (("full fine-tun", "fully fine-tun", "end-to-end fine"), "lit:full_ft",
     {"adaptation": "full_ft", "schedule": "cosine", "llrd": True, "epochs": 25}),
    (("fixres", "higher resolution", "two-stage training"), "lit:fixres",
     {"augmentation": "fixres", "epochs": 25, "schedule": "cosine"}),
    (("randaugment", "rand augment", "randaug"), "lit:randaug",
     {"augmentation": "randaug", "epochs": 20}),
    (("mixup",), "lit:mixup", {"augmentation": "mixup", "optimizer": "adamw", "weight_decay": 0.05}),
    (("cutmix",), "lit:cutmix", {"augmentation": "cutmix"}),
    (("model soup", "weight averaging", "averaging the weights", "model averaging"), "lit:model_soup",
     {"aggregation": "model_soup"}),
    (("logit ensemble", "ensemble"), "lit:ensemble", {"aggregation": "logit_ensemble"}),
    (("distillation", "distill"), "lit:distill", {"aggregation": "distill"}),
    (("sharpness-aware", "sharpness aware", "sam optimizer"), "lit:sam", {"optimizer": "sam"}),
    (("layer-wise learning rate", "layer-wise lr", "layerwise", "llrd"), "lit:llrd",
     {"llrd": True, "schedule": "cosine"}),
    (("class rebalanc", "class imbalance", "class-imbalance", "rebalancing", "resampling", "oversampling"),
     "lit:rebalance", {"data_strategy": "class_rebalance"}),
    (("label noise", "noisy label", "label-noise"), "lit:noise_repair",
     {"data_strategy": "label_noise_repair"}),
    (("active learning", "label acquisition", "label budget"), "lit:active", {"data_strategy": "active"}),
]

# typed validation: a proposed gene survives only if its key is a known axis AND its value is legal.
_CATEGORICAL_AXES: Dict[str, Tuple[str, ...]] = {
    "adaptation": ADAPTATIONS, "augmentation": AUGMENTATIONS, "optimizer": OPTIMIZERS,
    "schedule": SCHEDULES, "aggregation": AGGREGATIONS, "data_strategy": DATA_STRATEGIES, "head": HEADS,
}


def validate_genes(genes: Genome) -> Genome:
    """Drop any gene whose key is not a recipe axis or whose value is off-axis. Numeric axes are clamped to
    the recipe space. This is the firewall that makes a hallucinated/LLM-proposed motif safe: at worst it is
    emptied, never injected as an illegal recipe."""
    out: Genome = {}
    for k, v in genes.items():
        if k in _CATEGORICAL_AXES:
            if v in _CATEGORICAL_AXES[k]:
                out[k] = v
        elif k == "llrd":
            out[k] = bool(v)
        elif k == "epochs":
            try:
                out[k] = int(min(40, max(3, int(v))))
            except Exception:
                continue
        elif k in ("lr", "weight_decay"):
            try:
                out[k] = float(v)
            except Exception:
                continue
        # unknown key (incl. 'backbone' -- handled via extract_backbones, not motifs) -> dropped
    return out


def extract_motifs(findings: Sequence[Finding]) -> List[Tuple[Motif, List[Finding]]]:
    """Deterministic keyword extractor: scan finding text for technique keywords and emit validated motifs,
    each with the list of findings that mentioned it (provenance). De-duplicated by motif name."""
    by_name: Dict[str, Tuple[Motif, List[Finding]]] = {}
    blob = [(f, f.text.lower()) for f in findings]
    for keywords, name, genes in _MOTIF_RULES:
        srcs = [f for f, low in blob if any(kw in low for kw in keywords)]
        if not srcs:
            continue
        valid = validate_genes(genes)
        if not valid:
            continue
        if name not in by_name:
            by_name[name] = (Motif(name, valid), srcs)
    return list(by_name.values())


# ======================================================================= optional LLM bridge
_LLM_SYSTEM = (
    "You are an ML research assistant. Given retrieved paper/repo titles+abstracts and a target problem, "
    "propose training-recipe MOTIFS grounded in what the findings actually describe. Output STRICT JSON: "
    '{"motifs":[{"name":"lit:<short>","genes":{<axis>:<value>,...},"why":"<finding-grounded reason>"}],'
    '"backbones":["<hub or timm id>",...]}. '
    "Allowed gene axes ONLY: adaptation in [linear_probe,lora,adapter,vpt,partial_unfreeze,full_ft]; "
    "augmentation in [none,randaug,trivialaug,mixup,cutmix,fixres]; optimizer in [adamw,sam]; "
    "schedule in [cosine,step,constant]; llrd in [true,false]; aggregation in "
    "[single,logit_ensemble,model_soup,distill]; data_strategy in [none,class_rebalance,label_noise_repair,"
    "active]; head in [gbm,linear]; epochs int 3..40. Do not invent axes. Ground every motif in a finding."
)


def llm_motifs(findings: Sequence[Finding], problem: str, *, api_key: Optional[str] = None,
               model: Optional[str] = None, max_findings: int = 12
               ) -> Optional[Tuple[List[Tuple[Motif, List[Finding]]], List[str]]]:
    """Ask Claude to read the findings and propose motifs + backbone ids. Returns (motifs, backbone_ids) or
    None on any failure (no key, no client, parse error) so the caller falls back to the deterministic
    extractor. Every returned motif is validated against the typed axes; backbones are returned as ids."""
    api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        return None
    model = model or os.environ.get("ATTESTRA_STRATEGIST_MODEL", "claude-sonnet-4-5")
    digest = "\n".join(f"- [{f.source}:{f.ident}] {f.title}: {f.text[:300]}"
                       for f in list(findings)[:max_findings])
    prompt = f"TARGET PROBLEM: {problem}\n\nRETRIEVED FINDINGS:\n{digest}\n\nReturn the STRICT JSON now."
    try:
        import anthropic
        client = anthropic.Anthropic(api_key=api_key)
        msg = client.messages.create(model=model, max_tokens=1400, system=_LLM_SYSTEM,
                                     messages=[{"role": "user", "content": prompt}])
        text = "".join(getattr(b, "text", "") for b in msg.content)
    except Exception:
        return None
    m = re.search(r"\{.*\}", text, re.S)
    if not m:
        return None
    try:
        data = json.loads(m.group(0))
    except Exception:
        return None
    motifs: List[Tuple[Motif, List[Finding]]] = []
    for item in (data.get("motifs") or []):
        if not isinstance(item, dict):
            continue
        name = str(item.get("name") or "lit:llm")
        genes = validate_genes(item.get("genes") or {})
        if genes:
            motifs.append((Motif(name, genes), list(findings)[:max_findings]))
    backbones = [str(b) for b in (data.get("backbones") or []) if isinstance(b, (str,)) and b.strip()]
    return motifs, backbones


# ======================================================================= the scout
class LiteratureScout:
    """Ties retrieval -> extraction -> typed ingredients together, and exposes them to the generator with
    PROVENANCE. `backbones()` plugs into RecipeGenerator.discover_fn; `library()` supplies the grafted
    motifs. Every ingredient is recorded in `provenance` so the certificate can trace the champion to a
    real source. Deterministic given the same retrieval (cache/network)."""

    def __init__(self, problem: str, *, sources: Sequence[str] = ("arxiv", "github", "huggingface",
                 "paperswithcode"), limit: int = 8, online: bool = True, use_llm: bool = False,
                 api_key: Optional[str] = None, cache_dir: Optional[str] = None,
                 max_backbones: int = 12):
        self.problem = problem
        self.max_backbones = int(max_backbones)
        self.findings: List[Finding] = retrieve(problem, sources=sources, limit=limit, online=online,
                                                 cache_dir=cache_dir)
        # provenance: ingredient id/name -> {source, ident, url, title}
        self.provenance: Dict[str, dict] = {}
        self.used_llm = False

        bb_pairs = extract_backbones(self.findings)
        self._backbones: List[str] = []
        for mid, f in bb_pairs[: self.max_backbones]:
            self._backbones.append(mid)
            self.provenance[mid] = {"kind": "backbone", "source": f.source, "ident": f.ident,
                                    "url": f.url, "title": f.title}

        motifs_prov: List[Tuple[Motif, List[Finding]]] = []
        if use_llm:
            res = llm_motifs(self.findings, problem, api_key=api_key)
            if res is not None:
                motifs_prov, llm_backbones = res
                self.used_llm = True
                for b in llm_backbones[: self.max_backbones]:
                    if b not in self._backbones:
                        self._backbones.append(b)
                        self.provenance[b] = {"kind": "backbone", "source": "llm", "ident": "llm",
                                              "url": "", "title": "LLM-proposed from findings"}
        if not motifs_prov:
            motifs_prov = extract_motifs(self.findings)
        self._motifs: List[Motif] = []
        for motif, srcs in motifs_prov:
            self._motifs.append(motif)
            f = srcs[0] if srcs else None
            self.provenance[motif.name] = {"kind": "motif", "genes": motif.genes,
                                           "source": (f.source if f else "corpus"),
                                           "ident": (f.ident if f else "-"),
                                           "url": (f.url if f else ""),
                                           "title": (f.title if f else "")}

    # -- generator-facing surface -----------------------------------------------------------------------
    def backbones(self) -> List[str]:
        """Literature-discovered backbone ids (verbatim Hub ids + canonicalized family mentions)."""
        return list(self._backbones)

    def motifs(self) -> List[Motif]:
        return list(self._motifs)

    def library(self):
        """A regeneration.LiteratureLibrary built from the RETRIEVED motifs (replaces the hard-coded one)."""
        from .regeneration import LiteratureLibrary
        return LiteratureLibrary(self._motifs)

    def discover_fn(self):
        """A no-arg callable for RecipeGenerator.discover_fn (returns the literature backbone ids)."""
        return lambda: list(self._backbones)

    # -- reporting --------------------------------------------------------------------------------------
    def summary(self) -> dict:
        by_src: Dict[str, int] = {}
        for f in self.findings:
            by_src[f.source] = by_src.get(f.source, 0) + 1
        return {
            "problem": self.problem,
            "n_findings": len(self.findings),
            "findings_by_source": by_src,
            "n_backbones": len(self._backbones),
            "backbones": list(self._backbones),
            "n_motifs": len(self._motifs),
            "motifs": [{"name": m.name, "genes": m.genes} for m in self._motifs],
            "used_llm": self.used_llm,
            "provenance": self.provenance,
            "findings": [f.to_dict() for f in self.findings],
        }


__all__ = [
    "Finding", "retrieve", "fetch_arxiv", "fetch_paperswithcode", "fetch_github", "fetch_huggingface",
    "extract_backbones", "extract_motifs", "validate_genes", "llm_motifs", "LiteratureScout",
]
