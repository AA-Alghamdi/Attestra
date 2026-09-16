"""FOUNDATION-MODEL FEATURIZERS -- transfer beats search volume on small/medium commercial data.

WHY THIS EXISTS
---------------
On the commercial problems (100-call call-ender, 10k-image biopsy) the n is tiny relative to the modality's
complexity, so REPRESENTATION/TRANSFER is the whole game: a frozen pretrained encoder + a tiny head, not
training from raw pixels/waveforms. The existing vision featurizer flattens pixels (it learns position,
not shape); that caps quality on exactly the problems where transfer matters most.

This module defines one clean Featurizer interface per modality with:
  * a REAL pretrained-backbone injection point (`backbone=`/`encoder=`): pass any callable mapping raw
    inputs -> embedding (e.g. a frozen torchvision resnet, a sentence-transformer, a speech encoder) and
    the featurizer uses it. This is the intended production path.
  * a deterministic, numpy-only FALLBACK that needs no download and still captures transfer-flavored
    STRUCTURE (gradient-orientation / multi-scale pooling for images; hashed n-grams for text; log-mel
    spectral statistics for audio) -- markedly better than flatten-pixels, so the pipeline is runnable
    end-to-end on a slim CPU box while a real backbone is wired.

THE INVARIANT
-------------
A featurizer changes only HOW raw inputs become a fixed numeric matrix. It never certifies, never promotes,
and the downstream frozen certifier is unchanged. Embeddings are L2-normalized and fixed-dimension so a
linear head transfers cleanly. Deterministic: same input -> same embedding.
"""
from __future__ import annotations

import hashlib
import math
from abc import ABC, abstractmethod
from typing import Callable, Optional, Sequence, Union

import numpy as np

ArrayLike = Union[np.ndarray, Sequence[float], Sequence[Sequence[float]]]


def _l2(M: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(M, axis=1, keepdims=True)
    norms[norms == 0.0] = 1.0
    return M / norms


class Featurizer(ABC):
    """raw inputs (a list of items) -> a fixed (n, dim) float matrix."""

    @property
    @abstractmethod
    def name(self) -> str: ...

    @property
    @abstractmethod
    def dim(self) -> int: ...

    @abstractmethod
    def transform(self, inputs: Sequence) -> np.ndarray: ...


# -- vision ---------------------------------------------------------------------------------------------

def _to_gray_grid(img: ArrayLike, grid: int) -> np.ndarray:
    """Coerce an image to a grid x grid grayscale array in [0,1] via block-mean pooling (size-invariant)."""
    a = np.asarray(img, dtype=float)
    if a.ndim == 1:
        s = int(round(math.sqrt(a.size)))
        if s * s == a.size:
            a = a.reshape(s, s)
        else:
            a = a.reshape(1, -1)
    if a.ndim == 3:
        a = a.mean(axis=2)
    mn, mx = float(a.min()), float(a.max())
    if mx > mn:
        a = (a - mn) / (mx - mn)
    h, w = a.shape
    out = np.zeros((grid, grid), dtype=float)
    rs = np.linspace(0, h, grid + 1).astype(int)
    cs = np.linspace(0, w, grid + 1).astype(int)
    for i in range(grid):
        for j in range(grid):
            r0, r1 = rs[i], max(rs[i] + 1, rs[i + 1])
            c0, c1 = cs[j], max(cs[j] + 1, cs[j + 1])
            out[i, j] = a[r0:r1, c0:c1].mean()
    return out


class ImageFeaturizer(Featurizer):
    """Transfer-flavored image embedding. With a `backbone` callable -> uses it (the production path).
    Fallback (numpy): gradient-ORIENTATION histogram (position-invariant shape/edge signal) + multi-scale
    average-pool maps. Captures shape/texture, not raw pixel positions."""

    def __init__(self, *, grid: int = 8, orient_bins: int = 9,
                 backbone: Optional[Callable[[Sequence], np.ndarray]] = None, backbone_dim: int = 0):
        self.grid = int(grid)
        self.orient_bins = int(orient_bins)
        self.backbone = backbone
        self._backbone_dim = int(backbone_dim)
        self._pool2 = 2 * 2
        self._fallback_dim = self.orient_bins + self.grid * self.grid + self._pool2

    @property
    def name(self) -> str:
        return "image_backbone" if self.backbone is not None else "image_fallback_orient_pool"

    @property
    def dim(self) -> int:
        return self._backbone_dim if self.backbone is not None else self._fallback_dim

    def _embed_one(self, img: ArrayLike) -> np.ndarray:
        g = _to_gray_grid(img, self.grid)
        gy, gx = np.gradient(g)
        mag = np.sqrt(gx * gx + gy * gy)
        ang = np.mod(np.arctan2(gy, gx), math.pi)            # unsigned orientation [0, pi)
        hist = np.zeros(self.orient_bins, dtype=float)
        bin_idx = np.minimum((ang / math.pi * self.orient_bins).astype(int), self.orient_bins - 1)
        for b in range(self.orient_bins):
            hist[b] = mag[bin_idx == b].sum()
        # multi-scale pools: the full grid (flattened) + a coarse 2x2 average
        pool2 = _to_gray_grid(g, 2).reshape(-1)
        return np.concatenate([hist, g.reshape(-1), pool2])

    def transform(self, inputs: Sequence) -> np.ndarray:
        if self.backbone is not None:
            M = np.asarray(self.backbone(inputs), dtype=float)
            if M.ndim != 2:
                raise ValueError("backbone must return a 2-D (n, dim) embedding matrix")
            return _l2(M)
        return _l2(np.asarray([self._embed_one(x) for x in inputs], dtype=float))


class ResnetBackbone:
    """The production transfer path for ImageFeaturizer: a FROZEN ImageNet-pretrained torchvision CNN with
    its classification head removed, callable as inputs -> (n, feat_dim) embedding matrix. Requires torch +
    torchvision + Pillow (raises ImportError otherwise). Inputs may be PIL images or numpy HxW / HxWxC
    uint8/float arrays. The net is eval-mode and never trained here -- it only produces a fixed
    representation a downstream certifiable head consumes; it never touches the certifier."""

    _ARCHS = ("resnet18", "resnet34", "resnet50")

    def __init__(self, *, arch: str = "resnet18", resize: int = 112, batch_size: int = 64,
                 pretrained: bool = True):
        if arch not in self._ARCHS:
            raise ValueError(f"unsupported arch {arch!r}; choose from {list(self._ARCHS)}")
        try:
            import torch
            import torchvision.models as models
            from torchvision import transforms
            from PIL import Image
        except Exception as exc:  # noqa: BLE001
            raise ImportError("ResnetBackbone requires torch + torchvision + Pillow") from exc
        builders = {"resnet18": (models.resnet18, models.ResNet18_Weights),
                    "resnet34": (models.resnet34, models.ResNet34_Weights),
                    "resnet50": (models.resnet50, models.ResNet50_Weights)}
        self._torch = torch
        self._Image = Image
        self.arch = arch
        self.resize = int(resize)
        self.batch_size = int(batch_size)
        build, weights_enum = builders[arch]
        weights = weights_enum.IMAGENET1K_V1
        net = build(weights=weights if pretrained else None)
        self.feat_dim = int(net.fc.in_features)
        net.fc = torch.nn.Identity()
        net.eval()
        self._net = net
        if pretrained:
            tf = weights.transforms()
            mean, std = tf.mean, tf.std
        else:
            mean, std = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
        self._prep = transforms.Compose([transforms.Resize(self.resize),
                                         transforms.CenterCrop(self.resize),
                                         transforms.ToTensor(),
                                         transforms.Normalize(mean, std)])

    def _as_pil(self, x):
        if isinstance(x, self._Image.Image):
            return x.convert("RGB")
        a = np.asarray(x)
        if a.ndim == 2:
            a = np.stack([a, a, a], axis=-1)
        if a.dtype != np.uint8:
            mn, mx = float(a.min()), float(a.max())
            a = (((a - mn) / (mx - mn) * 255.0) if mx > mn else np.zeros_like(a, dtype=float)).astype(np.uint8)
        return self._Image.fromarray(a).convert("RGB")

    def __call__(self, inputs: Sequence) -> np.ndarray:
        buf = [self._prep(self._as_pil(x)) for x in inputs]
        if not buf:
            return np.zeros((0, self.feat_dim), dtype=float)
        out = []
        with self._torch.no_grad():
            for i in range(0, len(buf), self.batch_size):
                batch = self._torch.stack(buf[i:i + self.batch_size])
                out.append(self._net(batch).cpu().numpy())
        return np.concatenate(out, axis=0)


def _np_to_pil_rgb(x, Image):
    """Coerce a PIL image or numpy HxW / HxWxC (uint8 or float) array to an RGB PIL image. Shared by the
    transfer backbones so CIFAR-style uint8 arrays and PIL inputs are handled identically."""
    if isinstance(x, Image.Image):
        return x.convert("RGB")
    a = np.asarray(x)
    if a.ndim == 2:
        a = np.stack([a, a, a], axis=-1)
    if a.dtype != np.uint8:
        mn, mx = float(a.min()), float(a.max())
        a = (((a - mn) / (mx - mn) * 255.0) if mx > mn else np.zeros_like(a, dtype=float)).astype(np.uint8)
    return Image.fromarray(a).convert("RGB")


class ClipImageBackbone:
    """FROZEN CLIP image tower (language-aligned representation) as inputs -> (n, output_dim) embeddings,
    via open_clip. Requires torch + open_clip + Pillow (raises ImportError otherwise). The visual encoder is
    eval-mode and never trained here; it only emits a fixed representation a downstream certifiable head
    consumes -- it never touches the certifier. `pretrained=None` builds the architecture with random weights
    (offline; used by hermetic tests)."""

    def __init__(self, *, model_name: str = "ViT-B-32", pretrained: Optional[str] = "openai",
                 batch_size: int = 64):
        try:
            import torch
            import open_clip
            from PIL import Image
        except Exception as exc:  # noqa: BLE001
            raise ImportError("ClipImageBackbone requires torch + open_clip + Pillow") from exc
        self._torch = torch
        self._Image = Image
        self.model_name = model_name
        self.batch_size = int(batch_size)
        model, _, preprocess = open_clip.create_model_and_transforms(model_name, pretrained=pretrained)
        model.eval()
        self._model = model
        self._prep = preprocess
        # Most open_clip towers expose visual.output_dim; SigLIP's timm-backed tower does not, so fall back
        # to a single dummy forward pass (offline-safe) to read the true embedding width.
        try:
            self.feat_dim = int(model.visual.output_dim)
        except AttributeError:
            with torch.no_grad():
                probe = self._prep(Image.new("RGB", (32, 32)))
                self.feat_dim = int(model.encode_image(probe.unsqueeze(0)).shape[1])

    def __call__(self, inputs: Sequence) -> np.ndarray:
        buf = [self._prep(_np_to_pil_rgb(x, self._Image)) for x in inputs]
        if not buf:
            return np.zeros((0, self.feat_dim), dtype=float)
        out = []
        with self._torch.no_grad():
            for i in range(0, len(buf), self.batch_size):
                batch = self._torch.stack(buf[i:i + self.batch_size])
                out.append(self._model.encode_image(batch).cpu().numpy())
        return np.concatenate(out, axis=0)


class TimmBackbone:
    """FROZEN timm backbone (e.g. a DINOv2 self-supervised ViT) as inputs -> (n, num_features) pooled
    embeddings. `num_classes=0` makes the model return its pooled feature vector. Requires torch + timm +
    Pillow (raises ImportError otherwise). Eval-mode, never trained here, never touches the certifier.
    `pretrained=False` builds the architecture with random weights (offline; used by hermetic tests)."""

    def __init__(self, *, model_name: str = "vit_small_patch14_dinov2.lvd142m", pretrained: bool = True,
                 batch_size: int = 32, img_size: Optional[int] = 224):
        try:
            import torch
            import timm
            from PIL import Image
        except Exception as exc:  # noqa: BLE001
            raise ImportError("TimmBackbone requires torch + timm + Pillow") from exc
        self._torch = torch
        self._Image = Image
        self.model_name = model_name
        self.batch_size = int(batch_size)
        kw = {"pretrained": pretrained, "num_classes": 0}
        if img_size is not None:
            kw["img_size"] = int(img_size)        # DINOv2's timm default is 518px; 224 (interpolated pos-embed) is far cheaper on CPU
        model = timm.create_model(model_name, **kw)
        model.eval()
        cfg = timm.data.resolve_data_config({}, model=model)
        if img_size is not None:                  # resolve_data_config reports the model's DEFAULT input_size (e.g. 518); pin it to img_size
            cfg["input_size"] = (3, int(img_size), int(img_size))
        self._prep = timm.data.create_transform(**cfg)
        self._model = model
        self.feat_dim = int(model.num_features)

    def __call__(self, inputs: Sequence) -> np.ndarray:
        buf = [self._prep(_np_to_pil_rgb(x, self._Image)) for x in inputs]
        if not buf:
            return np.zeros((0, self.feat_dim), dtype=float)
        out = []
        with self._torch.no_grad():
            for i in range(0, len(buf), self.batch_size):
                batch = self._torch.stack(buf[i:i + self.batch_size])
                feats = self._model(batch)
                out.append(np.asarray(feats.cpu().numpy()))
        return np.concatenate(out, axis=0)


# -- text -----------------------------------------------------------------------------------------------

class TextFeaturizer(Featurizer):
    """Deterministic hashed n-gram embedding (the hashing trick), L2-normalized to a fixed dim. With an
    `encoder` callable -> uses it (e.g. a sentence-transformer). Fallback needs no vocabulary fit and no
    network, and similar documents get similar vectors."""

    def __init__(self, *, dim: int = 256, word_ngrams: int = 2, char_ngrams: int = 0,
                 encoder: Optional[Callable[[Sequence[str]], np.ndarray]] = None, encoder_dim: int = 0):
        self._dim = int(dim)
        self.word_ngrams = int(word_ngrams)
        self.char_ngrams = int(char_ngrams)
        self.encoder = encoder
        self._encoder_dim = int(encoder_dim)

    @property
    def name(self) -> str:
        return "text_encoder" if self.encoder is not None else "text_hashed_ngram"

    @property
    def dim(self) -> int:
        return self._encoder_dim if self.encoder is not None else self._dim

    def _hash(self, token: str) -> int:
        h = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
        return int.from_bytes(h, "big") % self._dim

    def _embed_one(self, text: str) -> np.ndarray:
        vec = np.zeros(self._dim, dtype=float)
        words = str(text).lower().split()
        for n in range(1, self.word_ngrams + 1):
            for i in range(len(words) - n + 1):
                vec[self._hash("w%d:" % n + " ".join(words[i:i + n]))] += 1.0
        if self.char_ngrams > 0:
            s = str(text).lower()
            for i in range(len(s) - self.char_ngrams + 1):
                vec[self._hash("c:" + s[i:i + self.char_ngrams])] += 1.0
        return vec

    def transform(self, inputs: Sequence) -> np.ndarray:
        if self.encoder is not None:
            M = np.asarray(self.encoder(inputs), dtype=float)
            if M.ndim != 2:
                raise ValueError("encoder must return a 2-D (n, dim) embedding matrix")
            return _l2(M)
        return _l2(np.asarray([self._embed_one(t) for t in inputs], dtype=float))


# -- audio ----------------------------------------------------------------------------------------------

class AudioFeaturizer(Featurizer):
    """Deterministic log-mel-ish spectral embedding for short clips (suited to the 50ms call-ender). With
    an `encoder` callable -> uses it (e.g. a frozen speech encoder). Fallback (numpy FFT): frame the
    waveform, triangular mel filterbank on the magnitude spectrum, log, then mean+std pooling over frames
    -> a fixed 2*n_mels vector."""

    def __init__(self, *, sample_rate: int = 16000, n_mels: int = 24, frame: int = 400, hop: int = 160,
                 encoder: Optional[Callable[[Sequence], np.ndarray]] = None, encoder_dim: int = 0):
        self.sample_rate = int(sample_rate)
        self.n_mels = int(n_mels)
        self.frame = int(frame)
        self.hop = int(hop)
        self.encoder = encoder
        self._encoder_dim = int(encoder_dim)
        self._fb = self._mel_filterbank(self.frame // 2 + 1, self.n_mels, self.sample_rate)

    @property
    def name(self) -> str:
        return "audio_encoder" if self.encoder is not None else "audio_logmel_stats"

    @property
    def dim(self) -> int:
        return self._encoder_dim if self.encoder is not None else 2 * self.n_mels

    @staticmethod
    def _hz_to_mel(f: float) -> float:
        return 2595.0 * math.log10(1.0 + f / 700.0)

    @staticmethod
    def _mel_to_hz(m: float) -> float:
        return 700.0 * (10.0 ** (m / 2595.0) - 1.0)

    def _mel_filterbank(self, n_fft_bins: int, n_mels: int, sr: int) -> np.ndarray:
        fmax = sr / 2.0
        mel_pts = np.linspace(0.0, self._hz_to_mel(fmax), n_mels + 2)
        hz_pts = np.array([self._mel_to_hz(m) for m in mel_pts])
        bins = np.floor((n_fft_bins - 1) * hz_pts / fmax).astype(int)
        fb = np.zeros((n_mels, n_fft_bins), dtype=float)
        for m in range(1, n_mels + 1):
            l, c, r = bins[m - 1], bins[m], bins[m + 1]
            c = max(c, l + 1)
            r = max(r, c + 1)
            for k in range(l, min(c, n_fft_bins)):
                fb[m - 1, k] = (k - l) / max(c - l, 1)
            for k in range(c, min(r, n_fft_bins)):
                fb[m - 1, k] = (r - k) / max(r - c, 1)
        return fb

    def _embed_one(self, wave: ArrayLike) -> np.ndarray:
        x = np.asarray(wave, dtype=float).reshape(-1)
        if x.size < self.frame:
            x = np.pad(x, (0, self.frame - x.size))
        window = np.hanning(self.frame)
        frames = []
        for start in range(0, x.size - self.frame + 1, self.hop):
            seg = x[start:start + self.frame] * window
            spec = np.abs(np.fft.rfft(seg))
            mel = self._fb @ spec
            frames.append(np.log1p(mel))
        if not frames:
            frames.append(np.zeros(self.n_mels))
        F = np.asarray(frames)
        return np.concatenate([F.mean(axis=0), F.std(axis=0)])

    def transform(self, inputs: Sequence) -> np.ndarray:
        if self.encoder is not None:
            M = np.asarray(self.encoder(inputs), dtype=float)
            if M.ndim != 2:
                raise ValueError("encoder must return a 2-D (n, dim) embedding matrix")
            return _l2(M)
        return _l2(np.asarray([self._embed_one(w) for w in inputs], dtype=float))


def get_featurizer(modality: str, **kw) -> Featurizer:
    """Factory: 'image'/'vision' -> ImageFeaturizer, 'text' -> TextFeaturizer, 'audio'/'speech' ->
    AudioFeaturizer. Pass backbone=/encoder= to use a real pretrained model."""
    m = modality.lower()
    if m in ("image", "vision"):
        return ImageFeaturizer(**kw)
    if m in ("text", "nlp"):
        return TextFeaturizer(**kw)
    if m in ("audio", "speech"):
        return AudioFeaturizer(**kw)
    raise ValueError(f"unknown modality {modality!r}; expected image/text/audio")


__all__ = ["Featurizer", "ImageFeaturizer", "TextFeaturizer", "AudioFeaturizer", "get_featurizer"]
