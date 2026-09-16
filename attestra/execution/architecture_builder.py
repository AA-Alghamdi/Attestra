"""Architecture builder: LLM-driven neural architecture construction.

The builder:
  1. Takes a problem spec (domain, data shape, constraints)
  2. Uses LLM to propose architectures (or selects from a library)
  3. Generates executable PyTorch code for the architecture
  4. Validates the architecture (forward pass, parameter count, memory)
  5. Returns a trainable module

This is the "build it" part of EXECUTOR — not just selecting from a menu,
but actually constructing novel architectures for the problem.
"""
from __future__ import annotations

import ast
import re
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np


@dataclass
class ArchitectureSpec:
    """Specification for a neural architecture."""
    name: str
    task: str                             # "classification" | "regression" | "generation"
    input_shape: Tuple[int, ...]          # e.g. (64,) for tabular, (3, 224, 224) for vision
    output_shape: Tuple[int, ...]         # e.g. (10,) for 10 classes
    # Architecture choices
    backbone: str = "mlp"                 # "mlp" | "resnet" | "transformer" | "unet" | "custom"
    hidden_dims: List[int] = field(default_factory=lambda: [256, 128])
    dropout: float = 0.1
    normalization: str = "batchnorm"      # "batchnorm" | "layernorm" | "groupnorm" | "none"
    activation: str = "gelu"
    # Constraints
    max_params: int = 10_000_000
    max_memory_mb: float = 500.0
    # Optional custom code
    custom_code: str = ""


@dataclass
class BuiltArchitecture:
    """A validated, ready-to-train architecture."""
    name: str
    module: Any                           # nn.Module
    n_parameters: int
    estimated_memory_mb: float
    code: str                             # source code that generated it
    validated: bool = False
    validation_error: str = ""


# ============================================================================== architecture library

_ARCH_LIBRARY = {
    "mlp_small": {
        "layers": [128, 64],
        "dropout": 0.1,
        "norm": "batchnorm",
    },
    "mlp_medium": {
        "layers": [512, 256, 128],
        "dropout": 0.15,
        "norm": "batchnorm",
    },
    "mlp_large": {
        "layers": [1024, 512, 256, 128],
        "dropout": 0.2,
        "norm": "layernorm",
    },
    "resnet_tabular": {
        "layers": [256, 256, 256],
        "dropout": 0.1,
        "norm": "batchnorm",
        "residual": True,
    },
    "ft_transformer": {
        "d_model": 64,
        "n_heads": 4,
        "n_layers": 3,
        "dropout": 0.1,
    },
}


class ArchitectureBuilder:
    """Build neural architectures from specs or LLM proposals.

    Usage:
        builder = ArchitectureBuilder()
        
        # From library
        arch = builder.from_library("resnet_tabular", input_dim=64, output_dim=10)
        
        # From LLM
        arch = builder.from_llm(spec, llm_call=my_llm)
        
        # From spec
        arch = builder.from_spec(ArchitectureSpec(...))
    """

    def __init__(self):
        self._library = _ARCH_LIBRARY

    def from_library(self, name: str, input_dim: int, output_dim: int,
                     task: str = "classification") -> BuiltArchitecture:
        """Build from the predefined architecture library."""
        if name not in self._library:
            name = "mlp_medium"  # fallback

        config = self._library[name]
        spec = ArchitectureSpec(
            name=name,
            task=task,
            input_shape=(input_dim,),
            output_shape=(output_dim,),
            hidden_dims=config.get("layers", [256, 128]),
            dropout=config.get("dropout", 0.1),
            normalization=config.get("norm", "batchnorm"),
        )
        return self.from_spec(spec)

    def from_spec(self, spec: ArchitectureSpec) -> BuiltArchitecture:
        """Build architecture from a specification."""
        try:
            import torch
            import torch.nn as nn
        except ImportError:
            return BuiltArchitecture(
                name=spec.name, module=None, n_parameters=0,
                estimated_memory_mb=0, code="",
                validated=False, validation_error="torch not installed",
            )

        input_dim = spec.input_shape[0] if len(spec.input_shape) == 1 else int(np.prod(spec.input_shape))
        output_dim = spec.output_shape[0] if len(spec.output_shape) == 1 else int(np.prod(spec.output_shape))

        if spec.backbone == "transformer":
            module = self._build_transformer(input_dim, output_dim, spec)
        elif spec.backbone == "resnet":
            module = self._build_resnet(input_dim, output_dim, spec)
        else:
            module = self._build_mlp(input_dim, output_dim, spec)

        n_params = sum(p.numel() for p in module.parameters())
        mem_mb = n_params * 4 / (1024 * 1024)  # float32

        # Validate
        validated = True
        validation_error = ""
        try:
            x_test = torch.randn(2, input_dim)
            out = module(x_test)
            assert out.shape == (2, output_dim), f"Expected (2, {output_dim}), got {out.shape}"
        except Exception as e:
            validated = False
            validation_error = str(e)

        return BuiltArchitecture(
            name=spec.name,
            module=module,
            n_parameters=n_params,
            estimated_memory_mb=mem_mb,
            code=f"# Architecture: {spec.name}\n# Params: {n_params:,}",
            validated=validated,
            validation_error=validation_error,
        )

    def from_llm(self, spec: ArchitectureSpec, llm_call: Callable,
                 max_retries: int = 3) -> BuiltArchitecture:
        """Use LLM to propose and build a custom architecture.

        The LLM generates PyTorch code which is validated and compiled.
        """
        system = f"""You are an expert neural architecture designer. Design a PyTorch nn.Module for:
- Task: {spec.task}
- Input shape: {spec.input_shape}
- Output shape: {spec.output_shape}
- Max parameters: {spec.max_params:,}
- Constraints: dropout={spec.dropout}, activation={spec.activation}

Write a COMPLETE Python class that:
1. Inherits from nn.Module
2. Has __init__(self, input_dim, output_dim) 
3. Has forward(self, x) method
4. Uses best practices (residual connections, normalization, proper initialization)

Output ONLY the Python code, no explanation."""

        user_msg = f"Design architecture for {spec.name}: input_dim={spec.input_shape[0]}, output_dim={spec.output_shape[0]}"

        for attempt in range(max_retries):
            try:
                raw, _ = llm_call(system, user_msg)
                code = self._extract_code(raw)
                module = self._compile_architecture(code, spec)
                if module is not None:
                    n_params = sum(p.numel() for p in module.parameters())
                    return BuiltArchitecture(
                        name=f"{spec.name}_llm",
                        module=module,
                        n_parameters=n_params,
                        estimated_memory_mb=n_params * 4 / (1024 * 1024),
                        code=code,
                        validated=True,
                    )
            except Exception as e:
                user_msg = f"Previous attempt failed: {e}. Try again with simpler architecture."

        # Fallback to spec-based
        return self.from_spec(spec)

    def recommend_architecture(self, n_samples: int, n_features: int,
                               n_outputs: int, task: str) -> str:
        """Recommend an architecture name from the library."""
        if n_samples < 500:
            return "mlp_small"
        elif n_samples < 5000:
            if n_features > 50:
                return "ft_transformer"
            return "mlp_medium"
        else:
            if n_features > 100:
                return "ft_transformer"
            return "resnet_tabular"

    # ========================================================================== builders

    def _build_mlp(self, input_dim: int, output_dim: int, spec: ArchitectureSpec):
        import torch.nn as nn

        layers = []
        prev_dim = input_dim
        act_fn = {"relu": nn.ReLU, "gelu": nn.GELU, "silu": nn.SiLU,
                  "elu": nn.ELU}.get(spec.activation, nn.GELU)

        for i, hidden_dim in enumerate(spec.hidden_dims):
            layers.append(nn.Linear(prev_dim, hidden_dim))
            if spec.normalization == "batchnorm":
                layers.append(nn.BatchNorm1d(hidden_dim))
            elif spec.normalization == "layernorm":
                layers.append(nn.LayerNorm(hidden_dim))
            layers.append(act_fn())
            layers.append(nn.Dropout(spec.dropout))
            prev_dim = hidden_dim

        layers.append(nn.Linear(prev_dim, output_dim))
        return nn.Sequential(*layers)

    def _build_resnet(self, input_dim: int, output_dim: int, spec: ArchitectureSpec):
        import torch
        import torch.nn as nn

        class ResBlock(nn.Module):
            def __init__(self, dim, dropout):
                super().__init__()
                self.net = nn.Sequential(
                    nn.Linear(dim, dim),
                    nn.BatchNorm1d(dim),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Linear(dim, dim),
                    nn.BatchNorm1d(dim),
                )
                self.act = nn.GELU()

            def forward(self, x):
                return self.act(x + self.net(x))

        class ResNet(nn.Module):
            def __init__(self, in_dim, out_dim, hidden_dims, dropout):
                super().__init__()
                self.input_proj = nn.Linear(in_dim, hidden_dims[0])
                self.blocks = nn.ModuleList([
                    ResBlock(hidden_dims[0], dropout)
                    for _ in range(len(hidden_dims))
                ])
                self.head = nn.Sequential(
                    nn.LayerNorm(hidden_dims[0]),
                    nn.Linear(hidden_dims[0], out_dim),
                )

            def forward(self, x):
                x = self.input_proj(x)
                for block in self.blocks:
                    x = block(x)
                return self.head(x)

        return ResNet(input_dim, output_dim, spec.hidden_dims, spec.dropout)

    def _build_transformer(self, input_dim: int, output_dim: int, spec: ArchitectureSpec):
        import torch
        import torch.nn as nn

        d_model = min(64, max(16, input_dim // 2))

        class TabTransformer(nn.Module):
            def __init__(self, in_features, out_features, d_model, n_heads=4,
                         n_layers=2, dropout=0.1):
                super().__init__()
                self.embedding = nn.Linear(1, d_model)
                self.pos = nn.Parameter(torch.randn(1, in_features, d_model) * 0.02)
                layer = nn.TransformerEncoderLayer(
                    d_model=d_model, nhead=n_heads,
                    dim_feedforward=d_model * 4,
                    dropout=dropout, batch_first=True,
                )
                self.encoder = nn.TransformerEncoder(layer, num_layers=n_layers)
                self.cls = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
                self.head = nn.Sequential(
                    nn.LayerNorm(d_model),
                    nn.Linear(d_model, out_features),
                )

            def forward(self, x):
                B = x.shape[0]
                x = x.unsqueeze(-1)
                x = self.embedding(x) + self.pos[:, :x.shape[1], :]
                cls = self.cls.expand(B, -1, -1)
                x = torch.cat([cls, x], dim=1)
                x = self.encoder(x)
                return self.head(x[:, 0])

        return TabTransformer(input_dim, output_dim, d_model, dropout=spec.dropout)

    # ========================================================================== code compilation

    def _extract_code(self, raw: str) -> str:
        """Extract Python code from LLM response."""
        # Try code fences first
        match = re.search(r'```(?:python)?\s*\n(.*?)```', raw, re.DOTALL)
        if match:
            return match.group(1).strip()
        # If no fences, try to find class definition
        match = re.search(r'(class \w+\(nn\.Module\).*)', raw, re.DOTALL)
        if match:
            return match.group(1).strip()
        return raw.strip()

    def _compile_architecture(self, code: str, spec: ArchitectureSpec) -> Any:
        """Compile LLM-generated architecture code safely."""
        import torch
        import torch.nn as nn
        import torch.nn.functional as F

        # Validate AST
        try:
            tree = ast.parse(code)
        except SyntaxError:
            return None

        # Find class name
        class_name = None
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                class_name = node.name
                break
        if class_name is None:
            return None

        # Execute in restricted namespace with builtins needed for class definitions
        import builtins as _b
        namespace = {
            "torch": torch,
            "nn": nn,
            "F": F,
            "np": np,
            "__builtins__": {"range": range, "int": int, "float": float,
                            "bool": bool, "str": str,
                            "list": list, "tuple": tuple, "dict": dict, "set": set,
                            "len": len, "min": min, "max": max, "sum": sum,
                            "isinstance": isinstance, "hasattr": hasattr,
                            "callable": callable,
                            "enumerate": enumerate, "sorted": sorted, "zip": zip,
                            "super": super, "type": type,
                            "__build_class__": _b.__build_class__,
                            "ValueError": ValueError, "RuntimeError": RuntimeError,
                            "NotImplementedError": NotImplementedError,
                            "print": lambda *a: None},
        }

        try:
            exec(code, namespace)
            cls = namespace[class_name]
            input_dim = spec.input_shape[0]
            output_dim = spec.output_shape[0]
            module = cls(input_dim, output_dim)

            # Validate forward pass
            x = torch.randn(2, input_dim)
            out = module(x)
            if out.shape != (2, output_dim):
                return None

            # Check parameter count
            n_params = sum(p.numel() for p in module.parameters())
            if n_params > spec.max_params:
                return None

            return module
        except Exception:
            return None
