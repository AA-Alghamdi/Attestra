"""The A8 static safety gate: a STRICT AST allowlist/denylist over generated connector code. This is
the primary defense before any execution. It rejects anything that could reach the OS, network,
filesystem, the import system, or the interpreter internals -- so only pure data-transformation code
(string/list/dict ops + json/re/csv parsing) ever runs.
"""
import ast
from dataclasses import dataclass, field

# Modules a data connector legitimately needs to parse text. Everything else is rejected.
ALLOWED_IMPORTS = {"json", "re", "csv", "io", "math", "datetime"}

# Builtins/identifiers that must never appear (code execution, FS, import, introspection escapes).
BANNED_NAMES = {
    "eval", "exec", "compile", "open", "__import__", "input", "breakpoint",
    "globals", "locals", "vars", "getattr", "setattr", "delattr", "hasattr",
    "memoryview", "help", "exit", "quit", "object", "type", "super", "classmethod",
    "staticmethod", "property",
}
# Attribute names that are introspection/escape vectors.
BANNED_ATTRS = {
    "__globals__", "__class__", "__bases__", "__base__", "__subclasses__", "__mro__",
    "__builtins__", "__dict__", "__code__", "__closure__", "__func__", "__self__",
    "__import__", "__getattribute__", "__reduce__", "__reduce_ex__", "__module__",
}


@dataclass
class StaticReport:
    ok: bool
    violations: list = field(default_factory=list)
    has_entrypoint: bool = False

    def as_dict(self):
        return {"ok": self.ok, "violations": list(self.violations),
                "has_entrypoint": self.has_entrypoint}


def static_check(code: str, *, entrypoint: str = "load") -> StaticReport:
    v = []
    try:
        tree = ast.parse(code)
    except SyntaxError as ex:
        return StaticReport(ok=False, violations=[f"syntax error: {ex}"])

    has_entry = False
    for node in ast.walk(tree):
        # imports
        if isinstance(node, ast.Import):
            for a in node.names:
                root = a.name.split(".")[0]
                if root not in ALLOWED_IMPORTS:
                    v.append(f"import of {a.name!r} not allowed (allowlist: {sorted(ALLOWED_IMPORTS)})")
        elif isinstance(node, ast.ImportFrom):
            root = (node.module or "").split(".")[0]
            if root not in ALLOWED_IMPORTS:
                v.append(f"import from {node.module!r} not allowed")
        # banned names
        elif isinstance(node, ast.Name):
            if node.id in BANNED_NAMES:
                v.append(f"use of banned name {node.id!r}")
        # banned / dunder attributes
        elif isinstance(node, ast.Attribute):
            if node.attr in BANNED_ATTRS or (node.attr.startswith("__") and node.attr.endswith("__")):
                v.append(f"access to forbidden attribute {node.attr!r}")
        # entrypoint presence + arity
        elif isinstance(node, ast.FunctionDef):
            if node.name == entrypoint:
                has_entry = True
                n_args = len(node.args.args)
                if n_args != 1:
                    v.append(f"entrypoint {entrypoint!r} must take exactly 1 argument, has {n_args}")

    if not has_entry:
        v.append(f"no entrypoint function named {entrypoint!r}")

    return StaticReport(ok=(len(v) == 0), violations=v, has_entrypoint=has_entry)
