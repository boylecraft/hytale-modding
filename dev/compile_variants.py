#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple, Union

Json = Union[Dict[str, Any], List[Any], str, int, float, bool, None]


# ----------------------------
# JSON Pointer / JSON Patch
# ----------------------------

def _unescape_json_pointer(token: str) -> str:
    # RFC 6901: ~1 => /, ~0 => ~
    return token.replace("~1", "/").replace("~0", "~")


def _split_pointer(ptr: str) -> List[str]:
    if ptr == "":
        return []
    if not ptr.startswith("/"):
        raise ValueError(f"Invalid JSON Pointer (must start with '/'): {ptr}")
    parts = ptr.split("/")[1:]
    return [_unescape_json_pointer(p) for p in parts]


def _resolve_parent(doc: Json, ptr: str) -> Tuple[Json, str]:
    """
    Returns (parent, last_token) for a pointer.
    Example: ptr '/a/b/0' returns (doc['a']['b'], '0')
    """
    parts = _split_pointer(ptr)
    if not parts:
        raise ValueError("Pointer refers to the document root; no parent exists.")
    parent_parts = parts[:-1]
    last = parts[-1]

    cur: Json = doc
    for t in parent_parts:
        if isinstance(cur, dict):
            if t not in cur:
                raise KeyError(f"Path token '{t}' not found while resolving '{ptr}'")
            cur = cur[t]
        elif isinstance(cur, list):
            if t == "-":
                raise KeyError(f"'-' is not valid in the middle of a pointer: '{ptr}'")
            idx = int(t)
            cur = cur[idx]
        else:
            raise KeyError(f"Cannot traverse into non-container at token '{t}' for '{ptr}'")
    return cur, last


def _get(doc: Json, ptr: str) -> Json:
    parts = _split_pointer(ptr)
    cur: Json = doc
    for t in parts:
        if isinstance(cur, dict):
            if t not in cur:
                raise KeyError(f"Token '{t}' not found while getting '{ptr}'")
            cur = cur[t]
        elif isinstance(cur, list):
            if t == "-":
                raise KeyError(f"'-' not valid for get: '{ptr}'")
            cur = cur[int(t)]
        else:
            raise KeyError(f"Cannot traverse into non-container for '{ptr}'")
    return cur


def apply_json_patch(doc: Json, ops: List[Dict[str, Any]], *, strict_paths: bool = True) -> Json:
    """
    Minimal JSON Patch (RFC 6902) support for: replace, add, remove.
    - strict_paths=True: missing path => error (recommended for catching drift)
    """
    for op in ops:
        operation = op.get("op")
        path = op.get("path")
        if operation not in ("replace", "add", "remove"):
            raise ValueError(f"Unsupported op '{operation}'. Supported: replace/add/remove")
        if not isinstance(path, str):
            raise ValueError(f"Patch op missing valid 'path': {op}")

        if operation == "remove":
            parent, token = _resolve_parent(doc, path)
            if isinstance(parent, dict):
                if strict_paths and token not in parent:
                    raise KeyError(f"remove failed; key '{token}' missing at '{path}'")
                parent.pop(token, None)
            elif isinstance(parent, list):
                if token == "-":
                    raise ValueError(f"remove does not support '-' index: '{path}'")
                idx = int(token)
                if strict_paths and not (0 <= idx < len(parent)):
                    raise IndexError(f"remove failed; index {idx} out of range at '{path}'")
                if 0 <= idx < len(parent):
                    parent.pop(idx)
            else:
                raise TypeError(f"remove failed; parent is not container at '{path}'")

        elif operation == "replace":
            if "value" not in op:
                raise ValueError(f"replace missing 'value': {op}")
            value = op["value"]
            parent, token = _resolve_parent(doc, path)

            if isinstance(parent, dict):
                if strict_paths and token not in parent:
                    raise KeyError(f"replace failed; key '{token}' missing at '{path}'")
                parent[token] = value
            elif isinstance(parent, list):
                if token == "-":
                    raise ValueError(f"replace does not support '-' index: '{path}'")
                idx = int(token)
                if strict_paths and not (0 <= idx < len(parent)):
                    raise IndexError(f"replace failed; index {idx} out of range at '{path}'")
                parent[idx] = value
            else:
                raise TypeError(f"replace failed; parent is not container at '{path}'")

        elif operation == "add":
            if "value" not in op:
                raise ValueError(f"add missing 'value': {op}")
            value = op["value"]
            parent, token = _resolve_parent(doc, path)

            if isinstance(parent, dict):
                # add allows creating new key
                parent[token] = value
            elif isinstance(parent, list):
                if token == "-":
                    parent.append(value)
                else:
                    idx = int(token)
                    if strict_paths and not (0 <= idx <= len(parent)):
                        raise IndexError(f"add failed; index {idx} out of range at '{path}'")
                    parent.insert(idx, value)
            else:
                raise TypeError(f"add failed; parent is not container at '{path}'")

    return doc


# ----------------------------
# Config model (lightweight)
# ----------------------------

@dataclass
class Sources:
    model_base: str
    role_base: str


@dataclass
class Outputs:
    model_dir: str
    role_dir: str


@dataclass
class Variant:
    id: str
    model_ops: List[Dict[str, Any]]
    role_ops: List[Dict[str, Any]]


@dataclass
class Config:
    server_version: str
    sources: Sources
    outputs: Outputs
    variants: List[Variant]


def load_config(path: Path) -> Config:
    raw = json.loads(path.read_text(encoding="utf-8"))

    server_version = raw.get("server_version")
    if not isinstance(server_version, str) or not server_version:
        raise ValueError("config missing non-empty 'server_version'")

    outputs_raw = raw.get("outputs") or {}
    outputs = Outputs(
        model_dir=_req_str(outputs_raw, "model_dir"),
        role_dir=_req_str(outputs_raw, "role_dir"),
    )

    sources_raw = raw.get("sources") or {}
    model_raw = sources_raw.get("model") or {}
    role_raw = sources_raw.get("role") or {}
    sources = Sources(
        model_base=_req_str(model_raw, "base"),
        role_base=_req_str(role_raw, "base"),
    )

    variants_raw = raw.get("variants")
    if not isinstance(variants_raw, list) or not variants_raw:
        raise ValueError("config 'variants' must be a non-empty array")

    variants: List[Variant] = []
    for v in variants_raw:
        if not isinstance(v, dict):
            raise ValueError(f"variant must be object, got: {v!r}")
        vid = v.get("id")
        if not isinstance(vid, str) or not vid:
            raise ValueError(f"variant missing non-empty 'id': {v}")

        patches = v.get("patches") or {}
        model_ops = patches.get("model") or []
        role_ops = patches.get("role") or []
        if not isinstance(model_ops, list) or not isinstance(role_ops, list):
            raise ValueError(f"variant patches must be arrays: {vid}")

        variants.append(Variant(id=vid, model_ops=model_ops, role_ops=role_ops))

    return Config(server_version=server_version, sources=sources, outputs=outputs, variants=variants)


def _req_str(obj: Dict[str, Any], key: str) -> str:
    val = obj.get(key)
    if not isinstance(val, str) or not val:
        raise ValueError(f"missing non-empty string '{key}'")
    return val


# ----------------------------
# Compiler
# ----------------------------

def read_json(path: Path) -> Json:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, data: Json) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def compile_variants(
        *,
        config_path: Path,
        versions_root: Path,
        mod_root: Path,
        strict_paths: bool = True,
) -> None:
    cfg = load_config(config_path)

    # Base assets root is the Hytale "Assets" directory for the given server version.
    base_assets_root = versions_root / cfg.server_version / "Assets"
    if not base_assets_root.exists():
        raise FileNotFoundError(
            f"Base assets root not found: {base_assets_root}\n"
            f"Expected layout: <versions_root>/<server_version>/Assets/Server/..."
        )

    # Load base docs from the base assets (NOT from the mod)
    base_model_path = base_assets_root / cfg.sources.model_base
    base_role_path = base_assets_root / cfg.sources.role_base

    if not base_model_path.exists():
        raise FileNotFoundError(f"Base model json not found: {base_model_path}")
    if not base_role_path.exists():
        raise FileNotFoundError(f"Base role json not found: {base_role_path}")

    base_model_doc = read_json(base_model_path)
    base_role_doc = read_json(base_role_path)

    # Output directories are inside the mod root
    out_model_dir = mod_root / cfg.outputs.model_dir
    out_role_dir = mod_root / cfg.outputs.role_dir

    for v in cfg.variants:
        # Clone docs so each variant starts from the same base
        model_doc = json.loads(json.dumps(base_model_doc))
        role_doc = json.loads(json.dumps(base_role_doc))

        # Apply patches
        apply_json_patch(model_doc, v.model_ops, strict_paths=strict_paths)
        apply_json_patch(role_doc, v.role_ops, strict_paths=strict_paths)

        # Write generated assets
        write_json(out_model_dir / f"{v.id}.json", model_doc)
        write_json(out_role_dir / f"{v.id}.json", role_doc)

        print(f"✅ Generated: {cfg.outputs.model_dir}/{v.id}.json")
        print(f"✅ Generated: {cfg.outputs.role_dir}/{v.id}.json")


def main() -> None:
    parser = argparse.ArgumentParser(description="Compile Hytale NPC texture variants into full mod assets.")
    parser.add_argument("--config", required=True, type=Path, help="Path to variants.json")
    parser.add_argument("--mod-root", required=True, type=Path,
                        help="Path to mod root (contains Server/, Common/, manifest.json)")
    parser.add_argument(
        "--versions-root",
        type=Path,
        default=Path(os.environ.get("HYTALE_VERSIONS_ROOT", "")) if os.environ.get("HYTALE_VERSIONS_ROOT") else None,
        help="Root folder that contains <server_version>/Assets/... (or set HYTALE_VERSIONS_ROOT)",
    )
    parser.add_argument(
        "--no-strict",
        action="store_true",
        help="If set, missing patch paths won't hard-fail (not recommended; hides upstream drift).",
    )
    args = parser.parse_args()

    if args.versions_root is None:
        raise SystemExit("ERROR: provide --versions-root or set env var HYTALE_VERSIONS_ROOT")

    compile_variants(
        config_path=args.config,
        versions_root=args.versions_root,
        mod_root=args.mod_root,
        strict_paths=not args.no_strict,
    )


if __name__ == "__main__":
    main()
