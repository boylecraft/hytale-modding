#!/usr/bin/env python3
from __future__ import annotations

from PIL import Image, ImageEnhance
import numpy as np
import argparse
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple, Union, Optional

Json = Union[Dict[str, Any], List[Any], str, int, float, bool, None]

# ----------------------------
# Image editing helpers
# ----------------------------

_BAYER8 = np.array([
    [0, 48, 12, 60, 3, 51, 15, 63],
    [32, 16, 44, 28, 35, 19, 47, 31],
    [8, 56, 4, 52, 11, 59, 7, 55],
    [40, 24, 36, 20, 43, 27, 39, 23],
    [2, 50, 14, 62, 1, 49, 13, 61],
    [34, 18, 46, 30, 33, 17, 45, 29],
    [10, 58, 6, 54, 9, 57, 5, 53],
    [42, 26, 38, 22, 41, 25, 37, 21],
], dtype=np.float32)


def _mask_to_float01(mask_img: Image.Image, *, channel: str) -> np.ndarray:
    """
    Returns HxW float mask in [0,1].
    channel: 'luma' | 'r' | 'g' | 'b' | 'a'
    """
    m = mask_img.convert("RGBA")
    arr = np.asarray(m).astype(np.float32) / 255.0  # H,W,4

    ch = channel.lower().strip()
    if ch == "r":
        return arr[..., 0]
    if ch == "g":
        return arr[..., 1]
    if ch == "b":
        return arr[..., 2]
    if ch == "a":
        return arr[..., 3]
    if ch == "luma":
        # Rec.709-ish luminance
        return 0.2126 * arr[..., 0] + 0.7152 * arr[..., 1] + 0.0722 * arr[..., 2]

    raise ValueError(f"mask.channel must be one of luma/r/g/b/a, got '{channel}'")


def _resolve_engine_path_to_file(
        base_assets_root: Path,
        mod_root: Path,
        engine_path: str,
) -> Path:
    """
    Prefer mask files in the mod (Common/...) if present, else fall back to base assets.
    engine_path like 'NPC/.../Masks/Foo.png'
    """
    mod_file = _npc_path_to_mod_common_any(mod_root, engine_path)
    if mod_file.exists():
        return mod_file
    return _npc_path_to_base_common_any(base_assets_root, engine_path)


# def _load_mask01(
#     *,
#     base_assets_root: Path,
#     mod_root: Path,
#     mask_spec: Dict[str, Any],
#     target_size: tuple[int, int],
#     texture_engine_path: str,
# ) -> np.ndarray:
#     """
#     Returns float mask in [0..1], shape (H,W).
#     White=1 apply effect, Black=0 protect. Optional invert.
#     """
#     path = mask_spec.get("path")
#     if not isinstance(path, str) or not path:
#         raise ValueError("mask.path must be a non-empty string")
#
#     invert = bool(mask_spec.get("invert", False))
#     channel = str(mask_spec.get("channel", "luma")).lower()  # "luma" or "alpha"
#
#     file_path = _resolve_engine_path_to_file(base_assets_root, mod_root, path)
#     if not file_path.exists():
#         raise FileNotFoundError(f"Mask not found: {file_path} (from '{path}')")
#
#     m = Image.open(file_path).convert("RGBA")
#
#     # match current working image size
#     if m.size != target_size:
#         m = m.resize(target_size, resample=Image.Resampling.NEAREST)
#
#     arr = np.asarray(m).astype(np.float32) / 255.0  # (H,W,4)
#
#     if channel == "alpha":
#         mask = arr[..., 3]
#     else:
#         # luma from RGB
#         mask = 0.2126 * arr[..., 0] + 0.7152 * arr[..., 1] + 0.0722 * arr[..., 2]
#
#     if invert:
#         mask = 1.0 - mask
#
#     return np.clip(mask, 0.0, 1.0)


def _apply_desaturate_masked(img: Image.Image, amount: float, mask01: np.ndarray | None) -> Image.Image:
    amount = float(amount)
    if amount <= 0:
        return img

    base = img.convert("RGBA")
    arr = np.asarray(base).astype(np.float32) / 255.0  # (H,W,4)

    # grayscale luma
    gray = (0.2126 * arr[..., 0] + 0.7152 * arr[..., 1] + 0.0722 * arr[..., 2])
    gray_rgb = np.stack([gray, gray, gray], axis=-1)

    if mask01 is None:
        w = np.clip(amount, 0.0, 1.0)
        blend = w
    else:
        blend = np.clip(amount, 0.0, 1.0) * mask01  # (H,W)

    # blend per pixel
    arr[..., 0:3] = arr[..., 0:3] * (1.0 - blend[..., None]) + gray_rgb * (blend[..., None])

    out = np.clip(arr * 255.0, 0, 255).astype(np.uint8)
    return Image.fromarray(out, mode="RGBA")


def _apply_tint_rgba_masked(
        img: Image.Image,
        *,
        rgba: tuple[float, float, float, float],
        amount: float = 1.0,
        mask01: np.ndarray | None,
) -> Image.Image:
    amount = float(amount)
    if amount <= 0:
        return img

    base = img.convert("RGBA")
    arr = np.asarray(base).astype(np.float32) / 255.0

    r, g, b, a = rgba
    tinted = arr.copy()
    tinted[..., 0] *= r
    tinted[..., 1] *= g
    tinted[..., 2] *= b
    tinted[..., 3] *= a

    if mask01 is None:
        w = np.clip(amount, 0.0, 1.0)
        blend = w
    else:
        blend = np.clip(amount, 0.0, 1.0) * mask01

    arr = arr * (1.0 - blend[..., None]) + tinted * (blend[..., None])

    out = np.clip(arr * 255.0, 0, 255).astype(np.uint8)
    return Image.fromarray(out, mode="RGBA")


def _load_mask_map(
        *,
        base_assets_root: Path,
        mod_root: Path,
        mask_spec: Dict[str, Any],
        target_size: tuple[int, int],
) -> np.ndarray:
    """
    Loads mask from engine path and returns HxW float in [0,1], resized to target_size.
    """
    path = mask_spec.get("path")
    if not isinstance(path, str) or not path:
        raise ValueError("opacity.mask.path must be a non-empty string engine path")

    channel = str(mask_spec.get("channel", "luma"))
    invert = bool(mask_spec.get("invert", False))

    # Prefer mod-root mask first (lets you ship masks), else fall back to base assets
    mod_file = _npc_path_to_mod_common_any(mod_root, path)
    base_file = _npc_path_to_base_common_any(base_assets_root, path)

    if mod_file.exists():
        f = mod_file
    elif base_file.exists():
        f = base_file
    else:
        raise FileNotFoundError(f"Mask image not found in mod or base assets: {path}")

    m = Image.open(f)
    # IMPORTANT: resize mask to match the already-transformed texture size
    if m.size != target_size:
        # nearest keeps crisp regions; change to bilinear if you prefer smoother edges
        m = m.resize(target_size, resample=Image.Resampling.NEAREST)

    mm = _mask_to_float01(m, channel=channel)
    if invert:
        mm = 1.0 - mm
    return np.clip(mm, 0.0, 1.0)


def _npc_path_to_base_common_any(base_assets_root: Path, engine_path: str) -> Path:
    """Maps engine path like 'NPC/.../Models/Model.blockymodel' to '<assets>/Common/NPC/.../Models/Model.blockymodel'"""
    if engine_path.startswith("/"):
        engine_path = engine_path[1:]
    return base_assets_root / "Common" / Path(engine_path)


def _npc_path_to_mod_common_any(mod_root: Path, engine_path: str) -> Path:
    """Maps engine path like 'NPC/.../Models/Texture.png' to '<mod>/Common/NPC/.../Models/Texture.png'"""
    if engine_path.startswith("/"):
        engine_path = engine_path[1:]
    return mod_root / "Common" / Path(engine_path)


def _scale_blockymodel_for_texture_resize(doc: Json, scale: int) -> Json:
    """
    When the texture image is resized by `scale`, adjust the blockymodel so
    UVs still cover the same *logical* regions:
      - textureLayout offsets *= scale
      - settings.size *= scale   (UV rect sizes grow)
      - stretch /= scale         (geometry stays same size)
    """
    if scale <= 1:
        return doc

    def scale_stretch_axis(v: float) -> float:
        # preserve sign for mirrored stretches (-1 etc)
        return v / scale

    def walk(x: Json) -> None:
        if isinstance(x, dict):
            # 1) scale offsets in textureLayout
            tl = x.get("textureLayout")
            if isinstance(tl, dict):
                for face in tl.values():
                    if isinstance(face, dict):
                        off = face.get("offset")
                        if isinstance(off, dict):
                            if isinstance(off.get("x"), (int, float)):
                                off["x"] = off["x"] * scale
                            if isinstance(off.get("y"), (int, float)):
                                off["y"] = off["y"] * scale

            # 2) scale UV rectangle sizes via settings.size, and neutralize via stretch
            # This lives under a "shape" object typically.
            if x.get("type") in ("box", "quad"):
                settings = x.get("settings")
                stretch = x.get("stretch")

                if isinstance(settings, dict) and isinstance(settings.get("size"), dict):
                    sz = settings["size"]

                    if x["type"] == "box":
                        # scale size x/y/z
                        for k in ("x", "y", "z"):
                            if isinstance(sz.get(k), (int, float)):
                                sz[k] = sz[k] * scale

                        # divide stretch x/y/z
                        if isinstance(stretch, dict):
                            for k in ("x", "y", "z"):
                                if isinstance(stretch.get(k), (int, float)):
                                    stretch[k] = scale_stretch_axis(float(stretch[k]))

                    elif x["type"] == "quad":
                        # quads have size.x, size.y only
                        for k in ("x", "y"):
                            if isinstance(sz.get(k), (int, float)):
                                sz[k] = sz[k] * scale

                        # divide stretch x/y so geometry stays same
                        if isinstance(stretch, dict):
                            for k in ("x", "y"):
                                if isinstance(stretch.get(k), (int, float)):
                                    stretch[k] = scale_stretch_axis(float(stretch[k]))
                        # leave stretch.z alone (often irrelevant / used differently)

            # recurse
            for v in x.values():
                walk(v)

        elif isinstance(x, list):
            for v in x:
                walk(v)

    walk(doc)
    return doc


def _scale_blockymodel_uv_offsets(doc: Json, scale: int) -> Json:
    """
    Multiply every textureLayout.*.offset.x/y by scale.
    Works for both box faces and quad 'front' layouts.
    """
    if scale <= 1:
        return doc

    def walk(x: Json) -> None:
        if isinstance(x, dict):
            # textureLayout node
            tl = x.get("textureLayout")
            if isinstance(tl, dict):
                for face in tl.values():
                    if isinstance(face, dict):
                        off = face.get("offset")
                        if isinstance(off, dict):
                            if "x" in off and isinstance(off["x"], (int, float)):
                                off["x"] = off["x"] * scale
                            if "y" in off and isinstance(off["y"], (int, float)):
                                off["y"] = off["y"] * scale

            for v in x.values():
                walk(v)

        elif isinstance(x, list):
            for v in x:
                walk(v)

    walk(doc)
    return doc


def _generate_scaled_blockymodel_from_base_appearance(
        *,
        base_assets_root: Path,
        mod_root: Path,
        base_appearance_doc: Json,  # <-- THIS is read from cfg.sources.model.base
        out_texture_engine_path: str,  # e.g. NPC/.../Bobby/Models/Texture.png
        scale: int,
) -> str:
    """
    Uses the base appearance JSON's "Model" field to locate the source .blockymodel,
    scales its textureLayout offsets by `scale`, and writes the result next to the
    generated texture. Returns the engine path to the written .blockymodel.
    """
    if scale <= 1:
        raise ValueError("scale must be > 1")

    if not isinstance(base_appearance_doc, dict):
        raise ValueError("base_appearance_doc must be an object")

    base_blocky_engine = base_appearance_doc.get("Model")
    if not isinstance(base_blocky_engine, str) or not base_blocky_engine:
        raise ValueError('Base appearance JSON missing non-empty "Model" path')

    # Load source blockymodel from base assets
    base_blocky_file = _npc_path_to_base_common_any(base_assets_root, base_blocky_engine)
    if not base_blocky_file.exists():
        raise FileNotFoundError(f"Base blockymodel not found: {base_blocky_file} (from '{base_blocky_engine}')")

    blocky_doc = json.loads(base_blocky_file.read_text(encoding="utf-8"))
    blocky_doc = _scale_blockymodel_for_texture_resize(blocky_doc, scale)

    # Output blockymodel path: same folder as output texture
    out_tex_p = Path(out_texture_engine_path)
    out_blocky_engine = str(out_tex_p.parent / "Model.blockymodel").replace("\\", "/")

    out_blocky_file = _npc_path_to_mod_common_any(mod_root, out_blocky_engine)
    _ensure_parent_dir(out_blocky_file)
    out_blocky_file.write_text(json.dumps(blocky_doc, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    return out_blocky_engine


def _apply_resize(img: Image.Image, *, scale: int, resample: str) -> Image.Image:
    if scale <= 1:
        return img

    resample_map = {
        "nearest": Image.Resampling.NEAREST,
        "bilinear": Image.Resampling.BILINEAR,
        "bicubic": Image.Resampling.BICUBIC,
        "lanczos": Image.Resampling.LANCZOS,
    }
    if resample not in resample_map:
        raise ValueError(f"resize.resample must be one of {list(resample_map.keys())}, got '{resample}'")

    w, h = img.size
    return img.resize((w * scale, h * scale), resample=resample_map[resample])


def _blue_noise_tile(size: int, seed: int) -> np.ndarray:
    """
    Returns a (size,size) threshold tile in [0,1), "blue-noise-ish".
    Deterministic given (size, seed).
    """
    if size <= 1:
        return np.array([[0.0]], dtype=np.float32)

    rng = np.random.default_rng(seed)
    noise = rng.random((size, size)).astype(np.float32)

    # High-pass filter in frequency domain to reduce low-frequency clumps
    f = np.fft.fft2(noise)
    fy = np.fft.fftfreq(size).reshape(-1, 1)
    fx = np.fft.fftfreq(size).reshape(1, -1)
    r2 = fx * fx + fy * fy

    # Smooth high-pass: suppress low frequencies
    # k controls cutoff-ish. This is a heuristic; tweak if needed.
    k = 6.0
    hp = 1.0 - np.exp(-k * r2)

    f_hp = f * hp
    filtered = np.fft.ifft2(f_hp).real.astype(np.float32)

    # Normalize to [0,1)
    mn = float(filtered.min())
    mx = float(filtered.max())
    if mx - mn < 1e-8:
        return np.zeros((size, size), dtype=np.float32)
    t = (filtered - mn) / (mx - mn)
    # Avoid exact 1.0
    t = np.clip(t, 0.0, np.nextafter(1.0, 0.0)).astype(np.float32)
    return t


def _blue_noise_threshold_map(h: int, w: int, *, tile: int = 64, seed: int = 0) -> np.ndarray:
    tile = int(tile)
    if tile <= 0:
        tile = 64
    t = _blue_noise_tile(tile, seed)
    return np.tile(t, (int(np.ceil(h / tile)), int(np.ceil(w / tile))))[:h, :w]


def _bayer_threshold_map(h: int, w: int, *, size: int = 8) -> np.ndarray:
    if size != 8:
        raise ValueError("Only bayer8 implemented here")
    # (m + 0.5) / 64 gives a nicer distribution (avoids threshold=0 exact)
    mat = (_BAYER8 + 0.5) / 64.0
    return np.tile(mat, (int(np.ceil(h / 8)), int(np.ceil(w / 8))))[:h, :w]


def _clamp01(x: float) -> float:
    return 0.0 if x < 0.0 else (1.0 if x > 1.0 else x)


def _bayer_matrix(n: int) -> np.ndarray:
    """
    Returns NxN Bayer threshold matrix normalized to [0,1).
    n must be power of 2 (e.g., 2,4,8,16).
    """
    if n == 1:
        return np.array([[0]], dtype=np.float32)
    if n & (n - 1) != 0:
        raise ValueError("Bayer matrix size must be a power of 2")
    # recursive construction
    prev = _bayer_matrix(n // 2)
    a = prev * 4 + 0
    b = prev * 4 + 2
    c = prev * 4 + 3
    d = prev * 4 + 1
    top = np.concatenate([a, b], axis=1)
    bot = np.concatenate([c, d], axis=1)
    m = np.concatenate([top, bot], axis=0).astype(np.float32)
    return m / (n * n)


def _apply_opacity_dither(
        img: Image.Image,
        *,
        amount: float,
        pattern: str = "bayer8",
        preserve_holes: bool = True,
        seed: int = 0,
        tile: int = 64,
        mask_map: Optional[np.ndarray] = None,
        mask_threshold: float = 0.5,
) -> Image.Image:
    """
    Converts intended opacity into cutout alpha via dithering.
    Output alpha will be only 0 or 255.

    - amount in [0..1] is a global coverage multiplier.
    - preserve_holes=True keeps originally transparent pixels (alpha==0) as holes.
    - pattern: "bayer8", "bayer16", "blue_noise"
    - seed/tile used for blue_noise
    """
    base = img.convert("RGBA")
    arr = np.asarray(base).astype(np.uint8)

    pat = pattern.lower().strip()

    if pat == "blue_noise":
        a = arr[..., 3].astype(np.float32) / 255.0
        amount = _clamp01(float(amount))
        coverage = np.clip(a * amount, 0.0, 1.0)
        thresh = _blue_noise_threshold_map(arr.shape[0], arr.shape[1], tile=tile, seed=seed)
        keep = coverage > thresh
        if preserve_holes:
            keep &= (a > 0.0)
    elif pat.startswith("bayer"):
        h, w = arr.shape[:2]
        th = _bayer_threshold_map(h, w, size=8)  # 0..1
        coverage = np.clip(float(amount), 0.0, 1.0)
        keep = th < coverage
        if preserve_holes:
            keep &= (arr[..., 3] > 0)
    else:
        raise ValueError(f"Unsupported dither pattern: {pattern!r} (use bayer4/bayer8/bayer16 or blue_noise)")

    if mask_map is not None:
        region = mask_map >= float(mask_threshold)

        # outside region: keep pixel opaque (255) unless it was already a hole and preserve_holes
        keep_outside = np.ones_like(keep, dtype=bool)
        if preserve_holes:
            keep_outside = (arr[..., 3] > 0)

        keep = np.where(region, keep, keep_outside)

    arr[..., 3] = np.where(keep, 255, 0).astype(np.uint8)

    out = arr.copy()
    out[..., 3] = np.where(keep, 255, 0).astype(np.uint8)
    return Image.fromarray(out, mode="RGBA")


def _apply_opacity_dither_rgba_worked(img: Image.Image, *, amount: float, preserve_holes: bool = True) -> Image.Image:
    base = img.convert("RGBA")
    arr = np.asarray(base).astype(np.uint8)
    h, w = arr.shape[:2]

    th = _bayer_threshold_map(h, w, size=8)  # 0..1

    alph = arr[..., 3].astype(np.float32) / 255.0
    # coverage = np.clip(alph * float(amount), 0.0, 1.0)
    coverage = np.clip(float(amount), 0.0, 1.0)

    keep = th < coverage

    if preserve_holes:
        keep &= (arr[..., 3] > 0)

    arr[..., 3] = np.where(keep, 255, 0).astype(np.uint8)
    return Image.fromarray(arr, "RGBA")


def _apply_opacity_dither_broke(
        img: Image.Image,
        *,
        amount: float,
        pattern: str = "bayer8",
        preserve_holes: bool = True,
) -> Image.Image:
    """
    Converts intended opacity into cutout alpha via ordered dithering.
    Output alpha will be only 0 or 255.

    - amount in [0..1] is a global coverage multiplier.
    - preserve_holes=True keeps originally transparent pixels (alpha==0) as holes.
    """
    amount = _clamp01(float(amount))
    if amount >= 1.0:
        # no change; but still ensure RGBA for consistency
        return img.convert("RGBA")

    base = img.convert("RGBA")
    arr = np.asarray(base).astype(np.uint8)  # H,W,4
    a = arr[..., 3].astype(np.float32) / 255.0  # existing alpha (holes/cutouts)

    # Combine existing alpha with global desired opacity
    coverage = a * amount  # per-pixel desired coverage in [0..1]

    # Optional: if you want to *not* reduce opaque pixels unless they're already holes:
    # you can use coverage = np.where(a > 0, amount, 0) instead.
    # Current choice preserves fine alpha detail from your pipeline (pre-dither).

    # Build threshold pattern
    pat = pattern.lower().strip()
    if pat.startswith("bayer"):
        # allow "bayer8", "bayer4", etc.
        size = int(pat.replace("bayer", ""))
        mat = _bayer_matrix(size)  # size x size in [0,1)
        h, w = coverage.shape
        # tile to image size
        tiled = np.tile(mat, (int(np.ceil(h / size)), int(np.ceil(w / size))))[:h, :w]
        thresh = tiled
    else:
        raise ValueError(f"Unsupported dither pattern: {pattern!r} (use bayer4/bayer8/bayer16)")

    # Dither: keep pixel if coverage > threshold
    keep = coverage > thresh

    if preserve_holes:
        keep = keep & (a > 0.0)

    out = arr.copy()
    out[..., 3] = np.where(keep, 255, 0).astype(np.uint8)
    return Image.fromarray(out, mode="RGBA")


def _ensure_parent_dir(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def _npc_path_to_common_file(mod_root: Path, npc_path: str) -> Path:
    """
    Converts an engine-style NPC path like:
      "NPC/Livestock/Chicken/Models/Texture.png"
    into a real file path under the mod:
      "<mod_root>/Common/NPC/Livestock/Chicken/Models/Texture.png"
    """
    if npc_path.startswith("/"):
        npc_path = npc_path[1:]
    return mod_root / "Common" / Path(npc_path)


def _npc_path_to_base_common_file(base_assets_root: Path, npc_path: str) -> Path:
    """
    Base assets path equivalent:
      "<base_assets_root>/Common/NPC/..."
    """
    if npc_path.startswith("/"):
        npc_path = npc_path[1:]
    return base_assets_root / "Common" / Path(npc_path)


def _apply_desaturate(img: Image.Image, amount: float) -> Image.Image:
    """
    amount=0 -> no change
    amount=1 -> fully desaturated (grayscale but kept in RGB)
    """
    amount = float(amount)
    if amount <= 0:
        return img
    if amount >= 1:
        return img.convert("L").convert("RGBA") if img.mode in ("RGBA", "LA") else img.convert("L").convert("RGB")
    # partial: use Color enhancer (saturation)
    # Pillow uses "Color" enhancer where factor=1 is original, factor=0 is grayscale.
    enhancer = ImageEnhance.Color(img)
    factor = 1.0 - amount
    return enhancer.enhance(factor)


def _apply_tint_rgba(img: Image.Image, *, rgba: tuple[float, float, float, float], amount: float = 1.0) -> Image.Image:
    """
    Multiplies RGB (and optionally alpha) by rgba factors in [0..1].
    amount blends between original and tinted: 0=original, 1=tinted.
    """
    amount = float(amount)
    if amount <= 0:
        return img

    # Always work in RGBA
    base = img.convert("RGBA")
    arr = np.asarray(base).astype(np.float32) / 255.0  # shape (H,W,4)

    r, g, b, a = rgba
    tinted = arr.copy()
    tinted[..., 0] *= r
    tinted[..., 1] *= g
    tinted[..., 2] *= b
    tinted[..., 3] *= a

    if amount < 1.0:
        tinted = arr * (1.0 - amount) + tinted * amount

    tinted = np.clip(tinted * 255.0, 0, 255).astype(np.uint8)
    return Image.fromarray(tinted, mode="RGBA")


def _generate_texture(
        *,
        base_assets_root: Path,
        mod_root: Path,
        base_model_doc: Json,
        spec: Dict[str, Any],
) -> tuple[str, str | None]:
    """
    Returns the engine texture path string that should be written into /Texture,
    and writes the generated file into mod_root/Common/...
    """
    scale_used = 1

    from_spec = spec.get("from")
    out_npc_path = spec.get("out")
    # transform = spec.get("transform") or {}

    xform = spec.get("transform") or []
    steps: list[dict[str, Any]]

    if isinstance(xform, list):
        steps = xform
    else:
        raise ValueError("transform must be an an array of transforms")

    out_texture_engine_path = out_npc_path  # what you already return today

    if not isinstance(out_npc_path, str) or not out_npc_path:
        raise ValueError("Texture gen spec missing non-empty 'out'")

    # Resolve input texture NPC path
    in_npc_path: Optional[str] = None

    if isinstance(from_spec, dict) and from_spec.get("$base_model_texture") is True:
        # Use base model json Texture field
        base_tex = base_model_doc
        if not isinstance(base_tex, dict) or "Texture" not in base_tex:
            raise ValueError("Base model doc missing 'Texture'")
        if not isinstance(base_tex["Texture"], str) or not base_tex["Texture"]:
            raise ValueError("Base model doc 'Texture' must be a non-empty string")
        in_npc_path = base_tex["Texture"]
    elif isinstance(from_spec, dict) and isinstance(from_spec.get("path"), str):
        in_npc_path = from_spec["path"]
    elif isinstance(from_spec, str):
        in_npc_path = from_spec

    if not in_npc_path:
        raise ValueError("Texture gen spec 'from' must be '$base_model_texture' or a path string")

    in_file = _npc_path_to_base_common_file(base_assets_root, in_npc_path)
    if not in_file.exists():
        raise FileNotFoundError(f"Input texture not found: {in_file} (from '{in_npc_path}')")

    out_file = _npc_path_to_common_file(mod_root, out_npc_path)
    _ensure_parent_dir(out_file)

    # Load + transform
    img = Image.open(in_file)

    # Preserve alpha if present; normalize to RGBA when alpha exists
    if img.mode not in ("RGB", "RGBA"):
        # keep alpha if paletted-with-transparency etc
        img = img.convert("RGBA") if "A" in img.getbands() else img.convert("RGB")

    # execute transforms
    for step in steps:
        if not isinstance(step, dict):
            raise ValueError("transform steps must be objects")

        op = step.get("op")
        if op == "desaturate":
            amount = step.get("amount", 1.0) if isinstance(step, dict) else 1.0
            mask01 = None
            if isinstance(step, dict) and isinstance(step.get("mask"), dict):
                mask01 = _load_mask_map(
                    base_assets_root=base_assets_root,
                    mod_root=mod_root,
                    mask_spec=step["mask"],
                    target_size=img.size,
                )
            img = _apply_desaturate_masked(img, amount, mask01)
            # img = _apply_desaturate(img, amount)

        elif op == "tint":
            if not isinstance(step, dict):
                raise ValueError("transform.tint must be an object")

            amount = float(step.get("amount", 1.0))

            if "rgba" in step:
                rgba = step["rgba"]
                if not (isinstance(rgba, list) and len(rgba) == 4):
                    raise ValueError("transform.tint.rgba must be a 4-element array")
                rgba_t = tuple(float(x) for x in rgba)
            elif "rgb" in step:
                rgb = step["rgb"]
                if not (isinstance(rgb, list) and len(rgb) == 3):
                    raise ValueError("transform.tint.rgb must be a 3-element array")
                rgba_t = (float(rgb[0]), float(rgb[1]), float(rgb[2]), 1.0)
            else:
                raise ValueError("transform.tint requires 'rgb' or 'rgba'")

            mask01 = None
            if isinstance(step.get("mask"), dict):
                mask01 = _load_mask_map(
                    base_assets_root=base_assets_root,
                    mod_root=mod_root,
                    mask_spec=step["mask"],
                    target_size=img.size,
                )

            img = _apply_tint_rgba_masked(img, rgba=rgba_t, amount=amount, mask01=mask01)

        elif op == "resize":
            if not isinstance(step, dict):
                raise ValueError("transform.resize must be an object")
            scale = int(step.get("scale", 1))
            resample = str(step.get("resample", "nearest")).lower()
            img = _apply_resize(img, scale=scale, resample=resample)
            scale_used = scale

        elif op == "opacity":
            if not isinstance(step, dict):
                raise ValueError("transform.opacity must be an object")
            op_amount = float(step.get("amount", 1.0))
            mode = str(step.get("mode", "dither")).lower()
            pattern = str(step.get("pattern", "bayer8"))
            preserve_holes = bool(step.get("preserve_holes", True))

            if mode in ("dither", "mask", "cutout"):
                seed = int(step.get("seed", 0))
                tile = int(step.get("tile", 64))
                mask_map = None
                mask_threshold = 0.5

                mask_spec = step.get("mask")
                if mask_spec is not None:
                    if not isinstance(mask_spec, dict):
                        raise ValueError("opacity.mask must be an object")
                    mask_threshold = float(mask_spec.get("threshold", 0.5))
                    mask_map = _load_mask_map(
                        base_assets_root=base_assets_root,
                        mod_root=mod_root,
                        mask_spec=mask_spec,
                        target_size=img.size,
                    )

                img = _apply_opacity_dither(
                    img,
                    amount=op_amount,
                    pattern=pattern,
                    preserve_holes=preserve_holes,
                    seed=seed,
                    tile=tile,
                    mask_map=mask_map,
                    mask_threshold=mask_threshold,
                )

            elif mode in ("none", "keep"):
                pass
            else:
                raise ValueError(f"Unsupported opacity mode: {mode!r}")

        else:
            raise ValueError(f"Unknown transform op: {op!r}")

    # Save
    img.save(out_file, format="PNG")

    out_blocky_engine_path: str | None = None
    if scale_used > 1:
        out_blocky_engine_path = _generate_scaled_blockymodel_from_base_appearance(
            base_assets_root=base_assets_root,
            mod_root=mod_root,
            base_appearance_doc=base_model_doc,
            out_texture_engine_path=out_texture_engine_path,
            scale=scale_used,
        )

    # Return engine-facing path (still "NPC/...")
    return out_texture_engine_path, out_blocky_engine_path


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

        # Preprocess model ops: allow generated texture specs
        processed_model_ops: List[Dict[str, Any]] = []
        for op in v.model_ops:
            # copy op so we can safely mutate
            op2 = json.loads(json.dumps(op))
            if op2.get("op") == "replace" and op2.get("path") == "/Texture":
                val = op2.get("value")
                if isinstance(val, dict) and "$gen_texture" in val:
                    spec = val["$gen_texture"]
                    out_tex, out_blocky = _generate_texture(
                        base_assets_root=base_assets_root,
                        mod_root=mod_root,
                        base_model_doc=base_model_doc,
                        spec=spec,
                    )

                    # Patch /Texture to new texture path
                    op2["value"] = out_tex
                    processed_model_ops.append(op2)

                    # ALSO patch /Model to new blockymodel (if we made one)
                    if out_blocky is not None:
                        processed_model_ops.append({
                            "op": "replace",
                            "path": "/Model",
                            "value": out_blocky,
                        })

                    continue
            processed_model_ops.append(op2)

        apply_json_patch(model_doc, processed_model_ops, strict_paths=strict_paths)
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
    texture = r'C:\dev\hytale\hytale-modding\packs\better_variants\Common\NPC\Livestock\Chicken_Variants\Bobby\Models\Texture.png'
    if not os.path.exists(texture):
        print(f"{texture} not exists")

    tex_img = Image.open(texture).convert("RGBA")
    alpha = np.asarray(tex_img)[..., 3]
    print("alpha==0:", (alpha == 0).sum(), " / ", alpha.size)
    print("alpha unique:", np.unique(alpha)[:20], "...")
