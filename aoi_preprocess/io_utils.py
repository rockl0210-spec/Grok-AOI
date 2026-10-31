"""Stage 1 - load / save.

I/O contract (baseline):
  * input  : single-channel grayscale PNG (8- or 16-bit)  <- primary path
  * output : single-channel grayscale PNG (cleaned image keeps the input bit depth),
             single-channel 0/255 mask PNG; overlay is RGB but *preview only*.
Other inputs (color PNG/BMP, TIFF incl. float) are an optional fallback and are converted to
single-channel gray with a warning flag recorded in the metadata.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

log = logging.getLogger("aoi")

SUPPORTED_EXT = (".png", ".bmp", ".tif", ".tiff")


@dataclass
class LoadedImage:
    """Result of `load_image`.

    data        : float32 (H, W) single channel, scaled to [0, 1] w.r.t. `full_scale`
    meta        : bit-depth / provenance info (JSON-serialisable)
    """
    data: np.ndarray
    meta: dict = field(default_factory=dict)


def _read_raw(path: Path) -> np.ndarray:
    """Read file unchanged. Uses np.fromfile so non-ASCII paths work on every OS."""
    ext = path.suffix.lower()
    if ext in (".tif", ".tiff"):
        import tifffile  # lazy: only needed for TIFF

        return tifffile.imread(str(path))
    buf = np.fromfile(str(path), dtype=np.uint8)
    img = cv2.imdecode(buf, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise IOError(f"cannot decode image: {path}")
    return img


def load_image(path: str | Path, cfg_input) -> LoadedImage:
    """Load an image and return float32 single-channel data in [0, 1] plus bit-depth metadata.

    cfg_input keys: full_scale (null|int), color_to_gray ('luma'|'mean'|'r'|'g'|'b')
    """
    path = Path(path)
    raw = _read_raw(path)
    is_tiff = path.suffix.lower() in (".tif", ".tiff")
    src_dtype = raw.dtype
    src_shape = tuple(int(s) for s in raw.shape)

    # Planar TIFF (C,H,W) -> (H,W,C)
    if is_tiff and raw.ndim == 3 and raw.shape[0] in (3, 4) and raw.shape[2] not in (3, 4):
        raw = np.moveaxis(raw, 0, -1)

    if raw.dtype.kind == "f":
        bits, nominal = 32, 1.0  # float data assumed already in ~[0,1]
    elif raw.dtype == np.uint8:
        bits, nominal = 8, 255.0
    elif raw.dtype == np.uint16:
        bits, nominal = 16, 65535.0
    else:
        raise ValueError(f"unsupported dtype {raw.dtype} in {path}")

    full_scale = cfg_input["full_scale"]  # e.g. 4095 for 12-bit data stored in 16-bit PNG
    scale = float(full_scale) if full_scale else nominal

    note = "single-channel"
    if raw.ndim == 3 and raw.shape[2] == 1:
        raw = raw[..., 0]
    if raw.ndim == 3:
        mode = cfg_input["color_to_gray"]
        f = raw[..., :3].astype(np.float32)
        rgb = f if is_tiff else f[..., ::-1]  # cv2 -> BGR, convert to RGB
        if mode == "luma":
            g = rgb @ np.array([0.299, 0.587, 0.114], np.float32)
        elif mode == "mean":
            g = rgb.mean(axis=2)
        else:
            g = rgb[..., "rgb".index(mode)]
        note = f"color({raw.shape[2]}ch)->gray[{mode}] (fallback path)"
        log.warning("%s is not single-channel grayscale; converted via %s", path.name, mode)
    elif raw.ndim == 2:
        g = raw.astype(np.float32)
    else:
        raise ValueError(f"unsupported array shape {raw.shape}")

    data = np.ascontiguousarray(g / np.float32(scale), dtype=np.float32)
    meta = {
        "path": str(path),
        "format": path.suffix.lower().lstrip("."),
        "src_dtype": str(src_dtype),
        "src_shape": list(src_shape),
        "bit_depth": bits,                 # 8 / 16 / 32(float)
        "full_scale": scale,
        "observed_min": float(g.min()),
        "observed_max": float(g.max()),
        "channels_note": note,
        "height": int(data.shape[0]),
        "width": int(data.shape[1]),
    }
    return LoadedImage(data=data, meta=meta)


def load_calibration_frame(path: str | Path, cfg_input) -> np.ndarray:
    """Load a flat/dark frame as float32 single channel in the same [0,1] convention."""
    return load_image(path, cfg_input).data


def to_uint(img01: np.ndarray, bit_depth: int) -> np.ndarray:
    """float [0,1] -> uint8 / uint16 (clipped, rounded). bit_depth 32(float input) -> 16."""
    if bit_depth == 8:
        return np.clip(np.rint(img01 * 255.0), 0, 255).astype(np.uint8)
    return np.clip(np.rint(img01 * 65535.0), 0, 65535).astype(np.uint16)


def write_png(path: str | Path, arr: np.ndarray) -> None:
    """Write a PNG (2-D uint8/uint16 = single-channel gray; 3-D uint8 BGR = preview). Unicode-path safe."""
    ok, buf = cv2.imencode(".png", arr)
    if not ok:
        raise IOError(f"PNG encode failed: {path}")
    buf.tofile(str(path))


def list_inputs(p: str | Path, recursive: bool = False) -> list[Path]:
    p = Path(p)
    if p.is_file():
        return [p]
    if not p.is_dir():
        raise FileNotFoundError(p)
    it = p.rglob("*") if recursive else p.glob("*")
    return sorted(f for f in it if f.is_file() and f.suffix.lower() in SUPPORTED_EXT)
