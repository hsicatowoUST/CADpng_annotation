#!/usr/bin/env python3
"""
FloorplanCAD → PNG instance annotations

Given a folder of SVG/PNG pairs (same basename), this script:
  1) Builds a single "instance-ID map" SVG by recoloring each primitive to its instance color.
  2) Rasterizes the instance map to PNG at the EXACT size of the dataset PNG (optionally oversampled).
  3) Extracts per-instance binary masks, tight crops, and cutout PNGs.
  4) Saves JSON metadata per instance (semantic id, class name, bbox, area, optional stroke length in SVG units).

Dependencies (conda-forge): lxml, cairosvg, pillow, numpy, scikit-image, svgpathtools, tqdm

Usage:
  python make_png_annotations.py \
    --in-dir /path/to/floorplancad \
    --out-dir /path/to/out \
    --class-map /path/to/semantic_id_to_name.json \
    --oversample 1

Outputs per drawing (basename = e.g., 000123):
  out/000123/instance_map.png              # color-coded by instance id
  out/000123/meta.json                     # summary listing of instances
  out/000123/instances/<id>_mask.png       # 1-channel mask (0/255)
  out/000123/instances/<id>_cutout.png     # RGBA cutout from original PNG
  out/000123/instances/<id>.json           # metadata for that instance

Notes:
- The mask/cutout PNGs are pixel-aligned to the dataset PNG. No resizing/stretching.
- If hairline strokes vanish, try --oversample 4 (will render 4× and downsample with NEAREST).
- Stroke length is computed from vector geometry per instance in SVG units (ignores transforms in a first pass).
"""

from __future__ import annotations
import argparse
import json
import math
import os
from pathlib import Path
from copy import deepcopy
from typing import Dict, Tuple, List, Optional, Iterable

import numpy as np
from lxml import etree as ET
from PIL import Image
from tqdm import tqdm

# Rasterizer
import cairosvg

# Metrics
from skimage.measure import label as sk_label
from skimage.measure import regionprops

# Vector geometry (optional lengths)
from svgpathtools import parse_path

SUPPORTED_DRAWABLES = {
    'path', 'line', 'polyline', 'polygon', 'rect', 'circle', 'ellipse'
}

# -----------------------------
# Utilities
# -----------------------------

def id_to_rgb(i: int) -> Tuple[int, int, int]:
    """Encode instance id into a 24-bit RGB color.
    Supports up to 16,777,215 instances (which is... enough).
    """
    if i < 0 or i > 0xFFFFFF:
        raise ValueError(f"instance-id {i} out of 24-bit range")
    return ( (i >> 16) & 255, (i >> 8) & 255, i & 255 )


def rgb_to_id(rgb: Tuple[int, int, int]) -> int:
    r, g, b = [int(v) & 255 for v in rgb]
    return (r << 16) | (g << 8) | b


def parse_float(s: Optional[str], default: float = 1.0) -> float:
    if s is None:
        return default
    try:
        # strip common units
        for u in ('px', 'mm', 'cm', 'in'):
            if s.endswith(u):
                s = s[:-len(u)]
                break
        return float(s)
    except Exception:
        return default


def ensure_rgba(img: Image.Image) -> Image.Image:
    return img.convert('RGBA') if img.mode != 'RGBA' else img


def np_to_pil_mask(mask: np.ndarray) -> Image.Image:
    """Boolean mask -> 8-bit L image 0/255"""
    return Image.fromarray((mask.astype(np.uint8) * 255), mode='L')


def tight_bbox(mask: np.ndarray, pad: int = 6) -> Tuple[int, int, int, int]:
    ys, xs = np.where(mask)
    if ys.size == 0:
        return 0, 0, 0, 0
    y0, y1 = int(ys.min()), int(ys.max())
    x0, x1 = int(xs.min()), int(xs.max())
    h, w = mask.shape
    x0 = max(0, x0 - pad)
    y0 = max(0, y0 - pad)
    x1 = min(w - 1, x1 + pad)
    y1 = min(h - 1, y1 + pad)
    # convert to inclusive bbox -> xywh
    return x0, y0, x1 - x0 + 1, y1 - y0 + 1


def crop_to_bbox(arr: np.ndarray, bbox_xywh: Tuple[int, int, int, int]) -> np.ndarray:
    x, y, w, h = bbox_xywh
    return arr[y:y+h, x:x+w, ...]


def load_class_map(path: Optional[Path]) -> Dict[str, str]:
    if not path:
        return {}
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    # normalize keys to str
    return {str(k): v for k, v in data.items()}


# -----------------------------
# SVG recoloring for instance map
# -----------------------------

def find_drawables(root: ET._Element) -> Iterable[ET._Element]:
    for el in root.iter():
        tag = el.tag.split('}')[-1]
        if tag in SUPPORTED_DRAWABLES:
            yield el


def get_attr(el: ET._Element, name: str) -> Optional[str]:
    # Try attribute directly, then in style="..." if present
    val = el.get(name)
    if val is not None:
        return val
    style = el.get('style')
    if style:
        # style="a:b;c:d" -> dict
        try:
            items = dict([kv.split(':', 1) for kv in style.split(';') if ':' in kv])
            return items.get(name)
        except Exception:
            return None
    return None


def set_style_hard(el: ET._Element, *, fill: Optional[str], stroke: Optional[str],
                   stroke_width: Optional[str], fill_opacity: Optional[str] = None,
                   stroke_opacity: Optional[str] = None, opacity: Optional[str] = None):
    # Remove style attr and set explicit attributes to avoid CSS inheritance
    if 'style' in el.attrib:
        del el.attrib['style']
    if fill is not None:
        el.set('fill', fill)
    if stroke is not None:
        el.set('stroke', stroke)
    if stroke_width is not None:
        el.set('stroke-width', stroke_width)
    if fill_opacity is not None:
        el.set('fill-opacity', fill_opacity)
    if stroke_opacity is not None:
        el.set('stroke-opacity', stroke_opacity)
    if opacity is not None:
        el.set('opacity', opacity)


def recolor_svg_to_instance_map(svg_tree: ET._ElementTree,
                                instance_sem_map: Dict[int, Optional[int]],
                                stroke_inflate: float = 1.0,
                                min_stroke: float = 0.6) -> ET._ElementTree:
    """Return a deep-copied SVG where each primitive is painted to its instance color.
    - Shapes are made solid (fill+stroke the same color) where possible.
    - Lines remain stroke-only.
    - Stroke width may be inflated to help thin strokes survive rasterization.
    """
    root = deepcopy(svg_tree.getroot())
    nsmap = root.nsmap

    for el in find_drawables(root):
        inst = get_attr(el, 'instance-id')
        if inst is None:
            # Hide non-instance geometry by making it fully transparent.
            set_style_hard(el, fill='none', stroke='none', stroke_width='0', opacity='0')
            continue
        try:
            iid = int(inst)
        except Exception:
            set_style_hard(el, fill='none', stroke='none', stroke_width='0', opacity='0')
            continue

        r, g, b = id_to_rgb(iid)
        color = f"rgb({r},{g},{b})"

        # Stroke width handling
        sw = parse_float(get_attr(el, 'stroke-width'), default=1.0)
        sw = max(min_stroke, sw * float(stroke_inflate))
        sw_str = f"{sw}"

        tag = el.tag.split('}')[-1]
        if tag in {'line', 'polyline'}:
            set_style_hard(el, fill='none', stroke=color, stroke_width=sw_str,
                           fill_opacity='1', stroke_opacity='1', opacity='1')
        elif tag in {'rect', 'polygon', 'circle', 'ellipse'}:
            set_style_hard(el, fill=color, stroke=color, stroke_width=sw_str,
                           fill_opacity='1', stroke_opacity='1', opacity='1')
        elif tag == 'path':
            # Closed paths will fill; open paths will effectively show stroke only.
            set_style_hard(el, fill=color, stroke=color, stroke_width=sw_str,
                           fill_opacity='1', stroke_opacity='1', opacity='1')
        else:
            # Fallback: show as stroke
            set_style_hard(el, fill='none', stroke=color, stroke_width=sw_str,
                           fill_opacity='1', stroke_opacity='1', opacity='1')

    # Ensure root has preserveAspectRatio default (avoid stretch); keep viewBox
    if root.get('preserveAspectRatio') is None:
        root.set('preserveAspectRatio', 'xMidYMid meet')

    return ET.ElementTree(root)


# -----------------------------
# Rasterize
# -----------------------------

def render_instance_map_png(svg_tree: ET._ElementTree,
                            out_png: Path,
                            width: int,
                            height: int,
                            oversample: int = 1) -> Path:
    """Rasterize the recolored SVG to a PNG at width×height (or oversample× larger)."""
    os.makedirs(out_png.parent, exist_ok=True)
    svg_bytes = ET.tostring(svg_tree, encoding='utf-8', xml_declaration=True)
    W = int(width * oversample)
    H = int(height * oversample)
    cairosvg.svg2png(bytestring=svg_bytes,
                     write_to=str(out_png),
                     output_width=W,
                     output_height=H,
                     background_color='transparent')
    return out_png


# -----------------------------
# Extract masks & cutouts
# -----------------------------

def unique_instance_colors(arr: np.ndarray) -> List[Tuple[int, int, int]]:
    """Return unique non-transparent RGB colors present in H×W×4/3 array.
    Background transparent pixels are ignored.
    """
    if arr.shape[-1] == 4:
        rgb = arr[..., :3]
        alpha = arr[..., 3]
        # keep where alpha>0
        m = alpha > 0
        if not np.any(m):
            return []
        rgb = rgb[m]
    else:
        rgb = arr.reshape(-1, 3)
    uniq = np.unique(rgb, axis=0)
    return [tuple(map(int, u)) for u in uniq.tolist()]


def extract_masks_and_cutouts(instance_map_png: Path,
                              original_png: Path,
                              out_dir: Path,
                              instance_sem_map: Dict[int, Optional[int]],
                              class_map: Dict[str, str],
                              oversample: int = 1,
                              min_pixels: int = 8) -> Dict:
    """Creates mask.png, cutout.png and per-instance json. Returns drawing-level summary.
    """
    imap_img = Image.open(instance_map_png)
    imap_arr = np.array(imap_img)

    # Downsample if oversampled
    original = Image.open(original_png)
    W, H = original.size

    if oversample > 1:
        # We'll build masks at oversampled resolution, then resize masks with NEAREST
        pass

    sums = []
    rgb_colors = unique_instance_colors(imap_arr)

    # Prepare output dirs
    inst_dir = out_dir / 'instances'
    os.makedirs(inst_dir, exist_ok=True)

    # Ensure original RGBA for cutouts
    original_rgba = ensure_rgba(original)
    orig_arr = np.array(original_rgba)

    summary = {
        'drawing_png': str(original_png),
        'drawing_width': W,
        'drawing_height': H,
        'instances': []
    }

    for rgb in rgb_colors:
        iid = rgb_to_id(rgb)
        # Build mask at oversample resolution (if any)
        mask_os = np.all(imap_arr[..., :3] == np.array(rgb, dtype=np.uint8), axis=-1)
        if oversample > 1:
            # Downsample with NEAREST to 1×
            m_img = np_to_pil_mask(mask_os)
            m_img_1x = m_img.resize((W, H), resample=Image.NEAREST)
            mask = np.array(m_img_1x) > 0
        else:
            mask = mask_os

        area = int(mask.sum())
        if area < min_pixels:
            continue

        bbox = tight_bbox(mask, pad=6)
        x, y, w, h = bbox
        if w <= 0 or h <= 0:
            continue

        # Crop mask and image
        mask_c = crop_to_bbox(mask, bbox)

        cutout = crop_to_bbox(orig_arr, bbox).copy()
        # apply alpha = 0 where mask is False
        if cutout.shape[-1] != 4:
            # Shouldn't happen, but guard.
            tmp = np.zeros((cutout.shape[0], cutout.shape[1], 4), dtype=np.uint8)
            tmp[..., :3] = cutout
            tmp[..., 3] = 255
            cutout = tmp
        cutout[..., 3] = np.where(mask_c, cutout[..., 3], 0)

        # Save files
        mask_path = inst_dir / f"{iid}_mask.png"
        cutout_path = inst_dir / f"{iid}_cutout.png"
        np_to_pil_mask(mask_c).save(mask_path)
        Image.fromarray(cutout, mode='RGBA').save(cutout_path)

        # Region metrics
        lab = sk_label(mask_c.astype(np.uint8), connectivity=1)
        props = regionprops(lab)
        area_px = int(props[0].area) if props else int(mask_c.sum())

        # Semantic/class name
        sem_id = instance_sem_map.get(iid)
        class_name = class_map.get(str(sem_id), None) if sem_id is not None else None

        inst_meta = {
            'instance_id': iid,
            'instance_color_rgb': rgb,
            'semantic_id': sem_id,
            'class_name': class_name,
            'bbox_xywh': [int(x), int(y), int(w), int(h)],
            'area_px': area_px,
        }

        # Perimeter (optional, pixels)
        try:
            perim = float(props[0].perimeter) if props else None
            inst_meta['perimeter_px'] = perim
        except Exception:
            pass

        # Stroke length in SVG units (filled in by caller if provided)
        # We'll leave a placeholder; main() will replace if it computed.
        inst_meta['total_stroke_length_svg_units'] = None

        # Write per-instance json
        with open(out_dir / 'instances' / f"{iid}.json", 'w', encoding='utf-8') as f:
            json.dump(inst_meta, f, ensure_ascii=False, indent=2)

        summary['instances'].append(inst_meta)

    # Save drawing-level summary
    with open(out_dir / 'meta.json', 'w', encoding='utf-8') as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    return summary


# -----------------------------
# Optional: compute stroke length per instance from SVG
# -----------------------------

def path_length(d: str) -> float:
    try:
        return parse_path(d).length(error=1e-3)
    except Exception:
        return 0.0


def poly_length(points: str) -> float:
    try:
        pts = []
        for tok in points.replace(',', ' ').split():
            pass
    except Exception:
        return 0.0


def compute_instance_lengths(svg_tree: ET._ElementTree) -> Dict[int, float]:
    """Sum approximate stroke lengths per instance (SVG units).
    Note: ignores transforms and line caps; good as a rough size proxy.
    """
    lengths: Dict[int, float] = {}
    root = svg_tree.getroot()

    def add(iid: int, val: float):
        lengths[iid] = lengths.get(iid, 0.0) + float(val)

    for el in find_drawables(root):
        inst = get_attr(el, 'instance-id')
        if inst is None:
            continue
        try:
            iid = int(inst)
        except Exception:
            continue
        tag = el.tag.split('}')[-1]
        if tag == 'line':
            x1 = parse_float(el.get('x1'), 0.0)
            y1 = parse_float(el.get('y1'), 0.0)
            x2 = parse_float(el.get('x2'), 0.0)
            y2 = parse_float(el.get('y2'), 0.0)
            add(iid, math.hypot(x2 - x1, y2 - y1))
        elif tag in {'polyline', 'polygon'}:
            pts = el.get('points') or ''
            coords: List[Tuple[float, float]] = []
            buf = []
            for tok in pts.replace(',', ' ').split():
                try:
                    buf.append(float(tok))
                    if len(buf) == 2:
                        coords.append((buf[0], buf[1]))
                        buf = []
                except Exception:
                    buf = []
            for (x0, y0), (x1, y1) in zip(coords, coords[1:]):
                add(iid, math.hypot(x1 - x0, y1 - y0))
            if tag == 'polygon' and len(coords) >= 3:
                # close the polygon
                (x0, y0), (x1, y1) = coords[-1], coords[0]
                add(iid, math.hypot(x1 - x0, y1 - y0))
        elif tag == 'rect':
            w = parse_float(el.get('width'), 0.0)
            h = parse_float(el.get('height'), 0.0)
            add(iid, 2 * (w + h))
        elif tag == 'circle':
            r = parse_float(el.get('r'), 0.0)
            add(iid, 2 * math.pi * r)
        elif tag == 'ellipse':
            rx = parse_float(el.get('rx'), 0.0)
            ry = parse_float(el.get('ry'), 0.0)
            # Ramanujan approx for ellipse circumference
            a, b = max(rx, ry), min(rx, ry)
            if a == 0 or b == 0:
                add(iid, 0.0)
            else:
                h = ((a - b)**2) / ((a + b)**2)
                add(iid, math.pi * (a + b) * (1 + (3*h)/(10 + math.sqrt(4 - 3*h))))
        elif tag == 'path':
            d = el.get('d') or ''
            add(iid, path_length(d))
        else:
            continue

    return lengths


# -----------------------------
# Main
# -----------------------------

def process_pair(svg_path: Path, png_path: Path, out_base: Path, class_map: Dict[str, str],
                 oversample: int = 1, stroke_inflate: float = 1.0, min_pixels: int = 8) -> None:
    # Load PNG size (golden truth)
    with Image.open(png_path) as im:
        W, H = im.size

    # Parse SVG
    parser = ET.XMLParser(remove_comments=False)
    svg_tree = ET.parse(str(svg_path), parser)

    # Collect instance → semantic map from SVG attributes
    instance_sem_map: Dict[int, Optional[int]] = {}
    for el in find_drawables(svg_tree.getroot()):
        inst = get_attr(el, 'instance-id')
        sem = get_attr(el, 'semantic-id')
        if inst is None:
            continue
        try:
            iid = int(inst)
        except Exception:
            continue
        sid = None
        if sem is not None:
            try:
                sid = int(sem)
            except Exception:
                sid = None
        if iid not in instance_sem_map:
            instance_sem_map[iid] = sid

    # Recolor to instance map
    inst_svg = recolor_svg_to_instance_map(svg_tree, instance_sem_map,
                                           stroke_inflate=stroke_inflate)

    # Render instance map PNG
    out_dir = out_base / svg_path.stem
    os.makedirs(out_dir, exist_ok=True)
    inst_map_png = out_dir / 'instance_map.png'

    tmp_png = out_dir / ('instance_map_os.png' if oversample > 1 else 'instance_map.png')
    render_instance_map_png(inst_svg, tmp_png, W, H, oversample=oversample)

    # If oversampled, keep os file and also resize to 1× for reference
    if oversample > 1:
        im_os = Image.open(tmp_png)
        im_1x = im_os.resize((W, H), resample=Image.NEAREST)
        im_1x.save(inst_map_png)
    else:
        inst_map_png = tmp_png

    # Extract masks & cutouts
    summary = extract_masks_and_cutouts(inst_map_png, png_path, out_dir,
                                        instance_sem_map, class_map,
                                        oversample=oversample, min_pixels=min_pixels)

    # Optional: compute stroke lengths and update per-instance jsons
    lengths = compute_instance_lengths(svg_tree)
    for inst_meta in summary['instances']:
        iid = inst_meta['instance_id']
        L = float(lengths.get(iid, 0.0))
        inst_meta['total_stroke_length_svg_units'] = L
        # rewrite per-instance file
        inst_json = out_dir / 'instances' / f"{iid}.json"
        with open(inst_json, 'w', encoding='utf-8') as f:
            json.dump(inst_meta, f, ensure_ascii=False, indent=2)
    # rewrite summary
    with open(out_dir / 'meta.json', 'w', encoding='utf-8') as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)



def main():
    ap = argparse.ArgumentParser(description='FloorplanCAD → PNG instance annotations')
    ap.add_argument('--in-dir', type=Path, required=True,
                    help='Folder containing SVG/PNG pairs (same basename).')
    ap.add_argument('--png-dir', type=Path, default=None,
                    help='(Optional) Separate folder for PNGs; matches basename of SVGs in --in-dir.')
    ap.add_argument('--out-dir', type=Path, required=True,
                    help='Output base directory.')
    ap.add_argument('--class-map', type=Path, default=None,
                    help='JSON mapping: {"semantic_id": "class_name", ...}')
    ap.add_argument('--oversample', type=int, default=1,
                    help='Render instance map at N×, then downsample with NEAREST (default 1).')
    ap.add_argument('--stroke-inflate', type=float, default=1.0,
                    help='Multiply stroke-width while building instance map (default 1.0).')
    ap.add_argument('--min-pixels', type=int, default=8,
                    help='Drop tiny instances with area < this many pixels (default 8).')
    ap.add_argument('--ext-svg', type=str, default='.svg', help='SVG extension (default .svg)')
    ap.add_argument('--ext-png', type=str, default='.png', help='PNG extension (default .png)')
    args = ap.parse_args()

    class_map = load_class_map(args.class_map)

    # Find all SVGs
    svg_paths = sorted([p for p in args.in_dir.rglob(f"*{args.ext_svg}")])
    if not svg_paths:
        raise SystemExit(f"No SVG files found under {args.in_dir}")

    for svg_path in tqdm(svg_paths, desc='Processing drawings'):
        stem = svg_path.stem
        png_path = None
        if args.png_dir:
            candidate = args.png_dir / f"{stem}{args.ext_png}"
            if candidate.exists():
                png_path = candidate
        else:
            candidate = svg_path.with_suffix(args.ext_png)
            if candidate.exists():
                png_path = candidate
        if png_path is None or not png_path.exists():
            print(f"[WARN] Missing PNG for {svg_path}")
            continue
        try:
            process_pair(svg_path, png_path, args.out_dir, class_map,
                         oversample=args.oversample,
                         stroke_inflate=args.stroke_inflate,
                         min_pixels=args.min_pixels)
        except Exception as e:
            print(f"[ERROR] {svg_path.name}: {e}")


if __name__ == '__main__':
    main()