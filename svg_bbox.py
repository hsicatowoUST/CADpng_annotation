# make_instance_masks.py

from lxml import etree
from copy import deepcopy
import os
import json

# ---- Configurable SVG namespace ----
SVG_NS = "http://www.w3.org/2000/svg"
NSMAP = {"svg": SVG_NS}

# ==== Core helpers ====

def load_svg(path):
    tree = etree.parse(path)
    root = tree.getroot()
    return tree, root

def new_svg_like(src_root):
    # Create a new <svg> element, copying width/height/viewBox
    new_root = etree.Element("{%s}svg" % SVG_NS, nsmap={None: SVG_NS})
    for attr in ("width", "height", "viewBox"):
        if attr in src_root.attrib:
            new_root.set(attr, src_root.attrib[attr])
    return new_root

def copy_defs(src_root, dst_root):
    # Copy <defs> section if exists (gradients, markers, clipPaths, symbols, etc.)
    defs = src_root.find("svg:defs", namespaces=NSMAP)
    if defs is not None:
        dst_root.append(deepcopy(defs))

def collect_instance_ids(root):
    # Collect all unique instance-id values in the SVG
    els = root.xpath(".//*[@instance-id]", namespaces=NSMAP)
    ids = []
    for el in els:
        v = el.get("instance-id")
        if v is not None:
            ids.append(v.strip())
    # Deduplicate and sort: numeric first (ascending), then non-numeric
    numeric, non_numeric = [], []
    for x in set(ids):
        try:
            numeric.append(int(x))
        except:
            non_numeric.append(x)
    numeric.sort()
    non_numeric.sort()
    return [str(n) for n in numeric] + non_numeric

def pick_elements_by_instance(root, iid):
    # Match any tag with instance-id == iid
    return root.xpath(f".//*[@instance-id='{iid}']", namespaces=NSMAP)

def solid_maskify(elem):
    """
    Turn an element into a solid white mask (ignore original style).
    """
    elem.set("fill", "#ffffff")
    elem.set("stroke", "none")
    if "style" in elem.attrib:
        del elem.attrib["style"]

def semantic_ids_for_instance(elements):
    """
    Return a sorted list of unique semantic-ids present in this instance.
    Elements without semantic-id are ignored.
    """
    sids = {el.get("semantic-id").strip() for el in elements if el.get("semantic-id")}
    def _key(x):
        return (not x.isdigit(), int(x) if x.isdigit() else x)
    return sorted(sids, key=_key)

# ==== BBox computation using svgelements ====

from svgelements import SVG as SVGElemDoc, Shape

def union_bbox(b1, b2):
    if b1 is None: return b2
    if b2 is None: return b1
    x1, y1, x2, y2 = b1
    X1, Y1, X2, Y2 = b2
    return (min(x1, X1), min(y1, Y1), max(x2, X2), max(y2, Y2))

def bbox_to_xywh(bbox):
    if bbox is None: return None, None, None, None
    x1, y1, x2, y2 = bbox
    return x1, y1, max(0.0, x2 - x1), max(0.0, y2 - y1)

def compute_instance_bbox(svg_file, iid):
    """
    Compute union bbox for a given instance-id using svgelements.
    Returns:
      bbox: (xmin, ymin, xmax, ymax) in SVG user units
      parts: list of per-element {semantic_id, bbox, length?}
    Notes:
      - svgelements applies transforms automatically.
      - Units are SVG coordinate system (viewBox/user units).
    """
    doc = SVGElemDoc.parse(svg_file)
    bbox = None
    parts = []
    for el in doc.elements():
        if not isinstance(el, Shape):
            continue
        inst = el.values.get("instance-id")
        if inst is None or str(inst).strip() != str(iid):
            continue
        eb = el.bbox()
        if eb is None:
            continue
        bbox = union_bbox(bbox, eb)
        length = None
        try:
            length = float(el.length())  # Works for Path/Circle/Ellipse/Polyline, etc.
        except Exception:
            pass
        parts.append({
            "semantic_id": (el.values.get("semantic-id") or None),
            "bbox": [eb[0], eb[1], eb[2], eb[3]],
            "length": length
        })
    return bbox, parts

def pad_bbox(x, y, w, h, padding):
    if None in (x, y, w, h): return x, y, w, h
    return x - padding, y - padding, max(0.0, w + 2*padding), max(0.0, h + 2*padding)

# ==== Exporters ====

def export_instance_assets(
    root,
    src_root,
    iid,
    out_dir,
    solid_mask=False,
    ref_png=None,
    input_svg_path=None,
    export_png=True,
    draw_bbox=True,
    bbox_fill="#ffffff",
    bbox_opacity="1.0",
    bbox_padding=0.0,   # padding around bbox in SVG units
):
    """
    Exports:
      - SVG: results/instance_<iid>.svg (with optional filled bbox rect)
      - (optional) PNG: results/instance_<iid>.png (rasterized SVG)
      - JSON: results/instance_<iid>.json
        * instance_id
        * semantic_ids (unique list for this instance)
        * bbox (xmin, ymin, xmax, ymax) & bbox_xywh (SVG units)
        * files (svg/png filenames)
    """
    os.makedirs(out_dir, exist_ok=True)

    # Collect elements and semantic-ids for this instance
    picked_elems = pick_elements_by_instance(root, iid)
    uniq_semantic_ids = semantic_ids_for_instance(picked_elems)

    # Compute bbox from original SVG (preferred) or from per-instance SVG as fallback
    svg_for_bbox = input_svg_path if input_svg_path else None
    bbox, parts = compute_instance_bbox(svg_for_bbox or "", iid) if svg_for_bbox else (None, [])
    # If we didn't compute from original (no path provided), compute from a temporary per-instance SVG later
    compute_after_build = (bbox is None)

    # Build per-instance SVG
    new_root = new_svg_like(src_root)
    copy_defs(src_root, new_root)
    group = etree.SubElement(new_root, "{%s}g" % SVG_NS, id=f"instance-{iid}")
    for el in picked_elems:
        node = deepcopy(el)
        if solid_mask:
            solid_maskify(node)
        group.append(node)

    # If bbox not computed yet, compute now from the just-built per-instance SVG structure
    if compute_after_build:
        # Write a temporary SVG to compute bbox reliably
        tmp_svg_path = os.path.join(out_dir, f"__tmp_instance_{iid}.svg")
        etree.ElementTree(new_root).write(tmp_svg_path, encoding="utf-8", xml_declaration=True, pretty_print=True)
        bbox, parts = compute_instance_bbox(tmp_svg_path, iid)
        try:
            os.remove(tmp_svg_path)
        except OSError:
            pass

    # Add filled bbox rect (on top) if requested and bbox exists
    out_svg = os.path.join(out_dir, f"instance_{iid}.svg")
    x, y, w, h = bbox_to_xywh(bbox) if bbox else (None, None, None, None)
    if draw_bbox and bbox and w > 0 and h > 0:
        if bbox_padding and bbox_padding != 0.0:
            x, y, w, h = pad_bbox(x, y, w, h, bbox_padding)
        bbox_layer = etree.SubElement(new_root, "{%s}g" % SVG_NS, id=f"bbox-{iid}")
        etree.SubElement(
            bbox_layer,
            "{%s}rect" % SVG_NS,
            {
                "x": f"{x}",
                "y": f"{y}",
                "width": f"{w}",
                "height": f"{h}",
                "fill": bbox_fill,
                "fill-opacity": bbox_opacity
            }
        )

    # Write final SVG (now includes bbox rect if enabled)
    etree.ElementTree(new_root).write(out_svg, encoding="utf-8", xml_declaration=True, pretty_print=True)

    # Optional PNG export (rasterizes SVG *with* the bbox rect)
    out_png = None
    if export_png:
        try:
            import cairosvg
            out_png = os.path.splitext(out_svg)[0] + ".png"
            if ref_png is not None:
                # Force output size to match a reference PNG
                from PIL import Image
                with Image.open(ref_png) as img:
                    ref_w, ref_h = img.size
                cairosvg.svg2png(url=out_svg, write_to=out_png,
                                 output_width=ref_w, output_height=ref_h)
            else:
                cairosvg.svg2png(url=out_svg, write_to=out_png)
        except ImportError:
            out_png = None  # cairosvg not installed; skip

    # JSON payload
    out_json = os.path.join(out_dir, f"instance_{iid}.json")
    payload = {
        "instance_id": str(iid),
        "semantic_ids": uniq_semantic_ids,
        "bbox": bbox if bbox else None,                 # [xmin, ymin, xmax, ymax] in SVG units
        "bbox_xywh": [x, y, w, h] if bbox else None,    # [xmin, ymin, width, height]
        "files": {
            "svg": os.path.basename(out_svg),
            "png": os.path.basename(out_png) if out_png else None
        }
    }
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    return out_svg, out_png, out_json

# ==== Driver ====

def main(
    input_svg,
    out_dir="results",
    solid_mask=False,
    export_png=True,
    ref_png=None,
    draw_bbox=True,
    bbox_fill="#ffffff",
    bbox_opacity="1.0",
    bbox_padding=0.0
):
    _, root = load_svg(input_svg)
    src_root = root

    instance_ids = collect_instance_ids(root)
    print(f"Found {len(instance_ids)} unique instance-id(s): {instance_ids}")

    for iid in instance_ids:
        svg_path, png_path, json_path = export_instance_assets(
            root, src_root, iid, out_dir,
            solid_mask=solid_mask,
            ref_png=ref_png if export_png else None,
            input_svg_path=input_svg,
            export_png=export_png,
            draw_bbox=draw_bbox,
            bbox_fill=bbox_fill,
            bbox_opacity=bbox_opacity,
            bbox_padding=bbox_padding
        )
        msg = f"[SVG] {svg_path}"
        msg += f" | [PNG] {png_path}" if png_path else " | (PNG skipped)"
        msg += f" | [JSON] {json_path}"
        print(msg)

if __name__ == "__main__":
    # Example usage: python make_instance_masks.py
    INPUT = "example_pair/0000-0002.svg"     # Replace with your input SVG filename
    OUTDIR = "results_bbox_off"                        # Output folder
    SOLID_MASK = False                        # True: convert shapes to solid white; False: keep original style
    EXPORT_PNG = True                         # If cairosvg is installed, export PNG
    REF_PNG = "example_pair/0000-0002.png"    # Optional: force PNG size to match this raster
    DRAW_BBOX = False                          # Draw a filled bbox onto the per-instance SVG/PNG
    BBOX_FILL = "#000000"                     # Fill color for bbox (white)
    BBOX_OPACITY = "1.0"                      # 1.0 = fully solid; e.g., "0.5" for 50% opacity
    BBOX_PADDING = 0.0                        # Optional padding around bbox in SVG units

    main(
        INPUT,
        OUTDIR,
        SOLID_MASK,
        EXPORT_PNG,
        REF_PNG,
        DRAW_BBOX,
        BBOX_FILL,
        BBOX_OPACITY,
        BBOX_PADDING
    )