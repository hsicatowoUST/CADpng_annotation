# make_instance_masks.py

from lxml import etree
from copy import deepcopy
import os
import json

# ---- Configurable SVG namespace ----
SVG_NS = "http://www.w3.org/2000/svg"
NSMAP = {"svg": SVG_NS}

# ========== Core helpers ==========

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

def is_primitive(el):
    # Consider common drawable primitives; extend as needed
    local = el.tag.split('}')[-1]
    # return local in {"path","ellipse","rect","circle","polyline","polygon","line","text"}
    return local in {"path","ellipse","rect","circle","polyline","polygon","line"}

def force_color(el, stroke="#000000", fill="#000000", keep_fill_none=True):
    """
    Force a shape's color. If keep_fill_none and element has fill='none',
    we keep it none (i.e., only stroke is colored).
    """
    if "style" in el.attrib:
        # Remove inline style so explicit attributes win
        del el.attrib["style"]
    el.set("stroke", stroke)
    if keep_fill_none and el.get("fill", "").lower() == "none":
        # keep none
        pass
    else:
        # For line/polyline which often don't have fill, we can skip setting fill
        local = el.tag.split('}')[-1]
        if local in {"line", "polyline"} and el.get("fill") is None:
            # no fill for line/polyline
            pass
        else:
            el.set("fill", fill)

def set_green(el, keep_fill_none=True, green="#00ff00"):
    force_color(el, stroke=green, fill="#00ff006d", keep_fill_none=keep_fill_none)

def set_black(el, keep_fill_none=True, black="#000000"):
    force_color(el, stroke=black, fill="#0000006d", keep_fill_none=keep_fill_none)

def semantic_ids_for_instance(elements):
    sids = {el.get("semantic-id").strip() for el in elements if el.get("semantic-id")}
    def _key(x):
        return (not x.isdigit(), int(x) if x.isdigit() else x)
    return sorted(sids, key=_key)

# ========== BBox computation using svgelements (still available) ==========

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
            length = float(el.length())
        except Exception:
            pass
        parts.append({
            "semantic_id": (el.values.get("semantic-id") or None),
            "bbox": [eb[0], eb[1], eb[2], eb[3]],
            "length": length
        })
    return bbox, parts

# ========== Exports (original per-instance mask + JSON) ==========

def solid_maskify(elem):
    elem.set("fill", "#ffffff")
    elem.set("stroke", "none")
    if "style" in elem.attrib:
        del elem.attrib["style"]

def export_instance_assets(
    root,
    src_root,
    iid,
    out_dir,
    solid_mask=False,
    ref_png=None,
    input_svg_path=None,
    export_png=True
):
    """
    Original per-instance mask + JSON (kept for compatibility).
    """
    os.makedirs(out_dir, exist_ok=True)

    picked_elems = pick_elements_by_instance(root, iid)
    uniq_semantic_ids = semantic_ids_for_instance(picked_elems)

    # Build per-instance SVG
    new_root = new_svg_like(src_root)
    copy_defs(src_root, new_root)
    group = etree.SubElement(new_root, "{%s}g" % SVG_NS, id=f"instance-{iid}")
    for el in picked_elems:
        node = deepcopy(el)
        if solid_mask:
            solid_maskify(node)
        group.append(node)

    out_svg = os.path.join(out_dir, f"instance_{iid}.svg")
    etree.ElementTree(new_root).write(out_svg, encoding="utf-8", xml_declaration=True, pretty_print=True)

    # Optional PNG
    out_png = None
    if export_png:
        try:
            import cairosvg
            out_png = os.path.splitext(out_svg)[0] + ".png"
            if ref_png is not None:
                from PIL import Image
                with Image.open(ref_png) as img:
                    ref_w, ref_h = img.size
                cairosvg.svg2png(url=out_svg, write_to=out_png,
                                 output_width=ref_w, output_height=ref_h)
            else:
                cairosvg.svg2png(url=out_svg, write_to=out_png)
        except ImportError:
            out_png = None

    # BBox + JSON
    svg_for_bbox = input_svg_path if input_svg_path else out_svg
    bbox, parts = compute_instance_bbox(svg_for_bbox, iid)
    x, y, w, h = bbox_to_xywh(bbox) if bbox else (None, None, None, None)

    out_json = os.path.join(out_dir, f"instance_{iid}.json")
    payload = {
        "instance_id": str(iid),
        "semantic_ids": uniq_semantic_ids,
        "bbox": bbox if bbox else None,                 # [xmin, ymin, xmax, ymax]
        "bbox_xywh": [x, y, w, h] if bbox else None,    # [xmin, ymin, width, height]
        "files": {
            "svg": os.path.basename(out_svg),
            "png": os.path.basename(out_png) if out_png else None
        }
    }
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    return out_svg, out_png, out_json

# ========== Per-instance "highlight" outputs (all black, target instance green) ==========

def remove_background_style(svg_root):
    # Remove inline background-color in <svg style="...">
    if "style" in svg_root.attrib:
        style_val = svg_root.attrib["style"]
        # Drop background-color if present
        new_style = ";".join(
            kv for kv in style_val.split(";")
            if "background" not in kv.strip().lower()
        )
        if new_style.strip():
            svg_root.attrib["style"] = new_style
        else:
            del svg_root.attrib["style"]

    # Also remove explicit attribute if exists (rare, but some files use it)
    if "background-color" in svg_root.attrib:
        del svg_root.attrib["background-color"]

def export_highlight_for_instance(
    src_root,
    iid,
    out_dir,
    ref_png=None,
    green="#00ff00",
    black="#000000",
    keep_fill_none=True
):
    """
    Creates highlight SVG/PNG that contains ALL primitives:
      - everything is colored BLACK
      - elements with instance-id == iid are recolored GREEN
    Output:
      results/highlight_instance_<iid>.svg
      results/highlight_instance_<iid>.png
    """
    os.makedirs(out_dir, exist_ok=True)

    # Start from a deep copy of the original SVG
    new_root = deepcopy(src_root)
    remove_background_style(new_root)

    # 1) Turn ALL primitives black
    for el in new_root.xpath(".//*", namespaces=NSMAP):
        if is_primitive(el):
            set_black(el, keep_fill_none=keep_fill_none, black=black)

    # 2) Recolor target instance to green
    for el in new_root.xpath(f".//*[@instance-id='{iid}']", namespaces=NSMAP):
        if is_primitive(el):
            set_green(el, keep_fill_none=keep_fill_none, green=green)

    # Write highlight SVG
    out_svg = os.path.join(out_dir, f"highlight_instance_{iid}.svg")
    etree.ElementTree(new_root).write(out_svg, encoding="utf-8", xml_declaration=True, pretty_print=True)

    # Rasterize to PNG
    out_png = None
    try:
        import cairosvg
        out_png = os.path.splitext(out_svg)[0] + ".png"
        if ref_png is not None:
            from PIL import Image
            with Image.open(ref_png) as img:
                ref_w, ref_h = img.size
            cairosvg.svg2png(url=out_svg, write_to=out_png,
                             output_width=ref_w, output_height=ref_h)
        else:
            cairosvg.svg2png(url=out_svg, write_to=out_png)
    except ImportError:
        out_png = None

    return out_svg, out_png

# ========== Driver ==========

def main(
    input_svg,
    out_dir="results",
    export_png=True,
    ref_png=None,
    # original mask/json exports
    export_masks=True,
    solid_mask=False,
    # highlight export (your new request)
    export_highlight=True,
    green="#00ff00",
    black="#000000",
    keep_fill_none=True
):
    _, root = load_svg(input_svg)
    src_root = root

    instance_ids = collect_instance_ids(root)
    print(f"Found {len(instance_ids)} unique instance-id(s): {instance_ids}")

    for iid in instance_ids:
        if export_masks:
            svg_path, png_path, json_path = export_instance_assets(
                root, src_root, iid, out_dir,
                solid_mask=solid_mask,
                ref_png=ref_png if export_png else None,
                input_svg_path=input_svg,
                export_png=export_png
            )
            msg = f"[MASK] svg={svg_path}, png={png_path}, json={json_path}"
            print(msg)

        if export_highlight:
            hl_svg, hl_png = export_highlight_for_instance(
                src_root, iid, out_dir,
                ref_png=ref_png if export_png else None,
                green=green,
                black=black,
                keep_fill_none=keep_fill_none
            )
            msg = f"[HIGHLIGHT] svg={hl_svg}, png={hl_png}"
            print(msg)

if __name__ == "__main__":
    # Example usage: python make_instance_masks.py
    INPUT = "example_pair/0000-0002.svg"   # Replace with your SVG path
    OUTDIR = "results_highlight"                     # Output folder
    EXPORT_PNG = True                      # If cairosvg is installed, export PNGs
    REF_PNG = "example_pair/0000-0002.png" # Optional: force PNG size to match this raster

    # If you ONLY want highlight images set EXPORT_MASKS=False
    EXPORT_MASKS = True                   # keep False if you don't need per-instance masks/JSON right now
    SOLID_MASK = False                     # ignored if EXPORT_MASKS=False

    EXPORT_HIGHLIGHT = True                # <— your new feature
    GREEN = "#00ff00"                      # highlight color
    BLACK = "#000000"                      # base color for other primitives
    KEEP_FILL_NONE = True                  # respect fill="none" (only stroke shows)

    main(
        INPUT,
        OUTDIR,
        EXPORT_PNG,
        REF_PNG,
        EXPORT_MASKS,
        SOLID_MASK,
        EXPORT_HIGHLIGHT,
        GREEN,
        BLACK,
        KEEP_FILL_NONE
    )