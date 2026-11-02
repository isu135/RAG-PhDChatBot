"""
Extract figures from the PDFs in Data/, then describe them with the Azure vision model.
  python build_figures.py extract    -> saves images to figures/ (review and delete junk)
  python build_figures.py describe   -> writes a description for each remaining image
"""
import os, sys, json, base64, re
import fitz  # PyMuPDF
from dotenv import load_dotenv
from openai import AzureOpenAI

DATA_DIR = "Data"
FIG_DIR = "figures"
FIG_JSON = os.path.join(FIG_DIR, "figures.json")
MIN_PART = 40     # ignore tiny image fragments (points; 72 points = 1 inch)
MIN_FIG = 120     # ignore final figures smaller than this (logos, icons)
DPI = 150         # image resolution: enough to read, small enough for GitHub


def load_figs():
    if os.path.exists(FIG_JSON):
        with open(FIG_JSON, encoding="utf-8") as f:
            return json.load(f)
    return []


def save_figs(figs):
    with open(FIG_JSON, "w", encoding="utf-8") as f:
        json.dump(figs, f, indent=2, ensure_ascii=False)


def find_caption(page, rect):
    """Find the 'Fig. N ...' text block closest below the figure."""
    best, best_dist = "", 1e9
    for x0, y0, x1, y1, text, *_ in page.get_text("blocks"):
        t = " ".join(text.split())
        if re.match(r"^(fig\.?|figure)\s*\d+", t, re.I):
            dist = y0 - rect.y1
            if -20 <= dist < 150 and dist < best_dist:
                best, best_dist = t, dist
    return best[:500]


def merge_rects(rects):
    """Join image pieces that touch or overlap (multi-panel figures)."""
    merged = []
    for r in rects:
        r = fitz.Rect(r)
        changed = True
        while changed:
            changed = False
            for m in merged:
                grown = fitz.Rect(m.x0 - 10, m.y0 - 10, m.x1 + 10, m.y1 + 10)
                if grown.intersects(r):
                    r = r | m
                    merged.remove(m)
                    changed = True
                    break
        merged.append(r)
    return merged


def extract():
    os.makedirs(FIG_DIR, exist_ok=True)
    figs = []
    for name in sorted(os.listdir(DATA_DIR)):
        if not name.lower().endswith(".pdf"):
            continue
        doc = fitz.open(os.path.join(DATA_DIR, name))
        stem = re.sub(r"[^A-Za-z0-9]+", "_", os.path.splitext(name)[0])[:40]
        count = 0
        for pno, page in enumerate(doc, start=1):
            parts = []
            for img in page.get_images(full=True):
                for r in page.get_image_rects(img[0]):
                    if r.width >= MIN_PART and r.height >= MIN_PART:
                        parts.append(r & page.rect)
            n = 0
            for rect in merge_rects(parts):
                too_small = rect.width < MIN_FIG or rect.height < MIN_FIG
                whole_page = rect.get_area() > 0.9 * page.rect.get_area()
                if too_small or whole_page:
                    continue
                n += 1
                path = f"{FIG_DIR}/{stem}_p{pno}_{n}.png"
                page.get_pixmap(clip=rect, dpi=DPI).save(path)
                figs.append({"file_name": name, "page": pno, "image_path": path,
                             "caption": find_caption(page, rect), "description": ""})
                count += 1
        print(f"{name}: {count} figures")
    save_figs(figs)
    print(f"\nSaved {len(figs)} figures in '{FIG_DIR}/'.")
    print("Open the folder, delete any images you don't want, then run:")
    print("  python build_figures.py describe")


def describe():
    load_dotenv(override=True)
    client = AzureOpenAI(api_key=os.getenv("AZURE_OPENAI_API_KEY"),
                         azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
                         api_version=os.getenv("AZURE_OPENAI_API_VERSION"))
    # keep only figures whose image file still exists (you deleted the rest)
    figs = [f for f in load_figs() if os.path.exists(f["image_path"])]
    todo = [f for f in figs if not f["description"]]
    print(f"{len(todo)} figures to describe.")
    for i, f in enumerate(todo, start=1):
        with open(f["image_path"], "rb") as fh:
            b64 = base64.b64encode(fh.read()).decode()
        prompt = (
            f"This figure is from the research paper '{f['file_name']}', page {f['page']}. "
            f"Its caption is: '{f['caption'] or 'not found'}'.\n"
            "In 2-4 sentences, describe what the figure shows: the type of figure "
            "(photo, plot, diagram, table), what is pictured or plotted, axis titles and key "
            "labels, and any clearly visible trend. Describe only what is visible or stated "
            "in the caption; do not guess results."
        )
        r = client.chat.completions.create(
            model=os.getenv("CHAT_DEPLOYMENT"),
            temperature=0,
            max_tokens=300,
            messages=[{"role": "user", "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}},
            ]}],
        )
        f["description"] = r.choices[0].message.content.strip()
        save_figs(figs)  # save after each one, so nothing is lost if it stops
        print(f"[{i}/{len(todo)}] {f['image_path']}")
    print("Done. Now delete the storage/ folder and run the app to rebuild the index.")


if __name__ == "__main__":
    if len(sys.argv) < 2 or sys.argv[1] not in ("extract", "describe"):
        print("Usage: python build_figures.py extract | describe")
    elif sys.argv[1] == "extract":
        extract()
    else:
        describe()