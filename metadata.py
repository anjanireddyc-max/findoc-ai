"""Checks the file itself (not the pixels) for signs of editing.

Digital documents - typed Word/PowerPoint files and PDFs with a text layer - carry
no scanning or pixel traces, so the image model cannot judge them. What they do
carry is metadata: which program created and last saved them, when, by whom, and
(for PDFs) whether the file was saved again after it was first created.
"""
import io
import re
import zipfile
from datetime import datetime

# PDF tools commonly used to edit existing PDFs (seen in the Creator/Producer fields)
PDF_EDITORS = ["ilovepdf", "smallpdf", "sejda", "pdfescape", "pdf-xchange", "foxit phantompdf", "foxit pdf editor",
               "nitro", "pdfelement", "wondershare", "inkscape", "pdffiller", "dochub", "pdf candy", "soda pdf",
               "libreoffice draw", "acrobat pro", "adobe acrobat"]


def _pdf_date(value):
    m = re.match(r"D:(\d{4})(\d{2})(\d{2})(\d{2})?(\d{2})?(\d{2})?", value or "")
    if not m:
        return None
    parts = [int(p) if p else 0 for p in m.groups()]
    try:
        return datetime(*parts)
    except ValueError:
        return None


def _office_date(value):
    try:
        return datetime.strptime(value[:19], "%Y-%m-%dT%H:%M:%S")
    except (TypeError, ValueError):
        return None


def inspect(data, ext, used_embedded_pictures=False):
    """Returns {"digital": bool, "kind": str, "findings": [str], "signals": int}.

    digital  - True when the pages were rendered from a digital file (no scan to analyse)
    signals  - number of findings that indicate the file was edited after creation
    """
    if ext == "pdf":
        return _inspect_pdf(data)
    if ext in ("docx", "pptx"):
        return _inspect_office(data, ext, used_embedded_pictures)
    if ext in ("doc", "ppt"):
        return {"digital": not used_embedded_pictures, "kind": f"legacy {ext.upper()} file",
                "findings": ["Metadata of legacy .doc/.ppt files is not checked; save the file as DOCX/PPTX or PDF for a metadata check."],
                "signals": 0}
    return _inspect_image(data)


IMAGE_EDITORS = ["photoshop", "gimp", "canva", "picsart", "snapseed", "lightroom", "paint.net", "pixlr",
                 "affinity", "fotor", "photopea", "inshot", "polarr", "illustrator", "coreldraw", "krita"]


def _inspect_image(data):
    """EXIF / XMP / PNG text information of an uploaded image."""
    from PIL import Image
    findings, signals = [], 0
    try:
        image = Image.open(io.BytesIO(data))
    except Exception:
        return {"digital": False, "kind": "image", "findings": [], "signals": 0}
    texts = []
    try:
        exif = image.getexif()
        for tag in (305, 271, 272, 306):                       # Software, Make, Model, DateTime
            if exif.get(tag):
                texts.append(str(exif.get(tag)))
        software = str(exif.get(305, "") or "")
    except Exception:
        software = ""
    for key, value in (image.info or {}).items():
        if isinstance(value, (str, bytes)) and key.lower() in ("software", "comment", "description", "xml:com.adobe.xmp"):
            texts.append(value.decode("utf-8", "ignore") if isinstance(value, bytes) else value)
            if key.lower() == "software" and not software:
                software = value if isinstance(value, str) else value.decode("utf-8", "ignore")
    xmp = data[:400000].decode("latin-1", "ignore")
    blob = " ".join(texts + [xmp[xmp.find("<x:xmpmeta"):xmp.find("</x:xmpmeta>")] if "<x:xmpmeta" in xmp else ""]).lower()

    if software:
        findings.append(f"The image was last saved by: {software.strip()}.")
    editor = next((e for e in IMAGE_EDITORS if e in blob), None)
    if editor:
        signals += 1
        findings.append(f"The image metadata shows it was processed with image-editing software ({editor}). "
                        "This is common for edited images, but also for designed templates.")
    if "xmpmm:history" in blob or "photoshop:history" in blob:
        signals += 1
        findings.append("The image contains an editing history (XMP history), so it was opened and saved in an editor.")
    if not findings:
        findings.append("The image carries no editing-software information in its metadata.")
    return {"digital": False, "kind": "image", "findings": findings, "signals": signals}


def _inspect_pdf(data):
    import pymupdf
    findings, signals = [], 0
    try:
        pdf = pymupdf.open(stream=data, filetype="pdf")
    except Exception:
        return {"digital": False, "kind": "PDF", "findings": [], "signals": 0}

    text_chars = sum(len(pdf[i].get_text().strip()) for i in range(min(pdf.page_count, 5)))
    digital = text_chars > 30
    meta = pdf.metadata or {}
    creator, producer = meta.get("creator", "") or "", meta.get("producer", "") or ""
    created, modified = _pdf_date(meta.get("creationDate")), _pdf_date(meta.get("modDate"))

    tools = ", ".join(t for t in [creator, producer] if t) or "not recorded"
    findings.append(f"Created with: {tools}.")
    if created and modified and (modified - created).total_seconds() > 60:
        signals += 1
        findings.append(f"The PDF was modified after it was created (created {created:%d %b %Y %H:%M}, "
                        f"modified {modified:%d %b %Y %H:%M}).")
    saves = data.count(b"%%EOF")
    if saves > 1:
        signals += 1
        findings.append(f"The PDF file was saved {saves} times (incremental updates). This happens when an "
                        "existing PDF is changed and saved again.")
    editor = next((e for e in PDF_EDITORS if e in (creator + " " + producer).lower()), None)
    if editor:
        signals += 1
        findings.append(f"The PDF was processed with a PDF editing tool ({editor}).")
    if not digital:
        findings.append("The pages contain no text layer, so they are treated as scanned images and analysed by the model.")
    return {"digital": digital, "kind": "digital PDF" if digital else "scanned PDF", "findings": findings, "signals": signals}


def _inspect_office(data, ext, used_embedded_pictures):
    findings, signals = [], 0
    kind = "Word document" if ext == "docx" else "PowerPoint presentation"
    try:
        with zipfile.ZipFile(io.BytesIO(data)) as z:
            core = z.read("docProps/core.xml").decode("utf-8", "ignore") if "docProps/core.xml" in z.namelist() else ""
            app = z.read("docProps/app.xml").decode("utf-8", "ignore") if "docProps/app.xml" in z.namelist() else ""
    except zipfile.BadZipFile:
        return {"digital": True, "kind": kind, "findings": [], "signals": 0}

    def tag(xml, name):
        m = re.search(rf"<(?:\w+:)?{name}[^>]*>([^<]*)</", xml)
        return m.group(1).strip() if m else ""

    creator, last_by = tag(core, "creator"), tag(core, "lastModifiedBy")
    created, modified = _office_date(tag(core, "created")), _office_date(tag(core, "modified"))
    revision, application = tag(core, "revision"), tag(app, "Application")

    findings.append(f"Created with: {application or 'not recorded'}; author: {creator or 'not recorded'}.")
    if last_by and creator and last_by != creator:
        signals += 1
        findings.append(f"Last modified by a different person ({last_by}) than the author ({creator}).")
    if created and modified and (modified - created).total_seconds() > 3600:
        signals += 1
        findings.append(f"The file was modified after it was created (created {created:%d %b %Y %H:%M}, "
                        f"modified {modified:%d %b %Y %H:%M}).")
    if revision.isdigit() and int(revision) > 1:
        findings.append(f"The file has been saved {revision} times (revision number).")
    if used_embedded_pictures:
        findings.append("The pictures inside the file were extracted and analysed by the image model.")
    return {"digital": not used_embedded_pictures, "kind": kind, "findings": findings, "signals": signals}
