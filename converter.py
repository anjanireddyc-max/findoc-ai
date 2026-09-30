"""Turns an uploaded file into page images for the detector.

Supported: PNG, JPG/JPEG, BMP, TIFF, WEBP images, PDF files, Word documents
(DOCX/DOC) and PowerPoint presentations (PPTX/PPT). PDF pages are rendered at
200 dpi, close to the resolution of the scanned training receipts. Pictures
embedded in a DOCX/PPTX (for example a pasted scan) are analysed directly;
otherwise the file is exported to PDF with Microsoft Word / PowerPoint, opened
read-only so that it also works when Office is not activated.
"""
import io
import os
import tempfile
import zipfile

from PIL import Image, ImageOps

IMAGE_EXTENSIONS = {"png", "jpg", "jpeg", "bmp", "tif", "tiff", "webp"}
DOCUMENT_EXTENSIONS = {"pdf", "docx", "doc", "pptx", "ppt"}
ALLOWED_EXTENSIONS = IMAGE_EXTENSIONS | DOCUMENT_EXTENSIONS

MAX_PAGES = 5
PDF_DPI = 200


class ConversionError(Exception):
    pass


def extension(filename):
    return filename.rsplit(".", 1)[-1].lower() if "." in filename else ""


def load_pages(data, filename):
    """Returns a list of (page_label, PIL image) for the uploaded bytes."""
    ext = extension(filename)
    if ext in IMAGE_EXTENSIONS:
        return [("Image", _open_image(data))]
    if ext == "pdf":
        return _pdf_pages(data)
    if ext in ("docx", "doc"):
        return _word_pages(data, ext)
    if ext in ("pptx", "ppt"):
        return _powerpoint_pages(data, ext)
    raise ConversionError("Unsupported file type.")


def _open_image(data):
    try:
        image = Image.open(io.BytesIO(data))
        image.seek(0)                                   # first frame of multi-page TIFFs
        return ImageOps.exif_transpose(image).convert("RGB")   # phone photos the right way up
    except Exception as exc:
        raise ConversionError("The image could not be read.") from exc


def _pdf_pages(data):
    import pymupdf
    try:
        pdf = pymupdf.open(stream=data, filetype="pdf")
    except Exception as exc:
        raise ConversionError("The PDF could not be opened.") from exc
    if pdf.page_count == 0:
        raise ConversionError("The PDF has no pages.")
    pages = []
    for number in range(min(pdf.page_count, MAX_PAGES)):
        pix = pdf[number].get_pixmap(dpi=PDF_DPI, alpha=False)
        pages.append((f"Page {number + 1}", Image.frombytes("RGB", (pix.width, pix.height), pix.samples)))
    return pages


def _word_pages(data, ext):
    if ext == "docx":                                   # pictures inside the document (e.g. a pasted scan)
        pictures = _docx_pictures(data)
        if pictures:
            return pictures
    pdf_bytes = _word_to_pdf(data, ext)                 # otherwise render the pages with Microsoft Word
    if pdf_bytes:
        return _pdf_pages(pdf_bytes)
    return [("Page 1", None)]                           # could not render: metadata check only


WORD_TIMEOUT = 60
_WORD_SCRIPT = r"""
import sys, pythoncom, win32com.client
pythoncom.CoInitialize()
word = win32com.client.DispatchEx("Word.Application")
word.Visible = False
word.DisplayAlerts = 0
try:
    word.AutomationSecurity = 3                      # never run macros
    doc = word.Documents.Open(sys.argv[1], False, True, False)   # ConfirmConversions, ReadOnly, AddToRecentFiles
    doc.ExportAsFixedFormat(sys.argv[2], 17)         # 17 = PDF
    doc.Close(0)
finally:
    word.Quit()
"""


def _powerpoint_pages(data, ext):
    if ext == "pptx":                                   # pictures on the slides (e.g. a pasted scan)
        pictures = _docx_pictures(data, "ppt/media/")
        if pictures:
            return pictures
    pdf_bytes = _office_to_pdf(data, ext, _POWERPOINT_SCRIPT, "POWERPNT.EXE")
    if pdf_bytes:
        return _pdf_pages(pdf_bytes)
    return [("Page 1", None)]                           # could not render: metadata check only


_POWERPOINT_SCRIPT = r"""
import sys, pythoncom, win32com.client
pythoncom.CoInitialize()
app = win32com.client.DispatchEx("PowerPoint.Application")
try:
    app.DisplayAlerts = 1                            # ppAlertsNone
    deck = app.Presentations.Open(sys.argv[1], True, False, False)   # ReadOnly, Untitled, WithWindow
    deck.SaveAs(sys.argv[2], 32)                     # 32 = ppSaveAsPDF
    deck.Close()
finally:
    app.Quit()
"""


def _office_pids(image_name):
    import subprocess
    out = subprocess.run(["tasklist", "/FI", f"IMAGENAME eq {image_name}", "/FO", "CSV", "/NH"],
                         capture_output=True, text=True).stdout
    return {int(line.split('","')[1]) for line in out.splitlines() if line.upper().startswith('"' + image_name.upper())}


def _word_to_pdf(data, ext):
    return _office_to_pdf(data, ext, _WORD_SCRIPT, "WINWORD.EXE")


def _office_to_pdf(data, ext, script, image_name):
    """Exports an Office file to PDF in a separate process with a time limit.
    If Office hangs, only the instance started for this conversion is closed."""
    import subprocess
    import sys
    if os.name != "nt":
        return None
    folder = tempfile.mkdtemp()
    source, target = os.path.join(folder, "upload." + ext), os.path.join(folder, "upload.pdf")
    with open(source, "wb") as f:
        f.write(data)
    before = _office_pids(image_name)
    try:
        subprocess.run([sys.executable, "-c", script, source, target], timeout=WORD_TIMEOUT,
                       capture_output=True)
    except subprocess.TimeoutExpired:
        pass
    finally:
        for pid in _office_pids(image_name) - before:
            subprocess.run(["taskkill", "/PID", str(pid), "/F"], capture_output=True)
    if os.path.exists(target):
        with open(target, "rb") as f:
            return f.read()
    return None


def _docx_pictures(data, folder="word/media/"):
    pages = []
    try:
        with zipfile.ZipFile(io.BytesIO(data)) as docx:
            for name in sorted(n for n in docx.namelist() if n.startswith(folder)):
                try:
                    image = Image.open(io.BytesIO(docx.read(name))).convert("RGB")
                except Exception:
                    continue
                if min(image.size) >= 300:          # skip logos and icons
                    pages.append((f"Picture {len(pages) + 1}", image))
                if len(pages) == MAX_PAGES:
                    break
    except zipfile.BadZipFile:
        return []
    return pages
