"""Uploads sample files to the app (Flask test client) and prints verdicts and reasons."""
import html as H
import re
import time

import app

client = app.app.test_client()
tests = [("test_files/tampered_receipt.pdf", "tampered_receipt.pdf"), ("test_files/two_pages.pdf", "two_pages.pdf"),
         ("test_files/receipt_in_word.docx", "receipt_in_word.docx"), ("test_files/text_invoice.docx", "text_invoice.docx"),
         ("static/uploads/sample_medical_template.png", "medical_template.png"), ("dataset/test/X00016469619.png", "genuine_receipt.png"),
         ("test_files/receipt_in_slides.pptx", "receipt_in_slides.pptx"), ("test_files/text_invoice_slides.pptx", "text_invoice_slides.pptx"),
         ("test_files/review_ppt.pptx", "review_ppt.pptx"), ("dataset/test/X51008099081.png", "tampered_detected.png"),
         ("test_files/digital_invoice.pdf", "digital_invoice.pdf"), ("test_files/digital_invoice_edited.pdf", "digital_invoice_edited.pdf"),
         ("test_files/text_invoice_modified.docx", "text_invoice_modified.docx")]
for path, name in tests:
    start = time.time()
    with open(path, "rb") as f:
        r = client.post("/predict", data={"document": (f, name)}, content_type="multipart/form-data")
    page = r.data.decode()
    verdict = re.search(r"<h2>\s*(Genuine|Tampered|Signs of editing found|No signs of editing found)\s*</h2>", page)
    error = re.search(r'class="error-message">\s*(.*?)\s*</div>', page, re.S)
    reasons = [H.unescape(re.sub(r"\s+", " ", x)).strip() for x in re.findall(r"<li>(.*?)</li>", page, re.S)]
    print(f"\n== {name}: HTTP {r.status_code}, {time.time() - start:.1f}s, verdict {verdict.group(1) if verdict else None}, "
          f"error {error.group(1) if error else None}", flush=True)
    for x in reasons[:4]:
        print("   -", x[:190], flush=True)
    print("   zoomed region shown:", "MOST SUSPICIOUS REGION" in page, "| type warning:", "DOCUMENT TYPE WARNING" in page, flush=True)
