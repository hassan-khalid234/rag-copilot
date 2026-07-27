import fitz  # PyMuPDF
import os

def extract_text_from_pdf(pdf_path: str) -> str:
    """Extracts clean text from a target PDF file."""
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"PDF file not found at: {pdf_path}")
    
    doc = fitz.open(pdf_path)
    full_text = ""
    for page in doc:
        text = page.get_text().strip()
        if text:
            full_text += text + "\n"
            
    return full_text.strip()