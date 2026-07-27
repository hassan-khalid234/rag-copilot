from typing import List

def chunk_text_sliding_window(text: str, chunk_size: int = 500, overlap: int = 50) -> List[str]:
    """
    Splits text into fixed-size chunks with a configurable overlap window.
    """
    if not text:
        return []
    
    chunks = []
    start = 0
    text_len = len(text)
    
    while start < text_len:
        end = start + chunk_size
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        start += (chunk_size - overlap)
        
    return chunks