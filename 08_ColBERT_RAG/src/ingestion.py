from pathlib import Path


# --------------------------------
# LOAD RAW DOCUMENTS
# --------------------------------
def load_documents():
    path = Path(__file__).resolve().parent.parent.parent / "_data" / "source.txt"
    text = path.read_text()

    chunks = text.split("\n")
    chunks = [
        c.strip()
        for c in chunks
        if c.strip()
    ]

    return chunks
