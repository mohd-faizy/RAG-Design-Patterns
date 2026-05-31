from pathlib import Path


# --------------------------------
# LOAD RAW DOCUMENTS
# --------------------------------
def load_documents():
    path = Path("data/knowledge.txt")
    text = path.read_text()

    chunks = text.split("\n")
    chunks = [
        c.strip()
        for c in chunks
        if c.strip()
    ]

    return chunks
