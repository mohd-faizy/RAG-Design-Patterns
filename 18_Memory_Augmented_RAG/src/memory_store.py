import json
from pathlib import Path

MEMORY_PATH = "memory/memory.json"

def load_memory() -> list[str]:
    """Loads the persistent memory list from disk."""
    path = Path(MEMORY_PATH)
    if not path.exists():
        # Bootstrap empty memory store if file is missing
        path.parent.mkdir(parents=True, exist_ok=True)
        return []
    with open(MEMORY_PATH, "r") as f:
        return json.load(f)

def save_memory(memories: list[str]):
    """Persists the memory list to disk as JSON."""
    Path(MEMORY_PATH).parent.mkdir(parents=True, exist_ok=True)
    with open(MEMORY_PATH, "w") as f:
        json.dump(memories, f, indent=2)
