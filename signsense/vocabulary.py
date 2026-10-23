"""Loads the fixed sign vocabulary from vocabulary.json."""
import json
from typing import Dict, List

from . import config

_CATEGORY_NAMES = {"alphabets": "alphabet", "numbers": "number", "words": "word"}


def load_vocabulary() -> List[Dict[str, str]]:
    """Return [{'label': 'A', 'category': 'alphabet'}, ...] in file order."""
    with open(config.VOCAB_PATH, "r", encoding="utf-8") as fh:
        raw = json.load(fh)

    vocab: List[Dict[str, str]] = []
    seen = set()
    for key, category in _CATEGORY_NAMES.items():
        for label in raw.get(key, []):
            label = str(label).strip()
            if not label:
                continue
            if label in seen:
                raise ValueError(f"Duplicate sign '{label}' in vocabulary.json")
            seen.add(label)
            vocab.append({"label": label, "category": category})
    if len(vocab) < 2:
        raise ValueError("vocabulary.json must list at least 2 signs.")
    return vocab
