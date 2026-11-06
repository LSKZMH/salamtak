"""app.py builds its detector at import; keep tests from loading the YOLOE model."""
import os

os.environ.setdefault("SAFESITE_OPEN_VOCAB", "0")
