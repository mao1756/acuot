"""Utility to regenerate PNG figures from their base64 encodings."""

import base64
from pathlib import Path

for b64_path in Path(__file__).parent.glob("*.base64"):
    png_path = b64_path.with_suffix(".png")
    png_path.write_bytes(base64.b64decode(b64_path.read_text()))
    print(f"Decoded {b64_path.name} -> {png_path.name}")
