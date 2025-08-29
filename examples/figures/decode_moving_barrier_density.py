import base64
from pathlib import Path

for b64_path in Path(__file__).parent.glob('*.base64'):
    png_path = b64_path.with_suffix('.png')
    png_path.write_bytes(base64.b64decode(b64_path.read_text()))
    print(f'Decoded image written to {png_path}')
