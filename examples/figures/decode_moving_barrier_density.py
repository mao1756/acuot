import base64
from pathlib import Path

b64_path = Path(__file__).with_name('moving_barrier_density_with_maze_snapshots.base64')
png_path = Path(__file__).with_name('moving_barrier_density_with_maze_snapshots.png')

png_path.write_bytes(base64.b64decode(b64_path.read_text()))
print(f'Decoded image written to {png_path}')
