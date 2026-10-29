"""Turn the full size place pictures into the app's library: places/<id>.jpg (what the app uses) and places/thumbs/<id>.jpg (the tile)."""
import sys
from pathlib import Path
from PIL import Image
src, app = Path(sys.argv[1]), Path(__file__).resolve().parent
for f in sorted(src.glob("place_*.png")):
    pid = f.stem.removeprefix("place_")
    im = Image.open(f).convert("RGB")
    im.save(app / "places" / f"{pid}.jpg", quality=92)
    s = min(im.size); x = (im.width - s) // 2
    im.crop((x, 0, x + s, s)).resize((256, 256), Image.LANCZOS).save(app / "places" / "thumbs" / f"{pid}.jpg", quality=85)
    print(pid)
