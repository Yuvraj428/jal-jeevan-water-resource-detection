
import sys
import base64
from pathlib import Path

def main():
    if len(sys.argv) < 2:
        print("Usage: python encode_image.py <path_to_image>")
        sys.exit(1)

    img_path = Path(sys.argv[1])
    if not img_path.exists():
        print(f"File not found: {img_path}")
        sys.exit(1)

    with open(img_path, "rb") as f:
        encoded = base64.b64encode(f.read()).decode("utf-8")

    out_path = Path("image_base64.txt")
    out_path.write_text(encoded)

    print(f"Base64 string saved to: {out_path.resolve()}")
    print(f"String length: {len(encoded)} characters")

if __name__ == "__main__":
    main()
