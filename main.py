# main.py — ultra-minimal Moondream2 runner (GPU if available, works on ARM/Jetson)

import sys
from PIL import Image
import torch
from transformers import AutoModelForCausalLM

MODEL_ID = "vikhyatk/moondream2"

def main():
    if len(sys.argv) < 2:
        print("Usage: python main.py <image_path> [optional freeform prompt]")
        raise SystemExit(1)

    image_path = sys.argv[1]
    # prompt is optional with Moondream's caption() API; it can caption with just an image
    prompt = " ".join(sys.argv[2:]) if len(sys.argv) > 2 else None

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32

    # Moondream exposes convenience methods (caption/query/detect/point) via remote code
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        trust_remote_code=True,
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
    ).to(device).eval()

    image = Image.open(image_path).convert("RGB")

    with torch.inference_mode():
        if prompt:
            # Visual Q&A (freeform question about the image)
            out = model.query(image, prompt)   # returns {"answer": "..."}
            print(out.get("answer", "").strip())
        else:
            # Plain captioning (no prompt needed)
            out = model.caption(image, length="normal")  # returns {"caption": "..."}
            print(out.get("caption", "").strip())

if __name__ == "__main__":
    main()

