# main.py — Moondream2 with an explicit VRAM cap (Jetson-safe)

import sys, os
from PIL import Image
import torch
from transformers import AutoModelForCausalLM

MODEL_ID = "vikhyatk/moondream2"

def downscale_max_dim(img: Image.Image, max_dim: int) -> Image.Image:
    if max_dim <= 0:
        return img
    w, h = img.size
    if max(w, h) <= max_dim:
        return img
    if w >= h:
        new_w, new_h = max_dim, int(h * (max_dim / w))
    else:
        new_h, new_w = max_dim, int(w * (max_dim / h))
    return img.resize((new_w, new_h), Image.BICUBIC)

def main():
    if len(sys.argv) < 2:
        print("Usage: python main.py <image_path> [optional prompt]")
        raise SystemExit(1)

    image_path = sys.argv[1]
    prompt = " ".join(sys.argv[2:]) if len(sys.argv) > 2 else None

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32

    # ---- VRAM cap settings ----
    # How much VRAM to allow this *process* to use (GiB). Default <8 to leave headroom.
    vram_gb = float(os.environ.get("VLM_VRAM_GB", "7.5"))  # change to "7.8" if you want tighter/looser
    max_dim = int(os.environ.get("VLM_MAX_IMAGE_DIM", "1024"))  # optional image downscale cap

    model_kwargs = dict(
        trust_remote_code=True,
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
    )

    if device == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.benchmark = True

        # Soft cap the CUDA caching allocator to a fraction of total VRAM
        total = torch.cuda.get_device_properties(0).total_memory  # bytes
        cap_bytes = int(vram_gb * (1024 ** 3))
        frac = min(0.98, max(0.05, cap_bytes / total))
        torch.cuda.set_per_process_memory_fraction(frac, 0)

        # Also tell 🤗 Accelerate/Transformers not to exceed this cap when placing weights
        # NOTE: when using device_map/max_memory, DO NOT call .to("cuda") on the model.
        model_kwargs.update(
            device_map="auto",
            max_memory={0: f"{vram_gb}GiB", "cpu": "8GiB"},
        )

    # ---- Load model ----
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, **model_kwargs).eval()

    # ---- Load & optionally downscale image (saves VRAM/compute) ----
    image = Image.open(image_path).convert("RGB")
    image = downscale_max_dim(image, max_dim)

    # ---- Inference ----
    try:
        with torch.inference_mode():
            if prompt:
                out = model.query(image, prompt)   # {"answer": "..."}
                text = out.get("answer", "")
            else:
                out = model.caption(image, length="normal")  # {"caption": "..."}
                text = out.get("caption", "")
        print(text.strip())
    except RuntimeError as e:
        msg = str(e)
        if "CUDA out of memory" in msg:
            print(
                "[OOM] CUDA ran out of memory.\n"
                f"- Current cap: {vram_gb} GiB (env VLM_VRAM_GB).\n"
                "- Try lowering VLM_VRAM_GB, reducing VLM_MAX_IMAGE_DIM (e.g., 768 or 640),\n"
                "- closing other GPU apps, or using a shorter caption length.\n"
            )
        raise

    # ---- Quick memory summary (optional) ----
    if device == "cuda":
        torch.cuda.synchronize()
        alloc = torch.cuda.memory_allocated(0) / (1024 ** 2)
        reserved = torch.cuda.memory_reserved(0) / (1024 ** 2)
        print(f"[MEM] allocated={alloc:.1f} MiB, reserved={reserved:.1f} MiB (cap ~{vram_gb} GiB)")

if __name__ == "__main__":
    main()

