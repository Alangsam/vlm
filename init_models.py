#!/usr/bin/env python3
"""
init_models.py — minimal, cross-platform model fetcher

What it does:
- Creates a .models/ folder
- (Optional) Prefetches Moondream2 to .models/moondream2
- Downloads Obsidian GGUF + projector to .models/obsidian
- Prints the environment variables you should export for your runner

Usage examples:
  python init_models.py --moondream
  python init_models.py --obsidian
  python init_models.py --obsidian --quant q6
  python init_models.py --all
"""

import argparse
import os
import shutil
from pathlib import Path

try:
    from huggingface_hub import snapshot_download, hf_hub_download
except Exception as e:
    raise SystemExit(
        "Missing dependency: huggingface_hub\n"
        "Install it with:  python -m pip install -U huggingface_hub\n"
    )

ROOT = Path(__file__).resolve().parent
MODELS_DIR = ROOT / ".models"
MD_OBSIDIAN = MODELS_DIR / "obsidian"
MD_MOONDREAM = MODELS_DIR / "moondream2"

def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def size_mb(p: Path) -> float:
    try:
        return p.stat().st_size / (1024 * 1024)
    except FileNotFoundError:
        return 0.0

def fetch_moondream():
    """
    Prefetch the full Moondream2 repo locally (optional).
    You could also skip this and let transformers download on first run.
    """
    ensure_dir(MD_MOONDREAM)
    print("[Moondream2] Downloading to", MD_MOONDREAM)
    snapshot_download(
        repo_id="vikhyatk/moondream2",
        local_dir=str(MD_MOONDREAM),
        local_dir_use_symlinks=False,
        resume_download=True,
    )
    print("[Moondream2] Ready at", MD_MOONDREAM)

def fetch_obsidian(quant: str = "q6"):
    """
    Download Obsidian-3B for llama.cpp:
    - GGUF model (quantized): from a community repo with the chosen quant level
    - Projector (mmproj): from the official GGUF repo

    quant options commonly available: q5, q6 (q6 uses more VRAM but is higher quality)
    """
    ensure_dir(MD_OBSIDIAN)

    # Choose GGUF source + filename based on quant
    if quant.lower() == "q6":
        gguf_repo = "nisten/obsidian-3b-multimodal-q6-gguf"
        gguf_filename = "obsidian-q6.gguf"
    elif quant.lower() == "q5":
        gguf_repo = "nisten/obsidian-3b-multimodal-q5-gguf"
        gguf_filename = "obsidian-q5.gguf"
    else:
        raise SystemExit(f"Unsupported quant '{quant}'. Try q6 or q5.")

    # Download GGUF
    print(f"[Obsidian] Downloading GGUF ({quant}) …")
    gguf_src = hf_hub_download(repo_id=gguf_repo, filename=gguf_filename)
    gguf_dst = MD_OBSIDIAN / gguf_filename
    shutil.copy(gguf_src, gguf_dst)
    print(f"[Obsidian] GGUF → {gguf_dst} ({size_mb(gguf_dst):.1f} MB)")

    # Download projector (mmproj) from official GGUF repo
    print("[Obsidian] Downloading projector (mmproj) …")
    mmproj_src = hf_hub_download(
        repo_id="NousResearch/Obsidian-3B-V0.5-GGUF",
        filename="mmproj-obsidian-f16.gguf",
    )
    mmproj_dst = MD_OBSIDIAN / "mmproj-obsidian-f16.gguf"
    shutil.copy(mmproj_src, mmproj_dst)
    print(f"[Obsidian] mmproj → {mmproj_dst} ({size_mb(mmproj_dst):.1f} MB)")

    # Print env exports the runner expects
    print("\n[Obsidian] Set these environment variables before running model 2:")
    print(f'  export OBS_GGUF="{gguf_dst}"')
    print(f'  export OBS_MMPROJ="{mmproj_dst}"')
    print("Optional (fits 8 GB better):")
    print('  export VLM_CTX=1024      # context length (lower uses less memory)')
    print('  export VLM_NGL=24        # layers on GPU (lower uses less GPU memory)')
    print('  export VLM_MAX_NEW=64    # max generated tokens')
    print('  export VLM_MAX_IMAGE_DIM=768  # downscale the image\n')

def main():
    ap = argparse.ArgumentParser(description="Minimal model initializer")
    ap.add_argument("--moondream", action="store_true", help="Prefetch Moondream2 locally")
    ap.add_argument("--obsidian", action="store_true", help="Download Obsidian GGUF + projector")
    ap.add_argument("--quant", default="q6", help="Obsidian quant (q6 or q5)")
    ap.add_argument("--all", action="store_true", help="Do both")
    args = ap.parse_args()

    ensure_dir(MODELS_DIR)

    if args.all or args.moondream:
        fetch_moondream()

    if args.all or args.obsidian:
        fetch_obsidian(args.quant)

    if not (args.all or args.moondream or args.obsidian):
        ap.print_help()

if __name__ == "__main__":
    main()
