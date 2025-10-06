# main.py — minimal runners (exactly one simple path for each backend)
# Usage:
#   python main.py --backend obsidian --image <path> "prompt"
#   python main.py --backend moondream --image <path> --maxtokens <num> "prompt"
#
# Requirements:
#   pip install pillow
#   For Obsidian: pip install llama-cpp-python
#                 export OBS_GGUF=/abs/path/to/obsidian-*.gguf
#                 export OBS_MMPROJ=/abs/path/to/mmproj-obsidian-f16.gguf
#                 [optional] export OBS_N_GPU_LAYERS=-1  # defaults to -1 for NVIDIA offload
#   For Moondream: pip install torch transformers safetensors, etc. from readme

import os, sys
from PIL import Image

def usage():
    print("Usage: python main.py --backend {obsidian|moondream} --image <path> --maxtokens <num> [prompt]")
    raise SystemExit(1)

def run_obsidian(image_path: str, prompt: str):
    from llama_cpp import Llama
    from llama_cpp.llama_chat_format import Llava15ChatHandler

    gguf = os.environ.get("OBS_GGUF")
    mmproj = os.environ.get("OBS_MMPROJ")
    if not gguf or not mmproj:
        print("Set OBS_GGUF and OBS_MMPROJ to your Obsidian *.gguf and projector *.gguf files.")
        raise SystemExit(2)

    chat_handler = Llava15ChatHandler(clip_model_path=mmproj)

    n_gpu_layers_env = os.environ.get("OBS_N_GPU_LAYERS")
    if n_gpu_layers_env is not None:
        try:
            n_gpu_layers = int(n_gpu_layers_env)
        except ValueError:
            raise SystemExit("OBS_N_GPU_LAYERS must be an integer")
    else:
        n_gpu_layers = -1  # request full GPU offload when llama.cpp was built with CUDA

    llama_kwargs = dict(
        model_path=gguf,
        chat_handler=chat_handler,
        n_ctx=2048,          # per docs: increase context so the prompt fits
        n_gpu_layers=n_gpu_layers,
        verbose=False,
    )

    try:
        llm = Llama(**llama_kwargs)
    except ValueError as err:
        err_text = str(err).lower()
        if n_gpu_layers != 0 and ("cuda" in err_text or "cublas" in err_text or "gpu" in err_text):
            print("Warning: llama.cpp was built without GPU support; retrying on CPU.")
            llama_kwargs["n_gpu_layers"] = 0
            llm = Llama(**llama_kwargs)
        else:
            raise

    messages = [{
        "role": "user",
        "content": [
            {"type": "text", "text": prompt or "Describe this image."},
            {"type": "image_url", "image_url": {"url": f"file://{os.path.abspath(image_path)}"}},
        ],
    }]
    out = llm.create_chat_completion(messages=messages, max_tokens=64, temperature=0.0)
    print(out["choices"][0]["message"]["content"].strip())

def run_moondream(image_path: str, prompt: str, max_tokens: str):
    import torch
    from transformers import AutoModelForCausalLM


    # bnb = BitsAndBytesConfig(
    #     load_in_4bit=True,
    #     bnb_4bit_use_double_quant=True,
    #     bnb_4bit_quant_type="nf4",
    #     bnb_4bit_compute_dtype=torch.float16,
    # )
    img = Image.open(image_path).convert("RGB")
    model = AutoModelForCausalLM.from_pretrained(
        "vikhyatk/moondream2",
        trust_remote_code=True,
        dtype=(torch.float16 if torch.cuda.is_available() else torch.float32),
        device_map="cuda"
    ).eval()
    settings = {"max_tokens": max_tokens}
    # print(max_tokens)

    if prompt and prompt.strip():
        out = model.query(img, prompt.strip(), settings)     # minimal/fast
        print(out.get("answer", "").strip())
    else:
        #print("hello")
        for t in model.caption(img, length="short", stream=True )["caption"]:
            print(t, end="", flush=True)
        print("\n")
        
        # out = model.caption(img, settings, length="short")
        # print(out.get("caption", "").strip())

def main():
    if len(sys.argv) < 5 or sys.argv[1] != "--backend" or sys.argv[3] != "--image" or sys.argv[5] != "--maxtokens":
        usage()
    #print(sys.argv)
    backend = sys.argv[2].lower()
    image_path = sys.argv[4]
    max_tokens = sys.argv[6]
    prompt = " ".join(sys.argv[7:])

    # quick file check
    try:
        Image.open(image_path).close()
    except Exception as e:
        raise SystemExit(f"Bad --image path: {e}")

    if backend == "obsidian":
        run_obsidian(image_path, prompt)
    elif backend == "moondream":
        run_moondream(image_path, prompt, max_tokens)
    else:
        usage()

if __name__ == "__main__":
    main()
