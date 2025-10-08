# main.py — minimal runners (exactly one simple path for each backend)
# Usage:
#   python main.py --backend obsidian --image <path> "prompt"
#   python main.py --backend moondream --image <path> --maxtokens <num> "prompt"
# New backends added: llava(tiny 0.5B for now)
#   python main.py --backend llava --image <path> --prompt --maxtokens
#
# Requirements:
#   pip install pillow
#   For Obsidian: pip install llama-cpp-python
#                 export OBS_GGUF=/abs/path/to/obsidian-*.gguf
#                 export OBS_MMPROJ=/abs/path/to/mmproj-obsidian-f16.gguf
#                 [optional] export OBS_N_GPU_LAYERS=-1  # defaults to -1 for NVIDIA offload
#   For Moondream: pip install torch transformers safetensors, etc. from readme

import os, sys
import argparse
import torch

from PIL import Image
from transformers import AutoProcessor


def usage():
    print("Usage: python main.py --backend {obsidian|moondream|llava} --image <path> --maxtokens <num> [prompt]")
    raise SystemExit(1)

def run_model_with_cpp(model: str, image_path: str, prompt: str, maxtokens:str):
    from llama_cpp import Llama
    from llama_cpp.llama_chat_format import Llava15ChatHandler
    #logic to enumerate .models folder to check if model requested has already been downloaded, and if not
    #show available models, and explain how to use ini_models
    
    models = []
    for name in os.listdir("./.models"):
        models.append(name)
    if model not in models:
        print(f'\n{model} not found. Check for typos\n')
        print(f'You wrote: {model}, but we only found: {models}\n')
        print(f'If you need to download a model\'s files, use init_models.py')
        raise SystemExit(1)
    # prompt user to then choose which model file is the actual modelfile(params+weights)
    # and which is the actual projector file 
    # (helps because user may want multiple versions of a model for testing)
    print("Please select which of the following is the model file, and which is projector file:")
    model_files = [name for name in os.listdir(f'./.models/{model}')]
    for idx, name in enumerate(model_files):
        print(f'[{idx}] {name}')
    
    def ask(label: str) -> str:
        while True:
            try:
                choice = int(input(f'Enter the index for the {label}: ').strip())
                return f'./.models/{model}/{model_files[choice]}'
            except (ValueError, IndexError):
                print("Invalid selection, please try again")
    
    model_path = ask("model file")
    projector_path = ask("projector file")
    #print(model_path,projector_path)
    chat_handler = Llava15ChatHandler(clip_model_path=projector_path)
    llm = Llama(
        model_path=model_path,
        n_gpu_layers=-1,
        chat_handler=chat_handler,
        # seed=1337,
        n_ctx=2048,
    )
    out = llm.create_chat_completion(
        messages= [
            {"role": "system", "content": "You are an assisstant who perfectly describes images."},
            {
                "role": "user",
                "content": [
                    {"type" : "text", "text" : prompt},
                    {"type" : "image_url" , "image_url" : {"url": f"file://{os.path.abspath(image_path)}"},}
                ]
            }
        ],
        max_tokens=int(maxtokens),
        temperature=0.1,
        stop=["\n### <|im_start|>", "\n<|im_end|>", "</s>"],
        #stream=True
    )
    print(out["choices"][0]["message"]["content"])



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

    message = out["choices"][0]["message"]
    content = message.get("content", "")
    if isinstance(content, list):
        content = "".join(
            block.get("text", "")
            for block in content
            if isinstance(block, dict) and block.get("type") == "text"
        )

    print((content or "").strip())

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

# Making sure that LLaVA doesn't run out of memory (Comment this function out if not needed)
#def _cast_fp16_inplace(inputs: dict):
#    """Cast float tensors to fp16 to save VRAM on Jetson."""
#    for k, v in inputs.items():
        # check each item that the processor returns (image tensors, text embeddings, etc.)
#        if torch.is_tensor(v) and v.is_floating_point():
            # if it’s a floating-point tensor, move it to half precision (FP16), help with cutting memory use roughly by half  
#            inputs[k] = v.to(torch.float16 if torch.cuda.is_available() else v.dtype)
            
# Added new backend: LLaaVA
def run_llava(image_path: str, prompt: str, max_tokens: str):
    """
    Uses llava-hf/llava-onevision-qwen2-0.5b-ov via Transformers.
    Prints output (same behavior style as existing functions).
    """
    from transformers import LlavaForConditionalGeneration

    # Open with Pillow and ensure 3 channels
    img = Image.open(image_path).convert("RGB")
    model_id = "llava-hf/llava-onevision-qwen2-0.5b-ov"
    processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
    model = LlavaForConditionalGeneration.from_pretrained(
        model_id,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
        torch_dtype=(torch.float16 if torch.cuda.is_available() else torch.float32),
        device_map="cuda" if torch.cuda.is_available() else None,
    ).eval()

    user_prompt = (prompt or "Describe this image.").strip()
    inputs = processor(images=img, text=user_prompt, return_tensors="pt")
    # Move to GPU if available
    # Converts floating tensors to FP16 to cut memory roughly in half (help with Jetson Orin Nano)
    if torch.cuda.is_available():
        inputs = {k: (v.to("cuda") if torch.is_tensor(v) else v) for k, v in inputs.items()}
        _cast_fp16_inplace(inputs)

    with torch.inference_mode():
        out_ids = model.generate(
            **inputs,
            max_new_tokens=int(max_tokens),
            do_sample=False,
            temperature=0.0,
        )
 
    text = processor.batch_decode(out_ids, skip_special_tokens=True)[0]
    print(text.replace(user_prompt, "").strip())

def main():
    if len(sys.argv) < 5 or sys.argv[1] != "--backend" or sys.argv[3] != "--image" or sys.argv[5] != "--maxtokens":
        usage()
    #print(sys.argv)
    backend = sys.argv[2].lower()
    image_path = sys.argv[4]
    max_tokens = sys.argv[6]
    prompt = " ".join(sys.argv[7:])

    # run_model_with_cpp(backend,image_path,prompt,max_tokens)
    # quick file check
    # try:
    #     Image.open(image_path).close()
    # except Exception as e:
    #     raise SystemExit(f"Bad --image path: {e}")

    if backend == "obsidian":
        run_model_with_cpp(backend,image_path,prompt,max_tokens)
    elif backend == "moondream":
        run_moondream(image_path, prompt, max_tokens)
    # Added elif for llava 
    elif backend == "llava":
        run_llava(image_path, prompt, max_tokens)
    else:
        usage()

if __name__ == "__main__":
    main()
