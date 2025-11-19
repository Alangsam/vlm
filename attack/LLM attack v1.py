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
        #stop=["\n### <|im_start|>", "\n<|im_end|>", "</s>"],
        #stream=True
    )
    print(out["choices"][0]["message"]["content"])

def trying():
    # from llama_cpp import Llama

    # llm = Llama(
    #     model_path="./.models/obsidian/obsidian-q6.gguf",
    #     n_gpu_layers=-1, # Uncomment to use GPU acceleration
    #     # seed=1337, # Uncomment to set a specific seed
    #     # n_ctx=2048, # Uncomment to increase the context window
    # )
    # output = llm(
    #     "Q: Name the planets in the solar system? A: ", # Prompt
    #     max_tokens=32, # Generate up to 32 tokens, set to None to generate up to the end of the context window
    #     #stop=["Q:", "\n"], # Stop generating just before the model would generate a new question
    #     echo=True # Echo the prompt back in the output
    # ) # Generate a completion, can also call create_completion
    # print(output)
    from llama_cpp import Llama
    from llama_cpp.llama_chat_format import Llava16ChatHandler
    chat_handler = Llava16ChatHandler(clip_model_path="./.models/obsidian/mmproj-obsidian-f16.gguf")
    llm = Llama(
    model_path="./.models/obsidian/obsidian-q6.gguf",
    chat_handler=chat_handler,
    n_gpu_layers=0,
    n_ctx=1024,
      # n_ctx should be increased to accommodate the image embedding
    )
    output = llm.create_chat_completion(
        messages = [
            {"role": "system", "content": "You are an assistant who perfectly describes images."},
            {
                "role": "user",
                "content": [
                    {"type" : "text", "text": "What's in this image?"},
                    {"type": "image_url", "image_url": {"url": "file:///home/mr/Projects/vlm/samples/cute_dog.jpg" } }
                ]
            }
        ]
    )
    print(output)
def mini(image_path: str, prompt: str, max_tokens: str):
    from llama_cpp import Llama
    from llama_cpp.llama_chat_format import MiniCPMv26ChatHandler
    import base64
    import cv2

    cap = cv2.VideoCapture(0)
    

    def img_to_b64(frame):
        frame_small = cv2.resize(frame, (384, 384))
        _, buffer = cv2.imencode(".jpg", frame_small, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
        b64 = base64.b64encode(buffer).decode("utf-8")
        return f"data:image/jpg;base64,{b64}"
    # Replace 'file_path.png' with the actual path to your PNG file
    # file_path = 'file_path.png'
    # data_uri = image_to_base64_data_uri(image_path)
    chat_handler = MiniCPMv26ChatHandler(clip_model_path="/home/mr/Projects/vlm/.models/minicpm/mmproj-model-f16.gguf")
    llm = Llama(
    model_path="/home/mr/Projects/vlm/.models/minicpm/ggml-model-Q4_K_M.gguf",
    chat_handler=chat_handler,
    n_ctx=800, # n_ctx should be increased to accommodate the image embedding
    n_gpu_layers=27,
    flash_attn=True,
    n_batch=128
    )

    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        frame_idx += 1

        if frame_idx % 5 == 0:
            img_b64 = img_to_b64(frame)
            out = llm.create_chat_completion(
                max_tokens=int(max_tokens),
                messages=[
                    {"role": "system", "content": "You are an assistant, identify what's in the image."},
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {"type": "image_url", "image_url": {"url": img_b64}},
                        ],
                    },
                ],
            )
            print(out["choices"][0]["message"]["content"])

        cv2.imshow("Camera", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


def run_moondream(image_path: str, prompt: str, max_tokens: str):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer


    # bnb = BitsAndBytesConfig(
    #     load_in_4bit=True,
    #     bnb_4bit_use_double_quant=True,
    #     bnb_4bit_quant_type="nf4",
    #     bnb_4bit_compute_dtype=torch.float16,
    # )
    img = Image.open(image_path).convert("RGB")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        "vikhyatk/moondream2",
        trust_remote_code=True,
        dtype=(torch.float16 if torch.cuda.is_available() else torch.float32),
        device_map="cuda"
    ).eval()
    settings = {"max_tokens": max_tokens}
    # print(max_tokens)

    if prompt and prompt.strip():
        print("\nTOKENIZED PROMPT:")
        print("Token IDs:", enc["input_ids"].tolist()[0])
        print("Tokens:", tokenizer.convert_ids_to_tokens(enc["input_ids"][0]))
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
#Uses llava-hf/llava-onevision-qwen2-0.5b-ov via Transformers.

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
        #_cast_fp16_inplace(inputs)

    with torch.inference_mode():
        out_ids = model.generate(
            **inputs,
            max_new_tokens=int(max_tokens),
            do_sample=False,
            temperature=0.0,
        )
 
    text = processor.batch_decode(out_ids, skip_special_tokens=True)[0]
    print(text.replace(user_prompt, "").strip())

import cv2

def camera():
    
    cap = cv2.VideoCapture(0)  # 0 for default camera
    if not cap.isOpened():
        print("Error: Could not open camera.")
        return
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to grab frame.")
            break
        cv2.imshow('Webcam Feed', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):  # Press 'q' to quit
            break
    cap.release()
    cv2.destroyAllWindows()

def main():
    # camera()
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
        trying()
        #run_model_with_cpp(backend,image_path,prompt,max_tokens)
        #run_obsidian(image_path, prompt)
    elif backend == "moondream":
        run_moondream(image_path, prompt, max_tokens)
    # Added elif for llava 
    # elif backend == "llava":
    #     run_llava(image_path, prompt, max_tokens)
    elif backend == "minicpm":
        mini(image_path, prompt, max_tokens)
    else:
        usage()
        

if __name__ == "__main__":
    main()
