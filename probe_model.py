# probe_model.py
import json, sys
from huggingface_hub import hf_hub_download

def main():
    if len(sys.argv) != 2:
        print("Usage: python probe_model.py <repo_id>"); raise SystemExit(1)
    repo_id = sys.argv[1]
    cfg_path = hf_hub_download(repo_id=repo_id, filename="config.json")
    with open(cfg_path, "r") as f:
        cfg = json.load(f)
    print(f"repo: {repo_id}")
    print("model_type:", cfg.get("model_type"))
    print("architectures:", cfg.get("architectures"))
    # LLaVA-ish repos sometimes expose processors under processing_config.json too:
    try:
        proc_path = hf_hub_download(repo_id=repo_id, filename="preprocessor_config.json")
        with open(proc_path, "r") as f:
            pp = json.load(f)
        print("preprocessor_config keys:", list(pp.keys()))
    except Exception:
        pass

if __name__ == "__main__":
    main()
