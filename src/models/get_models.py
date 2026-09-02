from huggingface_hub import hf_hub_download
import shutil
import os

# --- Configure these if your file is a different repo/quant ---
REPO_ID = "lmstudio-community/Meta-Llama-3.1-8B-Instruct-GGUF"
FILENAME = "Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf"
DEST_FOLDER = "src/models/local"
# ----------------------------------------------------------------

def main():
    os.makedirs(DEST_FOLDER, exist_ok=True)

    print(f"Downloading {FILENAME} from {REPO_ID} ...")
    downloaded_path = hf_hub_download(repo_id=REPO_ID, filename=FILENAME)

    dest_path = os.path.join(DEST_FOLDER, FILENAME)
    shutil.copy(downloaded_path, dest_path)

    print(f"Done. File saved to: {os.path.abspath(dest_path)}")

if __name__ == "__main__":
    main()