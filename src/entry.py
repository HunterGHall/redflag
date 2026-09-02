from models.model import *


transcript_path = 'transcript.txt'


if __name__ == "__main__":
    with open(transcript_path) as f:
        content = f.read()
    print(chat(
        message=content,
        conversation_id='default',
        model_path='src/models/local/Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf',
        system_prompt_path='src/models/prompts/detect.txt',
        temperature=0.5
        )
    )