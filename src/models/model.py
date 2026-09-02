import json
import os

try:
    from llama_cpp import Llama
except ImportError:
    Llama = None  # ignore

CONVERSATIONS_DIR = "src/models/conversation"
DEFAULT_MODEL_PATH = "src/models/local/Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf"
DEFAULT_SYSTEM_PROMPT_PATH = "src/models/prompts/detect.txt"

# Loaded models are kept here, keyed by their construction args, so repeated
# chat()/run_chat() calls with the same settings don't reload the model.
_llm_cache: dict[tuple, Llama] = {}


def _load_system_prompt(system_prompt_path: str) -> str:
    with open(system_prompt_path, "r", encoding="utf-8") as f:
        return f.read()


def _convo_path(conversation_id: str) -> str:
    return os.path.join(CONVERSATIONS_DIR, f"{conversation_id}.json")


def _load_history(conversation_id: str, system_prompt: str) -> list[dict]:
    path = _convo_path(conversation_id)
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            history = json.load(f)
        # Always keep the system prompt current, even if the file on disk changed.
        if history and history[0].get("role") == "system":
            history[0]["content"] = system_prompt
        else:
            history.insert(0, {"role": "system", "content": system_prompt})
        return history
    return [{"role": "system", "content": system_prompt}]


def _save_history(conversation_id: str, history: list[dict]) -> None:
    os.makedirs(CONVERSATIONS_DIR, exist_ok=True)
    with open(_convo_path(conversation_id), "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)


def _get_llm(
    model_path: str,
    n_ctx: int,
    n_threads: int,
    n_gpu_layers: int,
    verbose: bool,
) -> Llama:
    if Llama is None:
        raise RuntimeError(
            "llama-cpp-python is required. Install it with "
            "'pip install llama-cpp-python'."
        )

    key = (model_path, n_ctx, n_threads, n_gpu_layers, verbose)
    llm = _llm_cache.get(key)
    if llm is None:
        llm = Llama(
            model_path=model_path,
            n_ctx=n_ctx,
            n_threads=n_threads,
            n_gpu_layers=n_gpu_layers,
            verbose=verbose,
        )
        _llm_cache[key] = llm
    return llm


def chat(
    message: str,
    conversation_id: str = "default",
    model_path: str = DEFAULT_MODEL_PATH,
    system_prompt_path: str = DEFAULT_SYSTEM_PROMPT_PATH,
    n_ctx: int = 4096, # context size
    n_threads: int = 16,
    n_gpu_layers: int = 1, # -1 = yes gpu, 1 = no gpu
    temperature: float = 0.5, # creativity, 1 = very creative, 0.1 = boring
    max_tokens: int = 512,
    verbose: bool = False,
    persist: bool = True, # add to convo or not
) -> str:
    """Send one message to `conversation_id` and return the assistant's reply.

    History is persisted to disk under CONVERSATIONS_DIR, keyed by
    conversation_id, so repeated calls with the same id continue the same
    thread. Use a different conversation_id to start a fresh one. The model
    itself is cached in-process per settings combo, so back-to-back calls
    don't pay the load cost again.

    Set persist=False to read the conversation's existing history for
    context (e.g. for a classifier that needs to see what's actually being
    discussed) without writing this call's turn back to it — nothing is
    appended or saved to disk.
    """
    system_prompt = _load_system_prompt(system_prompt_path)
    history = _load_history(conversation_id, system_prompt)
    llm = _get_llm(model_path, n_ctx, n_threads, n_gpu_layers, verbose)

    history.append({"role": "user", "content": message})

    response = llm.create_chat_completion(
        messages=history,
        max_tokens=max_tokens,
        temperature=temperature,
    )
    reply = response["choices"][0]["message"]["content"]

    if persist:
        history.append({"role": "assistant", "content": reply})
        _save_history(conversation_id, history)

    return reply


def run_chat(
    conversation_id: str = "default",
    model_path: str = DEFAULT_MODEL_PATH,
    system_prompt_path: str = DEFAULT_SYSTEM_PROMPT_PATH,
    n_ctx: int = 4096,
    n_threads: int = 8,
    n_gpu_layers: int = 1,
    temperature: float = 0.5,
    max_tokens: int = 512,
    verbose: bool = False,
) -> None:
    """Run an interactive llama.cpp chat loop in the terminal.

    Same persistence/model-caching behavior as chat(), but streams tokens to
    stdout as they're generated instead of returning the full reply.
    """
    system_prompt = _load_system_prompt(system_prompt_path)
    history = _load_history(conversation_id, system_prompt)
    llm = _get_llm(model_path, n_ctx, n_threads, n_gpu_layers, verbose)

    print(f"Local llama.cpp chat. Conversation: '{conversation_id}'. Type 'exit' to quit.\n")

    while True:
        user_input = input("You: ").strip()
        if user_input.lower() in ("exit", "quit"):
            break
        if not user_input:
            continue

        history.append({"role": "user", "content": user_input})

        print("AI: ", end="", flush=True)
        full_response = ""

        for chunk in llm.create_chat_completion(
            messages=history,
            stream=True,
            max_tokens=max_tokens,
            temperature=temperature,
        ):
            delta = chunk["choices"][0]["delta"]
            token = delta.get("content", "")
            print(token, end="", flush=True)
            full_response += token

        print("\n")
        history.append({"role": "assistant", "content": full_response})
        _save_history(conversation_id, history)





if __name__ == "__main__":
    with open('transcript.txt') as f:
        content = f.read()
    print(chat(content, 'default', system_prompt_path='src/models/prompts/detect.txt'))