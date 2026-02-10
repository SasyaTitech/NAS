import pickle

_WARNED_UNKNOWN_CONTEXT_LENGTH_MODELS = set()

def save_data(filename, data):
    #Storing data with labels
    a_file = open(filename, "wb")
    pickle.dump(data, a_file)
    a_file.close()
    

def load_data(filename):
    a_file = open(filename, "rb")
    output = pickle.load(a_file)
    a_file.close()
    return output

# def load_data_split(filename, split):
#     a_file = open(filename, "rb")
#     output = pickle.load(a_file)
#     a_file.close()
#     return output[:split], output[split:]

FEW_SHOT_TEST_SPLIT = 10

def load_data_split(filename, number_of_few_shots, number_of_tests):
    a_file = open(filename, "rb")
    output = pickle.load(a_file)
    a_file.close()
    assert number_of_few_shots <= FEW_SHOT_TEST_SPLIT, f"The largest number of few shot can only be 100, we received {number_of_few_shots}"
    if not number_of_tests is None:
        assert number_of_tests <= len(output) - FEW_SHOT_TEST_SPLIT,  f"The largest number of test for this task can only be {len(output) - FEW_SHOT_TEST_SPLIT}, we received {number_of_tests}"
    else:
        number_of_tests = len(output) - FEW_SHOT_TEST_SPLIT
    allow_few_shots, allow_tests = output[:FEW_SHOT_TEST_SPLIT], output[FEW_SHOT_TEST_SPLIT:]
    return allow_few_shots[:number_of_few_shots], allow_tests[:number_of_tests]

MODEL_NAME_TO_MAXIMUM_CONTEXT_LENGTH_MAP = {
    "gpt2-xl": 1024,
    "llama-2-7b-hf": 4096,
    "llama3-8b-instruct": 4096,
    "meta-llama-3-8b-instruct": 4096,
    "eleutherai_gpt-j-6b": 2048,
    "gpt2-large": 1024,
    "gpt2-medium": 1024
}


def get_model_max_context_length(model, tokenizer=None, default: int = 4096) -> int:
    """
    Best-effort max context length for GLUE prompt construction.
    Prefer the existing name->length map for backwards compatibility, then fall back
    to model config / tokenizer, and finally a conservative default.
    """
    cfg = getattr(model, "config", None)
    model_id = str(getattr(cfg, "_name_or_path", "") or "")
    key = model_id.lower().split("/")[-1]

    if key in MODEL_NAME_TO_MAXIMUM_CONTEXT_LENGTH_MAP:
        return int(MODEL_NAME_TO_MAXIMUM_CONTEXT_LENGTH_MAP[key])

    for attr in ("max_position_embeddings", "n_positions", "n_ctx"):
        val = getattr(cfg, attr, None)
        if isinstance(val, int) and val > 0:
            return int(val)

    tok_max = getattr(tokenizer, "model_max_length", None) if tokenizer is not None else None
    if isinstance(tok_max, int) and 0 < tok_max < 10_000_000:
        return int(tok_max)

    if key and key not in _WARNED_UNKNOWN_CONTEXT_LENGTH_MODELS:
        _WARNED_UNKNOWN_CONTEXT_LENGTH_MODELS.add(key)
        print(
            f"Warning: unknown model {model_id!r} for context-length lookup; "
            f"using default={default}."
        )
    return int(default)
