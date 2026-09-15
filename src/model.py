import sys
from pathlib import Path

# Пути для корректных импортов как пакетом, так и напрямую
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

try:
    from config import load_params
except ImportError:
    from src.config import load_params

def load_model(p: dict | None = None):
    """Загружает токенизатор и модель. Возвращает (tok, model) для совместимости с tests/check.sh"""
    if p is None:
        p = load_params()

    torch.manual_seed(p["generate"]["seed"])
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(p["generate"]["seed"])

    model_name = p["model"]["name"]
    dtype_str = p["model"].get("dtype", "float32")
    dtype = getattr(torch, dtype_str) if hasattr(torch, dtype_str) else torch.float32

    tok = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=dtype,
        device_map=p["model"].get("device", "cpu"),
    )
    return tok, model

def load_model_and_tokenizer(p: dict | None = None):
    """Возвращает (model, tok) для generate.py и bench.py"""
    tok, model = load_model(p)
    return model, tok

def generate(model, tok, prompt: str, p: dict | None = None) -> tuple[str, int]:
    if p is None:
        p = load_params()

    messages = [{"role": "user", "content": prompt}]
    enable_thinking = p["generate"].get("enable_thinking", False)

    try:
        text = tok.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=enable_thinking,
        )
    except TypeError:
        text = tok.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

    inputs = tok(text, return_tensors="pt").to(model.device)
    prompt_len = inputs.input_ids.shape[1]

    temp = float(p["generate"]["temperature"])
    do_sample = temp > 0.0

    gen_kwargs = {
        "max_new_tokens": p["generate"]["max_new_tokens"],
        "do_sample": do_sample,
    }
    if do_sample:
        gen_kwargs["temperature"] = temp

    out = model.generate(**inputs, **gen_kwargs)
    new_tokens = out[0][prompt_len:]
    response_text = tok.decode(new_tokens, skip_special_tokens=True)

    return response_text, len(new_tokens)