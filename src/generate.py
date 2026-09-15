import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import load_params
from model import load_model_and_tokenizer, generate

def main():
    p = load_params()
    model, tok = load_model_and_tokenizer(p)
    response, _ = generate(model, tok, p["bench"]["prompt"], p)
    print(response)

if __name__ == "__main__":
    main()