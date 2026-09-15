import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import os
import time
import statistics
import psutil
from config import load_params
from model import load_model_and_tokenizer, generate

def get_rss_mb() -> float:
    return psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024)

def main():
    p = load_params()

    # 1. Время загрузки модели
    t0 = time.perf_counter()
    model, tok = load_model_and_tokenizer(p)
    load_time = time.perf_counter() - t0

    prompt = p["bench"]["prompt"]

    # 2. Прогревочный запуск (Warmup)
    for _ in range(p["bench"]["warmup_runs"]):
        generate(model, tok, prompt, p)

    # 3. Замеры генерации
    tokens_per_sec_list = []
    peak_rss = get_rss_mb()

    for _ in range(p["bench"]["measure_runs"]):
        t_start = time.perf_counter()
        _, n_tokens = generate(model, tok, prompt, p)
        elapsed = time.perf_counter() - t_start

        if elapsed > 0:
            tokens_per_sec_list.append(n_tokens / elapsed)

        current_rss = get_rss_mb()
        if current_rss > peak_rss:
            peak_rss = current_rss

    median_speed = statistics.median(tokens_per_sec_list) if tokens_per_sec_list else 0.0

    print(f"load_time: {load_time:.3f} s")
    print(f"tokens_per_sec: {median_speed:.2f}")
    print(f"peak_rss_mb: {peak_rss:.1f} MB")

if __name__ == "__main__":
    main()