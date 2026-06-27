import torch
import time
from transformers import AutoModelForCausalLM, AutoTokenizer
from speculative import speculative_decode

print("Loading models...")
tokenizer = AutoTokenizer.from_pretrained("facebook/opt-125m")
draft_model = AutoModelForCausalLM.from_pretrained("facebook/opt-125m", torch_dtype=torch.float32)
target_model = AutoModelForCausalLM.from_pretrained("facebook/opt-1.3b", torch_dtype=torch.float32)
draft_model.eval()
target_model.eval()

prompt = "The history of artificial intelligence began"
MAX_NEW_TOKENS = 50

# --- normal autoregressive generation (target only) ---
def normal_generate():
    input_ids = tokenizer(prompt, return_tensors = "pt").input_ids
    with torch.no_grad():
        out = target_model.generate(
            input_ids,
            max_new_tokens = MAX_NEW_TOKENS,
            do_sample=False
        )
    return out

print("\nBenchmarking normal generation (target only)...")
# warmup
normal_generate()
start = time.perf_counter()
for _ in range(3):
    normal_generate()
normal_ms = (time.perf_counter() - start) / 3 * 1000
print(f"Normal generation: {normal_ms:.2f} ms")

# --- speculative decoding ---
print("\nBenchmarking speculative decoding...")
# warmup
speculative_decode(draft_model, target_model, tokenizer, prompt, MAX_NEW_TOKENS, K=4)
start = time.perf_counter()
for _ in range(3):
    _, acc = speculative_decode(draft_model, target_model, tokenizer, prompt, MAX_NEW_TOKENS, K=4)

spec_ms = (time.perf_counter() - start) / 3 * 1000
print(f"Speculative decoding: {spec_ms:.2f} ms")

print(f"\n{'='*40}")
print(f"Normal generation:    {normal_ms:.2f} ms")
print(f"Speculative decoding: {spec_ms:.2f} ms")
print(f"Speedup:              {normal_ms/spec_ms:.2f}x")
print(f"Acceptance rate:      {acc:.2%}")