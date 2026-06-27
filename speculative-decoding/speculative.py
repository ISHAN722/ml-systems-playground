import torch
import time
from transformers import AutoModelForCausalLM, AutoTokenizer


def load_models():
    print("Loading draft model (OPT-125M)...")
    draft_tokenizer = AutoTokenizer.from_pretrained("facebook/opt-125m")
    draft_model = AutoModelForCausalLM.from_pretrained(
        "facebook/opt-125m", torch_dtype=torch.float32
    )
    draft_model.eval()

    print("Loading target model (OPT-1.3B)...")
    target_model = AutoModelForCausalLM.from_pretrained(
        "facebook/opt-1.3b", dtype=torch.float32
    )
    target_model.eval()
    return draft_model, target_model, draft_tokenizer

def speculative_decode(
    draft_model,
    target_model,
    tokenizer,
    prompt,
    max_new_tokens = 50,
    K = 4,
    temperature = 1.0
):
    input_ids = tokenizer(prompt, return_tensors = "pt").input_ids
    generated = input_ids.clone()

    accepted_total = 0
    steps = 0

    with torch.no_grad():
        while (generated.shape[1] - input_ids.shape[1]) < max_new_tokens:
            # --- draft: generate K tokens autoregressively ---
            draft_ids = generated.clone()
            draft_tokens = []

            for _ in range(K):
                out = draft_model(draft_ids)
                logits = out.logits[:, -1, :]
                next_token = torch.argmax(logits, dim=-1, keepdim=True)
                draft_tokens.append(next_token)
                draft_ids = torch.cat([draft_ids, next_token], dim=1)

            # --- target: verify all K draft tokens in one forward pass ---
            target_input = torch.cat([generated] + draft_tokens, dim = 1)
            target_out = target_model(target_input)
            target_logits = target_out.logits

            #check each draft token against target's prediction
            accepted = 0
            for i in range(K):
                pos = generated.shape[1] + i - 1
                target_token = torch.argmax(target_logits[:, pos, :], dim=-1, keepdim=True)
                if target_token.item() == draft_tokens[i].item():
                    accepted += 1
                else:
                    # reject - use target's token instead
                    generated = torch.cat([generated, target_token], dim=1)
                    break
            else:
                # all K accepted - apppend all draft tokens
                generated = torch.cat([generated] + draft_tokens, dim=1)
            accepted_total += accepted
            steps += 1

            if generated.shape[1] - input_ids.shape[1] >= max_new_tokens:
                break
    acceptance_rate = accepted_total / (steps * K) if steps > 0 else 0
    return tokenizer.decode(generated[0], skip_special_tokens=True), acceptance_rate

if __name__ == "__main__":
    draft_model, target_model, tokenizer = load_models()

    prompt = "The history of artifical intelligence began"
    print("\nRunning speculative decoding")
    start = time.perf_counter()
    text, acceptance_rate = speculative_decode(
        draft_model, target_model, tokenizer, prompt, max_new_tokens=50, K = 4
    )
    elapsed = (time.perf_counter() - start) * 1000

    print(f"\nOutput:\n{text}")
    print(f"\nAcceptance rate: {acceptance_rate:.2%}")
    print(f"Time: {elapsed:.2f} ms")