import time
from vllm import LLM

def measure_inference_latency_and_throughput(prompt, model="Qwen/Qwen2.5-0.5B-Instruct", dtype="float16" ):
    # Initialize the model
    llm = LLM(model=model, dtype=dtype)

    # Start the overall timer
    start_time = time.time()

    # Run generation with streaming to capture TTFT
    outputs = llm.generate(prompt,  stream=True)

    first_token_time = None
    tokens = []
    token_ids = []

    for output in outputs:
        # First token timing
        if first_token_time is None:
            first_token_time = time.time()
            ttft = first_token_time - start_time

        tokens.append(output.text)
        token_ids.append(output.token_id)

    end_time = time.time()

    # Final metrics
    total_time = end_time - start_time
    generation_time = total_time - ttft if first_token_time else 0.0
    tpot = len(tokens) / generation_time if generation_time > 0 else float("inf")
    generated_text = "".join(tokens)

    return generated_text, ttft, tpot, len(tokens)

if __name__ == "__main__":
    prompt = "What is the capital of France? Can you provide some details about it?"
    output, ttft, tpot, num_tokens = measure_inference_latency_and_throughput(prompt)

    print("\n--- Benchmark Results ---")
    print(f"Generated text:\n{output}")
    print(f"TTFT (Time To First Token): {ttft:.4f} seconds")
    print(f"TPOT (Tokens/sec After TTFT): {tpot:.2f} tokens/sec")
    print(f"Total Tokens Generated: {num_tokens}")
