import torch
import pandas as pd
import matplotlib.pyplot as plt
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
import math
import os
import copy
import time
import gc
from tqdm import tqdm
import numpy as np
from collections import defaultdict

# --- 1. Experimental Configuration ---

PRETRAINED_MODEL_NAME = "state-spaces/mamba-130m-hf"

# We'll discover the actual layer names dynamically
RANKS_TO_TEST = [8, 16, 32, 64]
CONTEXT_LENGTHS_TO_TEST = [128, 256, 512, 1024]

# Dataset Configuration
DATASET_NAME = "wikitext"
DATASET_CONFIG = "wikitext-2-raw-v1"
TOKENIZER_NAME = "state-spaces/mamba-130m-hf"

MAX_EVAL_SAMPLES = 50
BATCH_SIZE = 1

def inspect_model_architecture(model):
    """Inspect the model architecture to find the correct layer names."""
    print("\n" + "="*60)
    print("MODEL ARCHITECTURE INSPECTION")
    print("="*60)
    
    linear_layers = []
    all_layers = []
    
    for name, module in model.named_modules():
        all_layers.append((name, type(module).__name__))
        if isinstance(module, torch.nn.Linear):
            weight_shape = module.weight.shape
            linear_layers.append((name, weight_shape, module.weight.numel()))
    
    print(f"Total modules: {len(all_layers)}")
    print(f"Linear layers found: {len(linear_layers)}")
    
    print("\nAll modules (first 20):")
    for name, module_type in all_layers[:20]:
        print(f"  {name}: {module_type}")
    
    if len(all_layers) > 20:
        print(f"  ... and {len(all_layers) - 20} more")
    
    print(f"\nLinear layers (all {len(linear_layers)}):")
    total_params = 0
    for name, shape, params in linear_layers:
        print(f"  {name}: {shape} ({params:,} params)")
        total_params += params
    
    print(f"\nTotal parameters in Linear layers: {total_params:,}")
    
    # Look for common Mamba patterns
    mamba_patterns = ['mixer', 'proj', 'conv1d', 'in_proj', 'out_proj', 'x_proj', 'dt_proj']
    relevant_layers = []
    
    for name, shape, params in linear_layers:
        for pattern in mamba_patterns:
            if pattern in name.lower():
                relevant_layers.append((name, shape, params))
                break
    
    print(f"\nRelevant layers for compression ({len(relevant_layers)}):")
    for name, shape, params in relevant_layers:
        print(f"  {name}: {shape} ({params:,} params)")
    
    return [name for name, _, _ in relevant_layers]

def get_target_layers(model, min_params=1000):
    """Automatically identify target layers for compression."""
    target_layers = []
    
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear):
            # Skip very small layers (like biases or small projections)
            if module.weight.numel() >= min_params:
                # Focus on projection layers commonly found in transformers/mamba
                if any(keyword in name.lower() for keyword in 
                       ['proj', 'linear', 'dense', 'mixer', 'conv1d']):
                    target_layers.append(name)
    
    return target_layers

class ModelProfiler:
    def __init__(self, device="auto"):
        self.device = "cuda" if torch.cuda.is_available() and device == "auto" else device
        print(f"Using device: {self.device}")
        
        # Initialize tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME)
        if self.tokenizer.pad_token is None: 
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Load original model
        print(f"Loading pre-trained model: {PRETRAINED_MODEL_NAME}")
        self.original_model = AutoModelForCausalLM.from_pretrained(
            PRETRAINED_MODEL_NAME, 
            ignore_mismatched_sizes=True,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32
        )
        
        # Inspect architecture and get target layers
        self.target_layers = inspect_model_architecture(self.original_model)
        
        if not self.target_layers:
            print("No suitable layers found! Using automatic detection...")
            self.target_layers = get_target_layers(self.original_model)
        
        print(f"\nSelected {len(self.target_layers)} layers for compression:")
        for layer in self.target_layers:
            print(f"  - {layer}")
        
        if not self.target_layers:
            raise ValueError("No target layers found for compression!")
        
        # Pre-prepare evaluation datasets
        self.eval_datasets = {}
        self._prepare_all_eval_data()
        
    def _prepare_all_eval_data(self):
        """Pre-prepare evaluation data for all context lengths."""
        print("\nPreparing evaluation datasets...")
        
        # Load dataset
        dataset = load_dataset(DATASET_NAME, DATASET_CONFIG, split="test", streaming=True)
        
        # Tokenize and collect text
        tokenized_lines = []
        for example in dataset:
            if example['text'].strip():
                tokens = self.tokenizer(example['text'], add_special_tokens=False)['input_ids']
                tokenized_lines.append(tokens)
                if len(tokenized_lines) >= 1000:
                    break
        
        # Concatenate all tokens
        all_tokens = [token for line in tokenized_lines for token in line]
        print(f"Collected {len(all_tokens):,} total tokens")
        
        # Create datasets for each context length
        for max_length in CONTEXT_LENGTHS_TO_TEST:
            eval_samples = []
            for i in range(0, len(all_tokens) - max_length, max_length):
                chunk = all_tokens[i : i + max_length]
                eval_samples.append({"input_ids": chunk, "labels": chunk})
                if len(eval_samples) >= MAX_EVAL_SAMPLES:
                    break
            
            self.eval_datasets[max_length] = eval_samples
            print(f"  Prepared {len(eval_samples)} samples for context length {max_length}")

    def apply_low_rank_approximation(self, model, target_rank):
        """Apply SVD-based low-rank approximation with detailed logging."""
        print(f"\nApplying SVD with target rank = {target_rank}...")
        
        compression_stats = []
        layers_processed = 0
        
        with torch.no_grad():
            for name, module in model.named_modules():
                if name in self.target_layers and isinstance(module, torch.nn.Linear):
                    print(f"  Processing layer: {name}")
                    
                    # Get original weight
                    original_weight = module.weight.data.float()
                    original_shape = original_weight.shape
                    original_params = original_weight.numel()
                    
                    print(f"    Original shape: {original_shape} ({original_params:,} params)")
                    
                    # Perform SVD
                    try:
                        U, S, Vh = torch.linalg.svd(original_weight, full_matrices=False)
                        print(f"    SVD shapes: U{U.shape}, S{S.shape}, Vh{Vh.shape}")
                        
                        # Determine effective rank
                        max_possible_rank = min(U.shape[1], Vh.shape[0])
                        effective_rank = min(target_rank, max_possible_rank)
                        
                        print(f"    Target rank: {target_rank}, Max possible: {max_possible_rank}, Using: {effective_rank}")
                        
                        if effective_rank >= max_possible_rank:
                            print(f"    Warning: Target rank {target_rank} >= matrix rank {max_possible_rank}, no compression possible")
                            # No compression needed/possible
                            compressed_params = original_params
                        else:
                            # Truncate to target rank
                            U_r = U[:, :effective_rank]
                            S_r = S[:effective_rank]
                            Vh_r = Vh[:effective_rank, :]
                            
                            # Calculate compressed parameters
                            # Note: For actual deployment, you'd store U_r, S_r, Vh_r separately
                            # Here we reconstruct for simplicity but count the compressed params
                            compressed_params = U_r.numel() + S_r.numel() + Vh_r.numel()
                            
                            # Reconstruct the approximated weight
                            S_r_diag = torch.diag(S_r)
                            low_rank_weight = U_r @ S_r_diag @ Vh_r
                            
                            # Replace the weight
                            module.weight.data = low_rank_weight.to(model.dtype)
                            
                            print(f"    Compressed params: {compressed_params:,}")
                        
                        # Calculate compression ratio
                        compression_ratio = original_params / compressed_params if compressed_params > 0 else 1.0
                        
                        compression_stats.append({
                            'layer': name,
                            'original_params': original_params,
                            'compressed_params': compressed_params,
                            'compression_ratio': compression_ratio,
                            'effective_rank': effective_rank,
                            'original_shape': original_shape
                        })
                        
                        layers_processed += 1
                        print(f"    Compression ratio: {compression_ratio:.2f}x")
                        
                    except Exception as e:
                        print(f"    Error processing layer {name}: {e}")
                        continue
        
        print(f"\nProcessed {layers_processed} layers total")
        
        if not compression_stats:
            print("WARNING: No layers were processed!")
            return []
        
        # Calculate overall statistics
        total_original = sum(stat['original_params'] for stat in compression_stats)
        total_compressed = sum(stat['compressed_params'] for stat in compression_stats)
        
        if total_compressed > 0:
            overall_compression = total_original / total_compressed
            print(f"Overall compression ratio: {overall_compression:.2f}x")
            print(f"Parameter reduction: {(1 - total_compressed/total_original)*100:.1f}%")
        else:
            print("ERROR: Total compressed parameters is 0!")
            return []
        
        return compression_stats

    @torch.no_grad()
    def evaluate_model(self, model, context_length):
        """Evaluate model performance with detailed error handling."""
        model.eval()
        model.to(self.device)
        
        eval_data = self.eval_datasets[context_length]
        total_loss = 0
        total_samples = 0
        inference_times = []
        
        # Memory profiling
        if self.device == "cuda":
            torch.cuda.reset_peak_memory_stats()
            initial_memory = torch.cuda.memory_allocated()
        
        print(f"    Evaluating {len(eval_data)} samples...")
        
        for i, sample in enumerate(tqdm(eval_data, desc=f"Eval (ctx={context_length})", leave=False)):
            try:
                input_ids = torch.tensor([sample["input_ids"]]).to(self.device)
                labels = torch.tensor([sample["labels"]]).to(self.device)
                
                # Time the inference
                start_time = time.time()
                outputs = model(input_ids, labels=labels)
                inference_time = time.time() - start_time
                
                if torch.isfinite(outputs.loss):
                    total_loss += outputs.loss.item()
                    total_samples += 1
                    inference_times.append(inference_time)
                else:
                    print(f"    Warning: Invalid loss at sample {i}")
                    
            except Exception as e:
                print(f"    Error at sample {i}: {e}")
                continue
        
        if total_samples == 0:
            return {
                'perplexity': float('inf'),
                'avg_inference_time': 0,
                'throughput': 0,
                'memory_used_gb': 0,
                'num_samples': 0
            }
        
        # Calculate metrics
        avg_loss = total_loss / total_samples
        perplexity = math.exp(min(avg_loss, 100))  # Cap to prevent overflow
        avg_inference_time = np.mean(inference_times) if inference_times else 0
        throughput = context_length / avg_inference_time if avg_inference_time > 0 else 0
        
        # Memory metrics
        if self.device == "cuda":
            peak_memory = torch.cuda.max_memory_allocated()
            memory_used = (peak_memory - initial_memory) / 1024**3  # GB
        else:
            memory_used = 0
        
        print(f"    Completed: PPL={perplexity:.3f}, Throughput={throughput:.1f} tok/s")
        
        return {
            'perplexity': perplexity,
            'avg_inference_time': avg_inference_time,
            'throughput': throughput,
            'memory_used_gb': memory_used,
            'num_samples': total_samples
        }

    def profile_rank_configuration(self, rank):
        """Profile a single rank configuration with better error handling."""
        print(f"\n--- Profiling Rank = {rank} ---")
        
        try:
            # Create low-rank model
            model = copy.deepcopy(self.original_model)
            compression_stats = self.apply_low_rank_approximation(model, rank)
            
            if not compression_stats:
                raise ValueError("No compression statistics generated")
            
            results = []
            for context_length in CONTEXT_LENGTHS_TO_TEST:
                print(f"  Evaluating context length {context_length}...")
                
                metrics = self.evaluate_model(model, context_length)
                
                result = {
                    'rank': rank,
                    'context_length': context_length,
                    **metrics
                }
                results.append(result)
                
                print(f"    Result: {result}")
            
            # Clean up
            del model
            if self.device == "cuda":
                torch.cuda.empty_cache()
            gc.collect()
            
            return results
            
        except Exception as e:
            print(f"Error in profile_rank_configuration for rank {rank}: {e}")
            print(f"Error type: {type(e).__name__}")
            import traceback
            traceback.print_exc()
            raise

    def profile_original_model(self):
        """Profile the original unmodified model."""
        print("\n--- Profiling Original Model ---")
        
        results = []
        for context_length in CONTEXT_LENGTHS_TO_TEST:
            print(f"  Evaluating context length {context_length}...")
            
            metrics = self.evaluate_model(self.original_model, context_length)
            
            result = {
                'rank': 'Original',
                'context_length': context_length,
                **metrics
            }
            results.append(result)
            
            print(f"    Result: {result}")
        
        return results

    def run_full_profiling(self):
        """Run complete profiling experiment with better error handling."""
        all_results = []
        

        
        # Profile each rank configuration
        for rank in RANKS_TO_TEST:
            try:
                rank_results = self.profile_rank_configuration(rank)
                all_results.extend(rank_results)
            except Exception as e:
                print(f"Skipping rank {rank} due to error: {e}")
                continue
        
        # Profile original model first
        try:
            original_results = self.profile_original_model()
            all_results.extend(original_results)
        except Exception as e:
            print(f"Error profiling original model: {e}")
        
        return pd.DataFrame(all_results)

# --- Main Execution ---
if __name__ == "__main__":
    try:
        print("Starting diagnostic model profiling...")
        
        # Initialize profiler
        profiler = ModelProfiler()
        
        # Run profiling
        results = profiler.run_full_profiling()
        
        if not results.empty:
            # Save results
            results.to_csv('mamba_profiling_results_debug.csv', index=False)
            print("\nResults saved to mamba_profiling_results_debug.csv")
            print("\nFinal Results:")
            print(results)
        else:
            print("No results generated!")
        
    except Exception as e:
        print(f"Critical error during profiling: {e}")
        import traceback
        traceback.print_exc()