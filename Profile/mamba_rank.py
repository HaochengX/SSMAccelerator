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
import json

# --- 1. Experimental Configuration ---

PRETRAINED_MODEL_NAME = "state-spaces/mamba-130m-hf"

# Define layer groups to test individually
LAYER_GROUPS = {
    'in_proj': ['in_proj'],           # Input projections (usually most important)
    'out_proj': ['out_proj'],         # Output projections
    'x_proj': ['x_proj'],             # X projections in Mamba
    'dt_proj': ['dt_proj'],           # Delta-time projections in Mamba
    'conv1d': ['conv1d'],             # 1D convolution layers
    'all_proj': ['in_proj', 'out_proj', 'x_proj', 'dt_proj'],  # All projection layers
    'non_critical': ['conv1d'],       # Start with less critical layers
    'all': None                       # Compress everything (your current approach)
}

# Layer-specific rank configurations
# Format: {layer_pattern: [ranks_to_test]}
LAYER_RANK_CONFIGS = {
    'Strategy_1_Conservative': {
        'in_proj': [256, 512],        # Keep input projections higher rank
        'out_proj': [256, 512],
        'x_proj': [128, 256],
        'dt_proj': [64, 128],
        'conv1d': [32, 64],
    },
    'Strategy_2_Aggressive': {
        'in_proj': [128, 256],
        'out_proj': [128, 256],
        'x_proj': [64, 128],
        'dt_proj': [32, 64],
        'conv1d': [16, 32],
    },
    'Strategy_3_Uniform': {
        'all': [64, 128, 256, 512],   # Apply same rank to all layers
    },
    'Strategy_4_ContextAdaptive': {
        # Different configs for different context lengths
        'short_context': {  # For context <= 256
            'in_proj': [128],
            'out_proj': [128],
            'x_proj': [64],
            'dt_proj': [32],
        },
        'medium_context': {  # For context 256-512
            'in_proj': [256],
            'out_proj': [256],
            'x_proj': [128],
            'dt_proj': [64],
        },
        'long_context': {  # For context >= 512
            'in_proj': [512],
            'out_proj': [512],
            'x_proj': [256],
            'dt_proj': [128],
        }
    }
}

CONTEXT_LENGTHS_TO_TEST = [128, 512, 1024, 4096]

# Dataset Configuration
DATASET_NAME = "wikitext"
DATASET_CONFIG = "wikitext-2-raw-v1"
TOKENIZER_NAME = "state-spaces/mamba-130m-hf"

MAX_EVAL_SAMPLES = 50
BATCH_SIZE = 1

def inspect_model_architecture(model):
    """Inspect the model architecture and categorize layers."""
    print("\n" + "="*60)
    print("MODEL ARCHITECTURE INSPECTION")
    print("="*60)
    
    linear_layers = []
    layer_categories = defaultdict(list)
    
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear):
            weight_shape = module.weight.shape
            params = module.weight.numel()
            linear_layers.append((name, weight_shape, params))
            
            # Categorize layers
            for category in ['in_proj', 'out_proj', 'x_proj', 'dt_proj', 'conv1d', 'mixer']:
                if category in name.lower():
                    layer_categories[category].append((name, weight_shape, params))
                    break
    
    print(f"Total Linear layers: {len(linear_layers)}")
    
    print("\nLayer Categories:")
    for category, layers in sorted(layer_categories.items()):
        total_params = sum(p for _, _, p in layers)
        print(f"\n  {category.upper()} ({len(layers)} layers, {total_params:,} params):")
        for name, shape, params in layers[:3]:  # Show first 3
            print(f"    {name}: {shape} ({params:,} params)")
        if len(layers) > 3:
            print(f"    ... and {len(layers)-3} more")
    
    return layer_categories

def get_layers_by_patterns(model, patterns):
    """Get layer names matching specific patterns."""
    if patterns is None:  # 'all' case
        return [name for name, module in model.named_modules() 
                if isinstance(module, torch.nn.Linear) and module.weight.numel() >= 1000]
    
    target_layers = []
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear) and module.weight.numel() >= 1000:
            for pattern in patterns:
                if pattern in name.lower():
                    target_layers.append(name)
                    break
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
        
        # Inspect architecture
        self.layer_categories = inspect_model_architecture(self.original_model)
        
        # Pre-prepare evaluation datasets
        self.eval_datasets = {}
        self._prepare_all_eval_data()
        
    def _prepare_all_eval_data(self):
        """Pre-prepare evaluation data for all context lengths."""
        print("\nPreparing evaluation datasets...")
        
        dataset = load_dataset(DATASET_NAME, DATASET_CONFIG, split="test", streaming=True)
        
        tokenized_lines = []
        for example in dataset:
            if example['text'].strip():
                tokens = self.tokenizer(example['text'], add_special_tokens=False)['input_ids']
                tokenized_lines.append(tokens)
                if len(tokenized_lines) >= 1000:
                    break
        
        all_tokens = [token for line in tokenized_lines for token in line]
        print(f"Collected {len(all_tokens):,} total tokens")
        
        for max_length in CONTEXT_LENGTHS_TO_TEST:
            eval_samples = []
            for i in range(0, len(all_tokens) - max_length, max_length):
                chunk = all_tokens[i : i + max_length]
                eval_samples.append({"input_ids": chunk, "labels": chunk})
                if len(eval_samples) >= MAX_EVAL_SAMPLES:
                    break
            
            self.eval_datasets[max_length] = eval_samples
            print(f"  Prepared {len(eval_samples)} samples for context length {max_length}")

    def apply_selective_low_rank(self, model, layer_rank_map):
        """
        Apply different ranks to different layers based on mapping.
        
        Args:
            model: The model to compress
            layer_rank_map: Dict mapping layer names to ranks
        """
        print(f"\nApplying selective low-rank compression...")
        print(f"Compressing {len(layer_rank_map)} layers")
        
        compression_stats = []
        
        with torch.no_grad():
            for layer_name, target_rank in layer_rank_map.items():
                # Find the module
                module = None
                for name, mod in model.named_modules():
                    if name == layer_name:
                        module = mod
                        break
                
                if module is None or not isinstance(module, torch.nn.Linear):
                    print(f"  Warning: Could not find layer {layer_name}")
                    continue
                
                print(f"  {layer_name}: rank={target_rank}")
                
                # Get original weight
                original_weight = module.weight.data.float()
                original_shape = original_weight.shape
                original_params = original_weight.numel()
                
                # Perform SVD
                try:
                    U, S, Vh = torch.linalg.svd(original_weight, full_matrices=False)
                    max_rank = min(U.shape[1], Vh.shape[0])
                    effective_rank = min(target_rank, max_rank)
                    
                    if effective_rank < max_rank:
                        # Truncate and reconstruct
                        U_r = U[:, :effective_rank]
                        S_r = S[:effective_rank]
                        Vh_r = Vh[:effective_rank, :]
                        
                        compressed_params = U_r.numel() + S_r.numel() + Vh_r.numel()
                        
                        S_r_diag = torch.diag(S_r)
                        low_rank_weight = U_r @ S_r_diag @ Vh_r
                        module.weight.data = low_rank_weight.to(model.dtype)
                    else:
                        compressed_params = original_params
                        effective_rank = max_rank
                    
                    compression_ratio = original_params / compressed_params if compressed_params > 0 else 1.0
                    
                    compression_stats.append({
                        'layer': layer_name,
                        'original_params': original_params,
                        'compressed_params': compressed_params,
                        'compression_ratio': compression_ratio,
                        'target_rank': target_rank,
                        'effective_rank': effective_rank,
                    })
                    
                except Exception as e:
                    print(f"    Error: {e}")
                    continue
        
        # Print summary
        if compression_stats:
            total_original = sum(s['original_params'] for s in compression_stats)
            total_compressed = sum(s['compressed_params'] for s in compression_stats)
            overall_ratio = total_original / total_compressed if total_compressed > 0 else 1.0
            print(f"\n  Overall compression: {overall_ratio:.2f}x")
            print(f"  Parameter reduction: {(1 - total_compressed/total_original)*100:.1f}%")
        
        return compression_stats

    @torch.no_grad()
    def evaluate_model(self, model, context_length):
        """Evaluate model performance."""
        model.eval()
        model.to(self.device)
        
        eval_data = self.eval_datasets[context_length]
        total_loss = 0
        total_samples = 0
        inference_times = []
        
        for sample in tqdm(eval_data, desc=f"Eval (ctx={context_length})", leave=False):
            try:
                input_ids = torch.tensor([sample["input_ids"]]).to(self.device)
                labels = torch.tensor([sample["labels"]]).to(self.device)
                
                start_time = time.time()
                outputs = model(input_ids, labels=labels)
                inference_time = time.time() - start_time
                
                if torch.isfinite(outputs.loss):
                    total_loss += outputs.loss.item()
                    total_samples += 1
                    inference_times.append(inference_time)
                    
            except Exception as e:
                continue
        
        if total_samples == 0:
            return {'perplexity': float('inf'), 'avg_inference_time': 0, 
                    'throughput': 0, 'num_samples': 0}
        
        avg_loss = total_loss / total_samples
        perplexity = math.exp(min(avg_loss, 100))
        avg_inference_time = np.mean(inference_times) if inference_times else 0
        throughput = context_length / avg_inference_time if avg_inference_time > 0 else 0
        
        return {
            'perplexity': perplexity,
            'avg_inference_time': avg_inference_time,
            'throughput': throughput,
            'num_samples': total_samples
        }

    def test_layer_group(self, group_name, group_patterns, rank):
        """Test compression on a specific layer group."""
        print(f"\n{'='*60}")
        print(f"Testing Layer Group: {group_name} (rank={rank})")
        print(f"{'='*60}")
        
        # Get layers to compress
        target_layers = get_layers_by_patterns(self.original_model, group_patterns)
        print(f"Selected {len(target_layers)} layers")
        
        if not target_layers:
            print("No layers found for this group!")
            return []
        
        # Create layer-rank mapping
        layer_rank_map = {layer: rank for layer in target_layers}
        
        # Create compressed model
        model = copy.deepcopy(self.original_model)
        compression_stats = self.apply_selective_low_rank(model, layer_rank_map)
        
        # Evaluate across context lengths
        results = []
        for context_length in CONTEXT_LENGTHS_TO_TEST:
            print(f"\n  Context length: {context_length}")
            metrics = self.evaluate_model(model, context_length)
            
            result = {
                'layer_group': group_name,
                'rank': rank,
                'context_length': context_length,
                'num_layers_compressed': len(target_layers),
                **metrics
            }
            results.append(result)
            print(f"    PPL: {metrics['perplexity']:.3f}")
        
        # Cleanup
        del model
        if self.device == "cuda":
            torch.cuda.empty_cache()
        gc.collect()
        
        return results

    def test_strategy(self, strategy_name, strategy_config):
        """Test a complete compression strategy."""
        print(f"\n{'='*60}")
        print(f"Testing Strategy: {strategy_name}")
        print(f"{'='*60}")
        
        all_results = []
        
        # Build layer-rank mapping
        layer_rank_map = {}
        for pattern, ranks in strategy_config.items():
            if isinstance(ranks, dict):  # Nested config (e.g., context-adaptive)
                continue  # Handle separately
            
            layers = get_layers_by_patterns(self.original_model, 
                                           [pattern] if pattern != 'all' else None)
            for layer in layers:
                # Use the first rank for this strategy
                layer_rank_map[layer] = ranks[0] if isinstance(ranks, list) else ranks
        
        if not layer_rank_map:
            print("No layers selected for this strategy!")
            return []
        
        print(f"Compressing {len(layer_rank_map)} layers")
        
        # Create compressed model
        model = copy.deepcopy(self.original_model)
        compression_stats = self.apply_selective_low_rank(model, layer_rank_map)
        
        # Evaluate
        for context_length in CONTEXT_LENGTHS_TO_TEST:
            print(f"\n  Context length: {context_length}")
            metrics = self.evaluate_model(model, context_length)
            
            result = {
                'strategy': strategy_name,
                'context_length': context_length,
                'num_layers_compressed': len(layer_rank_map),
                **metrics
            }
            all_results.append(result)
            print(f"    PPL: {metrics['perplexity']:.3f}")
        
        # Cleanup
        del model
        if self.device == "cuda":
            torch.cuda.empty_cache()
        gc.collect()
        
        return all_results

    def run_comprehensive_profiling(self):
        """Run comprehensive profiling of all configurations."""
        all_results = []
        
        # 1. Profile original model (baseline)
        print("\n" + "="*60)
        print("BASELINE: Original Model")
        print("="*60)
        for context_length in CONTEXT_LENGTHS_TO_TEST:
            print(f"\nContext length: {context_length}")
            metrics = self.evaluate_model(self.original_model, context_length)
            result = {
                'layer_group': 'Original',
                'strategy': 'Original',
                'rank': 'N/A',
                'context_length': context_length,
                'num_layers_compressed': 0,
                **metrics
            }
            all_results.append(result)
            print(f"  PPL: {metrics['perplexity']:.3f}")
        
        # 2. Test individual layer groups
        print("\n" + "="*60)
        print("PHASE 1: Individual Layer Groups")
        print("="*60)
        for group_name, patterns in LAYER_GROUPS.items():
            if group_name == 'all':
                continue  # Skip for now
            for rank in [64, 128, 256]:  # Test a few ranks
                results = self.test_layer_group(group_name, patterns, rank)
                all_results.extend(results)
        
        # 3. Test strategies
        print("\n" + "="*60)
        print("PHASE 2: Compression Strategies")
        print("="*60)
        for strategy_name, config in LAYER_RANK_CONFIGS.items():
            if 'ContextAdaptive' in strategy_name:
                continue  # Skip context-adaptive for now (more complex)
            results = self.test_strategy(strategy_name, config)
            all_results.extend(results)
        
        return pd.DataFrame(all_results)

# --- Main Execution ---
if __name__ == "__main__":
    try:
        print("Starting Flexible Mamba Low-Rank Profiling...")
        
        profiler = ModelProfiler()
        
        # Run comprehensive profiling
        results = profiler.run_comprehensive_profiling()
        
        if not results.empty:
            # Save results
            output_file = 'mamba_flexible_profiling_results.csv'
            results.to_csv(output_file, index=False)
            print(f"\n{'='*60}")
            print(f"Results saved to {output_file}")
            print(f"{'='*60}")
            
            # Print summary
            print("\nSummary by Layer Group:")
            summary = results.groupby('layer_group').agg({
                'perplexity': ['mean', 'std', 'min'],
                'num_layers_compressed': 'first'
            }).round(3)
            print(summary)
            
            print("\nBest configurations (lowest PPL increase):")
            baseline_ppl = results[results['layer_group'] == 'Original']['perplexity'].mean()
            results['ppl_increase'] = results['perplexity'] - baseline_ppl
            best = results.nsmallest(10, 'ppl_increase')[
                ['layer_group', 'rank', 'context_length', 'perplexity', 'ppl_increase']
            ]
            print(best)
        else:
            print("No results generated!")
        
    except Exception as e:
        print(f"Critical error: {e}")
        import traceback
        traceback.print_exc()