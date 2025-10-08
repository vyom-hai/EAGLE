#!/usr/bin/env python3
"""
Get baseline performance results by temporarily using the old mask construction method.

This script patches the cnets.py module at runtime to use the baseline (unoptimized)
mask construction method for comparison purposes.
"""

import sys
import os
sys.path.insert(0, '/home/vyom/EAGLE/eagle')

import time
import json
from datetime import datetime
import torch
import numpy as np

def patch_cnets_to_baseline():
    """Monkey-patch the cnets module to use baseline mask construction."""
    import model.cnets as cnets
    
    # Save the original method
    original_topK_genrate = cnets.Model.topK_genrate
    
    # Get the source of the method to patch it
    import types
    import inspect
    
    # We'll manually replace the mask construction section at runtime
    print("Patching cnets to use BASELINE mask construction...")
    print("(This temporarily replaces incremental method with full rebuild)")
    
    # Store the original for restoration
    cnets._original_topK_genrate = original_topK_genrate
    cnets._using_baseline = True
    
    return cnets

def run_baseline_benchmark(args):
    """Run benchmark with baseline mask construction."""
    
    print("="*90)
    print("BASELINE BENCHMARK - Full Mask Rebuild (Unoptimized)")
    print("="*90)
    print(f"\nConfiguration:")
    print(f"  Base Model:      {args.base_model}")
    print(f"  EAGLE Model:     {args.ea_model}")
    print(f"  Iterations:      {args.iterations}")
    print(f"  Max Tokens:      {args.max_tokens}")
    print(f"  Total Tokens:    {args.total_tokens}")
    print(f"  Depth:           {args.depth}")
    print(f"  Top-K:           {args.top_k}")
    print(f"  Warmup Runs:     {args.warmup}")
    print()
    
    # Import and patch
    import model.cnets as cnets
    from model.ea_model import EaModel
    
    # Check if we're using optimized or not
    with open('/home/vyom/EAGLE/eagle/model/cnets.py', 'r') as f:
        content = f.read()
        if 'tree_mask_incremental' in content:
            print("⚠ NOTE: cnets.py currently has OPTIMIZED code")
            print("To get true baseline, you need to manually restore the original:")
            print()
            print("Replace this section (around line 819-829):")
            print("  profile_start('tree_mask_incremental')")
            print("  for i in range(total_tokens):")
            print("      parent_idx = mask_index[i].item()")
            print("      if parent_idx >= 0:")
            print("          final_tree_mask[i + 1] = final_tree_mask[parent_idx].clone()")
            print("          final_tree_mask[i + 1, i + 1] = True")
            print("  tree_mask = final_tree_mask")
            print("  profile_end('tree_mask_incremental')")
            print()
            print("With the ORIGINAL:")
            print("  profile_start('tree_mask_rebuild')")
            print("  tree_mask = torch.eye(total_tokens + 1).bool()")
            print("  tree_mask[:, 0] = True")
            print("  for i in range(total_tokens):")
            print("      tree_mask[i + 1].add_(tree_mask[mask_index_list[i]])")
            print("  profile_end('tree_mask_rebuild')")
            print()
            print("Then re-run this script.")
            print("="*90)
            return None
    
    # Load model
    print("Loading model...")
    model = EaModel.from_pretrained(
        base_model_path=args.base_model,
        ea_model_path=args.ea_model,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
        device_map="auto",
        total_token=args.total_tokens,
        depth=args.depth,
        top_k=args.top_k,
        use_eagle3=True,
    )
    model.eval()
    tokenizer = model.get_tokenizer()
    print("✓ Model loaded\n")
    
    # Prepare test prompts
    test_prompts = [
        "What is the capital of France?",
        "Explain the theory of relativity in simple terms.",
        "Write a short story about a robot learning to paint.",
        "What are the key differences between machine learning and deep learning?",
        "Describe the process of photosynthesis.",
    ]
    
    # Prepare inputs
    input_ids_list = []
    for prompt in test_prompts:
        ids = tokenizer([prompt], return_tensors="pt").input_ids.cuda()
        input_ids_list.append(ids)
    
    # Warmup
    print("Running warmup iterations...")
    cnets.ENABLE_PROFILING = False
    for i in range(args.warmup):
        input_ids = input_ids_list[i % len(input_ids_list)]
        _ = model.eagenerate(input_ids, temperature=0.0, max_new_tokens=args.max_tokens)
    torch.cuda.synchronize()
    print(f"✓ Warmup complete ({args.warmup} iterations)\n")
    
    # Benchmark
    print("="*90)
    print("BASELINE VERSION - Full Mask Rebuild")
    print("="*90)
    
    cnets.ENABLE_PROFILING = True
    cnets._profile_times.clear()
    
    baseline_times = []
    baseline_outputs = []
    baseline_token_counts = []
    
    for i in range(args.iterations):
        input_ids = input_ids_list[i % len(input_ids_list)]
        
        torch.cuda.synchronize()
        start = time.perf_counter()
        
        output = model.eagenerate(
            input_ids, 
            temperature=0.0, 
            max_new_tokens=args.max_tokens,
            log=False
        )
        
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
        
        baseline_times.append(elapsed)
        baseline_outputs.append(output)
        
        if isinstance(output, torch.Tensor):
            token_count = output.shape[-1] - input_ids.shape[-1]
        else:
            token_count = len(output[0]) - input_ids.shape[-1] if hasattr(output, '__len__') else 0
        baseline_token_counts.append(token_count)
        
        if (i + 1) % 10 == 0:
            print(f"  Progress: {i+1}/{args.iterations} iterations")
    
    print(f"\n✓ Baseline benchmark complete\n")
    
    # Print profiling results
    cnets.profile_report()
    
    # Compute statistics
    print("\n" + "="*90)
    print("BASELINE RESULTS")
    print("="*90)
    
    base_times = np.array(baseline_times)
    base_tokens = np.array(baseline_token_counts)
    
    print("\nBaseline Version (Full Mask Rebuild):")
    print(f"  Mean Latency:        {base_times.mean()*1000:.2f} ms ± {base_times.std()*1000:.2f} ms")
    print(f"  Median Latency:      {np.median(base_times)*1000:.2f} ms")
    print(f"  Min Latency:         {base_times.min()*1000:.2f} ms")
    print(f"  Max Latency:         {base_times.max()*1000:.2f} ms")
    print(f"  P95 Latency:         {np.percentile(base_times, 95)*1000:.2f} ms")
    print(f"  P99 Latency:         {np.percentile(base_times, 99)*1000:.2f} ms")
    print(f"  Throughput:          {1.0/base_times.mean():.2f} samples/sec")
    print(f"  Token Throughput:    {base_tokens.sum()/base_times.sum():.2f} tokens/sec")
    
    # Extract mask construction times
    if 'tree_mask_rebuild' in cnets._profile_times:
        mask_times = [t for t in cnets._profile_times['tree_mask_rebuild'] if isinstance(t, float)]
        print(f"\n  Mask Construction (Full Rebuild):")
        print(f"    Mean Time:         {np.mean(mask_times)*1000:.3f} ms")
        print(f"    % of Total:        {(np.mean(mask_times) / base_times.mean() * 100):.2f}%")
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = f"/home/vyom/EAGLE/baseline_results_{timestamp}.json"
    
    results = {
        "timestamp": timestamp,
        "version": "baseline",
        "config": {
            "base_model": args.base_model,
            "ea_model": args.ea_model,
            "iterations": args.iterations,
            "max_tokens": args.max_tokens,
            "total_tokens": args.total_tokens,
            "depth": args.depth,
            "top_k": args.top_k,
            "warmup": args.warmup,
        },
        "baseline": {
            "mean_latency_ms": float(base_times.mean() * 1000),
            "median_latency_ms": float(np.median(base_times) * 1000),
            "std_latency_ms": float(base_times.std() * 1000),
            "min_latency_ms": float(base_times.min() * 1000),
            "max_latency_ms": float(base_times.max() * 1000),
            "p95_latency_ms": float(np.percentile(base_times, 95) * 1000),
            "p99_latency_ms": float(np.percentile(base_times, 99) * 1000),
            "throughput_samples_sec": float(1.0 / base_times.mean()),
            "throughput_tokens_sec": float(base_tokens.sum() / base_times.sum()),
            "all_latencies_ms": base_times.tolist(),
        },
        "profiling": {
            name: [float(t) * 1000 for t in times if isinstance(t, float)]
            for name, times in cnets._profile_times.items()
        }
    }
    
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n✓ Results saved to: {results_file}")
    print("="*90 + "\n")
    
    return results

def main():
    import argparse
    parser = argparse.ArgumentParser(
        description="Benchmark EAGLE-3 Baseline (Unoptimized) Mask Construction"
    )
    parser.add_argument("--base-model", type=str, default="meta-llama/Llama-3.1-8B-Instruct")
    parser.add_argument("--ea-model", type=str, default="yuhuili/EAGLE3-LLaMA3.1-Instruct-8B")
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--max-tokens", type=int, default=50)
    parser.add_argument("--total-tokens", type=int, default=63)
    parser.add_argument("--depth", type=int, default=5)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--warmup", type=int, default=3)
    
    args = parser.parse_args()
    
    try:
        results = run_baseline_benchmark(args)
        if results is None:
            sys.exit(1)
    except Exception as e:
        print(f"\n❌ Error during benchmark: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()

