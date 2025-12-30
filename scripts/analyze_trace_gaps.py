#!/usr/bin/env python3
"""
Analyze PyTorch profiler trace for GPU utilization gaps.
"""
import json
import sys
from collections import defaultdict


def load_trace(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


def main():
    trace_path = sys.argv[1] if len(sys.argv) > 1 else 'profile_trace/trace.json'
    min_gap_us = float(sys.argv[2]) if len(sys.argv) > 2 else 10.0

    print(f"Loading {trace_path}...")
    trace = load_trace(trace_path)
    events = trace.get('traceEvents', [])

    # Single pass
    gpu_kernels = []
    cpu_ops = []
    memcpy_ops = []
    linear_ops = []
    cat_ops = []
    min_ts = float('inf')

    for e in events:
        if e.get('ph') != 'X':
            continue
        cat = e.get('cat', '')
        name = e.get('name', '')
        ts = e['ts']

        if ts < min_ts:
            min_ts = ts

        if cat == 'kernel':
            gpu_kernels.append(e)
        elif cat == 'gpu_memcpy':
            memcpy_ops.append(e)
        elif cat == 'cpu_op':
            cpu_ops.append(e)
            if 'linear' in name.lower():
                linear_ops.append(e)
            if 'cat' in name.lower():
                cat_ops.append(e)

    print(f"GPU kernels: {len(gpu_kernels):,}")
    print(f"GPU memcpy: {len(memcpy_ops):,}")
    print(f"CPU ops: {len(cpu_ops):,}")

    # Analyze linear ops (value head)
    print(f"\n=== LINEAR OPERATIONS (Value Head) ===")
    print(f"Total linear ops: {len(linear_ops)}")
    linear_times = [e.get('dur', 0) for e in linear_ops]
    if linear_times:
        print(f"Total time: {sum(linear_times)/1000:.2f}ms")
        print(f"Average: {sum(linear_times)/len(linear_times):.1f}us")
        print(f"Max: {max(linear_times):.1f}us")

    # Analyze cat ops
    print(f"\n=== CAT OPERATIONS ===")
    print(f"Total cat ops: {len(cat_ops)}")
    cat_times = [e.get('dur', 0) for e in cat_ops]
    if cat_times:
        print(f"Total time: {sum(cat_times)/1000:.2f}ms")
        print(f"Average: {sum(cat_times)/len(cat_times):.1f}us")

    # Analyze DtoD memcpy
    print(f"\n=== DtoD MEMORY COPIES ===")
    dtod = [e for e in memcpy_ops if 'DtoD' in e.get('name', '')]
    print(f"Total DtoD copies: {len(dtod)}")
    dtod_times = [e.get('dur', 0) for e in dtod]
    if dtod_times:
        print(f"Total time: {sum(dtod_times)/1000:.2f}ms")
        print(f"Average: {sum(dtod_times)/len(dtod_times):.1f}us")

    # Find gaps
    gpu_kernels.sort(key=lambda e: e['ts'])
    gaps = []
    for i in range(1, len(gpu_kernels)):
        prev = gpu_kernels[i-1]
        curr = gpu_kernels[i]
        prev_end = prev['ts'] + prev.get('dur', 0)
        gap = curr['ts'] - prev_end
        if gap >= min_gap_us:
            gaps.append({
                'gap_us': gap,
                'prev': prev['name'],
                'next': curr['name'],
            })

    print(f"\n=== GPU GAPS >= {min_gap_us}us ===")
    print(f"Total gaps: {len(gaps)}")
    total_gap = sum(g['gap_us'] for g in gaps)
    print(f"Total gap time: {total_gap/1000:.2f}ms")

    # Categorize gaps by what precedes them
    print(f"\n=== GAP PATTERNS ===")

    # Look for specific patterns
    linear_gaps = [g for g in gaps if 'linear' in g['prev'].lower() or 'addmm' in g['prev'].lower()]
    cat_gaps = [g for g in gaps if 'cat' in g['prev'].lower()]
    memcpy_gaps = [g for g in gaps if 'memcpy' in g['prev'].lower()]
    compiled_gaps = [g for g in gaps if 'triton' in g['prev'].lower()]

    print(f"After linear/addmm: {len(linear_gaps)} gaps, {sum(g['gap_us'] for g in linear_gaps)/1000:.2f}ms")
    print(f"After cat: {len(cat_gaps)} gaps, {sum(g['gap_us'] for g in cat_gaps)/1000:.2f}ms")
    print(f"After memcpy: {len(memcpy_gaps)} gaps, {sum(g['gap_us'] for g in memcpy_gaps)/1000:.2f}ms")
    print(f"After triton (compiled): {len(compiled_gaps)} gaps, {sum(g['gap_us'] for g in compiled_gaps)/1000:.2f}ms")

    # Summary
    trace_dur = max(e['ts'] + e.get('dur', 0) for e in gpu_kernels) - min_ts
    print(f"\n=== SUMMARY ===")
    print(f"Trace duration: {trace_dur/1000:.2f}ms")
    print(f"Total GPU gap time: {total_gap/1000:.2f}ms ({100*total_gap/trace_dur:.1f}%)")
    print(f"  - After compiled ops: {sum(g['gap_us'] for g in compiled_gaps)/1000:.2f}ms")
    print(f"  - After eager ops: {(total_gap - sum(g['gap_us'] for g in compiled_gaps))/1000:.2f}ms")


if __name__ == '__main__':
    main()
