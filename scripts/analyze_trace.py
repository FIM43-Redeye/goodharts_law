#!/usr/bin/env python3
"""
Analyze PyTorch profiler trace to find GPU gaps and their causes.

Loads a Chrome trace JSON and identifies:
1. GPU idle periods (gaps between kernel completions and next kernel start)
2. What CPU work is happening during those gaps
3. Summary of top gap causes

Usage:
    python scripts/analyze_trace.py [trace.json]
"""

import json
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path


@dataclass
class Event:
    name: str
    cat: str  # category
    ph: str   # phase: 'X' = complete, 'B' = begin, 'E' = end
    ts: float  # timestamp in microseconds
    dur: float  # duration in microseconds (for 'X' events)
    tid: int   # thread id
    pid: int   # process id

    @property
    def end_ts(self) -> float:
        return self.ts + self.dur


def load_trace(path: str) -> list[Event]:
    """Load Chrome trace JSON and parse events."""
    try:
        import orjson  # Much faster than json
        with open(path, 'rb') as f:
            data = orjson.loads(f.read())
    except ImportError:
        with open(path) as f:
            data = json.load(f)

    events = []
    trace_events = data.get('traceEvents', data)  # Handle both formats

    # Only keep complete events with duration
    for e in trace_events:
        if not isinstance(e, dict):
            continue
        ph = e.get('ph')
        if ph != 'X':  # Only complete events
            continue
        dur = e.get('dur', 0)
        if dur <= 0:
            continue

        events.append(Event(
            name=e.get('name', ''),
            cat=e.get('cat', ''),
            ph=ph,
            ts=e.get('ts', 0),
            dur=dur,
            tid=e.get('tid', 0),
            pid=e.get('pid', 0),
        ))

    return events


def find_gpu_gaps(events: list[Event], min_gap_us: float = 50) -> list[tuple[float, float]]:
    """
    Find gaps in GPU activity.

    Returns list of (gap_start_us, gap_duration_us) tuples.
    """
    # Find GPU events (CUDA kernels)
    # GPU events typically have 'cuda' or 'gpu' in category, or are on specific threads
    gpu_events = []
    for e in events:
        cat_lower = e.cat.lower() if e.cat else ''
        name_lower = e.name.lower() if e.name else ''

        # Identify GPU kernels by category or name patterns
        is_gpu = (
            'cuda' in cat_lower or
            'gpu' in cat_lower or
            'kernel' in cat_lower or
            'miopen' in name_lower or
            'hipLaunchKernel' in e.name or
            'void at::native::' in e.name or  # PyTorch CUDA kernels
            e.name.startswith('void ') or  # CUDA kernel naming convention
            'Memcpy' in e.name or
            'Memset' in e.name
        )

        if is_gpu and e.ph == 'X' and e.dur > 0:
            gpu_events.append(e)

    if not gpu_events:
        print("Warning: No GPU events found in trace")
        return []

    # Sort by start time
    gpu_events.sort(key=lambda e: e.ts)

    # Find gaps between consecutive GPU events
    gaps = []
    for i in range(1, len(gpu_events)):
        prev_end = gpu_events[i-1].end_ts
        curr_start = gpu_events[i].ts
        gap = curr_start - prev_end

        if gap >= min_gap_us:
            gaps.append((prev_end, gap))

    return gaps


def find_cpu_during_gaps(events: list[Event], gaps: list[tuple[float, float]]) -> dict[str, float]:
    """
    Find what CPU work is happening during GPU gaps.

    Returns dict of {cpu_event_name: total_time_during_gaps_us}
    """
    # Get CPU events (non-GPU complete events)
    cpu_events = []
    for e in events:
        cat_lower = e.cat.lower() if e.cat else ''

        is_cpu = (
            e.ph == 'X' and
            e.dur > 0 and
            'cuda' not in cat_lower and
            'gpu' not in cat_lower and
            'kernel' not in cat_lower and
            not e.name.startswith('void ')
        )

        if is_cpu:
            cpu_events.append(e)

    # For each gap, find overlapping CPU events
    cpu_time_in_gaps = defaultdict(float)

    for gap_start, gap_dur in gaps:
        gap_end = gap_start + gap_dur

        for e in cpu_events:
            # Check overlap
            overlap_start = max(e.ts, gap_start)
            overlap_end = min(e.end_ts, gap_end)
            overlap = overlap_end - overlap_start

            if overlap > 0:
                # Simplify event name for grouping
                name = simplify_name(e.name)
                cpu_time_in_gaps[name] += overlap

    return dict(cpu_time_in_gaps)


def simplify_name(name: str) -> str:
    """Simplify event name for grouping."""
    # Remove template parameters
    if '<' in name:
        name = name[:name.index('<')]

    # Remove function arguments
    if '(' in name:
        name = name[:name.index('(')]

    # Common patterns to group
    if 'autograd::engine' in name:
        return 'autograd::engine'
    if 'aten::' in name:
        # Keep the operation name
        parts = name.split('::')
        if len(parts) >= 2:
            return f"aten::{parts[-1]}"
    if 'Compiled' in name:
        return 'torch.compile overhead'
    if 'empty' in name.lower() or 'resize' in name.lower():
        return 'memory allocation'
    if 'copy' in name.lower():
        return 'memory copy'

    return name.strip()


def analyze_trace(path: str):
    """Main analysis function."""
    print(f"\nAnalyzing trace: {path}\n")
    print("=" * 70)

    events = load_trace(path)
    print(f"Loaded {len(events)} events")

    # Find GPU gaps
    gaps = find_gpu_gaps(events, min_gap_us=50)

    if not gaps:
        print("No significant GPU gaps found (>50us)")
        return

    total_gap_time = sum(g[1] for g in gaps)
    avg_gap = total_gap_time / len(gaps)
    max_gap = max(g[1] for g in gaps)

    print(f"\nGPU GAPS SUMMARY")
    print("-" * 70)
    print(f"Total gaps found:     {len(gaps)}")
    print(f"Total gap time:       {total_gap_time/1000:.2f} ms")
    print(f"Average gap:          {avg_gap:.1f} us")
    print(f"Max gap:              {max_gap:.1f} us")

    # Distribution of gap sizes
    print(f"\nGap size distribution:")
    buckets = [100, 200, 500, 1000, 2000, 5000, 10000]
    prev = 0
    for bucket in buckets:
        count = sum(1 for _, dur in gaps if prev <= dur < bucket)
        if count > 0:
            print(f"  {prev:>5} - {bucket:<5} us: {count:>4} gaps")
        prev = bucket
    count = sum(1 for _, dur in gaps if dur >= buckets[-1])
    if count > 0:
        print(f"  {buckets[-1]:>5}+ us:       {count:>4} gaps")

    # Find CPU work during gaps
    cpu_during_gaps = find_cpu_during_gaps(events, gaps)

    if cpu_during_gaps:
        print(f"\nCPU WORK DURING GPU GAPS (top 20)")
        print("-" * 70)

        # Sort by time spent
        sorted_cpu = sorted(cpu_during_gaps.items(), key=lambda x: -x[1])

        for name, time_us in sorted_cpu[:20]:
            pct = 100 * time_us / total_gap_time
            print(f"  {time_us/1000:>8.2f} ms ({pct:>5.1f}%)  {name}")

    # Find the largest gaps specifically
    print(f"\nLARGEST GAPS (top 10)")
    print("-" * 70)
    largest = sorted(gaps, key=lambda x: -x[1])[:10]
    for gap_start, gap_dur in largest:
        print(f"  {gap_dur:>8.1f} us at t={gap_start/1000:.2f} ms")

    # Detailed analysis of largest gap
    if largest:
        biggest_start, biggest_dur = largest[0]
        biggest_end = biggest_start + biggest_dur
        print(f"\nDETAIL: Largest gap ({biggest_dur:.1f} us)")
        print("-" * 70)

        # Find all CPU events overlapping this gap
        overlapping = []
        for e in events:
            if e.ph != 'X' or e.dur <= 0:
                continue
            cat_lower = e.cat.lower() if e.cat else ''
            if 'cuda' in cat_lower or 'gpu' in cat_lower:
                continue

            # Check overlap
            if e.ts < biggest_end and e.end_ts > biggest_start:
                overlap_start = max(e.ts, biggest_start)
                overlap_end = min(e.end_ts, biggest_end)
                overlap = overlap_end - overlap_start
                if overlap > 0:
                    overlapping.append((overlap, e.name, e.ts, e.dur))

        # Sort by overlap time
        overlapping.sort(key=lambda x: -x[0])
        for overlap, name, ts, dur in overlapping[:15]:
            name_short = simplify_name(name) if len(name) > 50 else name
            print(f"  {overlap:>8.1f} us overlap: {name_short}")

    print("\n" + "=" * 70)


if __name__ == '__main__':
    if len(sys.argv) > 1:
        trace_path = sys.argv[1]
    else:
        trace_path = './profile_trace/trace.json'

    if not Path(trace_path).exists():
        print(f"Error: {trace_path} not found")
        print("Run: python -m goodharts.training.train_ppo --mode ground_truth --profile-trace 5")
        sys.exit(1)

    analyze_trace(trace_path)
