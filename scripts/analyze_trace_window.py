#!/usr/bin/env python3
"""
Analyze a specific time window in the profiler trace.
Optimized for large traces.
"""
import json
import sys
from collections import defaultdict


def main():
    trace_path = sys.argv[1] if len(sys.argv) > 1 else 'profile_trace/trace.json'
    start_ms = float(sys.argv[2]) if len(sys.argv) > 2 else 161.0
    end_ms = float(sys.argv[3]) if len(sys.argv) > 3 else 161.5

    print(f"Loading trace from {trace_path}...")
    with open(trace_path) as f:
        trace = json.load(f)

    events = trace.get('traceEvents', [])
    print(f"Total events: {len(events):,}")

    # Single pass: collect all duration events and find min timestamp
    duration_events = []
    min_ts = float('inf')
    max_ts = 0

    for e in events:
        if e.get('ph') == 'X':
            ts = e['ts']
            dur = e.get('dur', 0)
            if ts < min_ts:
                min_ts = ts
            if ts + dur > max_ts:
                max_ts = ts + dur
            duration_events.append(e)

    print(f"Duration events: {len(duration_events):,}")
    print(f"Trace duration: {(max_ts - min_ts)/1000:.2f}ms")

    # Convert window to absolute timestamps
    window_start = min_ts + (start_ms * 1000)
    window_end = min_ts + (end_ms * 1000)

    print(f"\nAnalyzing window: {start_ms}ms - {end_ms}ms")

    # Filter events in window
    events_in_window = []
    for e in duration_events:
        e_start = e['ts']
        e_end = e_start + e.get('dur', 0)
        if e_start < window_end and e_end > window_start:
            events_in_window.append(e)

    print(f"Events in window: {len(events_in_window)}")

    # Skip annotations for detailed view
    skip_cats = {'user_annotation', 'gpu_user_annotation', 'python_function'}

    # Sort by start time
    events_in_window.sort(key=lambda e: e['ts'])

    # Print timeline
    print(f"\nTimeline (excluding annotations):")
    for e in events_in_window:
        cat = e.get('cat', 'unknown')
        if cat in skip_cats:
            continue
        rel_start = (e['ts'] - min_ts) / 1000
        dur_us = e.get('dur', 0)
        print(f"  [{rel_start:7.2f}ms +{dur_us:5.0f}us] {cat:15} | {e['name'][:55]}")


if __name__ == '__main__':
    main()
