#!/usr/bin/env python3
"""
Test CUDA/HIP graphs on AMD ROCm.

CUDA graphs capture a sequence of GPU operations and replay them with
minimal dispatch overhead. On AMD, this is implemented via HIP graphs.

This script tests whether graphs work for the operations we'd use in PPO:
1. Simple tensor ops
2. Convolution forward pass
3. Linear layers
4. The full inference step
"""

import torch
import torch.nn as nn
import time


def test_basic_graph():
    """Test basic CUDA graph capture and replay."""
    print("\n1. Basic tensor operations...")

    device = torch.device('cuda')

    # Static tensors (must not change between captures)
    a = torch.randn(1024, 1024, device=device)
    b = torch.randn(1024, 1024, device=device)
    c = torch.zeros(1024, 1024, device=device)

    # Warmup
    for _ in range(3):
        c = a @ b + c
    torch.cuda.synchronize()

    # Capture graph
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        c = a @ b + c

    # Replay
    for _ in range(10):
        g.replay()
    torch.cuda.synchronize()

    print("   Basic graph: OK")
    return True


def test_conv_graph():
    """Test CUDA graph with convolution."""
    print("\n2. Convolution forward pass...")

    device = torch.device('cuda')

    # Create a simple conv layer
    conv = nn.Conv2d(6, 32, 3, padding=1).to(device)
    x = torch.randn(192, 6, 11, 11, device=device)

    # Warmup
    for _ in range(3):
        y = conv(x)
    torch.cuda.synchronize()

    # Capture graph
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        y = conv(x)

    # Replay
    for _ in range(10):
        g.replay()
    torch.cuda.synchronize()

    print("   Conv graph: OK")
    return True


def test_inference_graph():
    """Test CUDA graph with full inference step."""
    print("\n3. Full inference (CNN + Linear)...")

    device = torch.device('cuda')

    # Simple CNN similar to base_cnn
    class SimpleCNN(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = nn.Conv2d(6, 32, 3, padding=1)
            self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
            self.conv3 = nn.Conv2d(64, 64, 3, padding=1)
            self.fc = nn.Linear(64 * 11 * 11, 512)
            self.policy = nn.Linear(512, 8)
            self.value = nn.Linear(512, 1)

        def forward(self, x):
            x = torch.relu(self.conv1(x))
            x = torch.relu(self.conv2(x))
            x = torch.relu(self.conv3(x))
            x = x.flatten(1)
            x = torch.relu(self.fc(x))
            return self.policy(x), self.value(x)

    model = SimpleCNN().to(device)
    x = torch.randn(192, 6, 11, 11, device=device)

    # Warmup
    for _ in range(3):
        logits, values = model(x)
    torch.cuda.synchronize()

    # Capture graph
    g = torch.cuda.CUDAGraph()

    # Static output buffers (graph outputs must be pre-allocated)
    static_logits = torch.zeros(192, 8, device=device)
    static_values = torch.zeros(192, 1, device=device)

    with torch.cuda.graph(g):
        logits, values = model(x)
        static_logits.copy_(logits)
        static_values.copy_(values)

    # Replay
    for _ in range(10):
        g.replay()
    torch.cuda.synchronize()

    print("   Inference graph: OK")
    return True


def test_graph_with_input_update():
    """Test updating input tensor and replaying graph."""
    print("\n4. Graph with input updates...")

    device = torch.device('cuda')

    # Create model
    model = nn.Linear(1024, 1024).to(device)

    # Static input buffer - we'll update this between replays
    static_input = torch.randn(256, 1024, device=device)
    static_output = torch.zeros(256, 1024, device=device)

    # Warmup
    for _ in range(3):
        out = model(static_input)
    torch.cuda.synchronize()

    # Capture graph
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        out = model(static_input)
        static_output.copy_(out)

    # Replay with different inputs
    for i in range(10):
        # Update input (copy new data into static buffer)
        new_data = torch.randn(256, 1024, device=device)
        static_input.copy_(new_data)

        # Replay graph
        g.replay()

    torch.cuda.synchronize()
    print("   Input update graph: OK")
    return True


def benchmark_graph_speedup():
    """Benchmark graph replay vs eager execution."""
    print("\n5. Benchmark: Graph vs Eager...")

    device = torch.device('cuda')

    # Model
    model = nn.Sequential(
        nn.Linear(1024, 1024),
        nn.ReLU(),
        nn.Linear(1024, 1024),
        nn.ReLU(),
        nn.Linear(1024, 512),
    ).to(device)

    x = torch.randn(256, 1024, device=device)
    y = torch.zeros(256, 512, device=device)

    # Warmup
    for _ in range(10):
        y = model(x)
    torch.cuda.synchronize()

    # Eager benchmark
    n_iters = 1000
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(n_iters):
        y = model(x)
    torch.cuda.synchronize()
    eager_time = time.perf_counter() - t0

    # Capture graph
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        out = model(x)
        y.copy_(out)

    # Graph benchmark
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(n_iters):
        g.replay()
    torch.cuda.synchronize()
    graph_time = time.perf_counter() - t0

    print(f"   Eager: {eager_time*1000:.2f} ms ({n_iters} iters)")
    print(f"   Graph: {graph_time*1000:.2f} ms ({n_iters} iters)")
    print(f"   Speedup: {eager_time/graph_time:.2f}x")

    return True


def main():
    print("=" * 60)
    print("CUDA/HIP Graph Test")
    print("=" * 60)

    if not torch.cuda.is_available():
        print("CUDA not available!")
        return

    print(f"Device: {torch.cuda.get_device_name()}")
    print(f"PyTorch: {torch.__version__}")

    tests = [
        ("Basic ops", test_basic_graph),
        ("Convolution", test_conv_graph),
        ("Full inference", test_inference_graph),
        ("Input updates", test_graph_with_input_update),
        ("Benchmark", benchmark_graph_speedup),
    ]

    results = []
    for name, test_fn in tests:
        try:
            success = test_fn()
            results.append((name, success, None))
        except Exception as e:
            print(f"   FAILED: {e}")
            results.append((name, False, str(e)))

    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    for name, success, error in results:
        status = "PASS" if success else f"FAIL: {error}"
        print(f"  {name}: {status}")

    all_passed = all(s for _, s, _ in results)
    print(f"\nOverall: {'ALL TESTS PASSED' if all_passed else 'SOME TESTS FAILED'}")

    if all_passed:
        print("\nCUDA/HIP graphs work on this system!")
        print("We can try using them for inference in the training loop.")


if __name__ == '__main__':
    main()
