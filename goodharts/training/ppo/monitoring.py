"""
GPU utilization monitoring for training.

Provides background thread that logs GPU utilization at fixed intervals.
Uses sysfs (AMD) or nvidia-smi (NVIDIA) for low-overhead sampling.
"""
import logging
import os
import time
import threading
from typing import Optional

logger = logging.getLogger(__name__)


class GPUMonitor:
    """
    Background thread that logs GPU utilization at fixed intervals.

    Uses sysfs (AMD, ~0.02ms) or nvidia-smi (NVIDIA) to sample GPU use %.
    Starts after warmup, stops when training ends. Output is CSV with
    millisecond timestamps for correlation with training events.
    """

    def __init__(self, interval_ms: int = 50, output_path: str = "gpu_utilization.csv"):
        self.interval_ms = interval_ms
        self.output_path = output_path
        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._start_time_ms: int = 0
        self._sysfs_path: Optional[str] = None  # Fast path for AMD
        self._nvidia_cmd: Optional[list] = None  # Fallback for NVIDIA

    def _detect_amd_card(self) -> Optional[str]:
        """
        Find the active AMD GPU's sysfs path by testing utilization.

        Returns path like '/sys/class/drm/card1/device/gpu_busy_percent'
        or None if not found.
        """
        import glob

        candidates = glob.glob('/sys/class/drm/card*/device/gpu_busy_percent')
        if not candidates:
            return None

        # If only one card, use it
        if len(candidates) == 1:
            return candidates[0]

        # Multiple cards: find the one PyTorch is using by running a quick workload
        try:
            import torch
            if not torch.cuda.is_available():
                return None

            # Baseline read
            baseline = {}
            for path in candidates:
                with open(path) as f:
                    baseline[path] = int(f.read().strip())

            # Quick GPU work
            x = torch.randn(1024, 1024, device='cuda')
            for _ in range(20):
                x = x @ x
            torch.cuda.synchronize()

            # Find which card spiked
            for path in candidates:
                with open(path) as f:
                    now = int(f.read().strip())
                if now > baseline[path] + 20:  # Significant increase
                    return path

            # Fallback: return first card with vendor 0x1002 (AMD)
            for path in candidates:
                vendor_path = os.path.join(os.path.dirname(path), 'vendor')
                if os.path.exists(vendor_path):
                    with open(vendor_path) as f:
                        if '0x1002' in f.read():
                            return path
        except Exception as e:
            logger.debug(f"GPU detection probe failed: {e}")

        return candidates[0]  # Last resort

    def start(self):
        """Start the monitoring thread."""
        import subprocess

        # Try AMD sysfs first (fast: ~0.02ms per read)
        self._sysfs_path = self._detect_amd_card()
        if self._sysfs_path:
            card = self._sysfs_path.split('/')[4]  # Extract 'card1' from path
            print(f"   [GPUMonitor] Using sysfs ({card}) - {self.interval_ms}ms intervals")
        else:
            # Fallback to nvidia-smi (slower: ~5ms per read)
            try:
                result = subprocess.run(
                    ["nvidia-smi", "--query-gpu=utilization.gpu",
                     "--format=csv,noheader,nounits"],
                    capture_output=True, text=True, timeout=2
                )
                if result.returncode == 0:
                    self._nvidia_cmd = ["nvidia-smi", "--query-gpu=utilization.gpu",
                                        "--format=csv,noheader,nounits"]
                    print(f"   [GPUMonitor] Using nvidia-smi - {self.interval_ms}ms intervals")
                else:
                    print(f"   [GPUMonitor] No supported GPU monitoring found")
                    return
            except (FileNotFoundError, subprocess.TimeoutExpired):
                print(f"   [GPUMonitor] No supported GPU monitoring found")
                return

        # Write CSV header
        with open(self.output_path, 'w') as f:
            f.write("timestamp_ms,gpu_use_pct\n")

        self._start_time_ms = int(time.time() * 1000)
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._worker, daemon=True, name="GPUMonitor")
        self._thread.start()

    def stop(self):
        """Stop the monitoring thread and wait for it to finish."""
        if self._thread is None:
            return
        self._stop_event.set()
        self._thread.join(timeout=2.0)
        self._thread = None
        print(f"   [GPUMonitor] Stopped")

    def _worker(self):
        """Background thread: samples GPU utilization."""
        import subprocess

        interval_s = self.interval_ms / 1000.0

        with open(self.output_path, 'a') as f:
            while not self._stop_event.is_set():
                try:
                    if self._sysfs_path:
                        # Fast path: direct sysfs read (~0.02ms)
                        with open(self._sysfs_path) as gpu_file:
                            use_pct = int(gpu_file.read().strip())
                    else:
                        # NVIDIA fallback
                        result = subprocess.run(
                            self._nvidia_cmd, capture_output=True, text=True, timeout=1
                        )
                        use_pct = int(result.stdout.strip().split('\n')[0])

                    elapsed_ms = int(time.time() * 1000) - self._start_time_ms
                    f.write(f"{elapsed_ms},{use_pct}\n")
                    f.flush()
                except Exception:
                    # Intentionally silent: skip failed samples during high-frequency
                    # background sampling. Logging every failure would be noisy.
                    pass

                self._stop_event.wait(interval_s)
