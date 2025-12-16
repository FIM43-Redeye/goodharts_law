"""
Centralized PyTorch device selection.

Provides a single function to get the configured compute device,
allowing users to easily switch between CPU, CUDA, ROCm, TPU (XLA), etc.

Priority order:
1. Explicit override parameter
2. GOODHARTS_DEVICE environment variable
3. config.toml [runtime] device setting
4. Auto-detect (CUDA > TPU > CPU)
"""
import os
import torch
import threading

# Module-level cache for the device (avoid repeated detection)
_cached_device: torch.device | None = None
_device_logged: bool = False

# Lock to ensure global optimizations are applied only once and thread-safely
_OPTIMIZATION_LOCK = threading.Lock()
_optimizations_applied = False

# TPU/XLA support (lazy import)
_xla_available: bool | None = None
_xla_module = None


def is_xla_available() -> bool:
    """Check if torch_xla is available for TPU support."""
    global _xla_available
    if _xla_available is None:
        try:
            import torch_xla
            _xla_available = True
        except ImportError:
            _xla_available = False
    return _xla_available


def _get_xla():
    """Get torch_xla.core.xla_model (lazy import)."""
    global _xla_module
    if _xla_module is None:
        import torch_xla.core.xla_model as xm
        _xla_module = xm
    return _xla_module


def is_tpu(device: torch.device) -> bool:
    """Check if device is a TPU (XLA device)."""
    return device.type == 'xla'


def is_cuda(device: torch.device) -> bool:
    """Check if device is CUDA."""
    return device.type == 'cuda'


def sync_device(device: torch.device):
    """
    Synchronize device operations.
    
    For TPU: calls xm.mark_step() to execute pending XLA operations.
    For CUDA: calls torch.cuda.synchronize().
    For CPU: no-op.
    """
    if is_tpu(device):
        _get_xla().mark_step()
    elif is_cuda(device):
        torch.cuda.synchronize()
    # CPU: no sync needed


def empty_cache(device: torch.device):
    """Clear device memory cache if applicable."""
    if is_cuda(device):
        torch.cuda.empty_cache()
    # TPU and CPU: no cache clearing API


def get_device(override: str | None = None, verbose: bool = True) -> torch.device:
    """
    Get the configured PyTorch device.
    
    Args:
        override: Explicit device string (e.g., 'cpu', 'cuda', 'cuda:1', 'tpu')
                  Takes highest priority if provided.
        verbose: If True, print device info on first call.
    
    Returns:
        torch.device configured for computation
    
    Examples:
        # Auto-detect
        device = get_device()
        
        # Force CPU
        device = get_device('cpu')
        
        # Use specific GPU
        device = get_device('cuda:1')
        
        # Use TPU (requires torch_xla)
        device = get_device('tpu')
        
        # Via environment variable
        # $ GOODHARTS_DEVICE=tpu python train.py
        device = get_device()  # Returns TPU
    """
    global _cached_device, _device_logged
    
    # If explicit override, use it directly (no caching)
    if override is not None:
        if override == 'tpu':
            if not is_xla_available():
                raise RuntimeError("TPU requested but torch_xla not installed. "
                                 "Install with: pip install torch-xla[tpu]")
            device = _get_xla().xla_device()
            if verbose:
                print(f"Device: TPU (XLA)")
            return device
        return torch.device(override)
    
    # Return cached device if already determined
    if _cached_device is not None:
        return _cached_device
    
    # Priority 1: Environment variable
    env_device = os.environ.get('GOODHARTS_DEVICE')
    if env_device:
        if env_device == 'tpu':
            if is_xla_available():
                _cached_device = _get_xla().xla_device()
                if verbose and not _device_logged:
                    print(f"Device: TPU (XLA) (from GOODHARTS_DEVICE env)")
                    _device_logged = True
                return _cached_device
            else:
                print("Warning: TPU requested but torch_xla not available. Falling back.")
        else:
            _cached_device = torch.device(env_device)
            if verbose and not _device_logged:
                print(f"Device: {_cached_device} (from GOODHARTS_DEVICE env)")
                _device_logged = True
            return _cached_device
    
    # Priority 2: Config file
    try:
        from goodharts.config import get_runtime_config
        runtime_cfg = get_runtime_config()
        if runtime_cfg and runtime_cfg.get('device'):
            cfg_device = runtime_cfg['device']
            if cfg_device == 'tpu' and is_xla_available():
                _cached_device = _get_xla().xla_device()
            else:
                _cached_device = torch.device(cfg_device)
            if verbose and not _device_logged:
                print(f"Device: {_cached_device} (from config)")
                _device_logged = True
            return _cached_device
    except (ImportError, KeyError):
        pass  # Config not available or no device setting
    
    # Priority 3: Auto-detect (CUDA > TPU > CPU)
    if torch.cuda.is_available():
        _cached_device = torch.device('cuda')
        if verbose and not _device_logged:
            gpu_name = torch.cuda.get_device_name(0)
            print(f"Device: {_cached_device} ({gpu_name})")
            _device_logged = True
    elif is_xla_available():
        _cached_device = _get_xla().xla_device()
        if verbose and not _device_logged:
            print(f"Device: TPU (XLA)")
            _device_logged = True
    else:
        _cached_device = torch.device('cpu')
        if verbose and not _device_logged:
            print(f"Device: {_cached_device}")
            _device_logged = True
    
    return _cached_device


def reset_device_cache():
    """
    Reset the cached device.
    
    Useful for testing or when device configuration changes.
    """
    global _cached_device, _device_logged
    _cached_device = None
    _device_logged = False


def probe_amp_support(device: torch.device = None, verbose: bool = True, timeout: float = 10.0) -> bool:
    """
    Test if AMP (Automatic Mixed Precision) actually works on the device.
    
    Runs the probe in a subprocess so it can be killed if it hangs
    (which happens on some ROCm configurations where GPU kernels block).
    
    Args:
        device: Device to test (default: auto-detect)
        verbose: Print result of probe
        timeout: Maximum seconds to wait for probe (default: 10)
    
    Returns:
        True if AMP is functional, False otherwise
    """
    if device is None:
        device = get_device(verbose=False)
    
    # CPU doesn't benefit from AMP
    if device.type != 'cuda':
        if verbose:
            print("   AMP: Disabled (CPU device)")
        return False
    
    # Check for ROCm (AMD GPU) - consumer GPUs have AMP issues even when probes pass
    if hasattr(torch.version, 'hip') and torch.version.hip:
        gpu_name = torch.cuda.get_device_name(0).lower()
        
        # Consumer RDNA GPUs (RX series) have known AMP issues
        # Pro/datacenter GPUs (MI, PRO W series) are more likely to work
        is_consumer_amd = 'rx' in gpu_name and any(x in gpu_name for x in ['7', '6'])
        is_pro_datacenter = any(x in gpu_name for x in ['mi100', 'mi200', 'mi250', 'mi300', 'w7', 'pro'])
        
        if is_consumer_amd and not is_pro_datacenter:
            if verbose:
                print(f"   AMP: Disabled (ROCm consumer GPU: {torch.cuda.get_device_name(0)})")
            return False
        elif verbose:
            print(f"   AMP: Probing AMD GPU: {torch.cuda.get_device_name(0)}")
    
    import subprocess
    import sys
    
    # The actual probe code that runs in subprocess
    probe_code = '''
import sys
import torch
from torch.amp import autocast, GradScaler
import torch.nn as nn

try:
    device = torch.device('cuda')
    
    # Create CNN-like test model
    test_model = nn.Sequential(
        nn.Conv2d(4, 16, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.Conv2d(16, 32, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.Flatten(),
        nn.Linear(32 * 8 * 8, 64),
        nn.ReLU(),
        nn.Linear(64, 5)
    ).to(device)
    
    test_input = torch.randn(8, 4, 8, 8, device=device)
    test_target = torch.randint(0, 5, (8,), device=device)
    
    scaler = GradScaler(enabled=True)
    optimizer = torch.optim.Adam(test_model.parameters(), lr=0.001)
    
    # Run multiple iterations
    for i in range(10):
        with autocast(device_type='cuda', enabled=True):
            output = test_model(test_input)
            loss = nn.functional.cross_entropy(output, test_target)
        
        if torch.isnan(loss) or torch.isinf(loss):
            print(f"FAIL:nan_loss:{i}")
            sys.exit(1)
        
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        
        for param in test_model.parameters():
            if param.grad is not None:
                if torch.isnan(param.grad).any() or torch.isinf(param.grad).any():
                    print(f"FAIL:nan_grad:{i}")
                    sys.exit(1)
        
        scaler.step(optimizer)
        scaler.update()
        
        if scaler.get_scale() < 1e-10:
            print(f"FAIL:scale_collapse:{i}")
            sys.exit(1)
    
    print("PASS")
    sys.exit(0)
    
except Exception as e:
    print(f"FAIL:exception:{e}")
    sys.exit(1)
'''
    
    try:
        result = subprocess.run(
            [sys.executable, '-c', probe_code],
            capture_output=True,
            text=True,
            timeout=timeout
        )
        
        output = result.stdout.strip()
        
        if output == "PASS":
            if verbose:
                print("   AMP: Enabled (probe passed)")
            return True
        else:
            if verbose:
                reason = output if output else "unknown failure"
                print(f"   AMP: Disabled ({reason})")
            return False
            
    except subprocess.TimeoutExpired:
        if verbose:
            print("   AMP: Disabled (probe timed out - GPU operations hung)")
        return False
    except Exception as e:
        if verbose:
            print(f"   AMP: Disabled (probe error: {e})")
        return False


# Cache the AMP probe result to avoid repeated subprocess calls
_amp_support_cached: bool | None = None

def get_amp_support(device: torch.device = None, verbose: bool = True) -> bool:
    """
    Get cached AMP support status. Runs probe once and caches result.
    """
    global _amp_support_cached
    if _amp_support_cached is None:
        _amp_support_cached = probe_amp_support(device, verbose)
    return _amp_support_cached

def reset_amp_cache():
    """Reset the cached AMP support status."""
    global _amp_support_cached
    _amp_support_cached = None


def get_device_info() -> dict:
    """
    Get detailed information about the current device.
    
    Returns:
        Dict with device info (useful for logging/debugging)
    """
    device = get_device(verbose=False)
    info = {
        'device': str(device),
        'type': device.type,
    }
    
    if device.type == 'cuda':
        info.update({
            'cuda_available': True,
            'gpu_name': torch.cuda.get_device_name(device),
            'gpu_count': torch.cuda.device_count(),
            'cuda_version': torch.version.cuda,
            'memory_allocated': torch.cuda.memory_allocated(device),
            'memory_reserved': torch.cuda.memory_reserved(device),
        })
    else:
        info['cuda_available'] = torch.cuda.is_available()
    
    return info


def apply_system_optimizations(device: torch.device = None, verbose: bool = True):
    """
    Apply global performance optimizations for the given device.
    
    Automatically enables:
    - TensorFloat32 (TF32) for matmuls on Ampere+ GPUs (huge speedup for FP32)
    - cuDNN benchmarking (finds best convolution algorithms)
    
    Args:
        device: Device to optimize for (default: auto-detect)
        verbose: Print applied optimizations
    """
    global _optimizations_applied
    
    # Fast path - check if already done
    if _optimizations_applied:
        return

    # Thread-safe initialization
    with _OPTIMIZATION_LOCK:
        # Check again inside lock
        if _optimizations_applied:
            return
            
        if device is None:
            device = get_device(verbose=False)
            
        if device.type == 'cuda':
            # 1. Enable TensorFloat32 (TF32)
            # This allows FP32 operations to run on Tensor Cores with slight precision reduction
            # (similar to BF16) but significant speedup. Standard for DL since Ampere.
            
            # Use the new API if available (PyTorch 1.12+)
            # 'high' maps to TF32 on Ampere+ and FP32 on older GPUs
            if hasattr(torch, 'set_float32_matmul_precision'):
                try:
                    torch.set_float32_matmul_precision('high')
                    if verbose:
                        print("   Performance: TF32 enabled (fast float32 matmul)")
                except Exception as e:
                    # Fallback or specific error handling if needed, but usually safe
                    print(f"   Note: Failed to set float32_matmul_precision: {e}")
            
            # 2. Enable cuDNN benchmark
            # This runs a quick benchmark on start to find the best convolution algo
            # for the specific hardware/input size. Great for fixed input sizes (standard RL).
            if torch.backends.cudnn.is_available():
                if not torch.backends.cudnn.benchmark:
                    torch.backends.cudnn.benchmark = True
                    if verbose:
                        print("   Performance: cuDNN benchmark enabled")
        
        _optimizations_applied = True
