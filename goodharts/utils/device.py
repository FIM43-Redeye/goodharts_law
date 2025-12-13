"""
Centralized PyTorch device selection.

Provides a single function to get the configured compute device,
allowing users to easily switch between CPU, CUDA, ROCm, etc.

Priority order:
1. Explicit override parameter
2. GOODHARTS_DEVICE environment variable
3. config.toml [runtime] device setting
4. Auto-detect (CUDA > CPU)
"""
import os
import torch

# Module-level cache for the device (avoid repeated detection)
_cached_device: torch.device | None = None
_device_logged: bool = False


def get_device(override: str | None = None, verbose: bool = True) -> torch.device:
    """
    Get the configured PyTorch device.
    
    Args:
        override: Explicit device string (e.g., 'cpu', 'cuda', 'cuda:1')
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
        
        # Via environment variable
        # $ GOODHARTS_DEVICE=cpu python train.py
        device = get_device()  # Returns CPU
    """
    global _cached_device, _device_logged
    
    # If explicit override, use it directly (no caching)
    if override is not None:
        return torch.device(override)
    
    # Return cached device if already determined
    if _cached_device is not None:
        return _cached_device
    
    # Priority 1: Environment variable
    env_device = os.environ.get('GOODHARTS_DEVICE')
    if env_device:
        _cached_device = torch.device(env_device)
        if verbose and not _device_logged:
            print(f"🖥️  Device: {_cached_device} (from GOODHARTS_DEVICE env)")
            _device_logged = True
        return _cached_device
    
    # Priority 2: Config file
    try:
        from goodharts.config import get_runtime_config
        runtime_cfg = get_runtime_config()
        if runtime_cfg and runtime_cfg.get('device'):
            _cached_device = torch.device(runtime_cfg['device'])
            if verbose and not _device_logged:
                print(f"🖥️  Device: {_cached_device} (from config)")
                _device_logged = True
            return _cached_device
    except (ImportError, KeyError):
        pass  # Config not available or no device setting
    
    # Priority 3: Auto-detect
    if torch.cuda.is_available():
        _cached_device = torch.device('cuda')
        if verbose and not _device_logged:
            gpu_name = torch.cuda.get_device_name(0)
            print(f"🖥️  Device: {_cached_device} ({gpu_name})")
            _device_logged = True
    else:
        _cached_device = torch.device('cpu')
        if verbose and not _device_logged:
            print(f"🖥️  Device: {_cached_device}")
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
