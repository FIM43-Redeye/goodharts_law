"""
Saliency visualization for learned behaviors.

Generates gradient-based saliency maps showing which parts of the input
the neural network is attending to when making decisions.

Crucial for demonstrating that proxy-trained models "look at" poison
while ground-truth-trained models avoid it.
"""
import torch
import torch.nn.functional as F
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pathlib import Path

from goodharts.behaviors.brains import load_brain
from goodharts.utils.device import get_device


# Dark theme colors
COLORS = {
    'background': '#1a1a2e',
    'paper': '#16213e',
    'text': '#e0e0e0',
    'grid': 'rgba(128, 128, 128, 0.15)',
}


def compute_gradient_saliency(
    model: torch.nn.Module,
    input_tensor: torch.Tensor,
    target_action: int | None = None,
) -> np.ndarray:
    """
    Compute gradient-based saliency map.

    The saliency shows how much each input pixel affects the output.
    Higher values = more important for the decision.

    Args:
        model: Neural network model
        input_tensor: Input of shape (1, C, H, W) with requires_grad=True
        target_action: Action index to compute saliency for.
                      If None, uses the model's predicted action.

    Returns:
        Saliency map as numpy array of shape (H, W) or (C, H, W)
    """
    model.eval()

    # Ensure input requires grad
    input_tensor = input_tensor.clone().detach().requires_grad_(True)

    # Forward pass
    output = model(input_tensor)

    if target_action is None:
        target_action = output.argmax(dim=1).item()

    # Backward pass to get gradients
    model.zero_grad()
    output[0, target_action].backward()

    # Get gradient w.r.t. input
    grad = input_tensor.grad.data.abs()

    # Remove batch dim: (1, C, H, W) -> (C, H, W) or (H, W)
    saliency = grad.squeeze(0).cpu().numpy()

    return saliency


def compute_guided_backprop_saliency(
    model: torch.nn.Module,
    input_tensor: torch.Tensor,
    target_action: int | None = None,
) -> np.ndarray:
    """
    Compute Guided Backpropagation saliency.

    This variant only backpropagates positive gradients through ReLUs,
    producing cleaner, more focused saliency maps.

    Works with arbitrary model architectures that use ReLU.
    """
    model.eval()

    # Store hooks to modify gradients
    handles = []

    def relu_backward_hook(module, grad_in, grad_out):
        # Only allow positive gradients
        return (F.relu(grad_in[0]),)

    # Register hooks on all ReLU layers
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.ReLU):
            handle = module.register_full_backward_hook(relu_backward_hook)
            handles.append(handle)

    try:
        input_tensor = input_tensor.clone().detach().requires_grad_(True)
        output = model(input_tensor)

        if target_action is None:
            target_action = output.argmax(dim=1).item()

        model.zero_grad()
        output[0, target_action].backward()

        grad = input_tensor.grad.data.abs()
        saliency = grad.squeeze(0).cpu().numpy()
    finally:
        # Remove hooks
        for handle in handles:
            handle.remove()

    return saliency


def compute_integrated_gradients(
    model: torch.nn.Module,
    input_tensor: torch.Tensor,
    target_action: int | None = None,
    baseline: torch.Tensor | None = None,
    steps: int = 50,
) -> np.ndarray:
    """
    Compute Integrated Gradients attribution.

    This is a more principled attribution method that satisfies
    desirable axioms like sensitivity and implementation invariance.

    Integrates gradients along the path from a baseline (default: zeros)
    to the actual input.
    """
    model.eval()

    if baseline is None:
        baseline = torch.zeros_like(input_tensor)

    # Determine target action from original input
    with torch.no_grad():
        output = model(input_tensor)
        if target_action is None:
            target_action = output.argmax(dim=1).item()

    # Compute integrated gradients
    scaled_inputs = [
        baseline + (float(i) / steps) * (input_tensor - baseline)
        for i in range(1, steps + 1)
    ]

    gradients = []
    for scaled_input in scaled_inputs:
        scaled_input = scaled_input.clone().detach().requires_grad_(True)
        output = model(scaled_input)

        model.zero_grad()
        output[0, target_action].backward()

        gradients.append(scaled_input.grad.data.clone())

    # Average gradients and multiply by (input - baseline)
    avg_gradients = torch.stack(gradients).mean(dim=0)
    integrated_grad = (input_tensor - baseline) * avg_gradients

    saliency = integrated_grad.abs().squeeze(0).cpu().numpy()

    return saliency


def _aggregate_channels(arr: np.ndarray) -> np.ndarray:
    """Aggregate multi-channel array to 2D for visualization."""
    if arr.ndim == 3:
        return arr.sum(axis=0)
    return arr


def visualize_saliency(
    model: torch.nn.Module,
    view: np.ndarray,
    method: str = 'gradient',
    save_path: str | None = None,
    title_prefix: str = '',
    show: bool = True,
) -> go.Figure:
    """
    Create a visualization of input view alongside saliency map.

    Args:
        model: Trained neural network
        view: Agent's view as numpy array (H, W) or (C, H, W)
        method: 'gradient', 'guided', or 'integrated'
        save_path: If provided, save figure to this path
        title_prefix: Prefix for plot title (e.g., "Ground Truth Model")
        show: If True, display the figure

    Returns:
        Plotly Figure object
    """
    device = next(model.parameters()).device

    # Handle different input shapes
    if view.ndim == 2:
        # Single channel: (H, W) -> (1, 1, H, W)
        input_tensor = torch.from_numpy(view).float().unsqueeze(0).unsqueeze(0).to(device)
        view_display = view
    elif view.ndim == 3:
        # Multi-channel: (C, H, W) -> (1, C, H, W)
        input_tensor = torch.from_numpy(view).float().unsqueeze(0).to(device)
        # For display, sum channels
        view_display = view.sum(axis=0)
    else:
        raise ValueError(f"Expected 2D or 3D view, got shape {view.shape}")

    # Get prediction
    with torch.no_grad():
        logits = model(input_tensor)
        probs = F.softmax(logits, dim=1)
        action_idx = logits.argmax(dim=1).item()
        confidence = probs[0, action_idx].item()

    # Compute saliency
    if method == 'gradient':
        saliency = compute_gradient_saliency(model, input_tensor)
    elif method == 'guided':
        saliency = compute_guided_backprop_saliency(model, input_tensor)
    elif method == 'integrated':
        saliency = compute_integrated_gradients(model, input_tensor)
    else:
        raise ValueError(f"Unknown method: {method}")

    # Aggregate multi-channel saliency to 2D
    saliency = _aggregate_channels(saliency)

    # Create subplot figure
    fig = make_subplots(
        rows=1, cols=3,
        subplot_titles=["Agent's View", f"Saliency ({method})",
                       f"Overlay (action={action_idx}, conf={confidence:.2f})"],
        horizontal_spacing=0.08,
    )

    # Plot 1: Original view (viridis colorscale)
    fig.add_trace(
        go.Heatmap(z=view_display, colorscale='Viridis', showscale=True,
                   colorbar=dict(x=0.28, len=0.9)),
        row=1, col=1
    )

    # Plot 2: Saliency map (hot colorscale)
    fig.add_trace(
        go.Heatmap(z=saliency, colorscale='Hot', showscale=True,
                   colorbar=dict(x=0.63, len=0.9)),
        row=1, col=2
    )

    # Plot 3: Overlay - normalize and blend
    view_norm = (view_display - view_display.min()) / (view_display.max() - view_display.min() + 1e-8)
    sal_norm = (saliency - saliency.min()) / (saliency.max() - saliency.min() + 1e-8)
    overlay = 0.5 * view_norm + 0.5 * sal_norm

    fig.add_trace(
        go.Heatmap(z=overlay, colorscale='Turbo', showscale=True,
                   colorbar=dict(x=0.98, len=0.9)),
        row=1, col=3
    )

    # Update layout
    title = f"{title_prefix} - {method.title()} Saliency" if title_prefix else f"{method.title()} Saliency"
    fig.update_layout(
        title=dict(text=title, x=0.5, font=dict(size=16, color=COLORS['text'])),
        plot_bgcolor=COLORS['paper'],
        paper_bgcolor=COLORS['background'],
        font=dict(color=COLORS['text']),
        height=400,
        width=1200,
    )

    # Make axes square and hide ticks
    for i in range(1, 4):
        fig.update_xaxes(showticklabels=False, showgrid=False, row=1, col=i)
        fig.update_yaxes(showticklabels=False, showgrid=False, autorange='reversed', row=1, col=i)

    if save_path:
        fig.write_image(save_path, scale=2)
        print(f"Saved: {save_path}")

    if show:
        fig.show()

    return fig


def compare_models_saliency(
    ground_truth_model: torch.nn.Module,
    proxy_model: torch.nn.Module,
    view: np.ndarray,
    method: str = 'gradient',
    save_path: str | None = None,
    show: bool = True,
) -> go.Figure:
    """
    Side-by-side comparison of saliency from ground-truth vs proxy models.

    This is the key visualization for demonstrating Goodhart's Law:
    show that the proxy model attends to poison (high interestingness)
    while the ground-truth model avoids it.
    """
    device = next(ground_truth_model.parameters()).device

    # Handle different input shapes
    if view.ndim == 2:
        input_tensor = torch.from_numpy(view).float().unsqueeze(0).unsqueeze(0).to(device)
    else:
        input_tensor = torch.from_numpy(view).float().unsqueeze(0).to(device)

    # Compute saliencies
    if method == 'gradient':
        sal_gt = compute_gradient_saliency(ground_truth_model, input_tensor)
        sal_proxy = compute_gradient_saliency(proxy_model, input_tensor.clone())
    elif method == 'guided':
        sal_gt = compute_guided_backprop_saliency(ground_truth_model, input_tensor)
        sal_proxy = compute_guided_backprop_saliency(proxy_model, input_tensor.clone())
    else:
        sal_gt = compute_integrated_gradients(ground_truth_model, input_tensor)
        sal_proxy = compute_integrated_gradients(proxy_model, input_tensor.clone())

    # Aggregate to 2D
    sal_gt = _aggregate_channels(sal_gt)
    sal_proxy = _aggregate_channels(sal_proxy)
    view_display = _aggregate_channels(view)

    # Difference map
    diff = sal_proxy - sal_gt
    max_abs = max(abs(diff.min()), abs(diff.max())) or 1

    # Create figure
    fig = make_subplots(
        rows=1, cols=4,
        subplot_titles=["Agent's View", "Ground Truth Model",
                       "Proxy Model", "Difference (Proxy - GT)"],
        horizontal_spacing=0.06,
    )

    # Plot 1: Original view
    fig.add_trace(
        go.Heatmap(z=view_display, colorscale='Viridis', showscale=False),
        row=1, col=1
    )

    # Plot 2: Ground truth saliency
    fig.add_trace(
        go.Heatmap(z=sal_gt, colorscale='Hot', showscale=False),
        row=1, col=2
    )

    # Plot 3: Proxy saliency
    fig.add_trace(
        go.Heatmap(z=sal_proxy, colorscale='Hot', showscale=False),
        row=1, col=3
    )

    # Plot 4: Difference (diverging colorscale)
    fig.add_trace(
        go.Heatmap(z=diff, colorscale='RdBu_r', zmid=0,
                   zmin=-max_abs, zmax=max_abs, showscale=True,
                   colorbar=dict(x=1.02, len=0.9)),
        row=1, col=4
    )

    # Update layout
    fig.update_layout(
        title=dict(text="Saliency Comparison: Ground Truth vs Proxy Training",
                  x=0.5, font=dict(size=16, color=COLORS['text'])),
        plot_bgcolor=COLORS['paper'],
        paper_bgcolor=COLORS['background'],
        font=dict(color=COLORS['text']),
        height=400,
        width=1400,
    )

    # Make axes square and hide ticks
    for i in range(1, 5):
        fig.update_xaxes(showticklabels=False, showgrid=False, row=1, col=i)
        fig.update_yaxes(showticklabels=False, showgrid=False, autorange='reversed', row=1, col=i)

    if save_path:
        fig.write_image(save_path, scale=2)
        print(f"Saved comparison to {save_path}")

    if show:
        fig.show()

    return fig


def load_model_from_path(model_path: str, device: torch.device = None) -> tuple[torch.nn.Module, dict]:
    """
    Load a brain model with automatic architecture detection.

    Wrapper around load_brain for backward compatibility.

    Returns:
        (model, metadata) where metadata contains architecture info
    """
    if device is None:
        device = get_device(verbose=False)

    brain, metadata = load_brain(model_path, device=device)

    # Print architecture info
    arch = metadata.get('architecture', brain.get_architecture_info())
    print(f"Loaded {metadata.get('brain_type', 'unknown')}: "
          f"{arch.get('input_channels')}ch x {arch.get('input_shape')} -> "
          f"hidden={arch.get('hidden_size')} -> {arch.get('output_size')} actions")

    # Merge architecture into metadata for backward compat
    if 'architecture' in metadata:
        metadata.update(metadata['architecture'])

    return brain, metadata


def find_default_model() -> str:
    """Find the first available model in the models directory."""
    models_dir = Path(__file__).parent.parent.parent / 'models'

    # Prefer ground_truth.pth if it exists
    preferred = models_dir / 'ground_truth.pth'
    if preferred.exists():
        return str(preferred)

    # Otherwise, find any .pth file
    pth_files = list(models_dir.glob('*.pth'))
    if pth_files:
        return str(pth_files[0])

    raise FileNotFoundError(f"No .pth files found in {models_dir}")


def analyze_model_attention(
    model_path: str | None = None,
    config: dict | None = None,
    output_dir: str = 'generated/visualizations',
    num_samples: int = 5,
    show: bool = True,
):
    """
    Load a model and generate saliency visualizations on sample views.

    Args:
        model_path: Path to model weights. If None, auto-detects from models/
        config: Simulation config. If None, uses default config.
        output_dir: Where to save visualization images
        num_samples: Number of views to visualize
        show: If True, display plots interactively
    """
    from goodharts.configs.default_config import get_simulation_config
    from goodharts.environments import create_vec_env
    from goodharts.modes import ObservationSpec

    # Auto-detect model if not specified
    if model_path is None:
        model_path = find_default_model()
        print(f"Auto-detected model: {model_path}")

    if config is None:
        config = get_simulation_config()

    device = get_device()

    # Load model with auto-detected architecture
    model, metadata = load_model_from_path(model_path, device)

    # Collect sample views using TorchVecEnv
    # Mode depends on input channels (4 = ground_truth, 1 = proxy)
    mode = 'ground_truth' if metadata['input_channels'] == 4 else 'proxy'
    print(f"Collecting {num_samples} sample views (mode: {mode})...")

    obs_spec = ObservationSpec.for_mode(mode, config)
    env = create_vec_env(n_envs=5, obs_spec=obs_spec, config=config, device=device)

    # Collect observations by stepping through environment
    sample_views = []
    obs = env.reset()
    sample_views.append(obs)
    for _ in range(max(20, num_samples // 5)):
        # Random actions to get diverse views
        actions = torch.randint(0, 8, (5,), device=device)
        obs, _, _, _ = env.step(actions)
        sample_views.append(obs)

    # Flatten: (num_steps, n_envs, C, H, W) -> (N, C, H, W)
    all_views = torch.cat(sample_views, dim=0)

    # Create output directory
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    model_name = Path(model_path).stem

    # Generate visualizations
    print(f"Generating saliency maps...")
    for i in range(min(num_samples, len(all_views))):
        view = all_views[i].cpu().numpy()  # Shape: (C, H, W) or (H, W)

        save_path = f"{output_dir}/{model_name}_saliency_{i}.png"
        visualize_saliency(
            model, view,
            method='gradient',
            save_path=save_path,
            title_prefix=model_name,
            show=show,
        )

    print(f"Generated {num_samples} saliency visualizations in {output_dir}/")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate saliency visualizations for trained models")
    parser.add_argument('--model', type=str, default=None,
                        help='Path to trained model (default: auto-detect from models/)')
    parser.add_argument('--output', type=str, default='generated/visualizations',
                        help='Output directory for images')
    parser.add_argument('--samples', type=int, default=5,
                        help='Number of sample views to visualize')
    parser.add_argument('--no-show', action='store_true',
                        help='Save images without displaying')
    args = parser.parse_args()

    analyze_model_attention(
        model_path=args.model,
        output_dir=args.output,
        num_samples=args.samples,
        show=not args.no_show,
    )
