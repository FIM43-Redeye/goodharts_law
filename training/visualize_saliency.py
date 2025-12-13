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
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Callable

from behaviors.brains.tiny_cnn import TinyCNN
from behaviors import LearnedBehavior


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
        input_tensor: Input of shape (1, 1, H, W) with requires_grad=True
        target_action: Action index to compute saliency for.
                      If None, uses the model's predicted action.
    
    Returns:
        Saliency map as numpy array of shape (H, W)
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
    
    # Remove batch and channel dims: (1, 1, H, W) -> (H, W)
    saliency = grad.squeeze().cpu().numpy()
    
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
        saliency = grad.squeeze().cpu().numpy()
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
    
    saliency = integrated_grad.abs().squeeze().cpu().numpy()
    
    return saliency


def visualize_saliency(
    model: torch.nn.Module,
    view: np.ndarray,
    method: str = 'gradient',
    figsize: tuple[int, int] = (12, 4),
    save_path: str | None = None,
    title_prefix: str = '',
) -> plt.Figure:
    """
    Create a visualization of input view alongside saliency map.
    
    Args:
        model: Trained neural network
        view: Agent's view as numpy array (H, W)
        method: 'gradient', 'guided', or 'integrated'
        figsize: Figure size
        save_path: If provided, save figure to this path
        title_prefix: Prefix for plot title (e.g., "Ground Truth Model")
    
    Returns:
        matplotlib Figure object
    """
    device = next(model.parameters()).device
    
    # Prepare input
    input_tensor = torch.from_numpy(view).float().unsqueeze(0).unsqueeze(0).to(device)
    
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
    
    # Create figure
    fig, axes = plt.subplots(1, 3, figsize=figsize)
    
    # Original view
    im0 = axes[0].imshow(view, cmap='viridis')
    axes[0].set_title("Agent's View")
    plt.colorbar(im0, ax=axes[0], fraction=0.046)
    
    # Saliency map
    im1 = axes[1].imshow(saliency, cmap='hot')
    axes[1].set_title(f"Saliency ({method})")
    plt.colorbar(im1, ax=axes[1], fraction=0.046)
    
    # Overlay
    axes[2].imshow(view, cmap='viridis', alpha=0.5)
    axes[2].imshow(saliency, cmap='hot', alpha=0.5)
    axes[2].set_title(f"Overlay (action={action_idx}, conf={confidence:.2f})")
    
    for ax in axes:
        ax.axis('off')
    
    fig.suptitle(f"{title_prefix} - {method.title()} Saliency" if title_prefix else f"{method.title()} Saliency")
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved saliency visualization to {save_path}")
    
    return fig


def compare_models_saliency(
    ground_truth_model: torch.nn.Module,
    proxy_model: torch.nn.Module,
    view: np.ndarray,
    method: str = 'gradient',
    save_path: str | None = None,
) -> plt.Figure:
    """
    Side-by-side comparison of saliency from ground-truth vs proxy models.
    
    This is the key visualization for demonstrating Goodhart's Law:
    show that the proxy model attends to poison (high interestingness)
    while the ground-truth model avoids it.
    """
    device = next(ground_truth_model.parameters()).device
    input_tensor = torch.from_numpy(view).float().unsqueeze(0).unsqueeze(0).to(device)
    
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
    
    # Create figure
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    
    # Original view
    im0 = axes[0].imshow(view, cmap='viridis')
    axes[0].set_title("Agent's View")
    
    # Ground truth saliency
    im1 = axes[1].imshow(sal_gt, cmap='hot')
    axes[1].set_title("Ground Truth Model")
    
    # Proxy saliency
    im2 = axes[2].imshow(sal_proxy, cmap='hot')
    axes[2].set_title("Proxy Model")
    
    # Difference (proxy - ground_truth)
    diff = sal_proxy - sal_gt
    max_abs = max(abs(diff.min()), abs(diff.max())) or 1
    im3 = axes[3].imshow(diff, cmap='RdBu_r', vmin=-max_abs, vmax=max_abs)
    axes[3].set_title("Difference (Proxy - GT)")
    
    for ax in axes:
        ax.axis('off')
    
    plt.colorbar(im3, ax=axes[3], fraction=0.046)
    fig.suptitle("Saliency Comparison: Ground Truth vs Proxy Training")
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved comparison to {save_path}")
    
    return fig


def analyze_model_attention(
    model_path: str,
    config: dict,
    output_dir: str = 'visualizations',
    num_samples: int = 5,
):
    """
    Load a model and generate saliency visualizations on sample views.
    
    Convenience function for command-line usage.
    """
    from training.collect import collect_from_expert
    from behaviors import OmniscientSeeker
    
    # Determine input shape
    view_range = config['AGENT_VIEW_RANGE']
    view_side = view_range * 2 + 1
    input_shape = (view_side, view_side)
    
    # Load model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TinyCNN(input_shape=input_shape, input_channels=1, output_size=8).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.eval()
    
    # Collect sample views
    buffer = collect_from_expert(config, OmniscientSeeker, num_steps=100, num_agents=5)
    
    # Create output directory
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    model_name = Path(model_path).stem
    
    # Generate visualizations
    for i in range(min(num_samples, len(buffer))):
        view = buffer.buffer[i].state
        save_path = f"{output_dir}/{model_name}_saliency_{i}.png"
        visualize_saliency(model, view, method='gradient', save_path=save_path, title_prefix=model_name)
    
    print(f"Generated {num_samples} saliency visualizations in {output_dir}/")
