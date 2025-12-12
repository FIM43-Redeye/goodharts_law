import torch
import torch.optim as optim
import torch.multiprocessing as mp
from behaviors.brains.tiny_cnn import TinyCNN
from simulation import Simulation
from configs.default_config import get_config
import numpy as np
import time

def simulation_worker(config, seed, model_state_dict=None):
    """
    Worker function to run a single simulation instance.
    
    Args:
        config (dict): Simulation configuration.
        seed (int): Random seed for reproducibility.
        model_state_dict (dict, optional): Weights to load into agents.
        
    Returns:
        list: Collected training data (e.g., [(view, action, reward), ...])
              OR simple stats for now if just testing throughput.
    """
    # Set seed
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    # Initialize simulation
    sim = Simulation(config)
    
    # Optional: Load model into agents if we were doing inference during sim
    # For now, we assume agents might be using random or heuristic behaviors
    # unless we explicitly set their brains.
    
    # Run simulation for a fixed number of steps
    max_steps = 100
    data = []
    
    for _ in range(max_steps):
        sim.step()
        
        # TODO: Collect data here.
        # For a "groundwork" task, we'll just demonstrate we ran the steps.
        # In a real RL loop, we'd extract (state, action, reward) from agents.
        
    # Return some dummy data or stats
    return {"steps": max_steps, "survivors": len(sim.agents)}

def train(config):
    """
    The main training loop with parallel data collection.
    """
    # -------------------------------------------------------------------------
    # EDUCATIONAL NOTE: Device Selection (GPU vs CPU)
    #
    # We check if CUDA (NVIDIA) or ROCm (AMD) is available.
    # PyTorch handles both via 'cuda' usually, or 'mps' for Mac.
    # -------------------------------------------------------------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Initialize the Model
    # Note: We need to know input shape. Let's calculate it from config.
    view_range = config['AGENT_VIEW_RANGE']
    view_side = view_range * 2 + 1
    input_shape = (view_side, view_side)
    
    model = TinyCNN(input_shape=input_shape, input_channels=1, output_size=4).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # -------------------------------------------------------------------------
    # EDUCATIONAL NOTE: Multiprocessing
    #
    # Running simulations is CPU-bound. To speed it up, we run multiple
    # simulations in parallel processes.
    # -------------------------------------------------------------------------
    num_workers = mp.cpu_count() - 1 or 1
    print(f"Starting training with {num_workers} parallel workers...")
    
    epochs = 5 # Short run for demo
    
    for epoch in range(epochs):
        start_time = time.time()
        
        # Prepare arguments for workers
        # We pass a different seed to each worker
        worker_args = [(config, epoch * 100 + i) for i in range(num_workers)]
        
        # Run workers
        with mp.Pool(num_workers) as pool:
            results = pool.starmap(simulation_worker, worker_args)
            
        # Process results
        total_steps = sum(r['steps'] for r in results)
        
        # TODO: Here we would take the collected data (states/actions)
        # move them to the GPU (device), and run the training step.
        
        # Example dummy training step:
        # optimizer.zero_grad()
        # dummy_input = torch.randn(32, 1, view_side, view_side).to(device)
        # output = model(dummy_input)
        # loss = output.mean() # Dummy loss
        # loss.backward()
        # optimizer.step()
        
        elapsed = time.time() - start_time
        print(f"Epoch {epoch+1}/{epochs}: Ran {total_steps} steps in {elapsed:.2f}s "
              f"({total_steps/elapsed:.0f} steps/s)")

    print("Training finished.")
    
    # Save the model
    # torch.save(model.state_dict(), "brain_model.pth")


if __name__ == "__main__":
    mp.set_start_method('spawn', force=True) # Recommended for PyTorch multiprocessing
    config = get_config()
    train(config)
