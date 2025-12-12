import torch
import torch.optim as optim
# from behaviors.brains.tiny_cnn import TinyCNN

def train(config):
    """
    The main training loop.
    
    1. Initialize the Environment (World)
    2. Initialize the Agents with LearnedBehavior
    3. Initialize the Model, Optimizer, and Loss Function
    4. Loop (Epochs):
       a. Run Simulation (Collect Data/Experience)
       b. Train Model (Update Weights)
    """
    
    # Setup
    print("Setting up training...")
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # model = TinyCNN().to(device)
    
    # -------------------------------------------------------------------------
    # EDUCATIONAL NOTE: Optimizers and Loss Functions
    #
    # The Optimizer updates the model's weights to minimize the error.
    # 'Adam' is a very popular, robust choice.
    #
    # The Loss Function measures how "wrong" the model's prediction was.
    # For action classification, CrossEntropyLoss is standard.
    # -------------------------------------------------------------------------
    
    # optimizer = optim.Adam(model.parameters(), lr=0.001)
    # criterion = torch.nn.CrossEntropyLoss()
    
    print("Starting training loop...")
    # for epoch in range(config['epochs']):
        # 1. Run Simulation & Collect Data
        # ...
        
        # 2. Training Step
        # optimizer.zero_grad()       # Reset gradients
        # outputs = model(inputs)     # Forward pass
        # loss = criterion(outputs, targets) # Calculate error
        # loss.backward()             # Backpropagation (calculate gradients)
        # optimizer.step()            # Update weights
        
        # print(f"Epoch {epoch}: Loss = {loss.item()}")
        pass

    print("Training finished.")
