import torch

# Load the state dictionary from the checkpoint
checkpoint = torch.load('ml_avt2_checkpoint_40000', map_location='cpu')
state_dict = checkpoint['state_dict']  # Adjust key based on how the checkpoint is structured

# Print each layer's name and dimensions
for layer_name, tensor in state_dict.items():
    print(f"{layer_name}: {tensor.size()}")
