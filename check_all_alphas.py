import os
import torch
import glob
from aqhm_net.model import AQHMNet

def check_alpha(ckpt_path):
    # Parse K and scale from path
    k = 1
    if "k4" in ckpt_path: k = 4
    if "k8" in ckpt_path: k = 8
    
    scale = "small"
    if "medium" in ckpt_path: scale = "medium"
    if "large" in ckpt_path: scale = "large"
    
    try:
        model = AQHMNet(in_channels=3, num_classes=10, n_quantum_heads=k, scale=scale)
        model.load_state_dict(torch.load(ckpt_path, map_location="cpu", weights_only=True))
        model.eval()
        
        alphas = []
        def hook(module, input, output):
            alphas.append(output.mean().item())
        
        handle = model.fusion.gate.register_forward_hook(hook)
        
        # Test on 5 random batches
        for _ in range(5):
            x = torch.randn(8, 3, 28, 28)
            with torch.no_grad():
                model(x)
                
        handle.remove()
        return sum(alphas)/len(alphas)
    except Exception as e:
        return f"Error: {e}"

print("Analyzing alpha values for trained models...\n")

patterns = [
    "c:/Users/nisha/OneDrive/Desktop/Research/Quantum_Images/results_scale*/cifar10*/checkpoints/*_best.pt"
]

for p in patterns:
    for ckpt in glob.glob(p):
        name = os.path.basename(os.path.dirname(os.path.dirname(ckpt)))
        alpha = check_alpha(ckpt)
        if isinstance(alpha, float):
            print(f"{name:20s} -> mean alpha = {alpha:.6f}")
        else:
            print(f"{name:20s} -> {alpha}")
