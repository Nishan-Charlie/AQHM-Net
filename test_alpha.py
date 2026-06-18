import torch
from aqhm_net.model import AQHMNet

# We know the checkpoint is K=4, large
model = AQHMNet(in_channels=3, num_classes=10, n_quantum_heads=4, scale="large")
ckpt = "c:/Users/nisha/OneDrive/Desktop/Research/Quantum_Images/results_scale_k4/cifar10_k4_large/checkpoints/aqhm_net_seed_000_best.pt"
model.load_state_dict(torch.load(ckpt, map_location="cpu", weights_only=True))
model.eval()

alphas = []
def hook(module, input, output):
    alphas.append(output.mean().item())
model.fusion.gate.register_forward_hook(hook)

# Let's test on 10 random batches
for _ in range(10):
    x = torch.randn(8, 3, 28, 28)
    out = model(x)

print(f"Mean alpha over 10 random batches: {sum(alphas)/len(alphas):.6f}")
