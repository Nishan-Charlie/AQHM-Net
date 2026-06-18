import torch
from aqhm_net.model import AQHMNet

model = AQHMNet(in_channels=3, num_classes=10, scale="small")
model.train()

# Add a hook to print alpha
def hook(module, input, output):
    print("alpha mean:", output.mean().item())
model.fusion.gate.register_forward_hook(hook)

x = torch.randn(8, 3, 28, 28)
out = model(x)
loss = out.sum()
loss.backward()

print("Gate weights grad:", model.fusion.gate[3].weight.grad.norm().item())
print("VQC params grad norm:", model.quantum.vqc_params.grad.norm().item())
