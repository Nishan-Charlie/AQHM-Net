"""Minimal end-to-end test: lightning.qubit adjoint circuit + model forward pass."""
import sys
print(f"Python: {sys.version}", flush=True)

print("1. Importing torch...", flush=True)
import torch
print(f"   torch {torch.__version__}, CUDA={torch.cuda.is_available()}", flush=True)

print("2. Importing pennylane...", flush=True)
import pennylane as qml
print(f"   pennylane {qml.__version__}", flush=True)

print("3. Creating lightning.qubit device...", flush=True)
dev = qml.device("lightning.qubit", wires=9)
print("   ok", flush=True)

print("4. Building circuit (adjoint)...", flush=True)
@qml.qnode(dev, interface="torch", diff_method="adjoint")
def circuit(x_enc, reup, vqc):
    qml.AngleEmbedding(x_enc, wires=range(9), rotation="Y")
    for l in range(3):
        qml.AngleEmbedding(reup[l], wires=range(9), rotation="Y")
        for q in range(9):
            qml.RX(vqc[l, q, 0], wires=q)
            qml.RY(vqc[l, q, 1], wires=q)
            qml.RZ(vqc[l, q, 2], wires=q)
        for q in range(8):
            qml.CZ(wires=[q, q+1])
        qml.CZ(wires=[8, 0])
    return ([qml.expval(qml.PauliX(q)) for q in range(6)] +
            [qml.expval(qml.PauliZ(0) @ qml.PauliZ(3)),
             qml.expval(qml.PauliZ(3) @ qml.PauliZ(6)),
             qml.expval(qml.PauliZ(0) @ qml.PauliZ(6))])

print("5. Running forward + backward...", flush=True)
vqc = torch.zeros(3, 9, 3, requires_grad=True)
x_enc = torch.rand(9)
reup  = torch.rand(3, 9)
meas  = circuit(x_enc, reup, vqc)
out   = torch.stack(meas).sum()
out.backward()
print(f"   output={out.item():.4f}  grad_norm={vqc.grad.norm().item():.4f}", flush=True)

print("6. Importing aqhm_net package...", flush=True)
sys.path.insert(0, r"c:\Users\nisha\OneDrive\Desktop\Research\Quantum_Images")
from aqhm_net import AQHMNet
print("   ok", flush=True)

print("7. Forward pass through full model (batch=2, MNIST)...", flush=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = AQHMNet(in_channels=1, num_classes=10).to(device)
imgs  = torch.randn(2, 1, 28, 28).to(device)
with torch.no_grad():
    logits = model(imgs)
print(f"   logits shape: {logits.shape}  (expected [2, 10])", flush=True)

print("\n=== ALL TESTS PASSED ===", flush=True)
