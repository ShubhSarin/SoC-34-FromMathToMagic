# Week 0 — Onboarding: Python + NumPy + PyTorch

Welcome to the project! This week is all about getting comfortable with the tools you’ll use for the next 13 weeks.  
By the end of Week 0, you should be able to:

- Write basic Python code (loops, functions, list comprehensions)
- Perform array operations with NumPy
- Understand PyTorch tensors, automatic differentiation (autograd), and a simple training loop
- Train a linear regression model from scratch in PyTorch

If you can do that, you’re ready for Week 1’s probability math.

---

## 📚 Required Resources

Work through these in order. Most are short (10–20 minutes).

| Type | Title | Why it’s useful |
|------|-------|------------------|
| 🎥 Video | [Python in 100 Seconds – Fireship](https://www.youtube.com/watch?v=x7X9w_GIm1s) | Fastest possible Python refresher |
| 📄 Docs | [NumPy Quickstart](https://numpy.org/doc/stable/user/quickstart.html) | Arrays, broadcasting, indexing – you’ll use NumPy only in Week 1–2 |
| 📄 Tutorial | [PyTorch: Learn the Basics – Official Docs](https://pytorch.org/tutorials/beginner/basics/intro.html) | Tensors → autograd → training loop (the official crash course) |
| 🎥 Video / Code | [60‑Minute Blitz – PyTorch](https://pytorch.org/tutorials/beginner/deep_learning_60min_blitz.html) | Train a small neural net end‑to‑end. **Stop after the “Training a Classifier” section** |

> **Optional but helpful** – [Python for Data Science Handbook (NumPy section)](https://jakevdp.github.io/PythonDataScienceHandbook/02.00-introduction-to-numpy.html) if you want deeper NumPy intuition.

---

## ✅ Checkpoint: Train a Linear Model in PyTorch

Copy the code below into a notebook (local or [Colab](https://colab.research.google.com/)).  
Run it, understand every line, then tweak the learning rate or number of epochs to see how the loss changes.

```python
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

# ---------- Synthetic data ----------
torch.manual_seed(42)
X = torch.linspace(-3, 3, 100).reshape(-1, 1)
y = 2 * X + 1 + torch.randn(X.size()) * 0.5   # y = 2x + 1 + noise

# ---------- Model ----------
model = nn.Linear(1, 1)        # one input, one output
loss_fn = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

# ---------- Training ----------
losses = []
for epoch in range(200):
    optimizer.zero_grad()
    y_pred = model(X)
    loss = loss_fn(y_pred, y)
    loss.backward()
    optimizer.step()
    losses.append(loss.item())

# ---------- Visualize ----------
plt.plot(losses)
plt.xlabel("Epoch")
plt.ylabel("MSE Loss")
plt.title("Training Loss")
plt.show()

print(f"Learned weight: {model.weight.item():.3f} (true: 2.0)")
print(f"Learned bias:   {model.bias.item():.3f} (true: 1.0)")
```
