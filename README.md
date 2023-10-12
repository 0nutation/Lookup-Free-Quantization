# Lookup-Free-Quantization

An implementation of Lookup-Free Quantization in [MAGVIT2](https://arxiv.org/pdf/2310.05737.pdf)

## Usage

```python
import torch
from .LFQ import LookupFreeQuantizer

LFQ = LookupFreeQuantizer(vocab_size=256)

z = torch.randn(5,10,8)  #[B, T, D]
q_z, index = LFQ.quantize(z)

print(f"z: \n{z}")
print(f"q(z): \n{q_z}")
print(f"Token Index: {index}")
```
