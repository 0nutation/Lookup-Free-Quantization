import torch

class LookupFreeQuantizer:
    def __init__(self, vocab_size: int=None):
        self.vocab_size = vocab_size

    def sign(self, z: torch.Tensor):
        return torch.sign(z)

    def token_index(self, q_z: torch.Tensor):
        indices = (torch.arange(q_z.size(-1), dtype=torch.float32)).to(q_z.device)
        tokens = torch.sum(2**indices * (q_z > 0).float(), dim=-1)
        return tokens

    def quantize(self, z: torch.Tensor):
        if self.vocab_size is not None:
            assert z.size(-1)==torch.log2(self.vocab_size)

        q_z = self.sign(z)
        index = self.token_index(q_z)
        return q_z, index.int()

