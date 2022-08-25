import torch
import torch.nn as nn

class SelfAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int, bias: bool=True, kv_bias: bool=False):
        super(SelfAttention, self).__init__()

        self.dim = dim
        self.num_heads = num_heads
        self.bias = bias
        self.kv_bias = kv_bias

        self.mha = nn.MultiheadAttention(dim, num_heads=num_heads, bias=bias, add_bias_kv=kv_bias)

    def forward(self, x, acquire_attention_scores: bool=False):
        attention_values, attention_scores = self.mha(x, x, x)
        if acquire_attention_scores:
            return attention_values, attention_scores
        else:
            return attention_values


def test():
    sample = torch.randn(4, 576, 768)
    attent = SelfAttention(768, num_heads=12, bias=True, kv_bias=True)
    out = attent(sample, acquire_attention_scores=False)

    assert out.shape == (4, 576, 768)
    print('PASSED: model/module/SelfAttention.py')


if __name__ == '__main__':
    test()
