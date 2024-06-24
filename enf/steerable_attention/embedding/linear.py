from flax import linen as nn


class FFNEmbedding(nn.Module):
    num_hidden: int
    num_out: int

    @nn.compact
    def __call__(self, x):
        x = nn.Dense(self.num_hidden)(x)
        x = nn.gelu(x)
        x = nn.Dense(self.num_out)(x)
        return x
