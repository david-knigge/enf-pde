from enf.steerable_attention.embedding.linear import FFNEmbedding
from enf.steerable_attention.embedding.rff import RFFEmbedding, RFFNet
from enf.steerable_attention.embedding.polynomial import PolynomialEmbedding


__all__ = [
    "RFFEmbedding",
    "PolynomialEmbedding",
    "FFNEmbedding",
    "get_embedding",
]


def get_embedding(embedding_type: str, num_in: int, num_hidden: int, num_emb_dim: int, freq_multiplier: float):
    """
    Get the embedding module.

    Args:
        embedding_type (str): The type of embedding to use. 'rff' or 'siren'.

    Returns:
        BaseEmbedding: The embedding module.

    """
    if embedding_type == "rff":
        return RFFNet(in_dim=num_in, output_dim=num_emb_dim, hidden_dim=num_hidden, num_layers=2,
                      learnable_coefficients=False, std=freq_multiplier)
    elif embedding_type == "ffn":
        return FFNEmbedding(num_hidden=num_hidden, num_out=num_emb_dim)
    elif embedding_type == "polynomial":
        return PolynomialEmbedding(num_hidden=num_hidden, num_out=num_emb_dim, degree=int(freq_multiplier))
    else:
        raise ValueError(f"Unknown embedding type: {embedding_type}.")
