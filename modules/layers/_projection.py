import torch
import torch.nn as nn


class ProjectionLayer(nn.Module):
    """
    A `nn.Module` that represents the Projection Layer of the Transformer model. This layer projects the output of the
    Decoder to the vocabulary size.
    """

    def __init__(
        self,
        d_model: int,
        vocab_size: int,
    ) -> None:
        """
        A `nn.Module` that represents the Projection Layer of the Transformer model. This layer projects the output of the
        Decoder to the vocabulary size.

        Args:
            d_model (int): The dimension of the model.
            vocab_size (int): The size of the vocabulary.
        """
        super().__init__()
        self.projection = nn.Linear(d_model, vocab_size)

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        """
        A forward pass of the Projection Layer of the Transformer model.

        Args:
            x (torch.Tensor): The input tensor. (N, seq_len, d_model)

        Returns:
            torch.Tensor: The output tensor. (N, seq_len, vocab_size)
        """
        return torch.log_softmax(self.projection(x), dim=-1)
