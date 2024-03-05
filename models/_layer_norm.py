import torch
import torch.nn as nn


class LayerNorm1D(nn.Module):
    """
    A `nn.Module` layer class which implements the Layer Normalization of the Transformer model. This will take the
    embedded sequence matrix of the input sentence and normalize it using the mean and standard deviation of the sequence
    matrix.
    """

    def __init__(
        self,
        epsilon: float = 1e-7,
    ) -> None:
        """
        A `nn.Module` layer class which implements the Layer Normalization of the Transformer model. This will take
        the embedded sequence matrix of the input sentence and normalize it using the mean and standard deviation of the
        sequence matrix.

        Args:
            epsilon (float, optional): The epsilon value to avoid division by zero. Defaults to 1e-7.
        """

        super().__init__()
        # Define the epsilon value
        self.epsilon = epsilon

        # Define the learnable parameters
        self.gamma = nn.Parameter(torch.ones(1))  # Scaling Factor [Learnable Parameter]
        self.beta = nn.Parameter(
            torch.zeros(1)
        )  # Additive Bias Term [Learnable Parameter]

    def forward(
        self,
        x: torch.Tensor,  # (N, seq_len, d_model) [Embedding vector of the i/p sentence]
    ) -> torch.Tensor:
        """
        A forward pass of the Layer Normalization layer.

        Args:
            x (torch.Tensor): The input sequence matrix of the i/p sentence. (N, seq_len, d_model)

        Returns:
            torch.Tensor: The normalized sequence matrix of the i/p sentence. (N, seq_len, d_model)
        """
        x__MEAN = x.mean(-1, keepdim=True)  # (N, seq_len, 1)
        x__STD = x.std(-1, keepdim=True)  # (N, seq_len, 1)
        x__NORM = (x - x__MEAN) / (x__STD + self.epsilon)
        return self.gamma * x__NORM + self.beta  # (N, seq_len, d_model)
