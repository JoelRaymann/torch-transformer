import torch
import torch.nn as nn


class FeedForwardBlock(nn.Module):
    """
    A simple Feed Forward Block, which is used in the Transformer Encoder and Decoder.
    """

    def __init__(
        self,
        d_model: int,
        d_ff: int,
        dropout: float = 0.1,
    ) -> None:
        """
        A simple Feed Forward Block, which is used in the Transformer Encoder and Decoder.
        This block consists of two Linear Layers with a PReLU activation function.

        Args:
            d_model (int): The number of expected features in the input (also called input size).
            d_ff (int): The number of features in the hidden layer of the Feed Forward Block.
            dropout (float, optional): The dropout probability. Defaults to 0.1.
        """
        super().__init__()

        # 1. Define the first Linear Layer
        self.linear1 = nn.Linear(d_model, d_ff)
        self.prelu1 = nn.PReLU(d_ff)

        # 2. Define the second Linear Layer
        self.linear2 = nn.Linear(d_ff, d_model)
        self.prelu2 = nn.PReLU(d_model)

        # Define the Dropout Layer
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,  # (N, seq_len, d_model)
    ) -> torch.Tensor:
        """
        Forward pass of the Feed Forward Block.

        Args:
            x (torch.Tensor): The input tensor. (N, seq_len, d_model)

        Returns:
            torch.Tensor: The output tensor. (N, seq_len, d_model)
        """
        # 1. Now, apply the first Linear Layer
        x = self.linear1(x)
        x = self.prelu1(x)
        # Apply the Dropout Layer
        x = self.dropout(x)

        # 2. Now, apply the second Linear Layer
        x = self.linear2(x)
        x = self.prelu2(x)

        return x  # (N, seq_len, d_model)
