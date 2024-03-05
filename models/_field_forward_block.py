import torch
import torch.nn as nn


class FeedForwardBlock(nn.Module):

    def __init__(
        self,
        d_model: int,
        d_ff: int,
        dropout: float = 0.1,
    ) -> None:

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
        # 1. Now, apply the first Linear Layer
        x = self.linear1(x)
        x = self.prelu1(x)
        # Apply the Dropout Layer
        x = self.dropout(x)

        # 2. Now, apply the second Linear Layer
        x = self.linear2(x)
        x = self.prelu2(x)

        return x  # (N, seq_len, d_model)
