from typing import Optional
import torch
import torch.nn as nn

# Import Layers
import modules.layers as CustomLayers

# Import Other Blocks
from ._feed_forward import FeedForwardBlock


class TransformerEncoderBlock(nn.Module):
    """
    The encoder block of the Transformer model. This is the block that is used in the Encoder of the Transformer model.
    It consists of a Multi-Head Attention Layer and a Feed Forward Block.
    """

    def __init__(
        self,
        d_model: int = 512,
        d_ff: int = 2048,
        num_heads: int = 8,
        dropout: float = 0.1,
    ) -> None:
        """
        A Transformer Encoder Block. This is the block that is used in the Encoder of the Transformer model.

        Args:
            d_model (int, optional): The number of features in the input tensor. Defaults to 512.
            d_ff (int, optional): The number of features in the feed forward layer. Defaults to 2048.
            num_heads (int, optional): The number of heads in the multi-head attention layer. Defaults to 8.
            dropout (float, optional): The dropout probability. Defaults to 0.1.
        """
        super().__init__()

        # 1. Define the layers
        self.self_attention = CustomLayers.MultiHeadAttentionLayer(
            d_model=d_model,
            num_heads=num_heads,
            dropout=dropout,
        )
        self.self_attention_norm = CustomLayers.LayerNorm1DLayer()
        # 2. Now, define the Feed Forward Block
        self.feed_forward = FeedForwardBlock(
            d_model=d_model,
            d_ff=d_ff,
            dropout=dropout,
        )
        self.feed_forward_norm = CustomLayers.LayerNorm1DLayer()

    def forward(
        self,
        x: torch.Tensor,  # (N, seq_len, d_model)
        mask: Optional[torch.Tensor] = None,  # (N, seq_len, seq_len)
    ) -> torch.Tensor:
        """
        The forward pass of the Transformer Encoder Block.

        Args:
            x (torch.Tensor): The input tensor. (N, seq_len, d_model)
            mask (Optional[torch.Tensor], optional): The mask tensor. Defaults to None.

        Returns:
            torch.Tensor: The output tensor. (N, seq_len, d_model)
        """
        # 1. Self Attention
        self_attention_output = self.self_attention(x, x, x, mask)
        # 2. Add and Norm
        x = self.self_attention_norm(x + self_attention_output)
        # 3. Feed Forward
        feed_forward_output = self.feed_forward(x)
        # 4. Add and Norm
        x = self.feed_forward_norm(x + feed_forward_output)
        # Return the output
        return x
