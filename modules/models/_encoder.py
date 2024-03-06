import torch
import torch.nn as nn
from typing import Optional

# Import Blocks
import modules.blocks as CustomBlocks


class EncoderModel(nn.Module):
    """
    A `nn.Module` that represents the Encoder of the Transformer model. This is a stack of `num_layers` Transformer encoder
    blocks.
    """

    def __init__(
        self,
        num_layers: int = 6,
        d_model: int = 512,
        d_ff: int = 2048,
        num_heads: int = 8,
        dropout: float = 0.1,
    ) -> None:
        """
        A `nn.Module` that represents the Encoder of the Transformer model. This is a stack of `num_layers` Transformer encoder
        blocks.

        Args:
            num_layers (int, optional): The number of Transformer encoder blocks to stack in the encoder. Defaults to 6.
        """
        self.encoder_blocks = nn.ModuleList(
            [
                CustomBlocks.TransformerEncoderBlock(
                    d_model=d_model,
                    d_ff=d_ff,
                    num_heads=num_heads,
                    dropout=dropout,
                )
                for _ in range(num_layers)
            ]
        )

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        A forward pass of the Encoder of the Transformer model. Set mask to 0 for the positions that you wish
        to mask.

        Args:
            x (torch.Tensor): The input tensor. (N, seq_len, d_model)
            mask (Optional[torch.Tensor], optional): The mask tensor. Defaults to None. (N, seq_len, seq_len)

        Returns:
            torch.Tensor: The output tensor. (N, seq_len, d_model)
        """
        for encoder_block in self.encoder_blocks:
            x = encoder_block(x, mask)
        return x
