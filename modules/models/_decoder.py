from typing import Optional
import torch
import torch.nn as nn

# Import Blocks
import modules.blocks as CustomBlocks


class DecoderModel(nn.Module):
    """
    A `nn.Module` that represents the Decoder of the Transformer model. This is a stack of `num_layers` Transformer decoder blocks.
    """

    def __init__(
        self,
        num_layers: int = 6,
    ) -> None:
        """
        A `nn.Module` that represents the Decoder of the Transformer model. This is a stack of `num_layers` Transformer decoder
        blocks.

        Args:
            num_layers (int, optional): The number of layers in the decoder. Defaults to 6.
        """
        self.decoder_blocks = nn.ModuleList(
            [
                CustomBlocks.TransformerDecoderBlock(
                    d_model=512,
                    d_ff=2048,
                    num_heads=8,
                    dropout=0.1,
                )
                for _ in range(num_layers)
            ]
        )

    def forward(
        self,
        x: torch.Tensor,
        encoder_output: torch.Tensor,
        encoder_mask: Optional[torch.Tensor] = None,
        decoder_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        A forward pass of the Decoder of the Transformer model.

        Args:
            x (torch.Tensor): The input tensor. (N, seq_len, d_model)
            encoder_output (torch.Tensor): The output tensor from the Encoder. (N, seq_len, d_model)
            encoder_mask (Optional[torch.Tensor], optional): The mask for the encoder. Defaults to None.
            decoder_mask (Optional[torch.Tensor], optional): The mask for the decoder. Defaults to None.

        Returns:
            torch.Tensor: The output tensor. (N, seq_len, d_model)
        """
        for decoder_block in self.decoder_blocks:
            x = decoder_block(x, encoder_output, encoder_mask, decoder_mask)
        return x
