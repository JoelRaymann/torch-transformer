import torch
import torch.nn as nn
from typing import Optional

# Import Layers
import modules.layers as CustomLayers

# Import Other Blocks
from ._feed_forward import FeedForwardBlock


class TransformerDecoderBlock(nn.Module):
    """
    A `nn.Module` that represents the Decoder block of the Transformer model. This block consists of three sub-layers:

    1. Self-attention with layer normalization
    2. Encoder-decoder attention/cross-attention with layer normalization
    3. Feed-forward block with layer normalization
    """

    def __init__(
        self,
        d_model: int = 512,
        num_heads: int = 8,
        d_ff: int = 2048,
        dropout: float = 0.1,
    ) -> None:
        """
        A `nn.Module` that represents the Decoder block of the Transformer model. This block consists of three sub-layers:

        1. Self-attention with layer normalization
        2. Encoder-decoder attention/cross-attention with layer normalization
        3. Feed-forward block with layer normalization

        Args:
            d_model (int, optional): The number of expected features in the input (default: 512). Defaults to 512.
            num_heads (int, optional): The number of heads in the multiheadattention models (default: 8). Defaults to 8.
            d_ff (int, optional): The dimension of the feedforward network model (default: 2048). Defaults to 2048.
            dropout (float, optional): The dropout value (default: 0.1).
        """

        super().__init__()

        # 1. Define the first sub-layer: self-attention with layer normalization
        self.self_attention_block = CustomLayers.MultiHeadAttentionLayer(
            d_model=d_model,
            num_heads=num_heads,
            dropout=dropout,
        )
        self.self_attention_layer_norm = CustomLayers.LayerNorm1DLayer()

        # 2. Now, define the second sub-layer: encoder-decoder attention/cross-attention with layer normalization
        self.cross_attention_block = CustomLayers.MultiHeadAttentionLayer(
            d_model=d_model,
            num_heads=num_heads,
            dropout=dropout,
        )
        self.cross_attention_layer_norm = CustomLayers.LayerNorm1DLayer()

        # 3. Finally, define the feed-forward block with layer normalization
        self.feed_forward_block = FeedForwardBlock(
            d_model=d_model,
            d_ff=d_ff,
            dropout=dropout,
        )
        self.feed_forward_layer_norm = CustomLayers.LayerNorm1DLayer()

    def forward(
        self,
        x: torch.Tensor,
        encoder_output: torch.Tensor,
        encoder_mask: Optional[torch.Tensor] = None,
        decoder_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        A forward pass of the Decoder block of the Transformer model. This block consists of three sub-layers:

        1. Self-attention with layer normalization
        2. Encoder-decoder attention/cross-attention with layer normalization
        3. Feed-forward block with layer normalization

        ---
        NOTE: The `mask` tensors are used to mask out certain positions in the input tensors. Please set the mask to 0 for the positions
        that you wish to mask out. The `mask` tensors must have a shape of (N, seq_len, seq_len) - where the values you wish to mask out
        are set to 0.

        Args:
            x (torch.Tensor): The input tensor to the decoder block.
            encoder_output (torch.Tensor): The output tensor from the encoder block, used in the cross-attention sub-layer.
            encoder_mask (Optional[torch.Tensor], optional): The mask tensor for the encoder output, used in the cross-attention sub-layer
                to mask out certain positions. Defaults to None.
            decoder_mask (Optional[torch.Tensor], optional): The mask tensor for the decoder input, used in the self-attention sub-layer
                to prevent attention to future positions. Defaults to None.

        Returns:
            torch.Tensor: The output tensor of the decoder block after processing through the self-attention, cross-attention, and feed-forward sub-layers.
        """
        # 1. Pass through the first sub-layer: self-attention with layer normalization
        # 1.1. Apply the self-attention mechanism
        self_attention_output = self.self_attention_block(x, x, x, decoder_mask)
        # 1.2. Now, apply the skip connection and layer normalization
        x = self.self_attention_layer_norm(x + self_attention_output)

        # 2. Pass through the second sub-layer: encoder-decoder attention/cross-attention with layer normalization
        # 2.1. Apply the cross-attention mechanism
        cross_attention_output = self.cross_attention_block(
            x, encoder_output, encoder_output, encoder_mask
        )
        # 2.2. Now, apply the skip connection and layer normalization
        x = self.cross_attention_layer_norm(x + cross_attention_output)

        # 3. Finally, pass through the feed-forward block with layer normalization
        # 3.1. Apply the feed-forward block
        feed_forward_output = self.feed_forward_block(x)
        # 3.2. Now, apply the skip connection and layer normalization
        x = self.feed_forward_layer_norm(x + feed_forward_output)
