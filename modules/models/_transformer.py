import torch
import torch.nn as nn

# Import Blocks
import modules.blocks as CustomBlocks

# Import Layers
import modules.layers as CustomLayers


class TransformerModel(nn.Module):
    """
    A `nn.Module` that represents the Transformer model.
    """

    def __init__(
        self,
        encoder: CustomBlocks.TransformerEncoderBlock,
        decoder: CustomBlocks.TransformerDecoderBlock,
        source_embedding: CustomLayers.InputEmbeddingLayer,
        target_embedding: CustomLayers.InputEmbeddingLayer,
        source_positional_encoding: CustomLayers.PositionalEncodingLayer,
        target_positional_encoding: CustomLayers.PositionalEncodingLayer,
        projection_layer: CustomLayers.ProjectionLayer,
    ) -> None:
        """
        A `nn.Module` that represents the Transformer model. This model is a stack of `num_layers` Transformer encoder and decoder blocks.

        Args:
            encoder (CustomBlocks.TransformerEncoderBlock): The encoder block of the Transformer model.
            decoder (CustomBlocks.TransformerDecoderBlock): The decoder block of the Transformer model.
            source_embedding (CustomLayers.InputEmbeddingLayer): The input embedding layer for the source language.
            target_embedding (CustomLayers.InputEmbeddingLayer): The input embedding layer for the target language.
            source_positional_encoding (CustomLayers.PositionalEncodingLayer): The positional encoding layer for the source language.
            target_positional_encoding (CustomLayers.PositionalEncodingLayer): The positional encoding layer for the target language.
            projection_layer (CustomLayers.ProjectionLayer): The projection layer for the Transformer model.
        """
        super().__init__()
        self.encoder_layer = encoder
        self.decoder_layer = decoder
        self.source_embedding_layer = source_embedding
        self.target_embedding_layer = target_embedding
        self.source_positional_encoding_layer = source_positional_encoding
        self.target_positional_encoding_layer = target_positional_encoding
        self.projection_layer = projection_layer

    def encode(
        self,
        source_tensor: torch.Tensor,
        source_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        A forward pass of the Encoder of the Transformer model.

        Args:
            source_tensor (torch.Tensor): The input tensor. (N, seq_len)
            source_mask (torch.Tensor): The mask for the input tensor. (N, seq_len)

        Returns:
            torch.Tensor: The output tensor. (N, seq_len, d_model)
        """
        source_tensor = self.source_embedding_layer(source_tensor)
        source_tensor = self.source_positional_encoding_layer(source_tensor)
        return self.encoder_layer(source_tensor, source_mask)

    def decode(
        self,
        encoded_output: torch.Tensor,
        source_mask: torch.Tensor,
        target_tensor: torch.Tensor,
        target_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        A forward pass of the Decoder of the Transformer model.

        Args:
            encoded_output (torch.Tensor): The output tensor from the Encoder. (N, seq_len, d_model)
            source_mask (torch.Tensor): The mask for the input tensor. (N, seq_len)
            target_tensor (torch.Tensor): The input tensor. (N, seq_len)
            target_mask (torch.Tensor): The mask for the input tensor. (N, seq_len)

        Returns:
            torch.Tensor: The output tensor. (N, seq_len, d_model)
        """
        target_tensor = self.target_embedding_layer(target_tensor)
        target_tensor = self.target_positional_encoding_layer(target_tensor)
        return self.decoder_layer(
            target_tensor, encoded_output, source_mask, target_mask
        )

    def project(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        """
        A forward pass of the projection layer of the Transformer model.

        Args:
            x (torch.Tensor): The input tensor. (N, seq_len, d_model)

        Returns:
            torch.Tensor: The output tensor. (N, seq_len, vocab_size)
        """
        return self.projection_layer(x)
