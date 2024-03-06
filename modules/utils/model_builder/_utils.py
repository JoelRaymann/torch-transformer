import torch
import torch.nn as nn

# Import Models
import modules.models as CustomModels

# Import Layers
import modules.layers as CustomLayers


def build_transformer(
    source_vocab_size: int,
    target_vocab_size: int,
    source_max_len: int,
    target_max_len: int,
    d_model: int = 512,
    num_encoder_layers: int = 6,
    num_decoder_layers: int = 6,
    num_heads: int = 8,
    dropout: float = 0.1,
    d_ff: int = 2048,
) -> CustomModels.TransformerModel:
    """
    A `Utility` function to build the Transformer model using the custom layers and blocks.

    Args:
        source_vocab_size (int): The size of the source vocabulary.
        target_vocab_size (int): The size of the target vocabulary.
        source_max_len (int): The maximum length of the source sequence.
        target_max_len (int): The maximum length of the target sequence.
        d_model (int, optional): The dimension of the model. Defaults to 512.
        num_encoder_layers (int, optional): The number of encoder layers. Defaults to 6.
        num_decoder_layers (int, optional): The number of decoder layers. Defaults to 6.
        num_heads (int, optional): The number of heads in the multi-head attention. Defaults to 8.
        dropout (float, optional): The dropout probability. Defaults to 0.1.
        d_ff (int, optional): The dimension of the feedforward network. Defaults to 2048.

    Returns:
        CustomModels.TransformerModel: The Transformer model.
    """
    # 1. First we build the embedding layers
    source_embedding = CustomLayers.InputEmbeddingLayer(
        d_model=d_model,
        vocab_size=source_vocab_size,
    )
    target_embedding = CustomLayers.InputEmbeddingLayer(
        d_model=d_model,
        vocab_size=target_vocab_size,
    )

    # 2. Create the positional encoding layers
    source_positional_encoding = CustomLayers.PositionalEncodingLayer(
        d_model=d_model,
        sequence_length=source_max_len,
        dropout=dropout,
    )
    target_positional_encoding = CustomLayers.PositionalEncodingLayer(
        d_model=d_model,
        sequence_length=target_max_len,
        dropout=dropout,
    )

    # 3. Create the encoder and decoder model
    encoder_model = CustomModels.EncoderModel(
        num_layers=num_encoder_layers,
        d_model=d_model,
        d_ff=d_ff,
        num_heads=num_heads,
        dropout=dropout,
    )
    decoder_model = CustomModels.DecoderModel(
        num_layers=num_decoder_layers,
        d_model=d_model,
        d_ff=d_ff,
        num_heads=num_heads,
        dropout=dropout,
    )

    # 4. Create the projection layer
    projection_layer = CustomLayers.ProjectionLayer(
        d_model=d_model,
        vocab_size=target_vocab_size,
    )

    # 5. Create the transformer model
    transformer_model = CustomModels.TransformerModel(
        encoder=encoder_model,
        decoder=decoder_model,
        source_embedding=source_embedding,
        target_embedding=target_embedding,
        source_positional_encoding=source_positional_encoding,
        target_positional_encoding=target_positional_encoding,
        projection_layer=projection_layer,
    )

    # 6. Initialize the weights using the Xavier initialization
    for param in transformer_model.parameters():
        if param.dim() > 1:
            nn.init.xavier_uniform_(param)

    # 7. Return the transformer model
    return transformer_model
