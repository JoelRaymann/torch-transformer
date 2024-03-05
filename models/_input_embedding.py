import torch
import torch.nn as nn


class InputEmbeddings(nn.Module):
    """
    A `nn.Module` model class which is used to create the input embeddings for the transformer model.

    NOTE: This uses the pytorch's `Embedding` layer to create the input embeddings.
    """

    def __init__(
        self,
        d_model: int,
        vocab_size: int,
    ) -> None:
        """
        A `nn.Module` model class which is used to create the input embeddings for the transformer model.

        Args:
            d_model (int): The dimension of the model
            vocab_size (int): The vocabulary size of the input data
        """
        super().__init__()
        self.d_model = d_model  # d_model is the dimension of the model
        self.vocab_size = vocab_size  # vocab_size is the size of the vocabulary
        # This is the embedding layer, which is a simple lookup table that stores embeddings of a fixed dictionary and size.
        self.embedding = nn.Embedding(vocab_size, d_model)

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        """
        This method implements the forward pass of the Model. According to the paper,
        the input embeddings are multiplied by sqrt(d_model) before passing them to the encoder.

        Formula: embedding * sqrt(d_model)

        Args:
            x (torch.Tensor): The input tensor

        Returns:
            torch.Tensor: The output embeddings
        """
        # Formula is from the original paper of Transformer - embedding * sqrt(d_model)
        return self.embedding(x) * (self.d_model**0.5)
