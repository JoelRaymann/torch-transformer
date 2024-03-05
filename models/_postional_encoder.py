import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    """
    A `nn.Module` layer class which implements the positional encoding of the Transformer model. This will take the embedded sequence matrix of the
    input sentence and apply positional encoding on it along with (optional) dropout.
    """

    def __init__(
        self,
        d_model: int,
        sequence_length: int,
        dropout: float = 0.1,
    ) -> None:
        """
        A `nn.Module` layer class which implements the positional encoding of the Transformer model. This will take the embedded sequence matrix of the
        input sentence and apply positional encoding on it along with (optional) dropout.

        Args:
            d_model (int): The size of the model.
            sequence_length (int): The length of the setences
            dropout (float, optional): The probability of the dropout - the rate of dropout. Defaults to 0.1.
        """

        super().__init__()
        self.d_model = d_model
        self.sequence_length = sequence_length
        self.dropout = nn.Dropout(p=dropout)

        # 1. Create the matrix of shape (sequence_length, d_model) [This is the positional encoding matrix] (shape: (sequence_length, d_model))
        self.positional_encoding = torch.zeros(self.sequence_length, self.d_model)
        # 2. Now, we need a vector to represent the position of the word in the sentence [This is the position vector] (shape: (sequence_length, 1))
        position_vector = torch.arange(
            0, self.sequence_length, dtype=torch.float
        ).unsqueeze(1)
        # 3. Let's calculate the "10000" raised to the power of 2i/d_model. For mathematical stability, we use the `exp` function and apply it to the `log` function.
        # Don't forget that it is applied to the even indices of the matrix. (shape: (d_model/2))
        div_term = torch.exp(
            torch.arange(0, self.d_model, 2).float()
            * (-torch.log(10000.0) / self.d_model)
        )
        # 4. Now, we need to apply the `sin` function to the even indices of the positional encoding matrix and the
        # `cos` function to the odd indices of the positional encoding matrix.
        self.positional_encoding[:, 0::2] = torch.sin(position_vector * div_term)
        self.positional_encoding[:, 1::2] = torch.cos(position_vector * div_term)
        self.positional_encoding = self.positional_encoding.unsqueeze(
            0
        )  # (1, seq_len, d_model)

        # 5. Store the positional_encoding as buffer
        self.register_buffer("positional_encoding", self.positional_encoding)

    def forward(
        self,
        x: torch.Tensor,  # (N, seq_len, d_model) [Embedding vector of the i/p sentence]
    ) -> torch.Tensor:
        """
        The forward pass of the positional encoding. This will take the input 'x' which is the sequence of embedded
        vector and convert them to positional encoding. Then, it adds the positional encoding with the embedded vector
        and apply dropout.

        Args:
            x (torch.Tensor): The input embedding matrix for the sequences.

        Returns:
            torch.Tensor: The output tensor including both the positional encoding and the embedding.
        """
        # 1. Get the positional encoding from the input embedding vector
        # (N, 0: seq_len, d_model)
        x__POSITION_ENCODING = self.positional_encoding[
            :, : x.shape[1], :
        ].requires_grad_(False)
        # 2. Add the input embedding with the positional encoding and apply dropout
        return self.dropout(x + x__POSITION_ENCODING)
