from typing import Optional
import torch
import torch.nn as nn


class MultiHeadAttentionLayer(nn.Module):
    """
    A `nn.Module` layer class which implements the Multi-Head Attention of the Transformer model. This will take the Query, Key and Value
    tensors and apply the Multi-Head Attention on them. The output of the Multi-Head Attention is then returned.

    The formula to calculate the attention scores is:

    Attention(Q, K, V) = softmax((Q @ K^T) / sqrt(d_k)) @ V

    The Multi-Head Attention is calculated as follows:
    1. Apply the Linear Layers to get the Query, Key and Value
    2. Now, split the Query, Key and Value into multiple heads
    3. Now, calculate the attention scores and the output for each head
    4. Now, concatenate the output of each head and apply the final linear layer
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        dropout: float = 0.1,
    ) -> None:
        """
        A `nn.Module` layer class which implements the Multi-Head Attention of the Transformer model. This will take the Query, Key and Value
        tensors and apply the Multi-Head Attention on them. The output of the Multi-Head Attention is then returned.

        Args:
            d_model (int): The dimension of the model
            num_heads (int): The number of heads to be used
            dropout (float, optional): The dropout probability. Defaults to 0.1.
        """
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        assert d_model % num_heads == 0  # d_model should be divisible by num_heads

        self.d_k = d_model // num_heads  # Total number of heads
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    @staticmethod
    def attention(
        query: torch.Tensor,  # (N, num_heads, seq_len, d_k)
        key: torch.Tensor,  # (N, num_heads, seq_len, d_k)
        value: torch.Tensor,  # (N, num_heads, seq_len, d_k)
        mask: Optional[torch.Tensor] = None,  # (N, num_heads, seq_len, seq_len)
        dropout: Optional[nn.Dropout] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        A helper function to calculate the attention scores and the output of the
        Multi-Head Attention Block. This function is static as it does not depend on
        the state of the object.

        The fomula to calculate the attention scores is:

        Attention(Q, K, V) = softmax((Q @ K^T) / sqrt(d_k)) @ V

        Where, Q, K and V are the Query, Key and Value tensors respectively. The @
        symbol denotes the matrix multiplication and the ^T denotes the transpose of the matrix.
        Now, Q shape: (N, num_heads, seq_len, d_k), K shape: (N, num_heads, seq_len, d_k)
        and V shape: (N, num_heads, seq_len, d_k).  When taking transpose of K, the shape
        becomes (N, num_heads, d_k, seq_len). The result of the matrix multiplication is
        (N, num_heads, seq_len, seq_len) which is then divided by sqrt(d_k). The softmax
        function is applied to the result which is our attention scores. The attention scores
        are then multiplied by the Value tensor to get the output. The output is then returned
        along with the attention scores.

        Args:
            query (torch.Tensor): The Query tensor of shape (N, num_heads, seq_len, d_k)
            key (torch.Tensor): The Key tensor of shape (N, num_heads, seq_len, d_k)
            value (torch.Tensor): The Value tensor of shape (N, num_heads, seq_len, d_k)
            mask (torch.Tensor, optional): The mask tensor of shape (N, num_heads, seq_len, seq_len). Set
                to 0 for the elements to be masked. Defaults to None.
            dropout (nn.Dropout, optional): The Dropout layer to be used. Defaults to None.

        Returns:
            tuple[torch.Tensor, torch.Tensor]: The output tensor and the attention scores tensor
        """
        # 1. Get the d_k value from the query tensor
        d_k = query.shape[-1]
        # 2. Calculate the attention scores
        attention_scores = (query @ key.transpose(-2, -1)) / torch.sqrt(
            torch.tensor(d_k, dtype=torch.float32)
        )
        if mask is not None:
            attention_scores = attention_scores.masked_fill(mask == 0, -torch.inf)
        attention_scores = nn.functional.softmax(attention_scores, dim=-1)
        if dropout is not None:
            attention_scores = dropout(attention_scores)

        output = attention_scores @ value
        return output, attention_scores

    def forward(
        self,
        q: torch.Tensor,  # (N, seq_len, d_model)
        k: torch.Tensor,  # (N, seq_len, d_model)
        v: torch.Tensor,  # (N, seq_len, d_model)
        mask: Optional[torch.Tensor] = None,  # (N, seq_len, seq_len)
    ) -> torch.Tensor:
        """
        A forward pass of the Multi-Head Attention Layer. This will take the Query, Key and Value tensors
        and apply the Multi-Head Attention on them. Optionally, a mask tensor can be passed to mask the
        attention scores.

        Args:
            q (torch.Tensor): The Query tensor of shape (N, seq_len, d_model)

        Returns:
            torch.Tensor: The output tensor of the Multi-Head Attention Layer
        """
        # 1. Apply the Linear Layers to get the Query, Key and Value
        q_prime: torch.Tensor = self.w_q(q)  # (N, seq_len, d_model)
        k_prime: torch.Tensor = self.w_k(k)  # (N, seq_len, d_model)
        v_prime: torch.Tensor = self.w_v(v)  # (N, seq_len, d_model)
        # 2. Now, split the Query, Key and Value into multiple heads
        # Torch view allows us to reshape the tensor without changing the data
        # We are keeping the number of batches and sequence length same, but we are splitting the d_model into num_heads and d_k
        # The result is (N, seq_len, num_heads, d_k) obtained from (N, seq_len, d_model)
        q_heads: torch.Tensor = q_prime.view(
            q_prime.shape[0],
            q_prime.shape[1],
            self.num_heads,
            self.d_k,
        )  # (N, seq_len, num_heads, d_k)
        # We need to transpose the tensor such that each head has the full sentence but only sees a fraction of the embedding
        # This means we need (N, seq_len, num_heads, d_k) -> (N, num_heads, seq_len, d_k)
        q_heads = q_heads.transpose(1, 2)
        # Similarly, we do the same for Key and Value
        k_heads: torch.Tensor = k_prime.view(
            k_prime.shape[0],
            k_prime.shape[1],
            self.num_heads,
            self.d_k,
        ).transpose(
            1, 2
        )  # (N, num_heads, seq_len, d_k)
        v_heads: torch.Tensor = v_prime.view(
            v_prime.shape[0],
            v_prime.shape[1],
            self.num_heads,
            self.d_k,
        ).transpose(
            1, 2
        )  # (N, num_heads, seq_len, d_k)
        # 3. Now, calculate the attention scores and the output
        output, attention_scores = self.attention(
            q_heads, k_heads, v_heads, mask, self.dropout
        )
        # 4. Now, concatenate the output of each head and apply the final linear layer
        # The output of each head is (N, num_heads, seq_len, d_k) and we need to convert it to (N, seq_len, d_model)
        # We first transpose the tensor to (N, seq_len, num_heads, d_k) and then reshape it to (N, seq_len, d_model)
        output = (
            output.transpose(1, 2).contiguous().view(output.shape[0], -1, self.d_model)
        )
        output = self.w_o(output)  # (N, seq_len, d_model)
        return output  # (N, seq_len, d_model)
