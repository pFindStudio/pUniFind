# Copyright (c) 2024, pUniFind Team
# Neural network modules compatible with unicore

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class LayerNorm(nn.LayerNorm):
    """
    Layer normalization with optional FP16 support.
    Compatible with unicore's LayerNorm.
    """

    def __init__(
        self,
        normalized_shape: int,
        eps: float = 1e-5,
        elementwise_affine: bool = True,
    ):
        super().__init__(normalized_shape, eps=eps, elementwise_affine=elementwise_affine)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Handle FP16 inputs
        input_dtype = x.dtype
        if x.dtype == torch.float16:
            x = x.float()
        output = super().forward(x)
        if input_dtype == torch.float16:
            output = output.half()
        return output


class SelfMultiheadAttention(nn.Module):
    """
    Self-attention mechanism compatible with unicore.
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout: float = 0.0,
        bias: bool = True,
        self_attention: bool = True,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == embed_dim, "embed_dim must be divisible by num_heads"

        self.scaling = self.head_dim ** -0.5

        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)

        self._reset_parameters()

    def _reset_parameters(self):
        nn.init.xavier_uniform_(self.q_proj.weight)
        nn.init.xavier_uniform_(self.k_proj.weight)
        nn.init.xavier_uniform_(self.v_proj.weight)
        nn.init.xavier_uniform_(self.out_proj.weight)

        if self.q_proj.bias is not None:
            nn.init.constant_(self.q_proj.bias, 0.0)
            nn.init.constant_(self.k_proj.bias, 0.0)
            nn.init.constant_(self.v_proj.bias, 0.0)
            nn.init.constant_(self.out_proj.bias, 0.0)

    def forward(
        self,
        query: torch.Tensor,
        key: Optional[torch.Tensor] = None,
        value: Optional[torch.Tensor] = None,
        key_padding_mask: Optional[torch.Tensor] = None,
        attn_mask: Optional[torch.Tensor] = None,
        need_weights: bool = False,
        need_head_weights: bool = False,
    ) -> tuple:
        """
        Forward pass for self-attention.

        Args:
            query: Query tensor of shape (batch, seq_len, embed_dim)
            key: Key tensor (optional, defaults to query for self-attention)
            value: Value tensor (optional, defaults to query for self-attention)
            key_padding_mask: Mask for key padding positions
            attn_mask: Attention mask
            need_weights: Whether to return attention weights
            need_head_weights: Whether to return per-head attention weights

        Returns:
            Tuple of (output, attention_weights)
        """
        if key is None:
            key = query
        if value is None:
            value = query

        batch_size, seq_len, embed_dim = query.size()

        # Project Q, K, V
        q = self.q_proj(query)
        k = self.k_proj(key)
        v = self.v_proj(value)

        # Scale query
        q = q * self.scaling

        # Reshape for multi-head attention
        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)

        # Compute attention scores
        attn_weights = torch.matmul(q, k.transpose(-2, -1))

        # Apply attention mask
        if attn_mask is not None:
            attn_weights = attn_weights + attn_mask

        # Apply key padding mask
        if key_padding_mask is not None:
            attn_weights = attn_weights.masked_fill(
                key_padding_mask.unsqueeze(1).unsqueeze(2),
                float('-inf'),
            )

        # Softmax and dropout
        attn_weights = F.softmax(attn_weights, dim=-1)
        attn_weights = F.dropout(attn_weights, p=self.dropout, training=self.training)

        # Apply attention to values
        attn_output = torch.matmul(attn_weights, v)

        # Reshape back
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, embed_dim)

        # Output projection
        output = self.out_proj(attn_output)

        if need_weights:
            if need_head_weights:
                return output, attn_weights
            else:
                return output, attn_weights.mean(dim=1)
        else:
            return output, None


def softmax_dropout(
    x: torch.Tensor,
    dropout_prob: float,
    training: bool = True,
    mask: Optional[torch.Tensor] = None,
    bias: Optional[torch.Tensor] = None,
    inplace: bool = False,
) -> torch.Tensor:
    """
    Fused softmax and dropout operation.
    Compatible with unicore's softmax_dropout.

    Args:
        x: Input tensor
        dropout_prob: Dropout probability
        training: Whether in training mode
        mask: Optional attention mask (additive, -inf for masked positions)
        bias: Optional bias to add before softmax
        inplace: Whether to operate in-place

    Returns:
        Tensor after softmax and dropout
    """
    # Add bias if provided
    if bias is not None:
        x = x + bias

    # Apply mask if provided
    if mask is not None:
        # Convert boolean mask to additive mask
        if mask.dtype == torch.bool:
            x = x.masked_fill(mask, float('-inf'))
        else:
            x = x + mask

    # Softmax
    x = F.softmax(x, dim=-1)

    # Dropout
    if training and dropout_prob > 0:
        x = F.dropout(x, p=dropout_prob, training=True, inplace=inplace)

    return x
