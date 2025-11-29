"""Base Transformer models for working with mass spectra and peptides"""

import copy
import itertools
import pickle
import re
from typing import Any, Callable, Dict, Optional, Tuple, Union

import numpy as np
import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torch.nn.modules.activation import MultiheadAttention
from torch.nn.modules.container import ModuleList
from torch.nn.modules.dropout import Dropout
from torch.nn.modules.linear import Linear
from torch.nn.modules.module import Module
from torch.nn.modules.normalization import LayerNorm

from .. import utils2 as utils
from .encoders import MassEncoder, PeakEncoder, PositionalEncoder
from .masses import PeptideMass

amino_acids_3to1_map = {
    "Ala": "A",
    "Arg": "R",
    "Asn": "N",
    "Asp": "D",
    "Cys": "C",
    "Gln": "Q",
    "Glu": "E",
    "Gly": "G",
    "His": "H",
    "Ile": "I",
    "Leu": "L",
    "Lys": "K",
    "Met": "M",
    "Phe": "F",
    "Pro": "P",
    "Ser": "S",
    "Thr": "T",
    "Trp": "W",
    "Tyr": "Y",
    "Val": "V",
}


def _get_activation_fn(activation: str) -> Callable[[Tensor], Tensor]:
    if activation == "relu":
        return F.relu
    elif activation == "gelu":
        return F.gelu

    raise RuntimeError(f"activation should be relu/gelu, not {activation}")


def initialize_embedding_to_zero(embedding_layer):
    with torch.no_grad():  # 确保在初始化过程中不计算梯度
        embedding_layer.weight.fill_(0)


class TransformerEncoderLayer(Module):
    r"""
    Args:
        d_model: the number of expected features in the input (required).
        nhead: the number of heads in the multiheadattention models (required).
        dim_feedforward: the dimension of the feedforward network model (default=2048).
        dropout: the dropout value (default=0.1).
        activation: the activation function of the intermediate layer, can be a string
            ("relu" or "gelu") or a unary callable. Default: relu
        layer_norm_eps: the eps value in layer normalization components (default=1e-5).
        batch_first: If ``True``, then the input and output tensors are provided
            as (batch, seq, feature). Default: ``False`` (seq, batch, feature).
        norm_first: if ``True``, layer norm is done prior to attention and feedforward
            operations, respectively. Otherwise it's done after. Default: ``False`` (after).
        bias: If set to ``False``, ``Linear`` and ``LayerNorm`` layers will not learn an additive
            bias. Default: ``True``.

    Examples::
        >>> encoder_layer = nn.TransformerEncoderLayer(d_model=512, nhead=8)
        >>> src = torch.rand(10, 32, 512)
        >>> out = encoder_layer(src)

    Alternatively, when ``batch_first`` is ``True``:
        >>> encoder_layer = nn.TransformerEncoderLayer(d_model=512, nhead=8, batch_first=True)
        >>> src = torch.rand(32, 10, 512)
        >>> out = encoder_layer(src)

    Fast path:
        forward() will use a special optimized implementation described in
        `FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness`_ if all of the following
        conditions are met:

        - Either autograd is disabled (using ``torch.inference_mode`` or ``torch.no_grad``) or no tensor
          argument ``requires_grad``
        - training is disabled (using ``.eval()``)
        - batch_first is ``True`` and the input is batched (i.e., ``src.dim() == 3``)
        - activation is one of: ``"relu"``, ``"gelu"``, ``torch.functional.relu``, or ``torch.functional.gelu``
        - at most one of ``src_mask`` and ``src_key_padding_mask`` is passed
        - if src is a `NestedTensor <https://pytorch.org/docs/stable/nested.html>`_, neither ``src_mask``
          nor ``src_key_padding_mask`` is passed
        - the two ``LayerNorm`` instances have a consistent ``eps`` value (this will naturally be the case
          unless the caller has manually modified one without modifying the other)

        If the optimized implementation is in use, a
        `NestedTensor <https://pytorch.org/docs/stable/nested.html>`_ can be
        passed for ``src`` to represent padding more efficiently than using a padding
        mask. In this case, a `NestedTensor <https://pytorch.org/docs/stable/nested.html>`_ will be
        returned, and an additional speedup proportional to the fraction of the input that
        is padding can be expected.

        .. _`FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness`:
         https://arxiv.org/abs/2205.14135

    """

    __constants__ = ["batch_first", "norm_first"]

    def __init__(
        self,
        d_model: int,
        nhead: int,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        activation: Union[str, Callable[[Tensor], Tensor]] = F.relu,
        layer_norm_eps: float = 1e-5,
        batch_first: bool = False,
        norm_first: bool = False,
        device=None,
        dtype=None,
        use_rope=False,
    ) -> None:
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.self_attn = MultiheadAttention(
            d_model, nhead, dropout=dropout, batch_first=batch_first, **factory_kwargs
        )
        # Implementation of Feedforward model
        self.linear1 = Linear(d_model, dim_feedforward, **factory_kwargs)
        self.dropout = Dropout(dropout)
        self.linear2 = Linear(dim_feedforward, d_model, **factory_kwargs)

        self.use_rope = use_rope
        self.norm_first = norm_first
        self.norm1 = LayerNorm(d_model, eps=layer_norm_eps, **factory_kwargs)
        self.norm2 = LayerNorm(d_model, eps=layer_norm_eps, **factory_kwargs)
        self.dropout1 = Dropout(dropout)
        self.dropout2 = Dropout(dropout)

        # Legacy string support for activation function.
        if isinstance(activation, str):
            activation = _get_activation_fn(activation)

        # We can't test self.activation in forward() in TorchScript,
        # so stash some information about it instead.
        if activation is F.relu or isinstance(activation, torch.nn.ReLU):
            self.activation_relu_or_gelu = 1
        elif activation is F.gelu or isinstance(activation, torch.nn.GELU):
            self.activation_relu_or_gelu = 2
        else:
            self.activation_relu_or_gelu = 0
        self.activation = activation

    def __setstate__(self, state):
        super().__setstate__(state)
        if not hasattr(self, "activation"):
            self.activation = F.relu

    def forward(
        self,
        src: Tensor,
        src_mask: Optional[Tensor] = None,
        src_key_padding_mask: Optional[Tensor] = None,
        is_causal: bool = False,
        rope_embeding: Tensor = None,
    ) -> Tensor:
        r"""Pass the input through the encoder layer.

        Args:
            src: the sequence to the encoder layer (required).
            src_mask: the mask for the src sequence (optional).
            is_causal: If specified, applies a causal mask as src_mask.
              Default: ``False``.
            src_key_padding_mask: the mask for the src keys per batch (optional).

        Shape:
            see the docs in Transformer class.
        """
        src_key_padding_mask = F._canonical_mask(
            mask=src_key_padding_mask,
            mask_name="src_key_padding_mask",
            other_type=F._none_or_dtype(src_mask),
            other_name="src_mask",
            target_type=src.dtype,
        )

        src_mask = F._canonical_mask(
            mask=src_mask,
            mask_name="src_mask",
            other_type=None,
            other_name="",
            target_type=src.dtype,
            check_other=False,
        )

        # see Fig. 1 of https://arxiv.org/pdf/2002.04745v1.pdf
        why_not_sparsity_fast_path = ""
        if not src.dim() == 3:
            why_not_sparsity_fast_path = (
                f"input not batched; expected src.dim() of 3 but got {src.dim()}"
            )
        elif self.training:
            why_not_sparsity_fast_path = "training is enabled"
        elif not self.self_attn.batch_first:
            why_not_sparsity_fast_path = "self_attn.batch_first was not True"
        elif not self.self_attn._qkv_same_embed_dim:
            why_not_sparsity_fast_path = "self_attn._qkv_same_embed_dim was not True"
        elif not self.activation_relu_or_gelu:
            why_not_sparsity_fast_path = "activation_relu_or_gelu was not True"
        elif not (self.norm1.eps == self.norm2.eps):
            why_not_sparsity_fast_path = "norm1.eps is not equal to norm2.eps"
        elif src.is_nested and (
            src_key_padding_mask is not None or src_mask is not None
        ):
            why_not_sparsity_fast_path = "neither src_key_padding_mask nor src_mask are not supported with NestedTensor input"
        elif self.self_attn.num_heads % 2 == 1:
            why_not_sparsity_fast_path = "num_head is odd"
        elif torch.is_autocast_enabled():
            why_not_sparsity_fast_path = "autocast is enabled"
        # if not why_not_sparsity_fast_path:
        #     assert 0
        #     tensor_args = (
        #         src,
        #         self.self_attn.in_proj_weight,
        #         self.self_attn.in_proj_bias,
        #         self.self_attn.out_proj.weight,
        #         self.self_attn.out_proj.bias,
        #         self.norm1.weight,
        #         self.norm1.bias,
        #         self.norm2.weight,
        #         self.norm2.bias,
        #         self.linear1.weight,
        #         self.linear1.bias,
        #         self.linear2.weight,
        #         self.linear2.bias,
        #     )

        #     # We have to use list comprehensions below because TorchScript does not support
        #     # generator expressions.
        #     if torch.overrides.has_torch_function(tensor_args):
        #         why_not_sparsity_fast_path = "some Tensor argument has_torch_function"
        #     elif not all((x.is_cuda or 'cpu' in str(x.device)) for x in tensor_args):
        #         why_not_sparsity_fast_path = "some Tensor argument is neither CUDA nor CPU"
        #     elif torch.is_grad_enabled() and any(x.requires_grad for x in tensor_args):
        #         why_not_sparsity_fast_path = ("grad is enabled and at least one of query or the "
        #                                       "input/output projection weights or biases requires_grad")

        #     if not why_not_sparsity_fast_path:
        #         merged_mask, mask_type = self.self_attn.merge_masks(src_mask, src_key_padding_mask, src)
        #         return torch._transformer_encoder_layer_fwd(
        #             src,
        #             self.self_attn.embed_dim,
        #             self.self_attn.num_heads,
        #             self.self_attn.in_proj_weight,
        #             self.self_attn.in_proj_bias,
        #             self.self_attn.out_proj.weight,
        #             self.self_attn.out_proj.bias,
        #             self.activation_relu_or_gelu == 2,
        #             self.norm_first,
        #             self.norm1.eps,
        #             self.norm1.weight,
        #             self.norm1.bias,
        #             self.norm2.weight,
        #             self.norm2.bias,
        #             self.linear1.weight,
        #             self.linear1.bias,
        #             self.linear2.weight,
        #             self.linear2.bias,
        #             merged_mask,
        #             mask_type,
        #         )

        x = src
        if self.norm_first:
            x = x + self._sa_block(
                self.norm1(x),
                src_mask,
                src_key_padding_mask,
                is_causal=is_causal,
                rope_embeding=rope_embeding,
            )
            x = x + self._ff_block(self.norm2(x))
        else:
            x = self.norm1(
                x
                + self._sa_block(
                    x,
                    src_mask,
                    src_key_padding_mask,
                    is_causal=is_causal,
                    rope_embeding=rope_embeding,
                )
            )
            x = self.norm2(x + self._ff_block(x))

        return x

    # self-attention block
    def _sa_block(
        self,
        x: Tensor,
        attn_mask: Optional[Tensor],
        key_padding_mask: Optional[Tensor],
        is_causal: bool = False,
        rope_embeding=None,
    ) -> Tensor:
        if not self.use_rope:
            q, k = x, x
        else:
            q, k = apply_rotary_emb(x, x, rope_embeding)
        # print(q.shape, k.shape, x.shape)
        x = self.self_attn(
            q,
            k,
            x,
            attn_mask=attn_mask,
            key_padding_mask=key_padding_mask,
            need_weights=False,
            is_causal=is_causal,
        )[0]
        return self.dropout1(x)

    # feed forward block
    def _ff_block(self, x: Tensor) -> Tensor:
        x = self.linear2(self.dropout(self.activation(self.linear1(x))))
        return self.dropout2(x)


def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0):
    """
    Precompute the frequency tensor for complex exponentials (cis) with given dimensions.

    This function calculates a frequency tensor with complex exponentials using the given dimension 'dim'
    and the end index 'end'. The 'theta' parameter scales the frequencies.
    The returned tensor contains complex values in complex64 data type.

    Args:
        dim (int): Dimension of the frequency tensor.
        end (int): End index for precomputing frequencies.
        theta (float, optional): Scaling factor for frequency computation. Defaults to 10000.0.

    Returns:
        torch.Tensor: Precomputed frequency tensor with complex exponentials.

    """
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(end, device=freqs.device)  # type: ignore
    freqs = torch.outer(t, freqs).float()  # type: ignore
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)  # complex64
    return freqs_cis


def reshape_for_broadcast(freqs_cis: torch.Tensor, x: torch.Tensor):
    """
    Reshape frequency tensor for broadcasting it with another tensor.

    This function reshapes the frequency tensor to have the same shape as the target tensor 'x'
    for the purpose of broadcasting the frequency tensor during element-wise operations.

    Args:
        freqs_cis (torch.Tensor): Frequency tensor to be reshaped.
        x (torch.Tensor): Target tensor for broadcasting compatibility.

    Returns:
        torch.Tensor: Reshaped frequency tensor.

    Raises:
        AssertionError: If the frequency tensor doesn't match the expected shape.
        AssertionError: If the target tensor 'x' doesn't have the expected number of dimensions.
    """
    ndim = x.ndim
    assert 0 <= 1 < ndim
    assert freqs_cis.shape == (x.shape[1], x.shape[-1])
    shape = [d if i == 1 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)]
    return freqs_cis.view(*shape)


def apply_rotary_emb(
    xq: torch.Tensor,
    xk: torch.Tensor,
    freqs_cis: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Apply rotary embeddings to input tensors using the given frequency tensor.

    This function applies rotary embeddings to the given query 'xq' and key 'xk' tensors using the provided
    frequency tensor 'freqs_cis'. The input tensors are reshaped as complex numbers, and the frequency tensor
    is reshaped for broadcasting compatibility. The resulting tensors contain rotary embeddings and are
    returned as real tensors.

    Args:
        xq (torch.Tensor): Query tensor to apply rotary embeddings.
        xk (torch.Tensor): Key tensor to apply rotary embeddings.
        freqs_cis (torch.Tensor): Precomputed frequency tensor for complex exponentials.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: Tuple of modified query tensor and key tensor with rotary embeddings.

    """
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
    # freqs_cis = reshape_for_broadcast(freqs_cis, xq_)
    xq_out = torch.view_as_real(xq_ * freqs_cis).reshape(
        *xq.shape[:-1], -1
    )  # .flatten(3)
    xk_out = torch.view_as_real(xk_ * freqs_cis).reshape(
        *xq.shape[:-1], -1
    )  # .flatten(3)
    # print(xq_.shape, xq_out.shape, torch.view_as_real(xq_ * freqs_cis).shape)
    return xq_out.type_as(xq), xk_out.type_as(xk)


def _get_clones(module, N):
    # FIXME: copy.deepcopy() is not defined on nn.module
    return ModuleList([copy.deepcopy(module) for i in range(N)])


class TransformerEncoder(Module):
    r"""TransformerEncoder is a stack of N encoder layers. Users can build the
    BERT(https://arxiv.org/abs/1810.04805) model with corresponding parameters.

    Args:
        encoder_layer: an instance of the TransformerEncoderLayer() class (required).
        num_layers: the number of sub-encoder-layers in the encoder (required).
        norm: the layer normalization component (optional).
        enable_nested_tensor: if True, input will automatically convert to nested tensor
            (and convert back on output). This will improve the overall performance of
            TransformerEncoder when padding rate is high. Default: ``True`` (enabled).

    Examples::
        >>> encoder_layer = nn.TransformerEncoderLayer(d_model=512, nhead=8)
        >>> transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=6)
        >>> src = torch.rand(10, 32, 512)
        >>> out = transformer_encoder(src)
    """

    __constants__ = ["norm"]

    def __init__(
        self,
        encoder_layer,
        num_layers,
        norm=None,
        enable_nested_tensor=True,
        mask_check=True,
    ):
        super().__init__()
        torch._C._log_api_usage_once(f"torch.nn.modules.{self.__class__.__name__}")
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm
        self.enable_nested_tensor = enable_nested_tensor
        self.mask_check = mask_check

    def forward(
        self,
        src: Tensor,
        mask: Optional[Tensor] = None,
        src_key_padding_mask: Optional[Tensor] = None,
        is_causal: Optional[bool] = None,
        rope_embeding: Tensor = None,
    ) -> Tensor:
        r"""Pass the input through the encoder layers in turn.

        Args:
            src: the sequence to the encoder (required).
            mask: the mask for the src sequence (optional).
            is_causal: If specified, applies a causal mask as mask (optional)
                and ignores attn_mask for computing scaled dot product attention.
                Default: ``False``.
            src_key_padding_mask: the mask for the src keys per batch (optional).

        Shape:
            see the docs in Transformer class.
        """
        src_key_padding_mask = F._canonical_mask(
            mask=src_key_padding_mask,
            mask_name="src_key_padding_mask",
            other_type=F._none_or_dtype(mask),
            other_name="mask",
            target_type=src.dtype,
        )

        mask = F._canonical_mask(
            mask=mask,
            mask_name="mask",
            other_type=None,
            other_name="",
            target_type=src.dtype,
            check_other=False,
        )

        output = src
        convert_to_nested = False
        first_layer = self.layers[0]
        src_key_padding_mask_for_layers = src_key_padding_mask
        why_not_sparsity_fast_path = ""
        str_first_layer = "self.layers[0]"
        if not isinstance(first_layer, torch.nn.TransformerEncoderLayer):
            why_not_sparsity_fast_path = (
                f"{str_first_layer} was not TransformerEncoderLayer"
            )
        elif first_layer.norm_first:
            why_not_sparsity_fast_path = f"{str_first_layer}.norm_first was True"
        elif first_layer.training:
            why_not_sparsity_fast_path = f"{str_first_layer} was in training mode"
        elif not first_layer.self_attn.batch_first:
            why_not_sparsity_fast_path = (
                f" {str_first_layer}.self_attn.batch_first was not True"
            )
        elif not first_layer.self_attn._qkv_same_embed_dim:
            why_not_sparsity_fast_path = (
                f"{str_first_layer}.self_attn._qkv_same_embed_dim was not True"
            )
        elif not first_layer.activation_relu_or_gelu:
            why_not_sparsity_fast_path = (
                f" {str_first_layer}.activation_relu_or_gelu was not True"
            )
        elif not (first_layer.norm1.eps == first_layer.norm2.eps):
            why_not_sparsity_fast_path = f"{str_first_layer}.norm1.eps was not equal to {str_first_layer}.norm2.eps"
        elif not src.dim() == 3:
            why_not_sparsity_fast_path = (
                f"input not batched; expected src.dim() of 3 but got {src.dim()}"
            )
        elif not self.enable_nested_tensor:
            why_not_sparsity_fast_path = "enable_nested_tensor was not True"
        elif src_key_padding_mask is None:
            why_not_sparsity_fast_path = "src_key_padding_mask was None"
        elif (
            (not hasattr(self, "mask_check")) or self.mask_check
        ) and not torch._nested_tensor_from_mask_left_aligned(
            src, src_key_padding_mask.logical_not()
        ):
            why_not_sparsity_fast_path = "mask_check enabled, and src and src_key_padding_mask was not left aligned"
        elif output.is_nested:
            why_not_sparsity_fast_path = "NestedTensor input is not supported"
        elif mask is not None:
            why_not_sparsity_fast_path = (
                "src_key_padding_mask and mask were both supplied"
            )
        elif first_layer.self_attn.num_heads % 2 == 1:
            why_not_sparsity_fast_path = "num_head is odd"
        elif torch.is_autocast_enabled():
            why_not_sparsity_fast_path = "autocast is enabled"

        if not why_not_sparsity_fast_path:
            tensor_args = (
                src,
                first_layer.self_attn.in_proj_weight,
                first_layer.self_attn.in_proj_bias,
                first_layer.self_attn.out_proj.weight,
                first_layer.self_attn.out_proj.bias,
                first_layer.norm1.weight,
                first_layer.norm1.bias,
                first_layer.norm2.weight,
                first_layer.norm2.bias,
                first_layer.linear1.weight,
                first_layer.linear1.bias,
                first_layer.linear2.weight,
                first_layer.linear2.bias,
            )

            if torch.overrides.has_torch_function(tensor_args):
                why_not_sparsity_fast_path = "some Tensor argument has_torch_function"
            elif not (src.is_cuda or "cpu" in str(src.device)):
                why_not_sparsity_fast_path = "src is neither CUDA nor CPU"
            elif torch.is_grad_enabled() and any(x.requires_grad for x in tensor_args):
                why_not_sparsity_fast_path = (
                    "grad is enabled and at least one of query or the "
                    "input/output projection weights or biases requires_grad"
                )

            if (not why_not_sparsity_fast_path) and (src_key_padding_mask is not None):
                convert_to_nested = True
                output = torch._nested_tensor_from_mask(
                    output, src_key_padding_mask.logical_not(), mask_check=False
                )
                src_key_padding_mask_for_layers = None

        # Prevent type refinement
        make_causal = is_causal is True

        if is_causal is None:
            if mask is not None:
                sz = mask.size(0)
                causal_comparison = torch.triu(
                    torch.ones(sz, sz, device=mask.device) * float("-inf"), diagonal=1
                ).to(mask.dtype)

                if torch.equal(mask, causal_comparison):
                    make_causal = True

        is_causal = make_causal

        for mod in self.layers:
            output = mod(
                output,
                src_mask=mask,
                is_causal=is_causal,
                src_key_padding_mask=src_key_padding_mask_for_layers,
                rope_embeding=rope_embeding,
            )

        if convert_to_nested:
            output = output.to_padded_tensor(0.0)

        if self.norm is not None:
            output = self.norm(output)

        return output


class SpectrumEncoder(torch.nn.Module):
    """A Transformer encoder for input mass spectra.

    Parameters
    ----------
    dim_model : int, optional
        The latent dimensionality to represent peaks in the mass spectrum.
    n_head : int, optional
        The number of attention heads in each layer. ``dim_model`` must be
        divisible by ``n_head``.
    dim_feedforward : int, optional
        The dimensionality of the fully connected layers in the Transformer
        layers of the model.
    n_layers : int, optional
        The number of Transformer layers.
    dropout : float, optional
        The dropout probability for all layers.
    peak_encoder : bool, optional
        Use positional encodings m/z values of each peak.
    dim_intensity: int or None, optional
        The number of features to use for encoding peak intensity.
        The remaining (``dim_model - dim_intensity``) are reserved for
        encoding the m/z value.
    """

    def __init__(
        self,
        dim_model=128,
        n_head=8,
        dim_feedforward=1024,
        n_layers=1,
        dropout=0,
        peak_encoder=True,
        dim_intensity=None,
        use_nce=False,
        use_ins=False,
        use_rope=False,
        norm_first=False,
    ):
        """Initialize a SpectrumEncoder"""
        super().__init__()
        self.use_nce = use_nce
        self.use_ins = use_ins
        self.use_rope = use_rope
        # self.latent_spectrum = torch.nn.Parameter(torch.randn(1, 1, dim_model))
        # self.spectrum_matrix = torch.nn.Parameter(torch.randn(dim_model,dim_model))

        # dim_intensity = 128
        self.zeroPeaks_intensity = torch.nn.Parameter(torch.randn(1, 1, 1))
        self.allPeaks_intensity = torch.nn.Parameter(torch.randn(1, 1, 1))

        if peak_encoder:
            self.peak_encoder = PeakEncoder(
                dim_model,
                dim_intensity=dim_intensity,
            )
        else:
            self.peak_encoder = torch.nn.Linear(2, dim_model)

        # The Transformer layers:
        layer = torch.nn.TransformerEncoderLayer(
            d_model=dim_model,
            nhead=n_head,
            dim_feedforward=dim_feedforward,
            batch_first=True,
            dropout=dropout,
        )

        self.transformer_encoder = torch.nn.TransformerEncoder(
            layer,
            num_layers=n_layers,
        )

        # Precursor Encoder
        # self.mass_encoder = MassEncoder(dim_model=256)
        if self.use_nce:
            self.nce_encoder = torch.nn.Embedding(101, dim_model)
            initialize_embedding_to_zero(self.nce_encoder)
        if self.use_ins:
            self.ins_encoder = torch.nn.Embedding(10, dim_model)
            initialize_embedding_to_zero(self.nce_encoder)

        self.charge_encoder = torch.nn.Embedding(50, dim_model)
        initialize_embedding_to_zero(self.charge_encoder)

    def forward(self, spectra, precursors, instrument, nce, rope_embeding):
        """The forward pass.

        Parameters
        ----------
        spectra : torch.Tensor of shape (n_spectra, n_peaks, 2)
            The spectra to embed. Axis 0 represents a mass spectrum, axis 1
            contains the peaks in the mass spectrum, and axis 2 is essentially
            a 2-tuple specifying the m/z-intensity pair for each peak. These
            should be zero-padded, such that all of the spectra in the batch
            are the same length.

        Returns
        -------
        latent : torch.Tensor of shape (n_spectra, n_peaks + 1, dim_model)
            The latent representations for the spectrum and each of its
            peaks.
        mem_mask : torch.Tensor
            The memory mask specifying which elements were padding in X.
        """

        # add percursors into encoder
        # masses = self.mass_encoder(precursors[:, None, [0]])
        # charges = self.charge_encoder(precursors[:, 1].int() - 1)
        # precursors = masses + charges[:, None, :]
        dtype = spectra.dtype

        zeroMass = torch.zeros([precursors.shape[0], 1, 1]).to(self.device).type(dtype)
        precursorMass = precursors[:, None, [0]]
        precursorCharge = precursors[:, [1]].long().squeeze(-1)
        zeroPeaksIntensities = self.zeroPeaks_intensity.expand(
            precursors.shape[0], -1, -1
        )
        allPeaksIntensities = self.allPeaks_intensity.expand(
            precursors.shape[0], -1, -1
        )
        zeros = torch.cat([zeroMass, zeroPeaksIntensities], dim=2)
        alls = torch.cat([precursorMass, allPeaksIntensities], dim=2)
        starts = torch.cat([zeros, alls], dim=1)
        spectra = torch.cat([starts, spectra], dim=1)
        zeros = ~(spectra.sum(dim=2).bool())
        mask = zeros
        # mask = [
        #     # add percursors into encoder
        #     # torch.tensor([[False]] * spectra.shape[0]).type_as(zeros),
        #     # torch.tensor([[False]] * spectra.shape[0]).type_as(zeros),
        #     zeros,
        # ]
        # mask = torch.cat(mask, dim=1)
        peaks = self.peak_encoder(spectra)
        # print(peaks[:, 0])
        if self.use_nce:
            peaks[:, 0] += self.nce_encoder(nce)
        if self.use_ins:
            peaks[:, 0] += self.ins_encoder(instrument)
        # print(peaks.shape, self.charge_encoder(precursorCharge).shape, precursorMass.shape)
        # print(peaks[:, 0])
        # assert precursorCharge < 50
        peaks[:, 0] += self.charge_encoder(precursorCharge)
        # print(peaks[:, 0])
        # assert 0
        # precursors = torch.matmul(precursors,self.spectrum_matrix)
        # peaks = torch.concat([precursors,peaks],dim = 1)

        # Add the spectrum representation to each input:

        # latent_spectra = self.latent_spectrum.expand(peaks.shape[0], -1, -1)
        # peaks = torch.cat([latent_spectra, peaks], dim=1)
        return self.transformer_encoder(peaks, src_key_padding_mask=mask), mask

    @property
    def device(self):
        """The current device for the model"""
        return next(self.parameters()).device


class _PeptideTransformer(torch.nn.Module):
    """A transformer base class for peptide sequences.

    Parameters
    ----------
    dim_model : int
        The latent dimensionality to represent the amino acids in a peptide
        sequence.
    pos_encoder : bool
        Use positional encodings for the amino acid sequence.
    residues: Dict or str {"massivekb", "canonical"}, optional
        The amino acid dictionary and their masses. By default this is only
        the 20 canonical amino acids, with cysteine carbamidomethylated. If
        "massivekb", this dictionary will include the modifications found in
        MassIVE-KB. Additionally, a dictionary can be used to specify a custom
        collection of amino acids and masses.
    max_charge : int
        The maximum charge to embed.
    """

    def __init__(
        self,
        dim_model,
        pos_encoder,
        residues,
        max_charge,
        non_auto_decoder=False,
    ):
        super().__init__()
        self.reverse = False
        self._peptide_mass = PeptideMass(residues=residues)
        self._amino_acids = list(self._peptide_mass.masses.keys()) + ["$"]
        self._idx2aa = {i + 1: aa for i, aa in enumerate(self._amino_acids)}
        self._aa2idx = {aa: i for i, aa in self._idx2aa.items()}

        if pos_encoder:
            self.pos_encoder = PositionalEncoder(dim_model)
        else:
            self.pos_encoder = torch.nn.Identity()
        self.charge_encoder = torch.nn.Embedding(max_charge, dim_model)
        self.aa_encoder = torch.nn.Embedding(
            len(self._amino_acids) + 1,
            dim_model,
            padding_idx=0,
        )

    def tokenize(self, sequence, partial=False):
        """Transform a peptide sequence into tokens

        Parameters
        ----------
        sequence : str
            A peptide sequence.

        Returns
        -------
        torch.Tensor
            The token for each amino acid in the peptide sequence.
        """
        if not isinstance(sequence, str):
            return sequence  # Assume it is already tokenized.

        # sequence = sequence.replace("I", "L")
        sequence = re.split(r"(?<=.)(?=[A-Z])", sequence)

        if self.reverse:
            assert 0
            sequence = list(reversed(sequence))

        if not partial:
            sequence += ["$"]

        tokens = [self._aa2idx[aa] for aa in sequence]
        tokens = torch.tensor(tokens, device=self.device)
        return tokens

    def deMass(self, sequence, modifi_mass=None):

        if not isinstance(sequence, str):
            sequence = [self._idx2aa.get(i.item(), "") for i in sequence]
            masses = [self._peptide_mass.masses[aa] for aa in sequence]
            # if(len(sequence) > 1):
            #     print(sequence,masses)
            masses = list(itertools.accumulate(masses))
            masses = torch.tensor(masses, device=self.device)

            return masses

        sequence = sequence.replace("I", "L")
        sequence = re.split(r"(?<=.)(?=[A-Z])", sequence)

        if self.reverse:
            assert 0
            sequence = list(reversed(sequence))
        if modifi_mass is None:
            masses = [self._peptide_mass.masses[aa] for aa in sequence]
        else:
            masses = [
                self._peptide_mass.masses[sequence[aa]] + modifi_mass[aa]
                for aa in range(len(sequence))
            ]

        masses = list(itertools.accumulate(masses))
        masses.append(0.0)

        masses = torch.tensor(masses, device=self.device)

        return masses

    def get_suffix_mass(self, sequence, premass, modifi_mass=None):

        if not isinstance(sequence, str):

            sequence = [self._idx2aa.get(i.item(), "") for i in sequence]
            masses = [self._peptide_mass.masses[aa] for aa in sequence]
            masses = list(itertools.accumulate(masses))
            masses = torch.tensor(masses, device=self.device)
            masses = premass - masses
            # print(sequence,masses)
            return masses

        sequence = sequence.replace("I", "L")
        sequence = re.split(r"(?<=.)(?=[A-Z])", sequence)

        if self.reverse:
            assert 0
            sequence = list(reversed(sequence))

        if modifi_mass is None:
            masses = [self._peptide_mass.masses[aa] for aa in sequence]
        else:
            masses = [
                self._peptide_mass.masses[sequence[aa]] + modifi_mass[aa]
                for aa in range(len(sequence))
            ]

        masses = list(itertools.accumulate(masses))
        masses.append(premass)

        masses = torch.tensor(masses, device=self.device)
        masses = premass - masses

        return masses

    def get_mass(self, sequence):

        if not isinstance(sequence, str):
            masses = [self._peptide_mass.masses[aa] for aa in sequence]
            masstemp = torch.tensor(masses, device=self.device)
            return masstemp

        sequence = sequence.replace("I", "L")
        sequence = re.split(r"(?<=.)(?=[A-Z])", sequence)

        if self.reverse:
            assert 0
            sequence = list(reversed(sequence))

        masses = [self._peptide_mass.masses[aa] for aa in sequence]
        masstemp = torch.tensor(masses, device=self.device)
        masstemp = torch.cat([masstemp, torch.tensor([0.0]).to(masstemp.device)])

        return masstemp

    def getAminoAcid(self):
        # print(self._idx2aa)
        AA_masslist = [self._peptide_mass.masses[self._idx2aa[i]] for i in range(1, 28)]
        AA_masslist = [0] + AA_masslist
        AA_masslist = torch.tensor(AA_masslist, device=self.device)
        return AA_masslist

    def detokenize(self, tokens):
        """Transform tokens back into a peptide sequence.

        Parameters
        ----------
        tokens : torch.Tensor of shape (n_amino_acids,)
            The token for each amino acid in the peptide sequence.

        Returns
        -------
        list of str
            The amino acids in the peptide sequence.
        """
        sequence = [self._idx2aa.get(i.item(), "") for i in tokens]
        if "$" in sequence:
            idx = sequence.index("$")
            sequence = sequence[: idx + 1]
        if self.reverse:
            assert 0
            sequence = list(reversed(sequence))
        return sequence

    @property
    def vocab_size(self):
        """Return the number of amino acids"""
        return len(self._aa2idx)

    @property
    def device(self):
        """The current device for the model"""
        return next(self.parameters()).device


class PeptideEncoder(nn.Module):
    def __init__(
        self,
        max_len=100,
        residues: Union[Dict[str, float], str] = "canonical",
        ptransformer_width: int = 512,
        ptransformer_heads: int = 8,
        ptransformer_layers: int = 9,
        norm_first: bool = False,
        dropout=0.4,
    ):

        super().__init__()

        self.reverse = False

        self.max_len = max_len
        self._peptide_mass = PeptideMass(residues=residues)
        self._amino_acids = list(self._peptide_mass.masses.keys())
        self._idx2aa = {i + 1: aa for i, aa in enumerate(self._amino_acids)}
        self._aa2idx = {aa: i for i, aa in self._idx2aa.items()}

        self.aminoEmbedDim = ptransformer_width // 2
        self.aa_encoder = torch.nn.Embedding(
            len(self._amino_acids) + 1,
            self.aminoEmbedDim,
            padding_idx=0,
        )

        # Mass/charge Encoder for precursors (Dim:256 = dim_model // 2)
        # self.mass_encoder = MassEncoder(ptransformer_width // 2)
        self.charge_encoder = torch.nn.Embedding(50, ptransformer_width // 2)
        self.nce_encoder = torch.nn.Embedding(101, ptransformer_width // 2)
        self.ins_encoder = torch.nn.Embedding(10, ptransformer_width // 2)
        # MassEncoder for prefix and suffix mass
        self.prefixMassEncoder = MassEncoder(ptransformer_width // 4)
        self.suffixMassEncoder = MassEncoder(ptransformer_width // 4)

        self.massEncoder = MassEncoder(ptransformer_width)

        self.pos_encoder = PositionalEncoder(ptransformer_width)

        self.modification_cls_encoder = torch.nn.Embedding(2609, ptransformer_width)
        self.modification_type_encoder = torch.nn.Embedding(6, ptransformer_width)
        # self.modification_cls_encoder = Linear(2, ptransformer_width, init="glorot")

        layer = torch.nn.TransformerEncoderLayer(
            d_model=ptransformer_width,
            nhead=ptransformer_heads,
            dim_feedforward=1024,
            batch_first=True,
            dropout=dropout,
            norm_first=norm_first,
        )

        self.peptideTransformer = TransformerEncoder(
            layer,
            num_layers=ptransformer_layers,
        )

        # self.peptideTransformer = Transformer(
        #     width=ptransformer_width,
        #     layers=ptransformer_layers,
        #     heads=ptransformer_heads,
        #     dropout=dropout
        # )

        # Massses Embedding
        """self.linearEmbedding = torch.nn.Linear(1,2)"""

    def forward(
        self, sequences, precursors, instrument, nce, modification, batch_index
    ):

        # Transformer Encoder For Peptide Sequence.
        # # Mass Encoder For Peptide Sequence.
        """if sequences is not None:
            sequences = utils.listify(sequences)
            tokens = [self.tokenize(s) for s in sequences]
            Masses = [self.deMass(s) for s in sequences]
            Masses = torch.nn.utils.rnn.pad_sequence(Masses, batch_first = True)
            tokens = torch.nn.utils.rnn.pad_sequence(tokens, batch_first = True)
        else:
            tokens = torch.tensor([[]]).to(self.device)
            Masses = torch.tensor([[]]).to(self.device)

        tgt = self.aa_encoder(tokens)

        Masses = Masses.unsqueeze(2)
        Masses = self.linearEmbedding(Masses)
        tgt = torch.concat([tgt,Masses],dim=2)

        tgt = self.pos_encoder(tgt)"""
        if sequences is not None:
            sequences = utils.listify(sequences)
            Masses = [
                self.deMass(sequences[i], modification[i])
                for i in range(len(sequences))
            ]
            Masses = torch.nn.utils.rnn.pad_sequence(Masses, batch_first=True)
            # print(len(sequences), len(precursors))
            # precursors = precursors.repeat_interleave(2, dim=0)
            precursors_list = []
            instrument_list = []
            nce_list = []
            for i in range(int(batch_index[-1] + 1)):
                num = (batch_index == i).sum()
                precursors_list.extend([precursors[i].unsqueeze(0)] * num)
                instrument_list.extend([instrument[i].unsqueeze(0)] * num)
                nce_list.extend([nce[i].unsqueeze(0)] * num)
            precursors = torch.cat(precursors_list, dim=0)
            instrument = torch.cat(instrument_list, dim=0)
            nce = torch.cat(nce_list, dim=0)
            assert precursors.shape[0] == batch_index.shape[0], (
                precursors.shape[0],
                batch_index.shape[0],
            )
            suffixMasses = [
                self.get_suffix_mass(sequences[i], precursors[i][0], modification[i])
                for i in range(len(sequences))
            ]
            suffixMasses = torch.nn.utils.rnn.pad_sequence(
                suffixMasses, batch_first=True
            )
            tokens = [self.tokenize(s) for s in sequences]
            tokens = torch.nn.utils.rnn.pad_sequence(tokens, batch_first=True)
        else:
            Masses = torch.tensor([[]]).to(self.device)
            suffixMasses = torch.tensor([[]]).to(self.device)
            tokens = torch.tensor([[]]).to(self.device)

        masses = self.mass_encoder(precursors[:, None, [0]])
        assert torch.all((precursors[:, 1].int() - 1) < 50)
        charges = self.charge_encoder(precursors[:, 1].int() - 1)
        ins = self.ins_encoder(instrument.int())
        nce = self.nce_encoder(nce.int())
        precursors = masses
        infos = charges[:, None, :] + ins[:, None, :] + nce[:, None, :]

        # preAndSufPrecursors = torch.tensor([[0]]).to(self.device)
        # preAndSufPrecursors = self.prefixMassEncoder(preAndSufPrecursors)
        # preAndSufPrecursors = preAndSufPrecursors.repeat(precursors.shape[0],1)
        # preAndSufPrecursors = preAndSufPrecursors.unsqueeze(1)
        precursors = torch.cat([precursors, infos], dim=2)

        Masses = Masses.unsqueeze(2)
        Masses = self.prefixMassEncoder(Masses)

        suffixMasses = suffixMasses.unsqueeze(2)
        suffixMasses = self.suffixMassEncoder(suffixMasses)

        Masses = torch.concat([Masses, suffixMasses], dim=2)

        assert torch.all(tokens < len(self._amino_acids))
        tgt = self.aa_encoder(tokens)
        tgt_key_padding_mask = tgt.sum(axis=2) == 0
        tgt = torch.concat([tgt, Masses], dim=2)

        tgt = torch.cat([precursors, tgt], dim=1)
        tgt_key_padding_mask = torch.cat(
            [
                torch.zeros(tgt.shape[0], 1).to(tgt_key_padding_mask.device),
                tgt_key_padding_mask,
            ],
            dim=1,
        ).bool()
        for idx in range(len(modification)):
            mods = modification[idx]
            for mod in mods:
                # try:
                tgt[idx][mod[0]] += self.modification_cls_encoder(
                    torch.Tensor(mod[1][2]).long().to(tgt.device)
                ) + self.modification_type_encoder(
                    torch.Tensor(mod[1][0]).long().to(tgt.device)
                )
                # except:
                #     print(tgt.shape)
                #     print(mod, sequences[idx], len(sequences[idx]))
                #     assert 0
        # tgt_key_padding_mask = tgt.sum(axis=2) == 0
        # Add positional code on peptide sequence.
        # tgt = self.pos_encoder(tgt) #(n_spectra, len(Peptide), dim_model)
        # Peptide input to Transformer.

        # tgt = tgt.permute(1,0,2)
        tgt, tgt_half = self.peptideTransformer(
            tgt, src_key_padding_mask=tgt_key_padding_mask.bool()
        )
        # tgt = tgt.permute(1,0,2) #Shape(B_s, Peptide_len, Feature_size)

        return tgt, tgt_half, tgt_key_padding_mask

    def tokenize(self, sequence, partial=False):
        """Transform a peptide sequence into tokens

        Parameters
        ----------
        sequence : str
            A peptide sequence.

        Returns
        -------
        torch.Tensor
            The token for each amino acid in the peptide sequence.
        """
        if not isinstance(sequence, str):
            return sequence  # Assume it is already tokenized.

        # sequence = sequence.replace("I", "L")
        sequence = re.split(r"(?<=.)(?=[A-Z])", sequence)

        if self.reverse:
            sequence = list(reversed(sequence))

        # if not partial:
        #     sequence += ["$"]

        tokens = [self._aa2idx[aa] for aa in sequence]
        tokens = torch.tensor(tokens, device=self.device)
        return tokens

    def deMass(self, sequence, modification=None):

        if not isinstance(sequence, str):

            sequence = [self._idx2aa.get(i.item(), "") for i in sequence]
            masses = [self._peptide_mass.masses[aa] for aa in sequence]
            if modification is not None:
                for idx in range(len(modification)):
                    for mod in modification[idx]:
                        masses[idx][mod[0] - 1] += mod[1][1]
            masses = list(itertools.accumulate(masses))
            masses = torch.tensor(masses, device=self.device)

            return masses

        sequence = sequence.replace("I", "L")
        sequence = re.split(r"(?<=.)(?=[A-Z])", sequence)

        if self.reverse:
            sequence = list(reversed(sequence))
        masses = [self._peptide_mass.masses[aa] for aa in sequence]
        if modification is not None:
            for mod in modification:
                masses[mod[0] - 1] += mod[1][1]
        masses = list(itertools.accumulate(masses))

        masses = torch.tensor(masses, device=self.device)

        return masses

    def get_suffix_mass(self, sequence, premass, modification):

        if not isinstance(sequence, str):

            sequence = [self._idx2aa.get(i.item(), "") for i in sequence]
            masses = [self._peptide_mass.masses[aa] for aa in sequence]
            if modification is not None:
                for idx in range(len(modification)):
                    for mod in modification[idx]:
                        masses[idx][mod[0] - 1] += mod[1][1]
            masses = list(itertools.accumulate(masses))
            masses = torch.tensor(masses, device=self.device)
            masses = premass - masses
            return masses

        sequence = sequence.replace("I", "L")
        sequence = re.split(r"(?<=.)(?=[A-Z])", sequence)

        if self.reverse:
            sequence = list(reversed(sequence))

        masses = [self._peptide_mass.masses[aa] for aa in sequence]
        if modification is not None:
            for mod in modification:
                masses[mod[0] - 1] += mod[1][1]
        masses = list(itertools.accumulate(masses))

        masses = torch.tensor(masses, device=self.device)
        masses = premass - masses

        return masses

    def get_mass(self, sequence):

        if not isinstance(sequence, str):
            masses = [self._peptide_mass.masses[aa] for aa in sequence]
            masstemp = torch.tensor(masses, device=self.device)
            return masstemp

        sequence = sequence.replace("I", "L")
        sequence = re.split(r"(?<=.)(?=[A-Z])", sequence)

        if self.reverse:
            sequence = list(reversed(sequence))

        masses = [self._peptide_mass.masses[aa] for aa in sequence]
        masstemp = torch.tensor(masses, device=self.device)
        # masstemp = torch.cat([masstemp,torch.tensor([0.0]).to(masstemp.device)])

        return masstemp

    def getAminoAcid(self):
        AA_masslist = [self._peptide_mass.masses[self._idx2aa[i]] for i in range(1, 28)]
        AA_masslist = [0] + AA_masslist
        AA_masslist = torch.tensor(AA_masslist, device=self.device)
        return AA_masslist

    @property
    def device(self):
        """The current device for the model"""
        return next(self.parameters()).device


class NonLinear(nn.Module):
    def __init__(self, input, output_size, hidden=None):
        super(NonLinear, self).__init__()

        if hidden is None:
            hidden = input
        self.layer1 = Linear(input, hidden)
        self.layer2 = Linear(hidden, output_size)

    def forward(self, x):
        x = self.layer1(x)
        x = F.gelu(x)
        x = self.layer2(x)
        return x

    def zero_init(self):
        nn.init.zeros_(self.layer2.weight)
        nn.init.zeros_(self.layer2.bias)


class PeptideDecoder(_PeptideTransformer):
    """A transformer decoder for peptide sequences.

    Parameters
    ----------
    dim_model : int, optional
        The latent dimensionality to represent peaks in the mass spectrum.
    n_head : int, optional
        The number of attention heads in each layer. ``dim_model`` must be
        divisible by ``n_head``.
    dim_feedforward : int, optional
        The dimensionality of the fully connected layers in the Transformer
        layers of the model.
    n_layers : int, optional
        The number of Transformer layers.
    dropout : float, optional
        The dropout probability for all layers.
    pos_encoder : bool, optional
        Use positional encodings for the amino acid sequence.
    reverse : bool, optional
        Sequence peptides from c-terminus to n-terminus.
    residues: Dict or str {"massivekb", "canonical"}, optional
        The amino acid dictionary and their masses. By default this is only
        the 20 canonical amino acids, with cysteine carbamidomethylated. If
        "massivekb", this dictionary will include the modifications found in
        MassIVE-KB. Additionally, a dictionary can be used to specify a custom
        collection of amino acids and masses.
    """

    def __init__(
        self,
        dim_model=128,
        n_head=8,
        dim_feedforward=1024,
        n_layers=1,
        dropout=0,
        pos_encoder=True,
        reverse=False,
        residues="canonical",
        max_charge=10,
        norm_first=False,
        modification_pred=False,
    ):
        """Initialize a PeptideDecoder"""
        super().__init__(
            dim_model=dim_model,
            pos_encoder=pos_encoder,
            residues=residues,
            max_charge=max_charge,
        )
        self.reverse = reverse

        self.aaDim = dim_model - 256
        # Additional model components

        # Mass/charge Encoder for precursors (Dim:256 = dim_model // 2)
        self.mass_encoder = MassEncoder(256)
        self.max_charge = max_charge
        assert max_charge == 10
        self.charge_encoder = torch.nn.Embedding(max_charge, 256)

        # MassEncoder for prefix and suffix mass
        self.prefixMassEncoder = MassEncoder(128)
        self.suffixMassEncoder = MassEncoder(128)

        self.aa_encoder = torch.nn.Embedding(
            len(self._amino_acids) + 1,
            self.aaDim,
            padding_idx=0,
        )

        layer = torch.nn.TransformerDecoderLayer(
            d_model=dim_model,
            nhead=n_head,
            dim_feedforward=dim_feedforward,
            batch_first=True,
            dropout=dropout,
        )

        self.transformer_decoder = torch.nn.TransformerDecoder(
            layer,
            num_layers=n_layers,
        )

        # self.startvector = torch.nn.Parameter(torch.randn(1, 1, dim_model))

        self.logit_scale = torch.nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        self.final_aa_Encoder = torch.nn.Embedding(
            len(self._amino_acids) + 1,
            self.aaDim,
            padding_idx=0,
        )

        finalLinears = []
        xin = dim_model
        for xout in [512, 1024, 1024]:
            finalLinears.append(torch.nn.Linear(xin, xout))
            finalLinears.append(torch.nn.PReLU())
            xin = xout
        finalLinears.append(torch.nn.Linear(xin, 512))

        self.finalLinears = torch.nn.Sequential(*finalLinears)
        # self.final_linear = torch.nn.Linear(dim_model,dim_model)

        self.final_mass_encoder = MassEncoder(256)
        # self.finalMassLinearLayer = torch.nn.Sequential(torch.nn.Linear(1,256),torch.nn.PReLU(),torch.nn.Linear(256,256))
        self.finalCharMass = torch.nn.Parameter(torch.randn(1))
        if modification_pred:
            self.modification_cls_decoder = NonLinear(dim_model, 2610)
            self.modification_mass_decoder = NonLinear(dim_model, 1)
        self.modification_pred = modification_pred

        # self.final = torch.nn.Linear(dim_model, len(self._amino_acids) + 1)

    def forward(self, sequences, precursors, memory, memory_key_padding_mask):
        """Predict the next amino acid for a collection of sequences.

        Parameters
        ----------
        sequences : list of str or list of torch.Tensor
            The partial peptide sequences for which to predict the next
            amino acid. Optionally, these may be the token indices instead
            of a string.
        precursors : torch.Tensor of size (batch_size, 2)
            The measured precursor mass (axis 0) and charge (axis 1) of each
            tandem mass spectrum
        memory : torch.Tensor of shape (batch_size, n_peaks, dim_model)
            The representations from a ``TransformerEncoder``, such as a
           ``SpectrumEncoder``.
        memory_key_padding_mask : torch.Tensor of shape (batch_size, n_peaks)
            The mask that indicates which elements of ``memory`` are padding.

        Returns
        -------
        scores : torch.Tensor of size (batch_size, len_sequence, n_amino_acids)
            The raw output for the final linear layer. These can be Softmax
            transformed to yield the probability of each amino acid for the
            prediction.
        tokens : torch.Tensor of size (batch_size, len_sequence)
            The input padded tokens.

        """
        # Prepare sequences
        """if sequences is not None:
            sequences = utils.listify(sequences)
            tokens = [self.tokenize(s) for s in sequences]
            tokens = torch.nn.utils.rnn.pad_sequence(tokens, batch_first=True)
        else:
            tokens = torch.tensor([[]]).to(self.device)"""

        if sequences is not None:
            # print(sequences[0],precursors[0])
            sequences = utils.listify(sequences)
            Masses = [self.deMass(s) for s in sequences]
            Masses = torch.nn.utils.rnn.pad_sequence(Masses, batch_first=True)
            suffixMasses = [
                self.get_suffix_mass(sequences[i], precursors[i][0])
                for i in range(len(sequences))
            ]
            suffixMasses = torch.nn.utils.rnn.pad_sequence(
                suffixMasses, batch_first=True
            )
            tokens = [self.tokenize(s) for s in sequences]
            tokens = torch.nn.utils.rnn.pad_sequence(tokens, batch_first=True)
        else:
            Masses = torch.tensor([[]]).to(self.device)
            suffixMasses = torch.tensor([[]]).to(self.device)
            tokens = torch.tensor([[]]).to(self.device)
        # print(tokens.shape) # torch.Size([64, 44])
        # Prepare mass and charge
        masses = self.mass_encoder(precursors[:, None, [0]])
        assert torch.all(precursors[:, 1].int() - 1 < self.max_charge)
        charges = self.charge_encoder(precursors[:, 1].int() - 1)
        precursors = masses + charges[:, None, :]

        preAndSufPrecursors = torch.tensor([[0]]).to(self.device)
        preAndSufPrecursors = self.prefixMassEncoder(preAndSufPrecursors)
        preAndSufPrecursors = preAndSufPrecursors.repeat(precursors.shape[0], 1)
        preAndSufPrecursors = preAndSufPrecursors.unsqueeze(1)
        precursors = torch.cat(
            [precursors, preAndSufPrecursors, preAndSufPrecursors], dim=2
        )

        # masses = self.mass_encoder(precursors[:, None, [0]])
        # charges = self.charge_encoder(precursors[:, 1].int() - 1)
        # precursors = masses + charges[:, None, :]

        Masses = Masses.unsqueeze(2)
        Masses = self.prefixMassEncoder(Masses)

        suffixMasses = suffixMasses.unsqueeze(2)
        suffixMasses = self.suffixMassEncoder(suffixMasses)

        Masses = torch.concat([Masses, suffixMasses], dim=2)

        assert torch.all(tokens.to(torch.long) < len(self._amino_acids) + 1)
        tgt = self.aa_encoder(tokens.to(torch.long))
        tgt_key_padding_mask = tgt.sum(axis=2) == 0

        tgtTemp = torch.concat([tgt, Masses], dim=2)

        # startVector = self.startvector.expand(tgtTemp.shape[0], -1, -1)
        # tgtTemp = self.aa_encoder(tokens)

        # Feed through model:
        if sequences is None:
            tgt = precursors
        else:
            tgt = torch.cat([precursors, tgtTemp], dim=1)
        # assert 0, tgt.shape

        tgt_key_padding_mask = tgt.sum(axis=2) == 0
        tgt = self.pos_encoder(tgt)
        tgt_mask = generate_tgt_mask(tgt.shape[1]).type_as(precursors)
        # assert 0, (tgt_mask, tgt_mask.shape)
        # print(tgt.shape, memory.shape, tgt_mask.shape) # torch.Size([64, 45, 512]) torch.Size([64, 163, 512]) torch.Size([45, 45])
        preds = self.transformer_decoder(
            tgt=tgt,
            memory=memory,
            tgt_mask=tgt_mask,
            tgt_key_padding_mask=tgt_key_padding_mask,
            memory_key_padding_mask=memory_key_padding_mask.to(self.device),
        )
        aa_masses = self.getAminoAcid()
        aa_idx = torch.arange(0, 29).to(torch.long).to(self.device)
        aa_masses = torch.concat([aa_masses, self.finalCharMass], dim=0)
        aa_masses = aa_masses.unsqueeze(1)
        # print(">>>>", preds.shape, tgt.shape, tokens.shape)

        # Mass Encoder for cos similar of Amino(with $ finalChar) and PepDecoder.
        # aa_masses = torch.cat([aa_masses,self.finalCharMass],dim = 0)
        aa_masses = self.final_mass_encoder(aa_masses)

        assert torch.all(aa_idx < len(self._amino_acids) + 1)
        aa_idx = self.final_aa_Encoder(aa_idx)
        final_martix = torch.concat([aa_masses, aa_idx], dim=-1)
        final_martix = self.finalLinears(final_martix)
        if self.modification_pred:
            # assert 0
            modification_preds = self.modification_cls_decoder(preds)
            modification_mass_preds = self.modification_mass_decoder(preds)
            assert modification_preds.shape[1] == preds.shape[1], (
                modification_preds.shape[1] == preds.shape[1]
            )
        else:
            modification_preds = None
            modification_mass_preds = None
        preds = self.logit_scale * preds @ final_martix.t()

        # return preds,tokens
        return (torch.softmax(preds, dim=2), tokens), (
            modification_preds,
            modification_mass_preds,
        )

        # return torch.softmax(self.final(preds),dim=2), tokens


class SwiGLU(nn.Module):
    def forward(self, x):
        x, gate = x.chunk(2, dim=-1)
        return F.silu(gate) * x


class NonARPeptideDecoderLayer(nn.Module):
    def __init__(
        self, dim_model=128, n_head=8, dim_feedforward=1024, dropout=0, preln=True
    ):
        super(NonARPeptideDecoderLayer, self).__init__()

        self.preln = preln
        self.cross_attention = nn.MultiheadAttention(
            dim_model, n_head, batch_first=True
        )
        self.self_attention = nn.MultiheadAttention(dim_model, n_head, batch_first=True)
        self.feed_forward = nn.Sequential(
            nn.Linear(dim_model, dim_feedforward),
            nn.SiLU(),
            nn.Linear(dim_feedforward, dim_model),
        )
        self.norm1 = nn.LayerNorm(dim_model)
        self.norm2 = nn.LayerNorm(dim_model)
        self.norm3 = nn.LayerNorm(dim_model)

        self.dropout1 = Dropout(dropout)
        self.dropout2 = Dropout(dropout)
        self.dropout3 = Dropout(dropout)

    def _sa_block(self, peptide: Tensor, peptide_mask: Tensor):
        # peptide_mask_ = peptide_mask.unsqueeze(-1) + peptide_mask.unsqueeze(-2)
        # peptide_mask_ = peptide_mask_ > 0
        # print(peptide_mask_.shape, peptide_mask.shape)
        peptide, _ = self.self_attention(
            query=peptide, key=peptide, value=peptide, key_padding_mask=peptide_mask
        )
        return self.dropout1(peptide)

    def _mha_block(
        self,
        peptide: Tensor,
        peptide_mask: Tensor,
        spectrum: Tensor,
        spectrum_mask: Tensor,
    ):
        # attn_mask_ = peptide_mask.unsqueeze(-1) + spectrum_mask.unsqueeze(-2)
        # attn_mask_ = attn_mask_ > 0
        peptide, _ = self.cross_attention(
            query=peptide, key=spectrum, value=spectrum, key_padding_mask=spectrum_mask
        )
        return self.dropout2(peptide)

    def _ff_block(self, peptide):
        peptide = self.feed_forward(peptide)
        return self.dropout3(peptide)

    def forward(self, peptide, peptide_mask, spectrum, spectrum_mask):
        if self.preln:
            peptide = peptide + self._sa_block(self.norm1(peptide), peptide_mask)
            peptide = peptide + self._mha_block(
                self.norm2(peptide), peptide_mask, spectrum, spectrum_mask
            )
            peptide = peptide + self._ff_block(self.norm3(peptide))
        else:
            peptide = self.norm1(peptide + self._sa_block(peptide, peptide_mask))
            peptide = self.norm2(
                peptide
                + self._mha_block(peptide, peptide_mask, spectrum, spectrum_mask)
            )
            peptide = self.norm3(peptide + self._ff_block(peptide))

        return peptide, spectrum, spectrum_mask


class CLSHead(nn.Module):
    def __init__(self, input_dim=768, output_cls=2, preln: bool = False):
        super().__init__()
        self.layer_norm = LayerNorm(input_dim)
        self.linear_in = nn.Linear(input_dim, input_dim)
        self.linear_out = nn.Linear(input_dim, output_cls)
        # nn.init.zeros_(self.linear_out.weight)
        # nn.init.zeros_(self.linear_out.bias)
        # self.logsoftmax = nn.LogSoftmax(dim=-1)
        self.preln = preln
        if self.preln:
            self.layer_norm_init = LayerNorm(input_dim)

    def forward(self, x):
        x = x.type(self.linear_in.weight.dtype)
        if self.preln:
            x = self.layer_norm_init(x)
        x = F.gelu(self.layer_norm(self.linear_in(x)))
        # x = self.logsoftmax(self.linear_out(x))
        x = self.linear_out(x)
        return x


class ModificationTokenizer(nn.Module):
    def __init__(self, tokenize_dict):
        super().__init__()


def get_token1(string):
    if "[III]" in string:
        return string.split("[")[0] + "[III]"
    elif "[II]" in string:
        return string.split("[")[0] + "[II]"
    elif (
        "Xlink_BS2G" in string
        or "Xlink_BuUrBu" in string
        or "Xlink_DMP" in string
        or "Xlink_DSS" in string
        or "Xlink_DST" in string
        or "Xlink_DTSSP" in string
        or "Xlink_EGS" in string
        or "Xlink_SMCC" in string
        or "Xlink_DTBP" in string
        or "Xlink_DSSO" in string
    ):
        # print("[".join(string.split("[")[:2]))
        return "[".join(string.split("[")[:2])
    else:
        return string.split("[")[0]


class NonARPeptideDecoder(_PeptideTransformer):
    """A transformer decoder for peptide sequences.

    Parameters
    ----------
    dim_model : int, optional
        The latent dimensionality to represent peaks in the mass spectrum.
    n_head : int, optional
        The number of attention heads in each layer. ``dim_model`` must be
        divisible by ``n_head``.
    dim_feedforward : int, optional
        The dimensionality of the fully connected layers in the Transformer
        layers of the model.
    n_layers : int, optional
        The number of Transformer layers.
    dropout : float, optional
        The dropout probability for all layers.
    pos_encoder : bool, optional
        Use positional encodings for the amino acid sequence.
    reverse : bool, optional
        Sequence peptides from c-terminus to n-terminus.
    residues: Dict or str {"massivekb", "canonical"}, optional
        The amino acid dictionary and their masses. By default this is only
        the 20 canonical amino acids, with cysteine carbamidomethylated. If
        "massivekb", this dictionary will include the modifications found in
        MassIVE-KB. Additionally, a dictionary can be used to specify a custom
        collection of amino acids and masses.
    """

    def __init__(
        self,
        args,
        dim_model=128,
        n_head=8,
        dim_feedforward=1024,
        n_layers=1,
        dropout=0,
        pos_encoder=True,
        reverse=False,
        residues="canonical",
        max_charge=5,
        norm_first=False,
        modification_pred=False,
        num_round_pred=2,
        modification_mass_pred=False,
        tokenize_mod=0,
        fix_mods={},
    ):
        """Initialize a PeptideDecoder"""
        super().__init__(
            dim_model=dim_model,
            pos_encoder=pos_encoder,
            residues=residues,
            max_charge=max_charge,
            non_auto_decoder=True,
        )
        self.reverse = reverse
        self.fix_mods = fix_mods
        self.aaDim = dim_model - 256
        self.num_round_pred = num_round_pred
        self.tokenize_mod = tokenize_mod
        if tokenize_mod == 0:
            self.modification_cls_num = 2610
        else:
            self.modification_cls_num = 1466

        # Additional model components

        # Mass/charge Encoder for precursors (Dim:256 = dim_model // 2)
        self.mass_encoder = MassEncoder(dim_model)
        # self.massmap2dim = NonLinear(2 * dim_model, dim_model)

        self.max_charge = max_charge
        # self.charge_encoder = torch.nn.Embedding(max_charge, 256)

        # MassEncoder for prefix and suffix mass
        self.mass_dim = 128
        self.prefixMassEncoder = MassEncoder(self.mass_dim)
        self.suffixMassEncoder = MassEncoder(self.mass_dim)

        self.stop_token = len(self._amino_acids)

        self.layers = nn.ModuleList(
            [
                NonARPeptideDecoderLayer(dim_model, n_head, dim_feedforward, dropout)
                for _ in range(n_layers)
            ]
        )

        if self.num_round_pred > 1:
            self.map2dim = NonLinear(2 * dim_model + self.mass_dim * 2, dim_model)

            self.modification_cls_encoder = torch.nn.Embedding(
                self.modification_cls_num, dim_model
            )

            self.refine_layers = nn.ModuleList(
                [
                    NonARPeptideDecoderLayer(
                        dim_model, n_head, dim_feedforward, dropout
                    )
                    for _ in range(n_layers // 2)
                ]
            )

        self.aa_cls_head = CLSHead(dim_model, len(self._amino_acids) + 1)

        modification_meta_dict_path = args.modification_meta_dict_path
        self.modification_meta_dict = pickle.load(
            open(modification_meta_dict_path, "rb")
        )

        self.modification_mass = torch.zeros(self.modification_cls_num)

        if self.tokenize_mod == 0:
            for key in self.modification_meta_dict.keys():
                cls = self.modification_meta_dict[key][2].item() + 1
                mass = self.modification_meta_dict[key][1].item()
                if (
                    "->" in key
                    and key.split("->")[1].split("[")[0] in amino_acids_3to1_map.keys()
                ):
                    mass = 0
                self.modification_mass[int(cls)] = mass
        else:
            tokenize1_pkl_path = args.tokenize1_pkl_path
            self.tokenize1_dict = pickle.load(open(tokenize1_pkl_path, "rb"))
            force_modification = "none"
            self.force_cls = None
            for key in self.modification_meta_dict.keys():
                cls = self.tokenize1_dict[self.modification_meta_dict[key][2].item()][
                    "token_idx"
                ]  # important
                if key == force_modification:
                    self.force_cls = cls
                    assert 0
                mass = self.modification_meta_dict[key][1].item()
                if (
                    "->" in key
                    and key.split("->")[1].split("[")[0] in amino_acids_3to1_map.keys()
                ):
                    mass = 0
                self.modification_mass[int(cls)] = mass
        print("modification mass", self.modification_mass)

        # print(self.tokenize1_dict)
        self.idx_2_token = {
            v["token_idx"]: v["token"] for k, v in self.tokenize1_dict.items()
        }

        if modification_pred:
            self.modification_cls_decoder = NonLinear(
                dim_model, self.modification_cls_num
            )
            self.modification_cls_decoder.zero_init()
            if modification_mass_pred:
                assert 0, "training error"
                self.modification_mass_decoder = NonLinear(dim_model, 1)
                self.modification_mass_decoder.zero_init()
        self.modification_pred = modification_pred
        self.modification_mass_pred = modification_mass_pred

    def activate_demod(self):
        self.detokenize_mod_dict = {}
        mod_names = list(self.modification_meta_dict.keys())
        for n in mod_names:
            if "])" in n:
                all_splits = n.split("](")
                assert len(all_splits) == 2
                append_name = "(" + all_splits[1]
                prefix_name = all_splits[0]
                position = prefix_name.split("[")[-1]
                assert (
                    get_token1(n) + f"[{position}]"
                    not in self.detokenize_mod_dict.keys()
                )
                self.detokenize_mod_dict[get_token1(n) + f"[{position}]"] = n
            else:
                assert n.endswith("]"), n
                position = n.split("[")[-1][:-1]
                # if get_token1(n) + f"[{position}]" in self.detokenize_mod_dict.keys():
                #     print("warning!!!", get_token1(n) + f"[{position}]")
                self.detokenize_mod_dict[get_token1(n) + f"[{position}]"] = n

    def block(self, peptide, peptide_mask, memory, memory_key_padding_mask):
        for layer in self.layers:
            peptide, memory, memory_key_padding_mask = layer(
                peptide, peptide_mask, memory, memory_key_padding_mask
            )
        return peptide

    def detokenize_mod(self, modification_pred, num_res):
        assert len(modification_pred.shape) == 2, modification_pred.shape
        modification_string = ""
        _, modification_type = torch.max(modification_pred, dim=1)
        for i in range(num_res):
            if modification_type[i] != 0:
                modification_string += (
                    f"_{i+1}_" + self.idx_2_token[modification_type[i].item()]
                )
        return modification_string

    def detokenize_mod_encode(self, peptide_seq, modification_pred, num_res):
        assert len(modification_pred.shape) == 2, modification_pred.shape
        modification_list = []
        _, modification_type = torch.max(modification_pred, dim=1)
        modification_name_list = []
        mod4spec_list = []
        for i in range(num_res):
            if modification_type[i] != 0:
                res = peptide_seq[i]
                if i == 0:
                    position = [
                        f"AnyN-term",
                        f"AnyN-term{res}",
                        f"ProteinN-term",
                        f"ProteinN-term{res}",
                        res,
                    ]
                elif i == num_res - 1:
                    position = [
                        f"AnyC-term",
                        f"AnyC-term{res}",
                        f"ProteinC-term",
                        f"ProteinC-term{res}",
                        res,
                    ]
                else:
                    position = [res]
                mod_token = self.idx_2_token[modification_type[i].item()]
                for p in position:
                    if f"{mod_token}[{p}]" in self.detokenize_mod_dict.keys():
                        modification_name = self.detokenize_mod_dict[
                            f"{mod_token}[{p}]"
                        ]
                        break
                    else:
                        modification_name = None
                if modification_name is None:
                    # print("warning denovo modification not good", mod_token, position, i, peptide_seq)
                    pass
                else:
                    # modification_list.append((num_res - i, self.modification_meta_dict[modification_name]))
                    # flip
                    modification_list.append(
                        (i + 1, self.modification_meta_dict[modification_name])
                    )
                    modification_name_list.append(modification_name)
                    mod4spec_list.append((i + 1, modification_name))
        return modification_list, modification_name_list, mod4spec_list

    def forward(self, sequences, precursors, memory, memory_key_padding_mask):
        """Predict the next amino acid for a collection of sequences.

        Parameters
        ----------
        sequences : list of str or list of torch.Tensor
            The partial peptide sequences for which to predict the next
            amino acid. Optionally, these may be the token indices instead
            of a string.
        precursors : torch.Tensor of size (batch_size, 2)
            The measured precursor mass (axis 0) and charge (axis 1) of each
            tandem mass spectrum
        memory : torch.Tensor of shape (batch_size, n_peaks, dim_model)
            The representations from a ``TransformerEncoder``, such as a
           ``SpectrumEncoder``.
        memory_key_padding_mask : torch.Tensor of shape (batch_size, n_peaks)
            The mask that indicates which elements of ``memory`` are padding.

        Returns
        -------
        scores : torch.Tensor of size (batch_size, len_sequence, n_amino_acids)
            The raw output for the final linear layer. These can be Softmax
            transformed to yield the probability of each amino acid for the
            prediction.
        tokens : torch.Tensor of size (batch_size, len_sequence)
            The input padded tokens.

        """
        # Prepare sequences
        """if sequences is not None:
            sequences = utils.listify(sequences)
            tokens = [self.tokenize(s) for s in sequences]
            tokens = torch.nn.utils.rnn.pad_sequence(tokens, batch_first=True)
        else:
            tokens = torch.tensor([[]]).to(self.device)"""

        ret = []
        dtype = memory.dtype

        assert not self.reverse
        if sequences is not None:
            # print(sequences[0],precursors[0])
            sequences = utils.listify(sequences)

            tokens = [self.tokenize(s) for s in sequences]
            tokens = torch.nn.utils.rnn.pad_sequence(tokens, batch_first=True)
        else:
            tokens = torch.tensor([[]]).to(self.device)
        # print(tokens.shape) # torch.Size([64, 44])
        # Prepare mass and charge
        masses = self.mass_encoder(precursors[:, [0]])
        charges = self.charge_encoder(precursors[:, 1].int() - 1)
        precursors = masses + charges
        # precursors = self.massmap2dim(torch.cat([masses, charges], dim=-1))

        init_tokens = torch.zeros(tokens.shape).long().to(memory.device)
        peptide_mask = torch.zeros(tokens.shape).bool().to(memory.device)
        for i in range(init_tokens.shape[0]):
            for j in range(len(sequences[i]), init_tokens.shape[1]):
                init_tokens[i][j] = self.stop_token
                if j > len(sequences[i]):
                    peptide_mask[i][j] = True
        peptide_mask = peptide_mask.bool()
        init_tokens = init_tokens.long()

        tgt = self.aa_encoder(init_tokens.to(torch.long))
        for i in range(len(sequences)):
            # tgt[i, len(sequences[i])] += precursors[i]
            tgt[i, len(sequences[i])] = precursors[i]

        peptide = self.pos_encoder(tgt)
        for layer in self.layers:
            peptide, memory, memory_key_padding_mask = layer(
                peptide, peptide_mask, memory, memory_key_padding_mask
            )

        for i in range(1, self.num_round_pred):
            if self.modification_pred:
                middle_modification_preds = self.modification_cls_decoder(peptide)
                if self.force_cls is not None:
                    middle_modification_preds[:, :, self.force_cls] *= 3
                if self.modification_mass_pred:
                    middle_modification_mass_preds = self.modification_mass_decoder(
                        peptide
                    )
                else:
                    middle_modification_mass_preds = None
            else:
                middle_modification_preds = None
                middle_modification_mass_preds = None
            middle_preds = torch.softmax(self.aa_cls_head(peptide), dim=2)
            ret.append(
                (
                    (middle_preds, tokens),
                    ~peptide_mask,
                    (middle_modification_preds, middle_modification_mass_preds),
                )
            )
            middle_cls_preds = torch.argmax(middle_preds, axis=2)
            middle_modification_cls_preds = torch.argmax(
                middle_modification_preds, axis=2
            )
            middle_modification_cls_preds_mass = self.modification_mass.to(
                middle_modification_cls_preds.device
            )[middle_modification_cls_preds]
            for i in range(len(sequences)):
                for j in range(0, len(sequences[i])):
                    if (
                        middle_cls_preds[i][j] == self.stop_token
                        or middle_cls_preds[i][j] == 0
                    ):
                        middle_cls_preds[i][j] = 1
                for j in range(len(sequences[i]), init_tokens.shape[1]):
                    middle_cls_preds[i][j] = self.stop_token
                middle_cls_preds[i][len(sequences[i])] = self.stop_token
            seqs = [
                "".join(self.detokenize(middle_cls_preds[i]))[:-1]
                for i in range(middle_cls_preds.shape[0])
            ]
            for i in range(len(seqs)):
                assert len(seqs[i]) == len(sequences[i]), (
                    self._idx2aa,
                    middle_cls_preds[i],
                    len(middle_cls_preds[i]),
                    seqs[i],
                    len(seqs[i]),
                    len(sequences[i]),
                )
            Masses = [
                self.deMass(seqs[s], modifi_mass=middle_modification_cls_preds_mass[s])
                for s in range(len(seqs))
            ]
            Masses = torch.nn.utils.rnn.pad_sequence(Masses, batch_first=True)
            suffixMasses = [
                self.get_suffix_mass(
                    seqs[i],
                    precursors[i][0],
                    modifi_mass=middle_modification_cls_preds_mass[i],
                )
                for i in range(len(seqs))
            ]
            suffixMasses = torch.nn.utils.rnn.pad_sequence(
                suffixMasses, batch_first=True
            )
            Masses = Masses.unsqueeze(2)
            Masses = self.prefixMassEncoder(Masses)
            suffixMasses = suffixMasses.unsqueeze(2)
            suffixMasses = self.suffixMassEncoder(suffixMasses)
            Masses = torch.concat([Masses, suffixMasses], dim=2)
            # Masses = torch.concat([Masses,precursors],dim=1)
            new_repre = self.aa_encoder(
                middle_cls_preds
            ) + self.modification_cls_encoder(middle_modification_cls_preds)
            # print(Masses.shape, peptide.shape, new_repre.shape)

            peptide = peptide + self.map2dim(
                torch.cat([Masses, peptide, new_repre], dim=2).type(dtype)
            )
            for layer in self.refine_layers:
                peptide, memory, memory_key_padding_mask = layer(
                    peptide, peptide_mask, memory, memory_key_padding_mask
                )

        if self.modification_pred:
            modification_preds = self.modification_cls_decoder(peptide)
            if self.force_cls is not None:
                modification_preds[:, :, self.force_cls] *= 3
            if self.modification_mass_pred:
                modification_mass_preds = self.modification_mass_decoder(peptide)
                assert modification_preds.shape[1] == peptide.shape[1], (
                    modification_preds.shape[1] == peptide.shape[1]
                )
            else:
                modification_mass_preds = None
        else:
            modification_preds = None
            modification_mass_preds = None
        # assert 0, (torch.argmax(modification_preds, dim=-1), None)
        # preds = self.logit_scale * preds @ final_martix.t()
        preds = self.aa_cls_head(peptide)
        ret.append(
            (
                (torch.softmax(preds, dim=2), tokens),
                ~peptide_mask,
                (modification_preds, modification_mass_preds),
            )
        )
        return ret


def generate_tgt_mask(sz):
    """Generate a square mask for the sequence. The masked positions
    are filled with float('-inf'). Unmasked positions are filled with
    float(0.0).

    This function is a slight modification of the version in the PyTorch
    repository.

    Parameters
    ----------
    sz : int
        The length of the target sequence.
    """
    mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
    mask = (
        mask.float()
        .masked_fill(mask == 0, float("-inf"))
        .masked_fill(mask == 1, float(0.0))
    )
    return mask


class PepMsJointEncoderLayer(nn.Module):
    def __init__(
        self, dim_model=128, n_head=8, dim_feedforward=1024, dropout=0, preln=True
    ):
        super(PepMsJointEncoderLayer, self).__init__()

        self.preln = preln
        self.cross_attention = nn.MultiheadAttention(
            dim_model, n_head, batch_first=True
        )
        self.self_attention = nn.MultiheadAttention(dim_model, n_head, batch_first=True)
        self.feed_forward = nn.Sequential(
            nn.Linear(dim_model, dim_feedforward),
            nn.SiLU(),
            nn.Linear(dim_feedforward, dim_model),
        )
        self.norm1 = nn.LayerNorm(dim_model)
        self.norm2 = nn.LayerNorm(dim_model)
        self.norm3 = nn.LayerNorm(dim_model)

        self.dropout1 = Dropout(dropout)
        self.dropout2 = Dropout(dropout)
        self.dropout3 = Dropout(dropout)

    def _sa_block(self, peptide: Tensor, peptide_mask: Tensor):
        # peptide_mask_ = peptide_mask.unsqueeze(-1) + peptide_mask.unsqueeze(-2)
        # peptide_mask_ = peptide_mask_ > 0
        # print(peptide_mask_.shape, peptide_mask.shape)
        peptide, _ = self.self_attention(
            query=peptide, key=peptide, value=peptide, key_padding_mask=peptide_mask
        )
        return self.dropout1(peptide)

    def _mha_block(
        self,
        peptide: Tensor,
        peptide_mask: Tensor,
        spectrum: Tensor,
        spectrum_mask: Tensor,
    ):
        # attn_mask_ = peptide_mask.unsqueeze(-1) + spectrum_mask.unsqueeze(-2)
        # attn_mask_ = attn_mask_ > 0
        peptide, _ = self.cross_attention(
            query=peptide, key=spectrum, value=spectrum, key_padding_mask=spectrum_mask
        )
        return self.dropout2(peptide)

    def _ff_block(self, peptide):
        peptide = self.feed_forward(peptide)
        return self.dropout3(peptide)

    def forward(self, peptide, peptide_mask, spectrum, spectrum_mask):
        if self.preln:
            peptide = peptide + self._sa_block(self.norm1(peptide), peptide_mask)
            peptide = peptide + self._mha_block(
                self.norm2(peptide), peptide_mask, spectrum, spectrum_mask
            )
            peptide = peptide + self._ff_block(self.norm3(peptide))
        else:
            peptide = self.norm1(peptide + self._sa_block(peptide, peptide_mask))
            peptide = self.norm2(
                peptide
                + self._mha_block(peptide, peptide_mask, spectrum, spectrum_mask)
            )
            peptide = self.norm3(peptide + self._ff_block(peptide))

        return peptide, spectrum, spectrum_mask


class PepMsJointEncoder(nn.Module):
    def __init__(
        self,
        dim_model=512,
        n_head=8,
        dim_feedforward=1024,
        n_layers=4,
        dropout=0,
    ):
        super(PepMsJointEncoder, self).__init__()
        self.layers = nn.ModuleList(
            [
                PepMsJointEncoderLayer(dim_model, n_head, dim_feedforward, dropout)
                for _ in range(n_layers)
            ]
        )

        self.proj_to_cls = nn.Linear(dim_model, 1)

    def forward(
        self, peptide, peptide_mask, spectrum, spectrum_mask, repeat_num, rt_pred=None
    ):
        # print(peptide.shape, spectrum.shape, repeat_num.shape)
        if rt_pred is not None:
            assert rt_pred.shape[0] == peptide.shape[0]
        spectrum = torch.repeat_interleave(spectrum, repeat_num.long(), dim=0)
        spectrum_mask = torch.repeat_interleave(spectrum_mask, repeat_num.long(), dim=0)
        for layer in self.layers:
            peptide, spectrum, spectrum_mask = layer(
                peptide, peptide_mask, spectrum, spectrum_mask
            )

        return self.proj_to_cls(peptide[:, 0, :])
        # return self.proj_to_cls((peptide[:, 0, :] * (~peptide_mask)).sum(dim=1) / (~peptide_mask).sum(dim=1))


class RankDecoderLayer(nn.Module):
    def __init__(
        self, dim_model=128, n_head=8, dim_feedforward=1024, dropout=0, preln=True
    ):
        super(RankDecoderLayer, self).__init__()

        self.preln = preln
        self.self_attention = nn.MultiheadAttention(dim_model, n_head, batch_first=True)
        self.feed_forward = nn.Sequential(
            nn.Linear(dim_model, dim_feedforward),
            nn.SiLU(),
            nn.Linear(dim_feedforward, dim_model),
        )
        self.norm1 = nn.LayerNorm(dim_model)
        self.norm3 = nn.LayerNorm(dim_model)

        self.dropout1 = Dropout(dropout)
        self.dropout3 = Dropout(dropout)

    def _sa_block(self, peptide: Tensor, peptide_mask: Tensor):
        peptide, _ = self.self_attention(
            query=peptide, key=peptide, value=peptide, key_padding_mask=peptide_mask
        )
        return self.dropout1(peptide)

    def _ff_block(self, peptide):
        peptide = self.feed_forward(peptide)
        return self.dropout3(peptide)

    def forward(self, ranklist, ranklist_mask):
        if self.preln:
            ranklist = ranklist + self._sa_block(self.norm1(ranklist), ranklist_mask)
            ranklist = ranklist + self._ff_block(self.norm3(ranklist))
        else:
            ranklist = self.norm1(ranklist + self._sa_block(ranklist, ranklist_mask))
            ranklist = self.norm3(ranklist + self._ff_block(ranklist))

        return ranklist


def process_spectra_peptides(
    spectra_feat_all,
    spectra_mask,
    peptide_feat_all,
    peptide_mask,
    scores,
    batch_index,
    n_peptide=9,
    use_mean=False,
):
    M, P, d = spectra_feat_all.shape
    device = spectra_feat_all.device

    # Step 1: Compute mean of non-padding peaks in each spectrum
    if use_mean:
        spectra_feat_mean = (spectra_feat_all * (~spectra_mask).unsqueeze(-1)).sum(
            dim=1
        ) / (~spectra_mask).sum(dim=1, keepdim=True).clamp(min=1)
    else:
        spectra_feat_mean = spectra_feat_all[:, 0, :]

    # Step 2: Get top n_peptide scores for each spectrum
    peptide_indices = torch.zeros(M, n_peptide, dtype=torch.long, device=device)
    peptide_mask_final = torch.ones(M, n_peptide, dtype=torch.bool, device=device)

    # 将batch_index转换为1D张量便于索引
    batch_index = batch_index.squeeze(-1)

    for m in range(M):
        # 获取当前spectrum对应的所有peptide索引
        mask = batch_index == m
        candidate_indices = torch.where(mask)[0]

        if len(candidate_indices) == 0:
            continue  # 无候选肽段，保持初始的mask=True

        # 提取对应scores并排序
        spectrum_scores = scores[candidate_indices].squeeze(-1)
        k = min(n_peptide, len(spectrum_scores))

        if k > 0:
            top_scores, top_relative_idx = torch.topk(spectrum_scores, k, largest=True)
            top_absolute_idx = candidate_indices[top_relative_idx]

            # 存储结果
            peptide_indices[m, :k] = top_absolute_idx
            peptide_mask_final[m, :k] = False  # False表示有效

    # Step 3: Compute peptide features
    # 使用安全索引，防止越界访问
    max_peptide_idx = peptide_feat_all.shape[0] - 1
    safe_indices = torch.clamp(peptide_indices, 0, max_peptide_idx)

    selected_peptides = peptide_feat_all[safe_indices]  # [M, n_peptide, S, d]
    selected_masks = peptide_mask[safe_indices]  # [M, n_peptide, S]

    # 计算非padding氨基酸的均值
    if use_mean:
        peptide_feat_mean = (selected_peptides * (~selected_masks).unsqueeze(-1)).sum(
            dim=2
        )
    else:
        peptide_feat_mean = selected_peptides[:, :, 0, :]
    # breakpoint()
    valid_counts = (~selected_masks).sum(dim=2).clamp(min=1)
    peptide_feat_mean = peptide_feat_mean / valid_counts.unsqueeze(-1)

    # Step 4: Concatenate features
    final_tensor = torch.cat(
        [
            spectra_feat_mean.unsqueeze(1),  # [M, 1, d]
            peptide_feat_mean,  # [M, n_peptide, d]
        ],
        dim=1,
    )  # 最终形状 [M, 1+n_peptide, d]

    # Step 5: Generate true labels
    true_labels_tensor = generate_true_labels(peptide_indices, batch_index, device)

    return final_tensor, peptide_indices, peptide_mask_final, true_labels_tensor


def generate_true_labels(peptide_indices, batch_index, device):
    M, n_peptide = peptide_indices.shape
    true_labels = torch.zeros(M, n_peptide, dtype=torch.bool, device=device)

    for m in range(M):
        # 找到当前spectrum对应的真实peptide索引
        true_idx = torch.where(batch_index == m)[0][0]  # 假设第一个匹配项是真实标签

        # 在top候选列表中标记真实标签
        matches = peptide_indices[m] == true_idx
        if matches.any():
            true_labels[m] = matches

    return true_labels.unsqueeze(-1)  # 添加最后一个维度以保持形状兼容


class Transition(nn.Module):
    def __init__(self, d_in, d_out, n=1, dropout=0.0):
        super(Transition, self).__init__()

        self.d_in = d_in
        self.d_out = d_out
        self.n = n

        self.linear_1 = nn.Linear(self.d_in, self.n * self.d_in)
        self.act = nn.GELU()
        self.linear_2 = nn.Linear(self.n * self.d_in, d_out)
        self.dropout = dropout

    def _transition(self, x):
        x = self.linear_1(x)
        x = self.act(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.linear_2(x)
        return x

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        x = self._transition(x=x)
        return x


class RankDecoder(nn.Module):
    def __init__(
        self,
        dim_model=512,
        n_head=8,
        dim_feedforward=1024,
        n_layers=4,
        dropout=0,
        residual=False,
    ):
        super(RankDecoder, self).__init__()
        self.layers = nn.ModuleList(
            [
                RankDecoderLayer(dim_model, n_head, dim_feedforward, dropout)
                for _ in range(n_layers)
            ]
        )

        self.proj_to_cls = nn.Linear(dim_model, 1)
        self.residual = residual
        if residual:
            self.transition = Transition(2 * dim_model, dim_model)

    def forward(
        self,
        spectra_feat_all,
        spectra_mask,
        peptide_feat_all,
        peptide_mask,
        scores,
        batch_index,
    ):
        ranklist, ranking_indices, ranklist_mask, true_labels_tensor = (
            process_spectra_peptides(
                spectra_feat_all,
                spectra_mask,
                peptide_feat_all,
                peptide_mask,
                scores,
                batch_index,
            )
        )
        residual = ranklist
        false_dimension = torch.zeros((ranklist_mask.shape[0], 1), dtype=torch.bool).to(
            ranklist_mask.device
        )
        result_mask = torch.cat((ranklist_mask, false_dimension), dim=1)
        for layer in self.layers:
            ranklist = layer(ranklist, result_mask)
        if self.residual:
            ranklist_cat = torch.cat([ranklist, residual], dim=-1)
            ranklist += self.transition(ranklist_cat)
        # ranklist = ranklist + residual
        # breakpoint()
        return (
            self.proj_to_cls(ranklist),
            ranking_indices,
            ranklist_mask.bool(),
            true_labels_tensor,
        )
