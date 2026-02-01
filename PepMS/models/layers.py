"""Simple encoders for input into Transformers and the like."""

import itertools
import re
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union

import einops
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.nn.modules import Module
from torch.nn.modules.transformer import _get_clones
from torch.utils.checkpoint import checkpoint
from compat.modules import LayerNorm, SelfMultiheadAttention, softmax_dropout

from . import utils


class Linear(nn.Linear):
    def __init__(
        self,
        d_in: int,
        d_out: int,
        bias: bool = True,
        init: str = "default",
    ):
        super(Linear, self).__init__(d_in, d_out, bias=bias)

        self.use_bias = bias

        if self.use_bias:
            with torch.no_grad():
                self.bias.fill_(0)

        if init == "default":
            self._trunc_normal_init(1.0)
        elif init == "relu":
            self._trunc_normal_init(2.0)
        elif init == "glorot":
            self._glorot_uniform_init()
        elif init == "gating":
            self._zero_init(self.use_bias)
        elif init == "normal":
            self._normal_init()
        elif init == "final":
            self._zero_init(False)
        else:
            raise ValueError("Invalid init method.")

    def _trunc_normal_init(self, scale=1.0):
        # Constant from scipy.stats.truncnorm.std(a=-2, b=2, loc=0., scale=1.)
        TRUNCATED_NORMAL_STDDEV_FACTOR = 0.87962566103423978
        _, fan_in = self.weight.shape
        scale = scale / max(1, fan_in)
        std = (scale**0.5) / TRUNCATED_NORMAL_STDDEV_FACTOR
        nn.init.trunc_normal_(self.weight, mean=0.0, std=std)

    def _glorot_uniform_init(self):
        nn.init.xavier_uniform_(self.weight, gain=1)

    def _zero_init(self, use_bias=True):
        with torch.no_grad():
            self.weight.fill_(0.0)
            if use_bias:
                with torch.no_grad():
                    self.bias.fill_(1.0)

    def _normal_init(self):
        torch.nn.init.kaiming_normal_(self.weight, nonlinearity="linear")


class NonLinear(nn.Module):
    def __init__(self, input, output_size, hidden=None):
        super(NonLinear, self).__init__()

        if hidden is None:
            hidden = input
        self.layer1 = Linear(input, hidden, init="relu")
        self.layer2 = Linear(hidden, output_size, init="final")

    def forward(self, x):
        x = self.layer1(x)
        x = F.gelu(x)
        x = self.layer2(x)
        return x

    def zero_init(self):
        nn.init.zeros_(self.layer2.weight)
        nn.init.zeros_(self.layer2.bias)


class CLSHead(nn.Module):
    def __init__(self, input_dim=768, output_cls=2, preln: bool = False):
        super().__init__()
        self.layer_norm = LayerNorm(input_dim)
        self.linear_in = Linear(input_dim, input_dim, init="relu")
        self.linear_out = Linear(input_dim, output_cls, bias=True, init="final")
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

        # self.charge_encoder = torch.nn.Embedding(max_charge, dim_model)
        self.aa_encoder = torch.nn.Embedding(
            len(self._amino_acids) + 2,
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
            sequence = list(reversed(sequence))

        if not partial:
            sequence += ["$"]

        tokens = [self._aa2idx[aa] for aa in sequence]
        tokens = torch.tensor(tokens, device=self.device)
        return tokens

    def detokenize(self, tokens):
        """Transform tokens back into a peptide sequence

        Parameters
        ----------
        tokens : torch.Tensor of shape (n_amino_acids)
        """
        sequence = [self._idx2aa.get(i.item(), "") for i in tokens]
        if "$" in sequence:
            idx = sequence.index("$")
            sequence = sequence[: idx + 1]

        if self.reverse:
            sequence = list(reversed(sequence))

        return "".join(sequence)

    @property
    def vocab_size(self):
        """Return the number of amino acids"""
        return len(self._aa2idx)

    @property
    def device(self):
        """The current device for the model"""
        return next(self.parameters()).device


class Dropout(nn.Module):
    def __init__(self, p):
        super().__init__()
        self.p = p

    def forward(self, x, inplace: bool = False):
        if self.p > 0 and self.training:
            return F.dropout(x, p=self.p, training=True, inplace=inplace)
        else:
            return x


class Attention(nn.Module):
    def __init__(
        self,
        q_dim: int,
        k_dim: int,
        v_dim: int,
        pair_dim: int,
        head_dim: int,
        num_heads: int,
        gating: bool = False,
        dropout: float = 0.0,
    ):
        super(Attention, self).__init__()

        self.num_heads = num_heads
        total_dim = head_dim * self.num_heads
        self.gating = gating
        self.linear_q = Linear(q_dim, total_dim, bias=False, init="glorot")
        self.linear_k = Linear(k_dim, total_dim, bias=False, init="glorot")
        self.linear_v = Linear(v_dim, total_dim, bias=False, init="glorot")
        self.linear_o = Linear(total_dim, q_dim, init="final")
        self.linear_g = None
        if self.gating:
            self.linear_g = Linear(q_dim, total_dim, init="gating")
        # precompute the 1/sqrt(head_dim)
        self.norm = head_dim**-0.5
        self.dropout = dropout
        self.linear_bias = Linear(pair_dim, num_heads)

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        pair: torch.Tensor,
        mask: torch.Tensor = None,
    ) -> torch.Tensor:
        g = None
        if self.linear_g is not None:
            # gating, use raw query input
            g = self.linear_g(q)

        q = self.linear_q(q)
        q *= self.norm
        k = self.linear_k(k)
        v = self.linear_v(v)

        q = q.view(q.shape[:-1] + (self.num_heads, -1)).transpose(-2, -3).contiguous()
        k = k.view(k.shape[:-1] + (self.num_heads, -1)).transpose(-2, -3).contiguous()
        v = v.view(v.shape[:-1] + (self.num_heads, -1)).transpose(-2, -3)

        attn = torch.matmul(q, k.transpose(-1, -2))
        del q, k
        bias = self.linear_bias(pair).permute(0, 3, 1, 2).contiguous()
        attn = softmax_dropout(attn, self.dropout, self.training, mask=mask, bias=bias)
        o = torch.matmul(attn, v)
        del attn, v

        o = o.transpose(-2, -3).contiguous()
        o = o.view(*o.shape[:-2], -1)

        if g is not None:
            o = torch.sigmoid(g) * o

        # merge heads
        o = self.linear_o(o)
        return o


class OuterProduct(nn.Module):
    def __init__(self, d_atom, d_pair, d_hid=32):
        super(OuterProduct, self).__init__()

        self.d_atom = d_atom
        self.d_pair = d_pair
        self.d_hid = d_hid

        self.linear_in = nn.Linear(d_atom, d_hid * 2)
        self.linear_out = nn.Linear(d_hid**2, d_pair)
        self.act = nn.GELU()
        self._memory_efficient = False

    def _opm(self, a, b):
        bsz, n, d = a.shape
        # outer = torch.einsum("...bc,...de->...bdce", a, b)
        a = a.view(bsz, n, 1, d, 1)
        b = b.view(bsz, 1, n, 1, d)
        outer = a * b
        outer = outer.view(outer.shape[:-2] + (-1,))
        outer = self.linear_out(outer)
        return outer

    def apply_memory_efficient(self):
        self._memory_efficient = True

    def forward(
        self,
        m: torch.Tensor,
        op_mask: Optional[torch.Tensor] = None,
        op_norm: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        ab = self.linear_in(m)
        ab = ab * op_mask
        a, b = ab.chunk(2, dim=-1)
        if self._memory_efficient and torch.is_grad_enabled():
            z = checkpoint(self._opm, a, b)
        else:
            z = self._opm(a, b)
        z *= op_norm
        return z


def get_activation_fn(activation: str) -> Callable:
    """Returns the activation function corresponding to `activation`"""

    if activation == "relu":
        return F.relu
    elif activation == "gelu":
        return F.gelu
    elif activation == "tanh":
        return torch.tanh
    elif activation == "linear":
        return lambda x: x
    else:
        raise RuntimeError("--activation-fn {} not supported".format(activation))


class Transition(nn.Module):
    def __init__(self, d_in, n, dropout=0.0):
        super(Transition, self).__init__()

        self.d_in = d_in
        self.n = n

        self.linear_1 = Linear(self.d_in, self.n * self.d_in, init="relu")
        self.act = nn.GELU()
        self.linear_2 = Linear(self.n * self.d_in, d_in, init="final")
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


class UniMSEncoder(nn.Module):
    def __init__(
        self,
        embedding_dim: int = 768,
        pair_dim: int = 64,
        pair_hidden_dim: int = 32,
        ffn_embedding_dim: int = 3072,
        num_attention_heads: int = 8,
        num_encoder_layers: int = 8,
        use_ins: bool = True,
        use_nce: bool = True,
        use_rope: bool = False,
        dropout: float = 0.1,
        attention_dropout: float = 0.1,
        activation_dropout: float = 0.1,
        activation_fn: str = "gelu",
        droppath_prob: float = 0.0,
        pair_dropout: float = 0.25,
    ) -> None:
        super().__init__()
        self.embedding_dim = embedding_dim
        self.num_head = num_attention_heads
        self.layer_norm = LayerNorm(embedding_dim)
        self.pair_layer_norm = LayerNorm(pair_dim)
        self.layers = nn.ModuleList([])

        if droppath_prob > 0:
            droppath_probs = [
                x.item() for x in torch.linspace(0, droppath_prob, num_encoder_layers)
            ]
        else:
            droppath_probs = None

        self.layers.extend(
            [
                UniMSEncoderLayer(
                    embedding_dim=embedding_dim,
                    pair_dim=pair_dim,
                    pair_hidden_dim=pair_hidden_dim,
                    ffn_embedding_dim=ffn_embedding_dim,
                    num_attention_heads=num_attention_heads,
                    dropout=dropout,
                    attention_dropout=attention_dropout,
                    activation_dropout=activation_dropout,
                    activation_fn=activation_fn,
                    droppath_prob=(
                        droppath_probs[i] if droppath_probs is not None else 0
                    ),
                    pair_dropout=pair_dropout,
                )
                for i in range(num_encoder_layers)
            ]
        )

    def forward(
        self,
        x,
        pair,
        atom_mask,
        pair_mask,
        attn_mask=None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.layer_norm(x)
        pair = self.pair_layer_norm(pair)
        op_mask = atom_mask.unsqueeze(-1)
        op_mask = op_mask * (op_mask.size(-2) ** -0.5)
        eps = 1e-3
        op_norm = 1.0 / (eps + torch.einsum("...bc,...dc->...bdc", op_mask, op_mask))

        zeroMass = torch.zeros([precursors.shape[0], 1, 1]).to(self.device)
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
        zeros = ~spectra.sum(dim=2).bool()
        mask = zeros

        for layer in self.layers:
            x, pair = layer(
                x,
                pair,
                pair_mask=pair_mask,
                self_attn_mask=attn_mask,
                op_mask=op_mask,
                op_norm=op_norm,
            )
        return x, pair


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
        output_layer=4,
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
        self.output_layer = output_layer

    def forward(
        self,
        src: Tensor,
        mask: Optional[Tensor] = None,
        src_key_padding_mask: Optional[Tensor] = None,
        is_causal: Optional[bool] = None,
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

        cnt_layer = 0
        output_half = None
        for mod in self.layers:
            output = mod(
                output,
                src_mask=mask,
                is_causal=is_causal,
                src_key_padding_mask=src_key_padding_mask_for_layers,
            )
            cnt_layer += 1
            if cnt_layer == self.output_layer:
                output_half = output.clone()
        if convert_to_nested:
            output = output.to_padded_tensor(0.0)
            output_half = output_half.to_padded_tensor(0.0)

        if self.norm is not None:
            output = self.norm(output)

        return output, output_half


def initialize_embedding_to_zero(embedding_layer):
    with torch.no_grad():  # 确保在初始化过程中不计算梯度
        embedding_layer.weight.fill_(0)


import pickle


class PeptideEncoder(nn.Module):
    def __init__(
        self,
        args,
        max_len=100,
        max_charge=10,
        residues: Union[Dict[str, float], str] = "canonical",
        ptransformer_width: int = 512,
        ptransformer_heads: int = 8,
        ptransformer_layers: int = 9,
        norm_first: bool = False,
        use_ins=False,
        use_nce=False,
        tokenize_mod=0,
        dropout=0.4,
        dtype=torch.float32,
    ):

        super().__init__()

        self.reverse = False
        self.dtype = dtype

        self.max_len = max_len
        self._peptide_mass = PeptideMass(residues=residues)
        self._amino_acids = list(self._peptide_mass.masses.keys())
        self._idx2aa = {i + 1: aa for i, aa in enumerate(self._amino_acids)}
        self._aa2idx = {aa: i for i, aa in self._idx2aa.items()}

        self.tokenize_mod = tokenize_mod
        if tokenize_mod == 0:
            self.modification_cls_num = 2610
        else:
            self.modification_cls_num = 1466
            tokenize1_pkl_path = args.tokenize1_pkl_path
            self.tokenize1_dict = pickle.load(open(tokenize1_pkl_path, "rb"))
        self.aminoEmbedDim = ptransformer_width // 2
        self.aa_encoder = torch.nn.Embedding(
            len(self._amino_acids) + 1,
            self.aminoEmbedDim,
            padding_idx=0,
        )

        # Mass/charge Encoder for precursors (Dim:256 = dim_model // 2)
        self.mass_encoder = MassEncoder(ptransformer_width // 2)
        # self.charge_encoder = torch.nn.Embedding(50, ptransformer_width // 2)
        self.charge_encoder = torch.nn.Embedding(max_charge, 256)
        self.use_ins = use_ins
        self.use_nce = use_nce
        if self.use_nce:
            self.nce_encoder = torch.nn.Embedding(101, ptransformer_width // 2)
            initialize_embedding_to_zero(self.nce_encoder)
        if self.use_ins:
            self.ins_encoder = torch.nn.Embedding(10, ptransformer_width // 2)
            initialize_embedding_to_zero(self.ins_encoder)
        # MassEncoder for prefix and suffix mass
        self.prefixMassEncoder = MassEncoder(ptransformer_width // 4)
        self.suffixMassEncoder = MassEncoder(ptransformer_width // 4)

        self.massEncoder = MassEncoder(ptransformer_width)

        self.pos_encoder = PositionalEncoder(ptransformer_width)

        self.modification_cls_encoder = torch.nn.Embedding(
            self.modification_cls_num, ptransformer_width
        )
        self.modification_type_encoder = torch.nn.Embedding(6, ptransformer_width)
        # self.modification_encoder = Linear(2, ptransformer_width, init="glorot")

        layer = torch.nn.TransformerEncoderLayer(
            d_model=ptransformer_width,
            nhead=ptransformer_heads,
            dim_feedforward=1024,
            batch_first=True,
            dropout=dropout,
        )

        self.peptideTransformer = torch.nn.TransformerEncoder(
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
        self,
        sequences,
        sequences_after_mod,
        precursors,
        instrument,
        nce,
        modification,
        batch_index,
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
        # assert 0, (sequences, sequences_after_mod)
        sequences = utils.listify(sequences)
        sequences_after_mod = utils.listify(sequences_after_mod)
        assert len(sequences) == len(modification), (len(sequences), len(modification))
        Masses = [
            self.deMass(sequences[i], modification[i]) for i in range(len(sequences))
        ]
        # assert 0, (Masses, sequences)
        Masses = torch.nn.utils.rnn.pad_sequence(Masses, batch_first=True)
        # print(len(sequences), len(precursors))
        # precursors = precursors.repeat_interleave(2, dim=0)
        precursors_list = []
        instrument_list = []
        nce_list = []
        for i in range(int(batch_index[-1] + 1)):
            num = (batch_index == i).sum()
            if num > 0:
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
        suffixMasses = torch.nn.utils.rnn.pad_sequence(suffixMasses, batch_first=True)
        tokens = []
        for i in range(len(sequences_after_mod)):
            if len(sequences[i]) == len(sequences_after_mod):
                tokens.append(self.tokenize(sequences_after_mod[i]))
            else:
                tokens.append(self.tokenize(sequences[i]))
        assert len(sequences) == len(sequences_after_mod)
        # tokens = [self.tokenize(s) for s in sequences_after_mod]
        tokens = torch.nn.utils.rnn.pad_sequence(tokens, batch_first=True)
        # assert 0, (tokens, sequences_after_mod)

        # assert 0,  (sequences[:3], tokens[:3])
        masses = self.mass_encoder(precursors[:, None, [0]])
        assert torch.all((precursors[:, 1].int() - 1) < 50)
        charges = self.charge_encoder(precursors[:, 1].int() - 1)
        infos = charges[:, None, :]
        if self.use_ins:
            ins = self.ins_encoder(instrument.int())
            infos += ins[:, None, :]
        if self.use_nce:
            nce = self.nce_encoder(nce.int())
            infos += nce[:, None, :]
        precursors = masses

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

        assert torch.all(tokens < len(self._amino_acids) + 1), (
            tokens,
            self._amino_acids,
            len(self._amino_acids),
        )
        tgt = self.aa_encoder(tokens)
        tgt_key_padding_mask = tgt.sum(axis=2) == 0
        tgt = torch.concat([tgt, Masses], dim=2)

        # print(precursors.shape, tgt.shape)
        tgt = torch.cat([precursors, tgt], dim=1).type(self.dtype)
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
                if self.tokenize_mod == 1:
                    cla = self.tokenize1_dict[mod[1][2].item()]["token_idx"]
                elif self.tokenize_mod == 0:
                    cla = mod[1][2]
                # print(cla, mod[1][2], self.modification_cls_encoder(torch.Tensor([cla]).long().to(tgt.device)).shape, self.modification_type_encoder(torch.Tensor(mod[1][0]).long().to(tgt.device)).shape, tgt.shape)
                tgt[idx][mod[0]] += self.modification_cls_encoder(
                    torch.Tensor([cla]).long().to(tgt.device)
                ).squeeze(0) + self.modification_type_encoder(
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
        # print(tgt)

        # tgt = tgt.permute(1,0,2)
        # assert 0, (tgt_key_padding_mask,tokens)
        tgt = self.peptideTransformer(
            tgt.type(self.dtype), src_key_padding_mask=tgt_key_padding_mask
        )
        # tgt = tgt.permute(1,0,2) #Shape(B_s, Peptide_len, Feature_size)
        # assert 0, tgt
        return tgt, tgt_key_padding_mask

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
                try:
                    masses[mod[0] - 1] += mod[1][1]
                except:
                    print(len(masses), mod, mod[0] - 1)
                    masses[mod[0] - 1]
                    assert 0

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
            assert 0
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


class MassEncoder(torch.nn.Module):
    """Encode mass values using sine and cosine waves.

    Parameters
    ----------
    dim_model : int
        The number of features to output.
    min_wavelength : float
        The minimum wavelength to use.
    max_wavelength : float
        The maximum wavelength to use.
    """

    def __init__(self, dim_model, min_wavelength=0.001, max_wavelength=10000):
        """Initialize the MassEncoder"""
        super().__init__()

        n_sin = int(dim_model / 2)
        n_cos = dim_model - n_sin

        if min_wavelength:
            base = min_wavelength / (2 * np.pi)
            scale = max_wavelength / min_wavelength
        else:
            base = 1
            scale = max_wavelength / (2 * np.pi)

        sin_term = base * scale ** (torch.arange(0, n_sin).float() / (n_sin - 1))
        cos_term = base * scale ** (torch.arange(0, n_cos).float() / (n_cos - 1))

        self.register_buffer("sin_term", sin_term)
        self.register_buffer("cos_term", cos_term)

    def forward(self, X):
        """Encode m/z values.

        Parameters
        ----------
        X : torch.Tensor of shape (n_masses)
            The masses to embed.

        Returns
        -------
        torch.Tensor of shape (n_masses, dim_model)
            The encoded features for the mass spectra.
        """
        sin_mz = torch.sin(X / self.sin_term)
        cos_mz = torch.cos(X / self.cos_term)
        return torch.cat([sin_mz, cos_mz], axis=-1)


class PositionalEncoder(torch.nn.Module):
    """The positional encoder for sequences.

    Parameters
    ----------
    dim_model : int
        The number of features to output.
    """

    def __init__(self, dim_model, max_wavelength=10000):
        """Initialize the MzEncoder"""
        super().__init__()

        n_sin = int(dim_model / 2)
        n_cos = dim_model - n_sin
        scale = max_wavelength / (2 * np.pi)

        sin_term = scale ** (torch.arange(0, n_sin).float() / (n_sin - 1))
        cos_term = scale ** (torch.arange(0, n_cos).float() / (n_cos - 1))
        self.register_buffer("sin_term", sin_term)
        self.register_buffer("cos_term", cos_term)

    def forward(self, X):
        """Encode positions in a sequence.

        Parameters
        ----------
        X : torch.Tensor of shape (batch_size, n_sequence, n_features)
            The first dimension should be the batch size (i.e. each is one
            peptide) and the second dimension should be the sequence (i.e.
            each should be an amino acid representation).

        Returns
        -------
        torch.Tensor of shape (batch_size, n_sequence, n_features)
            The encoded features for the mass spectra.
        """
        pos = torch.arange(X.shape[1]).type_as(self.sin_term)
        pos = einops.repeat(pos, "n -> b n", b=X.shape[0])
        sin_in = einops.repeat(pos, "b n -> b n f", f=len(self.sin_term))
        cos_in = einops.repeat(pos, "b n -> b n f", f=len(self.cos_term))

        sin_pos = torch.sin(sin_in / self.sin_term)
        cos_pos = torch.cos(cos_in / self.cos_term)
        encoded = torch.cat([sin_pos, cos_pos], axis=2)
        return encoded + X


class PeptideMass:
    """A simple class for calculating peptide masses

    Parameters
    ----------
    residues: Dict or str {"massivekb", "canonical"}, optional
        The amino acid dictionary and their masses. By default this is only
        the 20 canonical amino acids, with cysteine carbamidomethylated. If
        "massivekb", this dictionary will include the modifications found in
        MassIVE-KB. Additionally, a dictionary can be used to specify a custom
        collection of amino acids and masses.
    """

    canonical = {
        "G": 57.021463735,
        "A": 71.037113805,
        "S": 87.032028435,
        "P": 97.052763875,
        "V": 99.068413945,
        "T": 101.047678505,
        "C+57.021": 103.009184505 + 57.02146,
        "C": 103.009184505,
        "L": 113.084064015,
        "I": 113.084064015,
        "N": 114.042927470,
        "D": 115.026943065,
        "Q": 128.058577540,
        "K": 128.094963050,
        "E": 129.042593135,
        "M": 131.040484645,
        "H": 137.058911875,
        "F": 147.068413945,
        # "U": 150.953633405,
        "R": 156.101111050,
        "Y": 163.063328575,
        "W": 186.079312980,
        # "O": 237.147726925,
    }

    # Modfications found in MassIVE-KB
    massivekb = {
        # N-terminal mods:
        "+42.011": 42.010565,  # Acetylation
        "+43.006": 43.005814,  # Carbamylation
        "-17.027": -17.026549,  # NH3 loss
        "+43.006-17.027": (43.006814 - 17.026549),
        # AA mods:
        "M+15.995": canonical["M"] + 15.994915,  # Met Oxidation
        "N+0.984": canonical["N"] + 0.984016,  # Asn Deamidation
        "Q+0.984": canonical["Q"] + 0.984016,  # Gln Deamidation
    }

    # Constants
    hydrogen = 1.007825035
    oxygen = 15.99491463
    h2o = 2 * hydrogen + oxygen
    proton = 1.00727646688

    def __init__(self, residues="canonical"):
        """Initialize the PeptideMass object"""
        if residues == "canonical":
            self.masses = self.canonical
        elif residues == "massivekb":
            self.masses = self.canonical
            self.masses.update(self.massivekb)
        else:
            self.masses = residues

    def __len__(self):
        """Return the length of the residue dictionary"""
        return len(self.masses)

    def mass(self, seq, charge=None):
        """Calculate a peptide's mass or m/z.

        Parameters
        ----------
        seq : list or str
            The peptide sequence, using tokens defined in ``self.residues``.
        charge : int, optional
            The charge used to compute m/z. Otherwise the neutral peptide mass
            is calculated

        Returns
        -------
        float
            The computed mass or m/z.
        """
        if isinstance(seq, str):
            seq = re.split(r"(?<=.)(?=[A-Z])", seq)

        calc_mass = sum([self.masses[aa] for aa in seq]) + self.h2o
        if charge is not None:
            calc_mass = (calc_mass / charge) + self.proton

        return calc_mass
