# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


import logging
import math
import re
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from spectrum_utils.utils import mass_diff
from torch import Tensor
from compat import utils
from compat.models import BaseUnicoreModel, register_model, register_model_architecture

from PepMS.data.ms_pep import get_spectrum_prediction_label
from PepMS.models.decoders import (
    NonARPeptideDecoder,
    PepMsJointEncoder,
    PeptideDecoder,
    RankDecoder,
    SpectrumEncoder,
)

from .layers import CLSHead, NonLinear, PeptideEncoder, PeptideMass  # SpectrumEncoder,

logger = logging.getLogger(__name__)


def get_top_indices(ranklist, ranking_indices, ranklist_mask, batch_index):
    logits = ranklist[:, 1:, :]
    ranklist_mask = (~ranklist_mask.bool()).squeeze(-1).float()
    logits[ranklist_mask == 0] = -10000
    ranklist_mask = ranklist_mask.reshape(-1).unsqueeze(-1)
    pred_labels = logits.squeeze(-1).argmax(
        dim=1
    )  # Get the index of the max logit score
    # breakpoint()
    expanded_b = pred_labels.unsqueeze(1).expand(-1, ranking_indices.size(1))
    result = torch.gather(ranking_indices, 1, expanded_b)
    indices = result[:, 0]
    result_indices = torch.empty_like(indices)

    # 计算每个谱最好的样本在该谱中的相对位置
    for i in range(len(indices)):
        batch = batch_index[indices[i]]  # 该样本属于哪个谱
        mask = batch_index == batch  # 找出该谱的所有样本
        relative_position = mask[: indices[i]].sum()  # 在这些样本中找到它的位置
        result_indices[i] = relative_position
    return result_indices


def convert_submodule_to_half(module: nn.Module):
    """将指定模块及其所有子模块转为 FP16"""
    # 转换当前模块
    module.half()
    # 递归转换所有子模块
    for child in module.children():
        convert_submodule_to_half(child)


@register_model("pep_ms")
class MDProteinModel(BaseUnicoreModel):
    @staticmethod
    def add_args(parser):
        """Add model-specific arguments to the parser."""
        parser.add_argument(
            "--node-dim",
            default=512,
            type=int,
        )
        parser.add_argument(
            "--num-head",
            default=8,
            type=int,
        )
        parser.add_argument(
            "--dim-forward",
            default=1024,
            type=int,
        )
        parser.add_argument(
            "--num-layers",
            default=12,
            type=int,
        )
        parser.add_argument(
            "--cutoff-peptide",
            default=100,
            type=int,
        )
        parser.add_argument(
            "--cutoff-spectra",
            default=200,
            type=int,
        )
        parser.add_argument(
            "--debug",
            default=0,
            type=int,
        )
        parser.add_argument(
            "--use-nce",
            default=0,
            type=int,
        )
        parser.add_argument(
            "--use-ins",
            default=0,
            type=int,
        )
        parser.add_argument(
            "--use-rt",
            default=0,
            type=int,
        )
        parser.add_argument(
            "--pred-nce",
            default=0,
            type=int,
        )
        parser.add_argument(
            "--pred-ins",
            default=0,
            type=int,
        )
        parser.add_argument(
            "--pred-rt",
            default=0,
            type=int,
        )
        parser.add_argument(
            "--inference",
            default=0,
            type=int,
        )
        parser.add_argument(
            "--sample-test",
            default=1000,
            type=int,
        )
        parser.add_argument(
            "--denovo-pred",
            default=0,
            type=int,
        )
        parser.add_argument(
            "--use-clip",
            default=0,
            type=int,
        )
        parser.add_argument(
            "--use-hard-clip",
            default=0,
            type=int,
        )
        parser.add_argument(
            "--spectrum-pred",
            default=0,
            type=int,
        )

        parser.add_argument(
            "--triplet-loss-weight",
            default=1,
            type=float,
        )
        parser.add_argument(
            "--denovo-loss-weight",
            default=1,
            type=float,
        )
        parser.add_argument(
            "--spectrum-loss-weight",
            default=1,
            type=float,
        )
        parser.add_argument(
            "--peak-loss-weight",
            default=0.25,
            type=float,
        )
        parser.add_argument(
            "--num-res-loss-weight",
            default=0.25,
            type=float,
        )
        parser.add_argument(
            "--clip-loss-weight",
            default=0.1,
            type=float,
        )
        parser.add_argument(
            "--modification-loss-weight",
            default=1,
            type=float,
        )
        # TODO
        parser.add_argument(
            "--random-mask-prob",
            default=0,
            type=float,
        )
        parser.add_argument(
            "--intensity-pred",
            default=0,
            type=int,
        )
        parser.add_argument(
            "--intensity-prob",
            default=0.1,
            type=int,
        )
        parser.add_argument(
            "--denoise-pred",
            default=0,
            type=int,
        )
        parser.add_argument(
            "--denoise-prob",
            default=0.1,
            type=int,
        )
        parser.add_argument(
            "--dataset-name",
            default="",
            type=str,
        )
        parser.add_argument(
            "--enforce-triplet-loss",
            default=0,
            type=int,
        )
        parser.add_argument(
            "--use-rope",
            default=0,
            type=int,
        )
        parser.add_argument(
            "--noise-peak-pred",
            default=0,
            type=int,
        )
        parser.add_argument(
            "--multi-class-noise-peak-pred",
            default=1,
            type=int,
        )
        parser.add_argument(
            "--max-charges",
            default=2,
            type=int,
        )
        parser.add_argument(
            "--half-spectrum-pred",
            default=0,
            type=int,
        )
        parser.add_argument(
            "--from-scratch",
            default=0,
            type=int,
        )
        parser.add_argument(
            "--modification-pred",
            default=0,
            type=int,
        )
        parser.add_argument(
            "--modification-cls-pred",
            default=0,
            type=int,
        )
        parser.add_argument(
            "--modification-mass-pred",
            default=0,
            type=int,
        )
        parser.add_argument(
            "--norm-first",
            default=0,
            type=int,
        )

        parser.add_argument(
            "--massive",
            default=0,
            type=int,
        )
        parser.add_argument(
            "--res-num-pred",
            default=0,
            type=int,
        )
        parser.add_argument(
            "--res-seq-len-pred",
            default=0,
            type=int,
        )
        parser.add_argument(
            "--res-type-pred",
            default=0,
            type=int,
        )
        parser.add_argument(
            "--comp-res-num-pred",
            default=0,
            type=int,
        )
        parser.add_argument(
            "--no-mod",
            default=0,
            type=int,
        )
        parser.add_argument(
            "--train-pfind",
            default=0,
            type=int,
        )
        parser.add_argument(
            "--shift",
            default=0,
            type=int,
        )
        # Uni-Mol
        parser.add_argument(
            "--even-op",
            default=3,
            type=int,
        )
        parser.add_argument(
            "--pair-dim",
            default=64,
            type=int,
        )
        parser.add_argument(
            "--use-unimol-encoder",
            default=0,
            type=int,
        )
        parser.add_argument(
            "--head-clip",
            default=0,
            type=int,
        )
        parser.add_argument(
            "--non-ar",
            default=1,
            type=int,
        )
        parser.add_argument(
            "--max-prec-charge",
            default=10,
            type=int,
        )
        parser.add_argument(
            "--hard-neg",
            default=1,
            type=int,
        )
        parser.add_argument(
            "--num-round-pred",
            default=3,
            type=int,
        )
        parser.add_argument(
            "--tokenize-mod",
            default=1,
            type=int,
        )
        parser.add_argument(
            "--joint-encoding",
            default=0,
            type=int,
        )
        parser.add_argument(
            "--joint-encoding-layers",
            default=4,
            type=int,
        )
        parser.add_argument(
            "--num-pept-layers",
            default=9,
            type=int,
        )
        parser.add_argument(
            "--modification-meta-dict-path",
            default=r"./consts/modification_meta_dict.pkl",
            type=str,
        )
        parser.add_argument(
            "--tokenize1-pkl-path",
            default=r"./consts/tokenize1.pkl",
            type=str,
        )
        parser.add_argument(
            "--start-id",
            default=3,
            type=int,
        )
        parser.add_argument(
            "--unfreeze-step",
            default=-1,
            type=int,
        )
        parser.add_argument(
            "--range-pred",
            default=5,
            type=int,
        )
        parser.add_argument(
            "--tof",
            default=0,
            type=int,
        )
        parser.add_argument(
            "--only-decoy",
            default=0,
            type=int,
        )

        parser.add_argument(
            "--use-rankloss",
            default=1,
            type=int,
        )
        parser.add_argument(
            "--rank-encoding-layers",
            default=4,
            type=int,
        )
        parser.add_argument(
            "--rank-loss-weight",
            default=10,
            type=int,
        )
        parser.add_argument(
            "--special-rank-loss",
            default=0,
            type=int,
        )
        parser.add_argument(
            "--weighted",
            default=0,
            type=int,
        )
        parser.add_argument(
            "--dropout",
            default=0.18,
            type=float,
        )
        parser.add_argument(
            "--ckpt-file",
            default="",
            type=str,
        )

    def __init__(self, args):
        super().__init__()
        self.args = args
        self.dtype = torch.float32
        self.max_length = 110
        residues = {
            "G": 57.021464,
            "A": 71.037114,
            "S": 87.032028,
            "P": 97.052764,
            "V": 99.068414,
            "T": 101.047670,
            "C": 103.009185,
            "L": 113.084064,
            "I": 113.084064,
            "N": 114.042927,
            "D": 115.026943,
            "Q": 128.058578,
            "K": 128.094963,
            "E": 129.042593,
            "M": 131.040485,
            "H": 137.058912,
            "F": 147.068414,
            "R": 156.101111,
            "Y": 163.063329,
            "W": 186.079313,
        }
        self.residues = residues

        fix_mods = {}

        self.encoder = SpectrumEncoder(
            dim_model=self.args.node_dim,
            n_head=self.args.num_head,
            dim_feedforward=self.args.dim_forward,
            n_layers=self.args.num_layers,
            dropout=self.args.dropout,
            use_ins=self.args.use_ins,
            use_nce=self.args.use_nce,
            use_rope=self.args.use_rope,
            norm_first=self.args.norm_first == 1,
        )

        self.peptide_encoder = PeptideEncoder(
            args=args,
            max_len=self.args.cutoff_peptide,
            max_charge=self.args.max_prec_charge,
            residues=residues,
            use_ins=self.args.use_ins,
            use_nce=self.args.use_nce,
            ptransformer_width=self.args.node_dim,
            ptransformer_heads=self.args.num_head,
            ptransformer_layers=self.args.num_pept_layers,
            norm_first=self.args.norm_first == 1,
            tokenize_mod=self.args.tokenize_mod,
            dropout=0.18,
            dtype=torch.float32 if not self.args.fp16 else torch.half,
        )

        self.num_instrument = 10
        self.max_charge = 10
        self.residue_type = 23

        if self.args.denovo_pred:
            if not self.args.non_ar:
                self.decoder = PeptideDecoder(
                    args=args,
                    dim_model=args.node_dim,
                    n_head=args.num_head,
                    dim_feedforward=self.args.dim_forward,
                    n_layers=args.num_layers,
                    dropout=0.18,
                    residues=residues,
                    max_charge=self.args.max_prec_charge,
                    norm_first=self.args.norm_first == 1,
                    modification_pred=self.args.modification_pred,
                )
                self.softmax = torch.nn.Softmax(2)
                self.stop_token = self.decoder._aa2idx["$"]
                self.peptide_mass_calculator = PeptideMass(residues)
                self.n_beams = 5
                self.isotope_error_range = (0, 1)
                self.precursor_mass_tol = 50
                self.max_length = 100
            else:
                self.nonar_decoder = NonARPeptideDecoder(
                    args=args,
                    dim_model=args.node_dim,
                    n_head=args.num_head,
                    dim_feedforward=self.args.dim_forward,
                    n_layers=args.num_layers,
                    dropout=0.18,
                    residues=residues,
                    max_charge=self.args.max_prec_charge,
                    norm_first=self.args.norm_first == 1,
                    modification_pred=self.args.modification_pred,
                    num_round_pred=self.args.num_round_pred,
                    modification_mass_pred=self.args.modification_mass_pred,
                    tokenize_mod=self.args.tokenize_mod,
                    fix_mods=fix_mods,
                )
                # self.nonar_decoder = self.peptide_encoder
                self.softmax = torch.nn.Softmax(2)
                self.peptide_mass_calculator = PeptideMass(residues)
                self.n_beams = 5
                self.isotope_error_range = (0, 1)
                self.precursor_mass_tol = 50
                self.max_length = 100

        if self.args.spectrum_pred:
            if not self.args.no_mod:
                self.spectrum_decoder = NonLinear(
                    args.node_dim, self.args.max_charges * 8
                )
            else:
                self.spectrum_decoder = NonLinear(
                    args.node_dim, self.args.max_charges * 6
                )

        if self.args.noise_peak_pred:
            if not self.args.multi_class_noise_peak_pred:
                self.noise_peak_cls = CLSHead(input_dim=args.node_dim, output_cls=2)
            else:
                self.noise_peak_cls = CLSHead(
                    input_dim=args.node_dim, output_cls=self.args.max_charges * 8 + 1
                )

        if self.args.res_seq_len_pred:
            self.num_seq_head = CLSHead(input_dim=args.node_dim, output_cls=110)

        if self.args.joint_encoding:
            self.joint_encoder = PepMsJointEncoder(
                dim_model=args.node_dim,
                n_head=args.num_head,
                dim_feedforward=self.args.dim_forward,
                n_layers=args.joint_encoding_layers,
            )

        self.ion_types = [
            "b",
            "b-NH3",
            "b-H2O",
            "b-ModLoss",
            "y",
            "y-NH3",
            "y-H2O",
            "y-ModLoss",
        ]

    def half(self):
        super().half()
        self.dtype = torch.half
        return self

    def bfloat16(self):
        self.encoder = self.encoder.bfloat16()
        self.peptide_encoder = self.peptide_encoder.bfloat16()
        super().bfloat16()
        self.dtype = torch.bfloat16
        return self

    def float(self):
        self.encoder = self.encoder.float()
        self.peptide_encoder = self.peptide_encoder.float()
        super().float()
        self.dtype = torch.float32
        return self

    def forward_score(
        self,
        mz_array: torch.Tensor,
        intensity: torch.Tensor,
        residue_type: torch.Tensor,
        residue_type_after_mod: torch.Tensor,
        residue_type_after_mod_all,
        precursor_mz: torch.Tensor,
        precursor_charge: torch.Tensor,
        modification: List[Tuple],
        instrument: torch.Tensor,
        nce: torch.Tensor,
        batch_index: torch.Tensor = None,
        intensity_pred_mask: torch.Tensor = None,
        denoise_pred_mask: torch.Tensor = None,
        rope_embeding: torch.Tensor = None,
        label_index: torch.Tensor = None,
        repeat_num: torch.Tensor = None,
        RT_input: torch.Tensor = None,
        **kwargs,
    ):
        return_dict = {}
        batch_index = batch_index.long()
        first_appear = False
        if (
            self.args.use_clip
            and not self.args.use_hard_clip
            and not self.args.train_pfind
        ):
            residue_type = residue_type[::2]
            modification = modification[::2]
            batch_index = batch_index[::2]

        mz_array = mz_array.type(self.dtype)
        intensity = intensity.type(self.dtype)
        precursor_mz = precursor_mz.type(self.dtype)
        precursor_charge = precursor_charge.long()
        spectra = torch.cat([mz_array.unsqueeze(-1), intensity.unsqueeze(-1)], dim=-1)
        precursor = torch.cat(
            [precursor_mz.unsqueeze(-1), precursor_charge.unsqueeze(-1)], dim=-1
        )
        spectra_feat_all, spectra_mask = self.encoder(
            spectra, precursor, instrument, nce, rope_embeding
        )
        peptide_feat_all, peptide_mask = self.peptide_encoder(
            residue_type,
            residue_type_after_mod_all,
            precursor,
            instrument,
            nce,
            modification,
            batch_index,
        )
        return_dict["scores"] = self.joint_encoder(
            peptide_feat_all,
            peptide_mask,
            spectra_feat_all,
            spectra_mask,
            repeat_num,
            None,
        )
        return_dict["top_indices"] = None
        return return_dict

    def forward_faster_denovo(
        self,
        titles: List[str],
        mz_array: torch.Tensor,
        intensity: torch.Tensor,
        residue_type_after_mod: torch.Tensor,
        residue_type_after_mod_all,
        precursor_mz: torch.Tensor,
        precursor_charge: torch.Tensor,
        modification: List[Tuple],
        instrument: torch.Tensor,
        RT: torch.Tensor,
        nce: torch.Tensor,
        batch_index: torch.Tensor = None,
        rope_embeding: torch.Tensor = None,
        repeat_num: torch.Tensor = None,
        min_length: int = 7,
        max_length: int = 40,
        **kwargs,
    ):
        self.nonar_decoder.activate_demod()
        batch_index = batch_index.long()
        mz_array = mz_array.type(self.dtype)
        intensity = intensity.type(self.dtype)
        precursor_mz = precursor_mz.type(self.dtype)
        precursor_charge = precursor_charge.long()
        # nce = nce.type(self.dtype)
        # ins = ins.type(self.dtype)

        spectra = torch.cat([mz_array.unsqueeze(-1), intensity.unsqueeze(-1)], dim=-1)
        precursor = torch.cat(
            [precursor_mz.unsqueeze(-1), precursor_charge.unsqueeze(-1)], dim=-1
        )
        spectra_feat_all, spectra_mask = self.encoder(
            spectra, precursor, instrument, nce, rope_embeding
        )

        # num_seq_pred = torch.Tensor([len(_) for _ in residue_type_after_mod]).to(spectra.device).long()
        num_seq_pred = self.num_seq_head(spectra_feat_all[:, 1])
        _, num_seq_pred = torch.max(num_seq_pred, dim=1)  # B, 1
        num_seq_pred_original = num_seq_pred.clone()

        expanded_tensor = num_seq_pred.unsqueeze(1).repeat(1, self.args.range_pred)
        # offset = torch.tensor([-1, 0, 1]).to(expanded_tensor.device)
        offset = (
            torch.arange(0, self.args.range_pred).long().to(expanded_tensor.device)
            - self.args.range_pred // 2
        )
        result_tensor = expanded_tensor + offset
        num_seq_pred = result_tensor.flatten()  # 3B, 1

        valid_length_mask = (
            (min_length <= num_seq_pred) * (num_seq_pred <= max_length)
        ).bool()
        # print(num_seq_pred, torch.Tensor([len(_) for _ in residue_type_after_mod]).to(spectra.device).long())

        defualt_input = ["A" * _ for _ in num_seq_pred]
        spectra_feat_all_ = torch.repeat_interleave(
            spectra_feat_all, self.args.range_pred, dim=0
        )
        spectra_mask_ = torch.repeat_interleave(
            spectra_mask, self.args.range_pred, dim=0
        )
        precursor_ = torch.repeat_interleave(precursor, self.args.range_pred, dim=0)
        precursor_mz_ = torch.repeat_interleave(
            precursor_mz, self.args.range_pred, dim=0
        )

        spectra_feat_all_ = spectra_feat_all_[valid_length_mask]
        spectra_mask_ = spectra_mask_[valid_length_mask]
        precursor_ = precursor_[valid_length_mask]
        precursor_mz_ = precursor_mz_[valid_length_mask]

        str_np_array = np.array(defualt_input)
        defualt_input_ = [
            s for s, mask in zip(defualt_input, valid_length_mask) if mask
        ]
        if len(defualt_input_) == 0:
            assert valid_length_mask.sum() == 0, valid_mass_shift_mask.sum()
            return None, None, None

        rets = self.nonar_decoder(
            defualt_input_, precursor_, spectra_feat_all_, spectra_mask_
        )  # 3B

        predAndtruth, peptide_mask_, modification_preds_ = rets[-1]  # B
        modification_preds_ = modification_preds_[0]
        pred, _ = predAndtruth
        # print(pred.shape, truth.shape, peptide_mask.shape)
        tokens = torch.argmax(pred, axis=2)
        residue_type_pred = []

        modification = []
        modification_name = []
        mods = []
        mod_stat = {}
        for idx in range(tokens.size()[0]):
            # peptides_true.append(''.join(tokens_true)[:-1])
            tokens_pred = tokens[idx, :]
            residue_type_pred.append(self.nonar_decoder.detokenize(tokens_pred))
            mod, mod_name, mod4spec = self.nonar_decoder.detokenize_mod_encode(
                residue_type_pred[-1],
                modification_preds_[idx],
                len(residue_type_pred[-1]) - 1,
            )
            modification.append(mod)
            modification_name.append(mod_name)
            mods.append(mod4spec)
            for mod_n in mod_name:
                if mod_n in mod_stat.keys():
                    mod_stat[mod_n] += 1
                else:
                    mod_stat[mod_n] = 1

        # for i in range(len(residue_type_after_mod)):
        #     print(["".join(_) for _ in residue_type_pred][i* 3 + 1], residue_type_after_mod[i])
        #     print(modification[i* 3 + 1])
        continue_residue_type_pred = ["".join(_[:-1]) for _ in residue_type_pred]
        peptide_set = set()
        for seq in continue_residue_type_pred:
            peptide_set.add(seq)
        # print(">>>", continue_residue_type_pred)

        valid_mass_shift_mask, mass_shift = self._get_score_mask(
            precursor_mz_, continue_residue_type_pred, modification
        )

        continue_residue_type_pred = [
            s
            for s, mask in zip(continue_residue_type_pred, valid_mass_shift_mask)
            if mask
        ]
        modification = [
            s for s, mask in zip(modification, valid_mass_shift_mask) if mask
        ]
        precursor_ = precursor_[valid_mass_shift_mask]
        batch_index_ = batch_index[valid_length_mask]
        batch_index_ = batch_index_[valid_mass_shift_mask]
        if len(continue_residue_type_pred) == 0:
            assert valid_mass_shift_mask.sum() == 0, valid_mass_shift_mask.sum()
            return None, None, None

        peptide_feat_all, peptide_mask = self.peptide_encoder(
            continue_residue_type_pred,
            continue_residue_type_pred,
            precursor,
            instrument,
            nce,
            modification,
            batch_index_,
        )
        spec_pred = self.spectrum_decoder(peptide_feat_all)

        # Vectorized repeat_num computation (replaces per-sample GPU→CPU loop)
        num_samples = len(precursor)
        repeat_num_counts = torch.bincount(batch_index_, minlength=num_samples)[:num_samples]
        sample_mask = (repeat_num_counts > 0).cpu()  # CPU for downstream indexing
        assert repeat_num_counts.sum().item() == len(peptide_feat_all), (
            repeat_num_counts.sum().item(), len(peptide_feat_all)
        )
        repeat_num_ = repeat_num_counts.float().to(peptide_feat_all.device)
        scores = (
            self.joint_encoder(
                peptide_feat_all,
                peptide_mask,
                spectra_feat_all,
                spectra_mask,
                repeat_num_,
            )
            .squeeze(-1)
            .cpu()
        )

        final_valid_mask = torch.zeros_like(valid_length_mask, dtype=torch.bool).to(
            valid_mass_shift_mask.device
        )
        final_valid_mask[valid_length_mask] = valid_mass_shift_mask

        flat_valid_indices = torch.nonzero(final_valid_mask).squeeze(-1)  # [M]

        batch_size = len(precursor)
        range_pred = self.args.range_pred
        original_batch_idx = torch.arange(batch_size, device=scores.device)
        original_batch_idx = original_batch_idx.repeat_interleave(range_pred)[
            flat_valid_indices
        ]

        scores_all = (
            torch.zeros_like(valid_length_mask, dtype=torch.float32).to(
                valid_mass_shift_mask.device
            )
            - 50
        )
        scores_all[flat_valid_indices] = scores  # [M]

        _, best_match_index = torch.max(
            scores_all.unsqueeze(-1).reshape(-1, self.args.range_pred), dim=1
        )  #  B

        best_match_index += torch.arange(len(best_match_index), device=best_match_index.device) * range_pred

        best_match_index = best_match_index[sample_mask]  # sample * 5 -> return
        scores_max = scores_all[best_match_index]  # done

        # after length -> return
        len_valid_index = torch.arange(len(precursor_mz_))
        valid_length_mask = valid_length_mask.cpu()
        after_length_index = torch.zeros_like(valid_length_mask, dtype=torch.long) - 1
        after_length_index[valid_length_mask] = len_valid_index
        after_length_index = after_length_index[best_match_index]

        peptide_mask = peptide_mask.cpu()
        spec_pred = spec_pred.cpu()
        tokens = tokens[after_length_index]
        mass_shift = mass_shift.reshape(-1)[after_length_index]  #
        modification_preds_ = modification_preds_[after_length_index]  #

        mods_input = []
        mod_pred = []
        # mods = mods[after_length_index]
        mods_ = []
        for idx in after_length_index:
            mods_.append(mods[idx.item()])

        for i in range(len(best_match_index)):
            mods_input.append(mods_[i])
            m_str = ""
            for item in mods_[i]:
                m_str += f"{item[0]},{item[1]};"
            mod_pred.append(m_str)
        cos_sims = []
        miss_cleav = []

        len_valid_index = torch.arange(len(precursor_))
        valid_mass_mask_ = valid_length_mask.clone()
        length_indices = torch.nonzero(valid_length_mask, as_tuple=True)[0]
        mass_shift_false_subset = torch.nonzero(~valid_mass_shift_mask, as_tuple=True)[
            0
        ]
        combined_false = length_indices[mass_shift_false_subset]
        valid_mass_mask_[combined_false] = False
        # valid_mass_mask_[valid_length_mask][~valid_mass_shift_mask] = False
        after_length_index = torch.zeros_like(valid_length_mask, dtype=torch.long) - 1
        after_length_index[valid_mass_mask_] = len_valid_index
        after_length_index = after_length_index[best_match_index]
        peptide_mask = ~peptide_mask[after_length_index]
        spec_pred = spec_pred[after_length_index]

        residue_type_after_mod = [
            s for s, mask in zip(residue_type_after_mod, sample_mask) if mask
        ]
        residue_type_after_mod_all = [
            s for s, mask in zip(residue_type_after_mod_all, sample_mask) if mask
        ]
        recall_info = self._cal_infer_recall(
            tokens,
            residue_type_after_mod,
            modification_preds_,
            residue_type_after_mod_all,
        )
        recall_info["mod_pred"] = mod_pred
        recall_info["title"] = [s for s, mask in zip(titles, sample_mask) if mask]
        recall_info["mass_shift"] = mass_shift
        recall_info["score"] = scores_max
        recall_info["RT"] = [s for s, mask in zip(RT, sample_mask) if mask]
        # peptide level
        # peptide+mod level

        peptides_pred = recall_info["peptides_pred"]

        # Batch transfer mz_array and intensity to CPU once (instead of per-sample transfers)
        mz_array_cpu = mz_array.cpu()
        intensity_cpu = intensity.cpu()
        precursor_charge_cpu = precursor_charge.cpu()

        j = 0
        for i in range(len(mz_array_cpu)):
            if not sample_mask[i]:
                continue
            charge = precursor_charge_cpu[i]
            mz_i = mz_array_cpu[i]
            int_i = intensity_cpu[i]

            # Call with shift=True for cos_sim
            spec_exp, _, _, _, _, _, _ = get_spectrum_prediction_label(
                peptides_pred[j][0],
                mz_i,
                int_i,
                mods_input[j],
                self.ion_types,
                self.args.max_charges,
                flag_ITMS=False,
                shift=True,
                return_mz=True,
            )
            if charge < 4:
                spec_exp = spec_exp.reshape(spec_exp.shape[0], -1, 4)
                spec_exp = spec_exp[:, :, :charge].reshape(spec_exp.shape[0], -1)
            spec_pred_cal = spec_pred[j][1 : spec_exp.shape[0] + 1]
            if charge < 4:
                spec_pred_cal = spec_pred_cal.reshape(spec_pred_cal.shape[0], -1, 4)
                spec_pred_cal = spec_pred_cal[:, :, :charge].reshape(
                    spec_pred_cal.shape[0], -1
                )
            spec_pred_cal = spec_pred_cal.reshape(1, -1)
            cos_sims.append(
                F.cosine_similarity(
                    torch.Tensor(spec_exp).reshape(1, -1), spec_pred_cal
                )
            )

            # Call with shift=False for miss_cleav (reuse mz_i, int_i)
            spec_exp2, _, _, _, _, _, ions = get_spectrum_prediction_label(
                peptides_pred[j][0],
                mz_i,
                int_i,
                mods_input[j],
                self.ion_types,
                self.args.max_charges,
                flag_ITMS=False,
                shift=False,
                return_mz=True,
            )
            if charge < 4:
                spec_exp2 = spec_exp2.reshape(spec_exp2.shape[0], -1, 4)
                spec_exp2 = spec_exp2[:, :, :charge].reshape(spec_exp2.shape[0], -1)
            missed = torch.Tensor(spec_exp2).sum(-1) < 1e-3
            miss_cleav.append(
                "_".join(map(str, torch.nonzero(missed).squeeze(-1).tolist()))
            )
            j += 1

        recall_info["cos_sim"] = cos_sims
        recall_info["miss_cleav"] = miss_cleav
        assert (
            len(miss_cleav)
            == len(cos_sims)
            == len(scores_max)
            == len(recall_info["title"])
            == len(mass_shift)
            == len(peptides_pred)
            == len(recall_info["peptides_pred"])
        ), f"len(miss_cleav): {len(miss_cleav)}, len(cos_sims): {len(cos_sims)}, len(scores_max): {len(scores_max)}, len(titles): {len(titles)}, len(mass_shift): {len(mass_shift)}, len(peptides_pred): {len(peptides_pred)}"
        # modify range
        return recall_info, peptide_set, mod_stat

    def _precompute_mass_shift(
        self, precursor_mass, peptides, modifications, indices, range_pred
    ):
        mass_H2O = 2 * 1.0078246 + 15.9949141
        batch_idx = indices // range_pred
        pos_idx = indices % range_pred

        peptide_mass = torch.stack(
            [
                torch.sum(
                    torch.tensor([self.residues[c] for c in pep], dtype=torch.float32)
                )
                + sum(mod[1][1] for mod in mods)
                + mass_H2O
                for pep, mods in zip(peptides, [modifications[i] for i in indices])
            ]
        ).to(precursor_mass.device)

        return peptide_mass - precursor_mass[batch_idx]

    def _get_score_mask(
        self,
        precursor_mass: torch.Tensor,  # B
        seqs: List[str],  # B * 5
        modification,
    ):
        mass_shift = torch.zeros(len(seqs))
        mass_H2O = 2 * 1.0078246 + 15.9949141
        # print(precursor_mass.shape)
        precursor_mass = precursor_mass.cpu()
        mass_qualified = []
        for i in range(len(seqs)):
            seq = seqs[i]
            mods = modification[i]
            mass_seq = sum([self.residues[_] for _ in seq])
            mass_seq += sum([_[1][1] for _ in mods]) + mass_H2O
            mass_shift[i] = abs(precursor_mass[i] - mass_seq)
            if abs(precursor_mass[i] - mass_seq).item() > (mass_seq * 20 / 1e6):
                mass_qualified.append(False)
            else:
                mass_qualified.append(True)
        return torch.Tensor(mass_qualified).bool(), mass_shift

    def _refine_score(
        self,
        precursor_mass: torch.Tensor,  # B
        seqs: List[str],  # B * 5
        modification,
        score: torch.Tensor,
        range_pred,
    ):
        alert = torch.zeros(score.shape[0]).bool()
        mass_shift = torch.zeros(score.shape)
        mass_H2O = 2 * 1.0078246 + 15.9949141
        # print(precursor_mass.shape)
        precursor_mass = precursor_mass.cpu()
        for i in range(len(seqs)):
            seq = seqs[i]
            mods = modification[i]
            mass_seq = sum([self.residues[_] for _ in seq])
            mass_seq += sum([_[1][1] for _ in mods]) + mass_H2O
            mass_shift[i // range_pred, i % range_pred] = abs(
                precursor_mass[i // range_pred] - mass_seq
            )
            if abs(precursor_mass[i // range_pred] - mass_seq).item() > (
                mass_seq * 20 / 1e6
            ):
                score[i // range_pred, i % range_pred] -= 50
            else:
                alert[i // range_pred] = True
        # print(f"alert ratio: {torch.sum(alert) / score.shape[0]}")
        return score, mass_shift

    def _cal_recall(self, tokens, truth):
        peptides_pred = []
        peptides_true = []
        for idx in range(tokens.size()[0]):
            tokens_true = truth[idx, :]
            try:
                tokens_true = self.decoder.detokenize(tokens_true)[1:]
            except:
                tokens_true = self.nonar_decoder.detokenize(tokens_true)
            peptides_true.append("".join(tokens_true)[:-1])
            tokens_pred = tokens[idx, :]
            try:
                tokens_pred = self.decoder.detokenize(tokens_pred)
            except:
                tokens_pred = self.nonar_decoder.detokenize(tokens_pred)
            if tokens_pred[0] == "$":
                # print("strange $ ")
                tokens_pred = tokens_pred[1:]  # Remove stop token.
            peptides_pred.append(tokens_pred[:-1])
            # peptides_pred = peptides_pred[0]
        # assert 0, (["".join(_) for _ in peptides_pred], peptides_true,residue_type_after_mod, tokens, truth)
        # assert 0, ("".join(peptides_pred), peptides_true)
        try:
            aa_precision, aa_recall, pep_recall = aa_match_metrics(
                *aa_match_batch(
                    peptides_pred, peptides_true, self.decoder._peptide_mass.masses
                )
            )
        except:
            aa_precision, aa_recall, pep_recall = aa_match_metrics(
                *aa_match_batch(
                    peptides_pred,
                    peptides_true,
                    self.nonar_decoder._peptide_mass.masses,
                )
            )
        return aa_precision, aa_recall, pep_recall

    def _cal_infer_recall(
        self, tokens, peptides_true, modification_preds_, peptide_mod_true
    ):
        assert modification_preds_.shape[0] == len(peptides_true) == len(tokens), (
            modification_preds_.shape,
            len(peptides_true),
            len(tokens),
        )
        peptides_pred = []
        # peptides_true = []
        peptide_mod_pred = []
        # peptide_mod_true = []

        for idx in range(tokens.size()[0]):
            # peptides_true.append(''.join(tokens_true)[:-1])
            tokens_pred = tokens[idx, :]
            tokens_pred = self.nonar_decoder.detokenize(tokens_pred)[:-1]
            peptides_pred.append(tokens_pred)
            peptide_mod_pred.append(
                "".join(tokens_pred)
                + self.nonar_decoder.detokenize_mod(
                    modification_preds_[idx], len(tokens_pred) - 1
                )
            )
        for i in range(len(peptides_true)):
            peptides_true[i] = [_.replace("I", "L") for _ in peptides_true[i]]
            peptides_pred[i] = [_.replace("I", "L") for _ in peptides_pred[i]]
        # peptides_true = [_.replace("I", "L") for _ in peptides_true]
        # print(peptides_true, peptides_pred)
        aa_precision, aa_recall, pep_recall = aa_match_metrics(
            *aa_match_batch(
                peptides_pred, peptides_true, self.nonar_decoder._peptide_mass.masses
            )
        )
        for i in range(len(peptides_true)):
            peptides_true[i] = ["".join(peptides_true[i])]
            peptides_pred[i] = ["".join(peptides_pred[i])]
            # print(peptides_true[i], peptides_pred[i], num_seq_pred_original[i].item(), len(peptides_true[i][0]), len(peptides_pred[i][0]), peptide_mod_true[i], peptide_mod_pred[i])
        ret_info = {
            "aa_recall": aa_recall,
            "pep_recall": pep_recall,
            "peptides_true": peptides_true,
            "peptides_pred": peptides_pred,
            "peptide_mod_pred": peptide_mod_pred,
            "peptide_mod_true": peptide_mod_true,
        }
        # assert 0, (ret_info["aa_recall"], ret_info["pep_recall"])
        return ret_info

    @classmethod
    def build_model(cls, args, task):
        """Build a new model instance."""
        return cls(args)

    def set_num_updates(self, num_updates):
        """State from trainer to pass along to model at every update."""
        self._num_updates = num_updates


def aa_match_prefix(
    peptide1: List[str],
    peptide2: List[str],
    aa_dict: Dict[str, float],
    cum_mass_threshold: float = 0.5,
    ind_mass_threshold: float = 0.1,
) -> Tuple[np.ndarray, bool]:
    """
    Find the matching prefix amino acids between two peptide sequences.

    This is a similar evaluation criterion as used by DeepNovo.

    Parameters
    ----------
    peptide1 : List[str]
        The first tokenized peptide sequence to be compared.
    peptide2 : List[str]
        The second tokenized peptide sequence to be compared.
    aa_dict : Dict[str, float]
        Mapping of amino acid tokens to their mass values.
    cum_mass_threshold : float
        Mass threshold in Dalton to accept cumulative mass-matching amino acid
        sequences.
    ind_mass_threshold : float
        Mass threshold in Dalton to accept individual mass-matching amino acids.

    Returns
    -------
    aa_matches : np.ndarray of length max(len(peptide1), len(peptide2))
        Boolean flag indicating whether each paired-up amino acid matches across
        both peptide sequences.
    pep_match : bool
        Boolean flag to indicate whether the two peptide sequences fully match.
    """
    aa_matches = np.zeros(max(len(peptide1), len(peptide2)), np.bool_)
    # Find longest mass-matching prefix.
    i1, i2, cum_mass1, cum_mass2 = 0, 0, 0.0, 0.0
    while i1 < len(peptide1) and i2 < len(peptide2):
        aa_mass1 = aa_dict.get(peptide1[i1], 0)
        aa_mass2 = aa_dict.get(peptide2[i2], 0)
        if (
            abs(mass_diff(cum_mass1 + aa_mass1, cum_mass2 + aa_mass2, True))
            < cum_mass_threshold
        ):
            aa_matches[max(i1, i2)] = (
                abs(mass_diff(aa_mass1, aa_mass2, True)) < ind_mass_threshold
            )
            i1, i2 = i1 + 1, i2 + 1
            cum_mass1, cum_mass2 = cum_mass1 + aa_mass1, cum_mass2 + aa_mass2
        elif cum_mass2 + aa_mass2 > cum_mass1 + aa_mass1:
            i1, cum_mass1 = i1 + 1, cum_mass1 + aa_mass1
        else:
            i2, cum_mass2 = i2 + 1, cum_mass2 + aa_mass2
    return aa_matches, aa_matches.all()


def aa_match_prefix_suffix(
    peptide1: List[str],
    peptide2: List[str],
    aa_dict: Dict[str, float],
    cum_mass_threshold: float = 0.5,
    ind_mass_threshold: float = 0.1,
) -> Tuple[np.ndarray, bool]:
    """
    Find the matching prefix and suffix amino acids between two peptide
    sequences.

    Parameters
    ----------
    peptide1 : List[str]
        The first tokenized peptide sequence to be compared.
    peptide2 : List[str]
        The second tokenized peptide sequence to be compared.
    aa_dict : Dict[str, float]
        Mapping of amino acid tokens to their mass values.
    cum_mass_threshold : float
        Mass threshold in Dalton to accept cumulative mass-matching amino acid
        sequences.
    ind_mass_threshold : float
        Mass threshold in Dalton to accept individual mass-matching amino acids.

    Returns
    -------
    aa_matches : np.ndarray of length max(len(peptide1), len(peptide2))
        Boolean flag indicating whether each paired-up amino acid matches across
        both peptide sequences.
    pep_match : bool
        Boolean flag to indicate whether the two peptide sequences fully match.
    """
    # Find longest mass-matching prefix.
    aa_matches, pep_match = aa_match_prefix(
        peptide1, peptide2, aa_dict, cum_mass_threshold, ind_mass_threshold
    )
    # No need to evaluate the suffixes if the sequences already fully match.
    if pep_match:
        return aa_matches, pep_match
    # Find longest mass-matching suffix.
    i1, i2 = len(peptide1) - 1, len(peptide2) - 1
    i_stop = np.argwhere(~aa_matches)[0]
    cum_mass1, cum_mass2 = 0.0, 0.0
    while i1 >= i_stop and i2 >= i_stop:
        aa_mass1 = aa_dict.get(peptide1[i1], 0)
        aa_mass2 = aa_dict.get(peptide2[i2], 0)
        if (
            abs(mass_diff(cum_mass1 + aa_mass1, cum_mass2 + aa_mass2, True))
            < cum_mass_threshold
        ):
            aa_matches[max(i1, i2)] = (
                abs(mass_diff(aa_mass1, aa_mass2, True)) < ind_mass_threshold
            )
            i1, i2 = i1 - 1, i2 - 1
            cum_mass1, cum_mass2 = cum_mass1 + aa_mass1, cum_mass2 + aa_mass2
        elif cum_mass2 + aa_mass2 > cum_mass1 + aa_mass1:
            i1, cum_mass1 = i1 - 1, cum_mass1 + aa_mass1
        else:
            i2, cum_mass2 = i2 - 1, cum_mass2 + aa_mass2
    return aa_matches, aa_matches.all()


def aa_match(
    peptide1: List[str],
    peptide2: List[str],
    aa_dict: Dict[str, float],
    cum_mass_threshold: float = 0.5,
    ind_mass_threshold: float = 0.1,
    mode: str = "best",
) -> Tuple[np.ndarray, bool]:
    """
    Find the matching amino acids between two peptide sequences.

    Parameters
    ----------
    peptide1 : List[str]
        The first tokenized peptide sequence to be compared.
    peptide2 : List[str]
        The second tokenized peptide sequence to be compared.
    aa_dict : Dict[str, float]
        Mapping of amino acid tokens to their mass values.
    cum_mass_threshold : float
        Mass threshold in Dalton to accept cumulative mass-matching amino acid
        sequences.
    ind_mass_threshold : float
        Mass threshold in Dalton to accept individual mass-matching amino acids.
    mode : {"best", "forward", "backward"}
        The direction in which to find matching amino acids.

    Returns
    -------
    aa_matches : np.ndarray of length max(len(peptide1), len(peptide2))
        Boolean flag indicating whether each paired-up amino acid matches across
        both peptide sequences.
    pep_match : bool
        Boolean flag to indicate whether the two peptide sequences fully match.
    """
    if mode == "best":
        return aa_match_prefix_suffix(
            peptide1, peptide2, aa_dict, cum_mass_threshold, ind_mass_threshold
        )
    elif mode == "forward":
        return aa_match_prefix(
            peptide1, peptide2, aa_dict, cum_mass_threshold, ind_mass_threshold
        )
    elif mode == "backward":
        aa_matches, pep_match = aa_match_prefix(
            list(reversed(peptide1)),
            list(reversed(peptide2)),
            aa_dict,
            cum_mass_threshold,
            ind_mass_threshold,
        )
        return aa_matches[::-1], pep_match
    else:
        raise ValueError("Unknown evaluation mode")


def aa_match_batch(
    peptides1: Iterable,
    peptides2: Iterable,
    aa_dict: Dict[str, float],
    cum_mass_threshold: float = 0.5,
    ind_mass_threshold: float = 0.1,
    mode: str = "best",
) -> Tuple[List[Tuple[np.ndarray, bool]], int, int]:
    """
    Find the matching amino acids between multiple pairs of peptide sequences.

    Parameters
    ----------
    peptides1 : Iterable
        The first list of peptide sequences to be compared.
    peptides2 : Iterable
        The second list of peptide sequences to be compared.
    aa_dict : Dict[str, float]
        Mapping of amino acid tokens to their mass values.
    cum_mass_threshold : float
        Mass threshold in Dalton to accept cumulative mass-matching amino acid
        sequences.
    ind_mass_threshold : float
        Mass threshold in Dalton to accept individual mass-matching amino acids.
    mode : {"best", "forward", "backward"}
        The direction in which to find matching amino acids.

    Returns
    -------
    aa_matches_batch : List[Tuple[np.ndarray, bool]]
        For each pair of peptide sequences: (i) boolean flags indicating whether
        each paired-up amino acid matches across both peptide sequences, (ii)
        boolean flag to indicate whether the two peptide sequences fully match.
    n_aa1: int
        Total number of amino acids in the first list of peptide sequences.
    n_aa2: int
        Total number of amino acids in the second list of peptide sequences.
    """
    aa_matches_batch, n_aa1, n_aa2 = [], 0, 0
    for peptide1, peptide2 in zip(peptides1, peptides2):
        # Split peptides into individual AAs if necessary.
        if isinstance(peptide1, str):
            peptide1 = re.split(r"(?<=.)(?=[A-Z])", peptide1)
        if isinstance(peptide2, str):
            peptide2 = re.split(r"(?<=.)(?=[A-Z])", peptide2)
        n_aa1, n_aa2 = n_aa1 + len(peptide1), n_aa2 + len(peptide2)
        aa_matches_batch.append(
            aa_match(
                peptide1,
                peptide2,
                aa_dict,
                cum_mass_threshold,
                ind_mass_threshold,
                mode,
            )
        )
    return aa_matches_batch, n_aa1, n_aa2


def aa_match_metrics(
    aa_matches_batch: List[Tuple[np.ndarray, bool]],
    n_aa_true: int,
    n_aa_pred: int,
) -> Tuple[float, float, float]:
    """
    Calculate amino acid and peptide-level evaluation metrics.

    Parameters
    ----------
    aa_matches_batch : List[Tuple[np.ndarray, bool]]
        For each pair of peptide sequences: (i) boolean flags indicating whether
        each paired-up amino acid matches across both peptide sequences, (ii)
        boolean flag to indicate whether the two peptide sequences fully match.
    n_aa_true: int
        Total number of amino acids in the true peptide sequences.
    n_aa_pred: int
        Total number of amino acids in the predicted peptide sequences.

    Returns
    -------
    aa_precision: float
        The number of correct AA predictions divided by the number of predicted
        AAs.
    aa_recall: float
        The number of correct AA predictions divided by the number of true AAs.
    pep_precision: float
        The number of correct peptide predictions divided by the number of
        peptides.
    """
    n_aa_correct = sum([aa_matches[0].sum() for aa_matches in aa_matches_batch])
    aa_precision = n_aa_correct / (n_aa_pred + 1e-8)
    aa_recall = n_aa_correct / (n_aa_true + 1e-8)
    pep_precision = sum([aa_matches[1] for aa_matches in aa_matches_batch]) / (
        len(aa_matches_batch) + 1e-8
    )
    return aa_precision, aa_recall, pep_precision


def aa_precision_recall(
    aa_scores_correct: List[float],
    aa_scores_all: List[float],
    n_aa_total: int,
    threshold: float,
) -> Tuple[float, float]:
    """
    Calculate amino acid level precision and recall at a given score threshold.

    Parameters
    ----------
    aa_scores_correct : List[float]
        Amino acids scores for the correct amino acids predictions.
    aa_scores_all : List[float]
        Amino acid scores for all amino acids predictions.
    n_aa_total : int
        The total number of amino acids in the predicted peptide sequences.
    threshold : float
        The amino acid score threshold.

    Returns
    -------
    aa_precision: float
        The number of correct amino acid predictions divided by the number of
        predicted amino acids.
    aa_recall: float
        The number of correct amino acid predictions divided by the total number
        of amino acids.
    """
    n_aa_correct = sum([score > threshold for score in aa_scores_correct])
    n_aa_predicted = sum([score > threshold for score in aa_scores_all])
    return n_aa_correct / n_aa_predicted, n_aa_correct / n_aa_total


@register_model_architecture("pep_ms", "pep_ms")
def base_architecture(args):
    args.data_seed = getattr(args, "data_seed", args.seed)
