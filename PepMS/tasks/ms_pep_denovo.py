# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
#


import argparse
import math
import os
import pickle

import numpy as np
import torch
from unicore.data import EpochShuffleDataset, NestedDictionaryDataset
from unicore.tasks import UnicoreTask, register_task

from PepMS.data import (
    BatchIndexDataset,
    ClusteredDataset,
    ContrastIs2reDataset,
    DeNovoIs2reDataset,
    InferenceZipLMDB2Dataset,
    LMDB2Dataset,
    SampleZipLMDB2Dataset,
    ScalerDataset,
    SeqDataset,
    SeqStrDataset,
    ZipLMDB2Dataset,
    pFindIs2reDataset,
)


@register_task("denovo_inference")
class DeNovoProteinTask(UnicoreTask):
    @staticmethod
    def add_args(parser):
        parser.add_argument("data", metavar="FILE", help="file prefix for data")
        parser.add_argument("--fdr-thread", default=0.1, type=float)
        parser.add_argument("--pkl-path", default="", type=str)
        parser.add_argument("--check", default="", type=int)

    def __init__(self, args):
        super().__init__(args)
        self.seed = args.seed

    @classmethod
    def setup_task(cls, args, **kwargs):
        return cls(args)

    def load_dataset(self, split, combine=False, **kwargs):
        # assert split in [
        #     "train",
        #     "valid",
        # ], "Not Implemented: {}!".format(split)
        print(" > Loading {} ...".format(split))
        dataset_dict = {}
        dataset_dict["net_input"] = {}
        dataset_dict["targets"] = {}

        # db_path = os.path.join(self.args.data, f"{split}.lmdb")
        db_path = os.path.join(self.args.data, f"{split}.lmdb")

        # if self.args.
        lmdb_dataset = InferenceZipLMDB2Dataset(
            db_path,
            split,
            check=self.args.check,
            fdr_thread=self.args.fdr_thread,
            pkl_path=self.args.pkl_path,
            spec_pred=self.args.spectrum_pred,
        )

        is_train = split == "train" or split == "valid"

        is2re_dataset = DeNovoIs2reDataset(lmdb_dataset, self.args, is_train=is_train)

        dataset_dict["net_input"]["mz_array"] = SeqDataset(is2re_dataset, "mz_array")
        dataset_dict["net_input"]["intensity"] = SeqDataset(is2re_dataset, "intensity")
        dataset_dict["net_input"]["residue_type"] = SeqStrDataset(
            is2re_dataset, "residue_type"
        )
        dataset_dict["net_input"]["residue_type_after_mod"] = SeqStrDataset(
            is2re_dataset, "residue_type_after_mod"
        )
        dataset_dict["net_input"]["residue_type_after_mod_all"] = SeqStrDataset(
            is2re_dataset, "residue_type_after_mod_all"
        )
        dataset_dict["net_input"]["modification"] = SeqStrDataset(
            is2re_dataset, "modification"
        )
        dataset_dict["net_input"]["precursor_mz"] = ScalerDataset(
            is2re_dataset, "precursor_mz"
        )
        dataset_dict["net_input"]["precursor_charge"] = ScalerDataset(
            is2re_dataset, "precursor_charge"
        )
        dataset_dict["net_input"]["instrument"] = ScalerDataset(
            is2re_dataset, "instrument"
        )
        dataset_dict["net_input"]["nce"] = ScalerDataset(is2re_dataset, "nce")
        dataset_dict["net_input"]["RT"] = ScalerDataset(is2re_dataset, "RT")
        dataset_dict["net_input"]["idx"] = ScalerDataset(is2re_dataset, "idx")
        dataset_dict["net_input"]["batch_index"] = BatchIndexDataset(
            is2re_dataset, "batch_index"
        )

        dataset_dict["net_input"]["index"] = ScalerDataset(is2re_dataset, "idx")
        dataset_dict["net_input"]["fdr"] = ScalerDataset(is2re_dataset, "fdr")
        dataset_dict["net_input"]["titles"] = SeqStrDataset(is2re_dataset, "title")

        dataset_dict["targets"]["seq_len_label"] = SeqDataset(
            is2re_dataset, "seq_len_label"
        )
        dataset_dict["net_input"]["repeat_num"] = ScalerDataset(
            is2re_dataset, "repeat_num"
        )
        if self.args.spectrum_pred:
            dataset_dict["net_input"]["spec_label"] = SeqDataset(
                is2re_dataset, "spec_label"
            )

        dataset = NestedDictionaryDataset(
            dataset_dict,
            # {
            #     "net_input": {
            #         "mz_array": mz_array,
            #         "intensity": intensity,
            #         "residue_type": residue_type,
            #         "precursor_mz": precursor_mz,
            #         "precursor_charge": precursor_charge,
            #         "modification": modification,
            #         "batch_index": batch_index,
            #         "instrument": instrument,
            #         "residue_type_after_mod": residue_type_after_mod,
            #         "nce": nce,
            #         "denoise_pred_mask": denoise_pred_mask,
            #         "intensity_pred_mask": intensity_pred_mask,
            #         "rope_embeding": rope_embeding,
            #         "label_index":
            #         # "nce": nce,
            #         # "ins": ins,
            #         # "rt_normalized": rt_normalized,
            #     },
            #     "targets": {
            #         "residue_type_after_mod": residue_type_after_mod,
            #         "spec_label": spec_label,
            #         "denoise_pred_mask": denoise_pred_mask,
            #         "intensity_pred_mask": intensity_pred_mask,
            #         "noise_peak_mask": noise_peak_mask,
            #         "denoise_label": denoise_label,
            #         "intensity_label": intensity_label,
            #         "noise_peak_label": noise_peak_label,
            #         "mod_label": mod_label,
            #         "ion_res_num": ion_res_num,
            #         "ion_comp_res_num": ion_comp_res_num,
            #         "modification_cls_label": modification_cls_label,
            #         # "batch_index": batch_index
            #     },
            # },
        )

        if split == "train":
            dataset = EpochShuffleDataset(
                dataset,
                size=len(dataset),
                seed=self.args.data_seed,
            )

        print("| Loaded {} with {} samples".format(split, len(dataset)))

        self.datasets[split] = dataset

    def build_model(self, args):
        from unicore import models

        model = models.build_model(args, self)
        return model
