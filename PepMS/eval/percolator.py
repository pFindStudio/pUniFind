import copy
import os
import pickle
from multiprocessing import Pool
from os.path import join

import numba
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

# import matplotlib.pyplot as plt
import torch.nn.init as init
from tqdm import tqdm

from compat.parquet_storage import ParquetReader, get_storage_path
from PepMS.eval.eval_draw import *
from PepMS.eval.process_result import *

# from alphabase.peptide.fragment import get_charged_frag_types
# from alphabase.io.psm_reader import psm_reader_provider

# from peptdeep.rescore.feature_extractor import (
#     ScoreFeatureExtractor,
#     ScoreFeatureExtractorMP
# )

# from peptdeep.rescore.fdr import (
#     fdr_from_ref, fdr_to_q_values, calc_fdr_for_df
# )

# from peptdeep.pretrained_models import ModelManager

# from peptdeep.utils import logging


def try_remove(paths):
    for path in paths:
        try:
            os.remove(path)
        except:
            pass


class PercolatorConfig:
    def __init__(
        self,
        run_name,
        dataset_name,
        debug_mode=True,
        fdr_threashold=0.1,
        prefix="",
        processes=64,
        key_pkl_path=None,
        res_path_base=r"/mnt/vepfs/fs_ckps/zhaojiale/dataset/mol_spec/tmp",
        lmdb_path_base=r"/mnt/vepfs/fs_ckps/zhaojiale/dataset/mol_spec/dataset/lmdbs_full",
        middle_path=r"/mnt/vepfs/fs_users/zhaojiale/Contrast_MS_Pep/results/stat_results",
        use_rank=False,
        debug=False,
        reset_pfind=True,
        reset_pscore=True,
        use_joint_scores=True,
        use_pred_spec=False,
        mgf_path_root=None,
        res_path=None,
    ):
        self.debug = debug
        self.debug_raw = None
        self.run_name = run_name
        self.dataset_name = dataset_name
        self.fdr_threashold = fdr_threashold
        self.prefix = prefix
        self.processes = processes
        if res_path is None:
            self.res_path = join(res_path_base, run_name, "results")
        else:
            self.res_path = res_path
        if key_pkl_path is None:
            self.key_pkl_path = join(lmdb_path_base, "pkls")
        else:
            self.key_pkl_path = key_pkl_path
        self.use_joint_scores = use_joint_scores
        self.use_pred_spec = use_pred_spec
        self.mgf_path_root = mgf_path_root
        self.use_rank = use_rank

        self.middle_path = middle_path
        self.pfind_result_middle_path = join(
            middle_path, f"pfind_res_{run_name}_{dataset_name}_{fdr_threashold}.pkl"
        )
        self.pscore_result_middle_path = join(
            middle_path, f"pscore_res_{run_name}_{dataset_name}_{fdr_threashold}.pkl"
        )
        self.pscore_per_result_middle_path = join(
            middle_path,
            f"pscore_per_res_{run_name}_{dataset_name}_{fdr_threashold}.pkl",
        )
        self.per_filtered_result_middle_path = join(
            middle_path,
            f"per_filtered_res_{run_name}_{dataset_name}_{fdr_threashold}.pkl",
        )

        self.debug_pfind_result_middle_path = join(
            middle_path, f"pfind_debug_res_{run_name}_{dataset_name}.pkl"
        )
        self.debug_pscore_result_middle_path = join(
            middle_path, f"pscore_debug_res_{run_name}_{dataset_name}.pkl"
        )
        self.debug_pscore_per_result_middle_path = join(
            middle_path, f"pscore_per_debug_res_{run_name}_{dataset_name}.pkl"
        )
        self.debug_per_filtered_result_middle_path = join(
            middle_path, f"per_filtered_debug_res_{run_name}_{dataset_name}.pkl"
        )

        if reset_pscore:
            try_remove(
                [
                    self.pscore_result_middle_path,
                    self.pscore_per_result_middle_path,
                    self.per_filtered_result_middle_path,
                    self.debug_pscore_result_middle_path,
                    self.debug_pscore_per_result_middle_path,
                    self.debug_per_filtered_result_middle_path,
                ]
            )

        if reset_pfind:
            try_remove([self.pfind_result_middle_path])

        self.lmdb_path_base = lmdb_path_base
        self.lmdb_path = get_storage_path(join(lmdb_path_base, dataset_name + ".lmdb"))

        self.debug_mode = debug_mode

        files = os.listdir(self.res_path)
        self.files = [
            _ for _ in files if f"{dataset_name}_{dataset_name}" in _ and ".pkl" in _
        ]
        print("result files: ", self.files)


perc_settings = {
    "percolator_iter_num": 5,
    "cv_fold": 1,
    "percolator_model": "linear",
    "percolator_backend": "pytorch",
    "lr_percolator_torch_model": 0.05,
    "frag_types": [
        "b",
        "b-NH3",
        "b-H2O",
        "b-ModLoss",
        "y",
        "y-NH3",
        "y-H2O",
        "y-ModLoss",
    ],
    "charges": 4,
    "fdr": 0.01,
    "multiprocessing": True,
    "max_perc_train_sample": 1000000,
    "min_perc_train_sample": 100,
    "use_fdr_for_each_raw": False,
}

global_settings = {
    "peak_matching": {
        "ms2_ppm": 20,
        "ms2_tol_value": 0.5,
    },
}


@numba.njit
def fdr_to_q_values(fdr_values: np.ndarray) -> np.ndarray:
    """convert FDR values to q_values.

    Parameters
    ----------
    fdr_values : np.ndarray
        FDR values, they should be
        sorted according to the descending order of the `score`

    Returns
    -------
    np.ndarray
        q_values

    """
    q_values = np.zeros_like(fdr_values)
    min_q_value = np.max(fdr_values)
    for i in range(len(fdr_values) - 1, -1, -1):
        fdr = fdr_values[i]
        if fdr < min_q_value:
            min_q_value = fdr
        q_values[i] = min_q_value
    return q_values


class LogisticRegressionTorch(torch.nn.Module):
    """Torch-based rescore model"""

    def __init__(self, input_dim, **kwargs):
        super().__init__()
        torch.manual_seed(1337)
        self.linear = torch.nn.Linear(input_dim, 1)

    def forward(self, x):
        return self.linear(x).squeeze(1)

    def mode_init(self):
        # 遍历模块中的每一层，并应用 Xavier 初始化
        for m in self.modules():
            if isinstance(m, nn.Linear):
                init.xavier_uniform_(m.weight)  # Xavier 均匀分布初始化
                if m.bias is not None:
                    init.zeros_(m.bias)  # 将偏置初始化为0


class NNRescore:
    def __init__(self, feature_list):
        self.nn_model: nn.Module = LogisticRegressionTorch(len(feature_list))
        self.train_batch_size = 10000
        self.infer_batch_size = 100000

        self.optimizer = torch.optim.Adam(
            self.nn_model.parameters(), lr=perc_settings["lr_percolator_torch_model"]
        )
        self.loss_func = torch.nn.BCEWithLogitsLoss()

        if torch.cuda.is_available():
            self.device = torch.device("cuda")
            self.nn_model.to(self.device)
        else:
            self.device = torch.device("cpu")
        self.epoch = 10

    def collator(self, data_to_collator, infer=False) -> torch.Tensor:
        """
        "title": data_lmdb[k]["title"],
        "index": data["index"],
        "peptide": peptide,
        "spectrum_score": spectrum_score,
        "pfind_score": pfind_score,
        "is_target": is_target,
        "pscore_score": data["score"],
        "pfind_qvalue": qvalue,
        """
        inputs = []
        labels = []
        batch_index = []
        idx = 0

        for data in data_to_collator:
            # print(data)
            # print(torch.Tensor([data["pfind_score"][0]]).unsqueeze(0).shape)
            # print(torch.Tensor([data["pscore_score"][0]]).unsqueeze(0).shape)
            # print(torch.Tensor(data["spectrum_score"][0]).unsqueeze(0).shape)
            # print(torch.Tensor([data["is_target"]]).shape)
            # try:
            if infer:
                data = pickle.loads(data)
            # except:
            #     pass
            if not infer:
                inputs.append(
                    torch.cat(
                        [
                            torch.Tensor([data["pfind_score"][0]]).unsqueeze(0),
                            torch.Tensor([data["pscore_score"][0]]).unsqueeze(0),
                            torch.Tensor(data["spectrum_score"][0]).unsqueeze(0),
                        ],
                        dim=1,
                    )
                )
                labels.append(torch.Tensor([data["is_target"][0]]))
            else:
                # print(torch.Tensor(data["pfind_score"]).shape)
                # print(torch.Tensor(data["pscore_score"]).shape)
                # print(torch.cat([_.unsqueeze(0) for _ in data["spectrum_score"]], dim=0).shape)
                inputs.append(
                    torch.cat(
                        [
                            torch.Tensor(data["pfind_score"]).unsqueeze(1),
                            torch.Tensor(data["pscore_score"]).unsqueeze(1),
                            torch.cat(
                                [_.unsqueeze(0) for _ in data["spectrum_score"]], dim=0
                            ),
                        ],
                        dim=1,
                    )
                )
                labels.append(torch.Tensor(data["is_target"]))
            batch_index.append(torch.ones(len(labels[-1])) * idx)
            idx += 1
        inputs = torch.cat(inputs, dim=0)
        labels = torch.cat(labels, dim=0)
        batch_index = torch.cat(batch_index, dim=0)
        assert inputs.shape[0] == labels.shape[0] == batch_index.shape[0], (
            inputs.shape,
            labels.shape,
            batch_index.shape,
        )
        return inputs, labels, batch_index.long()

    def train(self, percolator_data_all, valid_ratio=0.1):
        self.nn_model.mode_init()
        num_samples = len(percolator_data_all)
        num_train = int((1 - valid_ratio) * num_samples)
        percolator_data = percolator_data_all[:num_train]
        percolator_data_valid = percolator_data_all[num_train:]
        for _ in range(self.epoch):
            print(f"training epoch {_}")
            sample_idxes = (
                np.random.RandomState(1337)
                .permutation(len(percolator_data))
                .astype(np.int64)
            )
            self.nn_model.train()
            for i in tqdm(
                range(0, len(percolator_data), self.train_batch_size),
                total=len(percolator_data) // self.train_batch_size,
            ):
                self.optimizer.zero_grad()
                # print(len(percolator_data), self.train_batch_size, int(i), int(min(i+self.train_batch_size, len(percolator_data))), sample_idxes)
                data_input, labels, batch_index = self.collator(
                    [
                        percolator_data[_]
                        for _ in sample_idxes[
                            int(i) : int(
                                min(i + self.train_batch_size, len(percolator_data))
                            )
                        ]
                    ]
                )

                outputs = self.nn_model(
                    data_input,
                )
                loss = self.loss_func(outputs, labels)
                loss.backward()
                print(f"training loss epoch {_} batch {i}: {loss.detach().item()}")
                self.optimizer.step()

            # valid
            self.nn_model.eval()
            data_input_valid, labels_valid, batch_index_valid = self.collator(
                percolator_data_valid
            )
            outputs_valid = self.nn_model(
                data_input_valid,
            )
            loss = self.loss_func(outputs_valid, labels_valid)
            print(f"valid loss epoch {_}: {loss.item()}")

    def infer(self, data_with_pscore_feature):
        outputs = []
        # batch_indexs = []
        for i in tqdm(
            range(0, len(data_with_pscore_feature), self.infer_batch_size),
            total=len(data_with_pscore_feature) // self.infer_batch_size,
        ):
            self.optimizer.zero_grad()
            data_input, labels, batch_index = self.collator(
                data_with_pscore_feature[
                    i : min(i + self.train_batch_size, len(data_with_pscore_feature))
                ],
                infer=True,
            )
            pred = self.nn_model(
                data_input,
            )
            # batch_indexs.append(batch_index)

            for j in range(batch_index[-1] + 1):
                label_ = labels[batch_index == j]
                pred_ = pred[batch_index == j]

                idx = torch.argmax(pred_)
                score_max = pred_[idx]
                outputs.append((score_max, label_[idx]))
        return outputs

    def train_and_infer(self, percolator_data, data_with_pscore_feature):
        pscore_per_results = {}

        for raw in percolator_data.keys():
            self.train(percolator_data[raw])
            pscore_per_results[raw] = self.infer(data_with_pscore_feature[raw])

        return pscore_per_results


class Percolator:
    """Percolator model.
    In parameter list, perc_settings is
    ```
    perc_settings = peptdeep.settings.global_settings['percolator']
    ```
    """

    def __init__(
        self,
        config: PercolatorConfig,
        percolator_model: str = perc_settings["percolator_model"],
        percolator_backend: str = perc_settings["percolator_backend"],
        cv_fold: int = perc_settings["cv_fold"],
        iter_num: int = perc_settings["percolator_iter_num"],
        ms2_ppm: bool = global_settings["peak_matching"]["ms2_ppm"],
        ms2_tol: float = global_settings["peak_matching"]["ms2_tol_value"],
    ):
        """
        Parameters
        ----------
        percolator_model : str, optional
            machine learning
            model type for rescoring, could be:
            "linear": logistic regression
            "random_forest": random forest
            Defaults to perc_settings['percolator_model'].

        percolator_backend : str, optional
            `sklearn` or `pytorch`.
            Defaults to perc_settings['percolator_backend']

        cv_fold : int, optional
            cross-validation fold.
            Defaults to perc_settings['cv_fold'].

        iter_num : int, optional
            percolator iteration number.
            Defaults to perc_settings['percolator_iter_num'].

        ms2_ppm : bool, optional
            is ms2 tolerance the ppm.
            Defaults to perc_settings['ms2_ppm'].

        ms2_tol : float, optional
            ms2 tolerance.
            Defaults to perc_settings['ms2_tol'].

        model_mgr : ModelManager, optional
            peptdeep.pretrained_model.ModelManager.
            If None, self.model_mgr will be init by default (see `peptdeep.pretrained_models.ModelManager`).
            Defaults to None.
        """

        self.config = config
        self.charged_frag_types = perc_settings["frag_types"]
        self.ms2_ppm = ms2_ppm
        self.ms2_tol = ms2_tol
        # self.fdr_level = perc_settings['fdr_level']
        self.fdr = perc_settings["fdr"]
        self.cv_fold = cv_fold
        self.iter_num = iter_num

        self.feature_list = [
            "pfind",
            "pscore",
            "pcc",
            "cos",
            "sa",
            "spc",
        ]
        # self.feature_list += ['score','nAA','charge']

        self.max_train_sample = perc_settings["max_perc_train_sample"]
        self.min_train_sample = perc_settings["min_perc_train_sample"]
        self.per_raw_fdr = perc_settings["use_fdr_for_each_raw"]

        self.model = NNRescore(self.feature_list)

    def load_data(self):
        """Load data from Parquet storage."""
        self._reader = ParquetReader(self.config.lmdb_path)
        filtered_keys = pickle.load(
            open(
                join(
                    self.config.key_pkl_path,
                    f"{self.config.dataset_name}_FDR{self.config.fdr_threashold}_keys.pkl",
                ),
                "rb",
            )
        )
        self.lmdb_keys = filtered_keys

    def get_pfind_results(self):
        pfind_results = {}
        pfind_best_score_dict = {}

        pfind_result_middle_path = self.config.pfind_result_middle_path

        if not self.config.debug_mode:
            try:
                os.remove(join(pfind_result_middle_path))
            except:
                pass

        def input_pfind():
            for key in self.lmdb_keys:
                yield key, self._reader.get(key), False

        set_all_001 = {}
        if os.path.exists(pfind_result_middle_path):
            pfind_results = pickle.load(open(pfind_result_middle_path, "rb"))
        else:
            # with Pool(self.config.processes) as pool:
            #     for ret in tqdm(
            #         pool.imap(get_pfind_result, input_pfind(), chunksize=10), total=len(self.lmdb_keys)
            #     ):
            if True:
                for item in tqdm(input_pfind(), total=len(self.lmdb_keys)):
                    ret = get_pfind_result(item)

                    res, raw_name = ret
                    if raw_name in pfind_results.keys():
                        pfind_results[raw_name].append(res)
                        # if res[2] in pfind_best_score_dict[raw_name].keys():
                        #     pfind_best_score_dict[raw_name][res[2]] = min(res[-1], pfind_best_score_dict[raw_name][res[2]])
                        # else:
                        #     pfind_best_score_dict[raw_name][res[2]] = res[-1]
                    else:
                        pfind_results[raw_name] = [res]
                        # pfind_best_score_dict[raw_name] = {res[2]: res[-1]}
                    if res[0] < 0.01:
                        if raw_name in set_all_001.keys():
                            set_all_001[raw_name].add(res[2])
                        # except:
                        else:
                            set_all_001[raw_name] = set()
                            set_all_001[raw_name].add(res[2])

            for raw_name in pfind_results.keys():
                pfind_results[raw_name] = sorted(
                    pfind_results[raw_name], key=lambda x: x[-1]
                )

            # ##
            # for raw in pfind_results.keys():
            #     for i in range(len(pfind_results[raw])):
            #         pep_str = pfind_results[raw][i][2]
            #         pfind_results[raw][i][-1] = pfind_best_score_dict[raw][pep_str]
            # for raw_name in pfind_results.keys():
            #     pfind_results[raw_name] = sorted(pfind_results[raw_name], key=lambda x: x[-1])
            pfind_results = process_pfind_qvalue(pfind_results)
            # ##
            # assert 0, len(set_all_001)
            # for raw_name in set_all_001.keys():
            #     tmp = set()
            #     for item in pfind_results[raw_name]:
            #         if item[0] < 0.01:
            #             tmp.add(item[2])
            #     print("pFind 0.01: ", raw_name, len(tmp), len(set_all_001[raw_name]), list(set_all_001[raw_name])[:4])
            # assert 0
            pickle.dump(pfind_results, open(pfind_result_middle_path, "wb"))
        return pfind_results

    def merge_pfind_results(self, pfind_results):
        all_pfind_results = {"all": []}

        for k in pfind_results.keys():
            all_pfind_results["all"].extend(pfind_results[k])
        all_pfind_results["all"] = sorted(all_pfind_results["all"], key=lambda x: x[-1])
        all_pfind_results = process_pfind_qvalue(all_pfind_results)

        set_all_001 = set()
        for item in all_pfind_results["all"]:
            if item[0] <= 0.01:
                set_all_001.add(item[2])
        # assert 0, len(set_all_001)
        # print(all_pfind_results["all"][:10])
        return all_pfind_results["all"], set_all_001

    def get_pscore_feat_and_results(self):
        """
        return:
            pscore_results:
                raw_name:
                    score,
                    is_target
            pscore_results_detail:
                raw_name:
                    spectrum name,
                    peptide pfind,
                    peptide pscore,
                    modification pfind,
                    modification pscore,
                    pfind fdr,
                    pscore fdr,
                    pfind_is_target,
                    pscore_is_target
            features:
                item:
                    title,
                    peptide,
                    index,
                    fdr,
                    spectrum_score,
                    pscore_score,
                    pfind_score,
                    pfind_qvalue,
        """

        # pscore_result_middle_path = self.config.pscore_result_middle_path
        if not self.config.debug:
            pscore_result_middle_path = self.config.pscore_result_middle_path
        else:
            pscore_result_middle_path = self.config.debug_pscore_result_middle_path

        if os.path.exists(pscore_result_middle_path):
            pscore_results, features, pscore_results_detail = pickle.load(
                open(pscore_result_middle_path, "rb")
            )
        else:
            features = {}
            pscore_results = {}
            pscore_results_detail = {}
            pscore_output = {}
            mgf_infos = {}

            for file in self.config.files:
                with open(os.path.join(self.config.res_path, file), "rb") as f:
                    pscore_output.update(pickle.load(f))
            pscore_keys = list(pscore_output.keys())

            print("num keys", len(pscore_keys), len(self.lmdb_keys))
            assert len(pscore_keys) == len(self.lmdb_keys), (
                f"Inference results ({len(pscore_keys)}) != dataset entries ({len(self.lmdb_keys)}). "
                f"Possibly duplicate keys or incomplete inference."
            )

            def input_pscore():
                for pscore_key in pscore_keys:
                    raw = pscore_output[pscore_key]
                    # Support both raw dicts (new format) and pickled bytes (legacy)
                    pscore_res = raw if isinstance(raw, dict) else safe_unpickle(raw)
                    idx = pscore_res["index"]
                    yield pscore_res, pscore_key, self._reader.get(
                        self.lmdb_keys[idx]
                    ), False, self.config

                # for key in keys:
                # yield key, txn_read, large_flag

            print("num samples:", len(pscore_output.keys()))

            # with Pool(self.config.processes) as pool:
            #     for ret in tqdm(
            #         pool.imap(get_pscore_feat_and_result, input_pscore(), chunksize=10), total=len(pscore_keys)
            #     ):
            if True:
                for item in tqdm(input_pscore(), total=len(pscore_keys)):
                    ret = get_pscore_feat_and_result(item)

                    # pass
                    res, raw_name, qvalue, feat, detail, info, peptide_str = ret

                    # if self.config.debug_raw is None:
                    #     self.config.debug_raw = raw_name
                    #     flag_in = True
                    # else:
                    #     if self.config.debug_raw == raw_name:
                    #         flag_in = True
                    #     else:
                    #         flag_in = False
                    # if not self.config.debug or flag_in:

                    # if qvalue < 1 and (not self.config.debug or flag_in):
                    if qvalue < 1:
                        if raw_name not in pscore_results.keys():
                            pscore_results[raw_name] = {}

                        if peptide_str not in pscore_results[raw_name].keys():
                            pscore_results[raw_name][peptide_str] = {
                                "score": res[0],
                                "num_psm": 1,
                                "is_target": res[1],
                            }

                        else:
                            if pscore_results[raw_name][peptide_str]["score"] < res[0]:
                                pscore_results[raw_name][peptide_str]["score"] = res[0]

                            pscore_results[raw_name][peptide_str]["num_psm"] += 1

                        if raw_name not in pscore_results_detail.keys():
                            pscore_results_detail[raw_name] = [detail]
                        else:
                            pscore_results_detail[raw_name].append(detail)

                        if info is not None:
                            if raw_name in mgf_infos.keys():
                                mgf_infos[raw_name][detail["spectrum_name"]] = info
                            else:
                                mgf_infos[raw_name] = {detail["spectrum_name"]: info}

                        # if raw_name in features.keys():
                        #     features[raw_name].append(feat)
                        # else:
                        #     features[raw_name] = [feat]
            # greater_target, greater_decoy = 0, 0

            # for raw in pscore_results_detail.keys():
            #     for i in range(len(pscore_results_detail[raw])):
            #         pep_str = pscore_results_detail[raw][i]["peptide_str"]
            #         pscore_results_detail[raw][i]["pscore_max_score"] = pscore_results[raw][pep_str]["score"]

            """
            pscore_results:
                raw_name:
                    peptide_str:
                        score: score_max
                        num_psm: number of psm
                        is_target: True/False
            mgf_infos:
                raw_name:
                    peptide_str:
                        infos
                        infos_score_max
            """

            # def input_data():
            # with Pool(self.config.processes) as pool:
            #     for ret in tqdm(
            #         pool.imap(get_td, input_data(), chunksize=10), total=len(pscore_keys)
            #     ):
            #         numt, numd = ret
            #         greater_target += numt
            #         greater_decoy += numd

            pscore_results, pscore_results_detail = process_qvalue(
                pscore_results, pscore_results_detail
            )

            #############################################
            # for raw_name in pscore_results_detail:
            #     # 想要看的是高分的mgf，检查为什么虚高
            #     for item in pscore_results_detail[raw_name]:
            #         if item["pscore_fdr"] > 0.01:
            #             break
            #         else:
            #             if not item["pscore_is_target"]:
            #                 self.write_mgf(join(self.config.mgf_path_root, raw_name + ".mgf"), item["pscore_fdr"], item["spectrum_name"], mgf_infos[raw_name][item["spectrum_name"]])
            ###############################################
            # check modification
            modification = "Deamidated[Q]"
            for raw_name in pscore_results_detail:
                # 想要看的是高分的mgf，检查为什么虚高
                for item in pscore_results_detail[raw_name]:
                    if item["pscore_fdr"] > 0.01:
                        break
                    else:
                        if (
                            modification in item["modification_pscore"]
                            and modification not in item["modification_pfind"]
                            and item["pscore_is_target"]
                        ):
                            # if "delPBP1_DB_NaN3_2.60992.60992.3." in item["spectrum_name"]:
                            self.write_mgf(
                                join(
                                    self.config.mgf_path_root, raw_name + "_check.mgf"
                                ),
                                item["pscore_fdr"],
                                item["spectrum_name"],
                                mgf_infos[raw_name][item["spectrum_name"]],
                            )

            ###############################################

            pickle.dump(
                (pscore_results, features, pscore_results_detail),
                open(pscore_result_middle_path, "wb"),
            )

        return pscore_results, pscore_results_detail, features

    def merge_pscore_results(self, pscore_results, pscore_results_detail):
        all_pscore_results = {"all": []}
        all_score_results_detail = {"all": []}
        # set_all_peptide = set()

        for k in pscore_results.keys():
            # for res in pscore_results[k]:
            all_pscore_results["all"].extend(pscore_results[k])
            all_score_results_detail["all"].extend(pscore_results_detail[k])
        all_pscore_results, all_score_results_detail = process_qvalue(
            all_pscore_results, all_score_results_detail
        )
        set_all_001 = set()
        for item in all_score_results_detail["all"]:
            if item["pscore_fdr"] <= 0.01:
                set_all_001.add(item["peptide_str"])

        return all_score_results_detail["all"], set_all_001

    def write_mgf(self, filename, pscore_fdr, title, info):
        charge = info["precursor_charge"]
        pepmass = info["precursor_mz"]
        mz_values = info["mz_array"]
        intensities = info["intensity"]
        with open(filename, "a") as f:
            f.write("BEGIN IONS\n")
            f.write(f"TITLE={title}\n")
            f.write(f"CHARGE={charge}\n")
            f.write(f"PEPMASS={pepmass:.6f}\n")
            f.write(f'PFIND_FDR={info["pfind_fdr"]:.6f}\n')
            f.write(f"SCORE_FDR={pscore_fdr:.6f}\n")
            f.write(f'PEPSEQ={info["peptide_pscore"]}\n')
            f.write(f'MODIFICATION={info["modification_pscore"]}\n')
            f.write(f'PFIND_PEPSEQ={info["peptide_pfind"]}\n')
            f.write(f'PFIND_IS_TARGET={info["pfind_is_target"]}\n')
            f.write(f'PFIND_MODIFICATION={info["modification_pfind"]}\n')
            # f.write(f"RTINSECONDS={rtinseconds:.7f}\n")

            for mz, intensity in zip(mz_values, intensities):
                f.write(f"{mz:.5f} {intensity:.1f}\n")

            f.write("END IONS\n")

    def filter_data4percolator(self, datas, thread=0.01):
        if not self.config.debug:
            per_filtered_result_middle_path = (
                self.config.per_filtered_result_middle_path
            )
        else:
            per_filtered_result_middle_path = (
                self.config.debug_per_filtered_result_middle_path
            )

        if os.path.exists(per_filtered_result_middle_path):
            ret_data = pickle.load(open(per_filtered_result_middle_path, "rb"))
        else:
            ret_data = {}
            for raw in datas.keys():
                print(f"filtering {raw}")
                target_num = 0
                decoy_num = 0
                ret_data[raw] = []
                for data_ in tqdm(datas[raw], total=len(datas[raw])):
                    data = pickle.loads(data_)
                    if self.config.debug_raw is None:
                        self.config.debug_raw = raw
                        flag_in = True
                    else:
                        if self.config.debug_raw == raw:
                            flag_in = True
                        else:
                            flag_in = False
                    if not self.config.debug or flag_in:
                        if data["is_target"][0] and data["pfind_qvalue"] < 0.01:
                            ret_data[raw].append(data)
                            target_num += 1
                        elif data["is_target"][0] and data["pfind_qvalue"] > 0.01:
                            ret_data[raw].append(data)
                            decoy_num += 1
                print(
                    f"# of training set for percolator: target{target_num}, decoy{decoy_num}"
                )
            pickle.dump(ret_data, open(per_filtered_result_middle_path, "wb"))
        return ret_data

    def run_percolator(self, percolator_data, data_with_pscore_feature):
        pscore_per_result_middle_path = self.config.pscore_per_result_middle_path
        # if os.path.exists(pscore_per_result_middle_path):
        # pscore_per_results = pickle.load(open(pscore_per_result_middle_path, 'rb'))
        # else:
        if True:
            pscore_per_results = self.model.train_and_infer(
                percolator_data, data_with_pscore_feature
            )
            pscore_per_results = process_qvalue(pscore_per_results)
            pickle.dump(pscore_per_results, open(pscore_per_result_middle_path, "wb"))

        return pscore_per_results

    def run(self, return_pfind=True):
        """
        read and save:
            pscore outputs: title, score, spectrum pred
            pfind outputs: title, score, fdr
            data: spectrum to compute

        pecolater

        store and output:
            a) pscore: title, score, fdr, peptide
            a) pscore with percolator: title, score, fdr, peptide
            b) pfind: title, score, fdr, peptide
        """

        self.load_data()

        print("processing pfind results")
        if return_pfind:
            pfind_results = self.get_pfind_results()
        else:
            pfind_results = None

        print("processing pscore results")
        pscore_results, pscore_results_detail, data_with_pscore_feature = (
            self.get_pscore_feat_and_results()
        )

        all_pfind_results, set_all_001_pfind = self.merge_pfind_results(
            copy.deepcopy(pfind_results)
        )

        all_pscore_results_detail, set_all_001_pscore = self.merge_pscore_results(
            copy.deepcopy(pscore_results), copy.deepcopy(pscore_results_detail)
        )

        print(f"pfind # of peptide {len(set_all_001_pfind)}")
        print(f"pscore # of peptide {len(set_all_001_pscore)}")
        print(f"intersect # of peptide {len(set_all_001_pfind & set_all_001_pscore)}")

        # for raw_name in pfind_results.keys():
        #     tmp = set()
        #     for item in pfind_results[raw_name]:
        #         if item[0] < 0.01:
        #             tmp.add(item[2])
        #     print("pFind 0.01: ", raw_name, len(tmp))
        # assert 0

        return (
            pfind_results,
            pscore_results,
            pscore_results_detail,
            None,
            all_pfind_results,
            all_pscore_results_detail,
        )

        print("filtering percolator data")
        percolator_data = self.filter_data4percolator(data_with_pscore_feature)

        print("running percolator")
        pscore_per_results = self.run_percolator(
            percolator_data, data_with_pscore_feature
        )

        return pfind_results, pscore_results, pscore_per_results
