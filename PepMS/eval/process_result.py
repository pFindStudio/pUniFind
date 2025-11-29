import gzip
import logging
import os
import pickle
import sys
from multiprocessing import Pool
from os.path import join

import lmdb
import numpy as np
import spectrum_utils.spectrum as sus
import torch
from tqdm import tqdm

# import matplotlib.pyplot as plt
from PepMS.data.mass_calculation import *
from PepMS.eval.draw_pred_spec import plot_spectra

mass_calculator = PeptideIonCalculator()

# decoy_thread = -1
# decoy_thread = 38141 # Apis
# decoy_thread = 41194 # human
# decoy_thread = 159480 # human
# decoy_thread = 40870
# decoy_thread = 38140 # Apis
decoy_thread = 70070  # Vigna
# decoy_thread = 12120
force_trap_decoy = False

title_to_trap = None
# title_to_trap = {
#     "HeLa": [0, 41194],
#     "Yeast": [41195, 53314],
# }
residues = {
    "G": 57.021464,
    "A": 71.037114,
    "S": 87.032028,
    "P": 97.052764,
    "V": 99.068414,
    "T": 101.047670,
    "C": 103.009185,  # 103.009185 + 57.021464 # SQ215.090606 TN215.090597
    # "C+57.021": 160.030649, # 103.009185 + 57.021464
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
mass_H2O = 2 * 1.0078246 + 15.9949141
modification_meta = pickle.load(open("./consts/modification_meta_dict.pkl", "rb"))


def data_2_str(data, idx):
    peptide = data["small"][f"{idx}"]["peptide"]

    mod_info = data["small"][f"{idx}"]["mods"]

    mod_str = "_"
    for mod in mod_info:
        mod_type = mod[1]
        mod_pos = str(int(mod[0]))
        mod_str += mod_type + "_" + mod_pos + "_"

    return peptide + mod_str


def process_pfind_qvalue(unsorted_results):
    for k in unsorted_results.keys():

        unsorted_results[k] = sorted(unsorted_results[k], key=lambda x: x[-1])
        fdr = []
        set_target = set()
        set_decoy = set()
        for res in unsorted_results[k]:
            if res[1]:
                set_target.add(res[2])
            else:
                set_decoy.add(res[2])
            fdr.append(len(set_decoy) / (len(set_target) + 0.0001))
        min_fdr = 1
        for i in range(len(fdr) - 1, -1, -1):
            min_fdr = min(min_fdr, fdr[i])
            fdr[i] = min_fdr
        # ret_sub = []
        for i in range(len(fdr)):
            unsorted_results[k][i][0] = fdr[i]

    return unsorted_results


def process_single_pfind_qvalue(unsorted_results):
    unsorted_results = sorted(unsorted_results, key=lambda x: x[-1])
    fdr = []
    set_target = set()
    set_decoy = set()
    for res in unsorted_results[k]:
        if res[1]:
            set_target.add(res[2])
        else:
            set_decoy.add(res[2])
        fdr.append(len(set_decoy) / (len(set_target) + 0.0001))
    min_fdr = 1
    for i in range(len(fdr) - 1, -1, -1):
        min_fdr = min(min_fdr, fdr[i])
        fdr[i] = min_fdr
    # ret_sub = []
    for i in range(len(fdr)):
        unsorted_results[i][0] = fdr[i]
    return unsorted_results


def process_qvalue(unsorted_results, pscore_results_detail):
    for k in unsorted_results.keys():
        num_target = 0
        num_decoy = 0
        # one_raw_result = unsorted_results[k]
        # one_raw_result_detail = pscore_results_detail[k]
        # print(one_raw_result[:10])
        # one_raw_result = sorted(one_raw_result, key=lambda x: x[0], reverse=True)
        pscore_results_detail[k] = sorted(
            pscore_results_detail[k], key=lambda x: x["pscore_score"], reverse=True
        )
        fdr = []
        set_target = set()
        set_decoy = set()
        for res in pscore_results_detail[k]:
            if res["pscore_is_target"]:
                set_target.add(res["peptide_str"])
            else:
                set_decoy.add(res["peptide_str"])
            fdr.append(len(set_decoy) / (len(set_target) + 0.0001))
        min_fdr = 1
        for i in range(len(fdr) - 1, -1, -1):
            min_fdr = min(min_fdr, fdr[i])
            fdr[i] = min_fdr
        # ret_sub = []
        for i in range(len(fdr)):
            pscore_results_detail[k][i]["pscore_fdr"] = fdr[i]
            # ret_sub.append((fdr[i], one_raw_result[i][1]))
            pep_str = pscore_results_detail[k][i]["peptide_str"]
            try:
                unsorted_results[k][pep_str]["qvalue"] = fdr[i]
            except:
                pass

    return unsorted_results, pscore_results_detail


def get_pfind_result(input):
    key, data_ziped, large_flag = input
    # data_ziped = txn_read.get(key)
    data = pickle.loads(gzip.decompress(data_ziped))
    if large_flag:
        k = "large"
    else:
        k = "small"
    # assert len(data[k]["1"]["proteins"]) > 0, data[k]
    if title_to_trap is None:
        is_trap = False
        if decoy_thread > 0 and data[k]["1"]["is_target"]:
            is_trap = True
            try:
                for id_p in data[k]["1"]["proteins"]:
                    if id_p <= decoy_thread:
                        is_trap = False
            except:
                is_trap = False
        is_trap_decoy = False
        if decoy_thread > 0 and not data[k]["1"]["is_target"]:
            is_trap_decoy = True
            try:
                for id_p in data[k]["1"]["proteins"]:
                    if id_p <= decoy_thread:
                        is_trap_decoy = False
            except:
                pass
    else:
        is_trap = False
        not_trap_range = None

        for title in title_to_trap.keys():
            if title in data[k]["title"]:
                not_trap_range = title_to_trap[title]
        # assert not_trap_range is not None

        is_trap = False
        if data[k]["1"]["is_target"]:
            is_trap = True
            for id_p in data[k]["1"]["proteins"]:
                if id_p >= not_trap_range[0] and id_p <= not_trap_range[1]:
                    is_trap = False
        is_trap_decoy = False
        if not data[k]["1"]["is_target"]:
            is_trap_decoy = True
            for id_p in data[k]["1"]["proteins"]:
                if id_p >= not_trap_range[0] and id_p <= not_trap_range[1]:
                    is_trap_decoy = False

    fdr = data[k]["1"]["fdr_value"]
    pfind_score = data[k]["1"]["pfind_score"]
    is_target = data[k]["1"]["is_target"]
    raw_name = data[k]["title"].split(".")[0]
    if force_trap_decoy and is_trap:
        is_target = False
    try:
        proteins = data[k]["1"]["proteins"]
    except:
        proteins = ["-1"]
    return [
        fdr,
        is_target,
        data_2_str(data, 1),
        is_trap,
        is_trap_decoy,
        data[k]["title"],
        proteins,
        pfind_score,
    ], raw_name


def get_pscore_result(input):
    assert 0
    data, key, data_ziped, large_flag = input
    dist = data["score"]
    # is_target = data["is_target"]

    idx = np.argmax(dist)
    score_max = dist[idx]
    # is_target_max = is_target[idx]
    # data_ziped = txn_read.get(key)
    # data_ziped = txn_read.get(keys[int(index)])
    data_lmdb = pickle.loads(gzip.decompress(data_ziped))
    if large_flag:
        k = "large"
    else:
        k = "small"
    raw_name = data_lmdb[k]["title"].split(".")[0]
    is_target_max = data_lmdb[k][f"{idx + 1}"]["is_target"]
    qvalue = data_lmdb[k]["1"]["fdr_value"]
    # print(len(dist), data_lmdb[k]["num_res"], data_lmdb[k]["title"], key)
    assert len(dist) >= data_lmdb[k]["num_res"], (
        dist,
        len(dist),
        data_lmdb[k]["num_res"],
        data_lmdb,
    )
    return (score_max, is_target_max), raw_name, qvalue


def get_pscore_feat_and_result(input):
    data, key, data_ziped, large_flag, config = input
    if not config.use_joint_scores:
        dist = data["score"]
        assert 0
    else:
        dist = data["joint_scores"]

    try:
        spec_preds = data["spec_pred"]
    except:
        spec_preds = None
    if not config.use_rank or data["best_rank"] == -1:
        idx = np.argmax(dist)
    else:
        idx = data["best_rank"]
    try:
        idx = idx.item()
    except:
        pass

    score_max = dist[idx]
    data_lmdb = pickle.loads(gzip.decompress(data_ziped))

    if large_flag:
        k = "large"
    else:
        k = "small"

    if title_to_trap is None:
        is_trap = False
        # assert len(data_lmdb[k][str(idx+1)]["proteins"]) > 0, data_lmdb[k]
        if decoy_thread > 0 and data_lmdb[k][str(idx + 1)]["is_target"]:
            is_trap = True
            try:
                for id_p in data_lmdb[k][str(idx + 1)]["proteins"]:
                    if id_p <= decoy_thread:
                        is_trap = False
            except:
                pass

        is_trap_decoy = False
        if decoy_thread > 0 and not data_lmdb[k][str(idx + 1)]["is_target"]:
            is_trap_decoy = True
            try:
                for id_p in data_lmdb[k][str(idx + 1)]["proteins"]:
                    if id_p <= decoy_thread:
                        is_trap_decoy = False
            except:
                pass
    else:
        is_trap = False
        not_trap_range = None

        for title in title_to_trap.keys():
            if title in data_lmdb[k]["title"]:
                not_trap_range = title_to_trap[title]
        assert not_trap_range is not None, data_lmdb[k]["title"]

        is_trap = False
        if data_lmdb[k]["1"]["is_target"]:
            is_trap = True
            for id_p in data_lmdb[k]["1"]["proteins"]:
                if id_p >= not_trap_range[0] and id_p <= not_trap_range[1]:
                    is_trap = False
        is_trap_decoy = False
        if not data_lmdb[k]["1"]["is_target"]:
            is_trap_decoy = True
            for id_p in data_lmdb[k]["1"]["proteins"]:
                if id_p >= not_trap_range[0] and id_p <= not_trap_range[1]:
                    is_trap_decoy = False

    raw_name = data_lmdb[k]["title"].split(".")[0]
    is_target_max = data_lmdb[k][f"{idx + 1}"]["is_target"]
    peptide_max = data_lmdb[k][f"{idx + 1}"]["peptide"]
    qvalue = data_lmdb[k]["1"]["fdr_value"]
    # print(len(dist), data_lmdb[k]["num_res"], data_lmdb[k]["title"], key)
    assert len(dist) >= data_lmdb[k]["num_res"], (
        dist,
        len(dist),
        data_lmdb[k]["num_res"],
        data_lmdb,
    )
    if force_trap_decoy and is_trap:
        is_target_max = False
    peptide = []
    # spectrum_score = []
    pfind_score = []
    mods = []
    is_target = []

    if data_lmdb[k]["num_res"] != 1:
        for i in range(1, data_lmdb[k]["num_res"] + 1):
            peptide.append(data_lmdb[k][str(i)]["peptide"])
            pfind_score.append(data_lmdb[k][str(i)]["pfind_score"])
            mods.append(data_lmdb[k][str(i)]["mods"])
            is_target.append(data_lmdb[k][str(i)]["is_target"])

    else:
        peptide.append(data_lmdb[k][str(1)]["peptide"])
        pfind_score.append(data_lmdb[k][str(1)]["pfind_score"])
        mods.append(data_lmdb[k][str(1)]["mods"])
        peptide.append(data_lmdb[k][str(1)]["peptide"])
        pfind_score.append(data_lmdb[k][str(1)]["pfind_score"])
        mods.append(data_lmdb[k][str(1)]["mods"])
        is_target.append(data_lmdb[k][str(1)]["is_target"])
        is_target.append(data_lmdb[k][str(1)]["is_target"])

    if config.use_pred_spec:
        precursor_mz = data_lmdb["small"]["mgf"]["precursor_mz"]
        precursor_charge = data_lmdb["small"]["mgf"]["precursor_charge"]

        spectrum_ = sus.MsmsSpectrum(
            "",
            precursor_mz,
            precursor_charge,
            data_lmdb["small"]["mgf"]["mz_array"].astype(np.float64),
            data_lmdb["small"]["mgf"]["intensity"].astype(np.float32),
        )
        spectrum_.set_mz_range(50.5, 4500.0)
        if len(spectrum_.mz) == 0:
            raise ValueError
        spectrum_.remove_precursor_peak(2.0, "Da")
        if len(spectrum_.mz) == 0:
            raise ValueError
        spectrum_.filter_intensity(0, 300)
        if len(spectrum_.mz) == 0:
            raise ValueError
        spectrum_.scale_intensity("root", 1)
        intensity = spectrum_.intensity / np.linalg.norm(spectrum_.intensity)
        mz_array = spectrum_.mz
        # mz_array, intensity = torch.Tensor(mz_array), torch.Tensor(intensity)

        # intensity = data_lmdb["small"]["mgf"]["intensity"] / np.linalg.norm(
        #     data_lmdb["small"]["mgf"]["intensity"]
        # )

        spectrum_score, pred_mz, pred_intensity = mass_calculator.cal_inference_feature(
            mz=mz_array,
            intensity=intensity,
            spectrum_preds=spec_preds,
            peptides=peptide,
            modinfos=mods,
        )
        pred_mask = pred_intensity > 1e-3
        pred_mz = pred_mz[pred_mask]
        pred_intensity = pred_intensity[pred_mask] / pred_intensity.max()

        if qvalue < 0.001:
            random_number = np.random.rand()
            if random_number < 0.00001:
                plot_spectra(
                    mz_array, intensity, pred_mz, pred_intensity, save_name=key
                )
                # assert 0, (mz_array, intensity, pred_mz, pred_intensity, spectrum_score[0])

        feat = pickle.dumps(
            {
                "title": data_lmdb[k]["title"],
                "index": data["index"],
                "peptide": peptide,
                "spectrum_score": spectrum_score,
                "pfind_score": pfind_score,
                "is_target": is_target,
                "pscore_score": data["score"],
                "pfind_qvalue": qvalue,
            }
        )
    else:
        spectrum_score = None
        feat = None

    precursor_mz = data_lmdb["small"]["mgf"]["precursor_mz"]
    precursor_charge = data_lmdb["small"]["mgf"]["precursor_charge"]
    MH = (precursor_mz - 1.007276) * (precursor_charge) + 1.007276
    try:
        proteins = data_lmdb[k][f"{idx + 1}"]["proteins"]
    except:
        proteins = ["-1"]
    mass_seq = sum([residues[_] for _ in data_lmdb[k][str(idx + 1)]["peptide"]])
    try:
        mass_seq += (
            sum(
                [modification_meta[_[1]][1] for _ in data_lmdb[k][str(idx + 1)]["mods"]]
            )
            + mass_H2O
            + 1.007276
        )
    except:
        # print("error cal mass")
        pass
    peptide_str = data_2_str(data_lmdb, idx + 1)
    mod_str = ""
    for m in data_lmdb[k][str(idx + 1)]["mods"]:
        mod_str += str(m[0]) + "," + m[1] + ";"

    detail = {
        "spectrum_name": data_lmdb[k]["title"],
        "pscore_max_score": None,
        "pscore_score": score_max,
        "peptide_str": peptide_str,
        "peptide_pfind": data_lmdb[k][str(1)]["peptide"],
        "peptide_pscore": peptide_max,
        "mods": mod_str,
        "mass_seq": mass_seq,
        "MH": MH,
        "proteins": proteins,
        "modification_pfind": "_".join(
            [str(mod[0]) + "," + mod[1] + ";" for mod in data_lmdb[k][str(1)]["mods"]]
        ),
        "modification_pscore": "_".join(
            [mod[1] + str(mod[0]) for mod in data_lmdb[k][str(idx + 1)]["mods"]]
        ),
        "pfind_fdr": data_lmdb[k]["1"]["fdr_value"],
        "pscore_fdr": None,
        "pfind_is_target": data_lmdb[k]["1"]["is_target"],
        "pscore_is_target": is_target_max,
        "is_trap": is_trap,
        "is_trap_decoy": is_trap_decoy,
    }

    if is_target_max:
        info = {
            "mz_array": data_lmdb["small"]["mgf"]["mz_array"].astype(np.float64),
            "intensity": data_lmdb["small"]["mgf"]["intensity"].astype(np.float32),
            "peptide_pfind": data_lmdb[k][str(1)]["peptide"],
            "peptide_pscore": peptide_max,
            "modification_pfind": "_".join(
                [mod[1] + str(mod[0]) for mod in data_lmdb[k][str(1)]["mods"]]
            ),
            "modification_pscore": "_".join(
                [mod[1] + str(mod[0]) for mod in data_lmdb[k][str(idx + 1)]["mods"]]
            ),
            "pfind_fdr": data_lmdb[k]["1"]["fdr_value"],
            "precursor_mz": data_lmdb["small"]["mgf"]["precursor_mz"],
            "precursor_charge": data_lmdb["small"]["mgf"]["precursor_charge"],
            "pfind_is_target": data_lmdb[k]["1"]["is_target"],
            "is_trap": is_trap,
        }
    else:
        info = None
    return (score_max, is_target_max), raw_name, qvalue, feat, detail, info, peptide_str


def get_td(data):
    data = pickle.loads(gzip.decompress(data))
    if data["small"]["1"]["fdr_value"] < 0.1:
        return 0, 0

    if data["small"]["1"]["is_target"]:
        return 1, 0
    else:
        return 0, 1
