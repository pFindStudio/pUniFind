import copy
import csv
import gzip
import linecache
import os
import pickle
import sys
from multiprocessing import Pool
from os import listdir
from os.path import isdir, join
from pathlib import Path

import numpy as np
import pandas as pd
import spectrum_utils.spectrum as sus
import torch
from pyteomics.mgf import MGF
from tqdm import tqdm

strQE = "QE"
strVelos = "Velos"
strElite = "Elite"
strLumos = "Lumos"
strFusion = "Fusion"
strTOF = "TOF"

# mass_cal = PeptideIonCalculator()

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

ins2num = {
    strQE: 0,
    strVelos: 1,
    strElite: 2,
    strLumos: 3,
    strTOF: 4,
}


def extract_string_between_brackets(input_string):
    start_index = input_string.find(
        "["
    )  # Get the starting index of the opening bracket
    end_index = input_string.find("]")  # Get the ending index of the closing bracket

    return input_string[start_index + 1 : end_index]


def contains_non_uppercase(s):
    """
    检查字符串s中是否包含非大写英文字母的字符。
    如果包含，返回True；否则返回False。
    """
    for char in s:
        if not char.isupper():  # 检查是否不是大写英文字母
            return True
    return False


def get_aa(s):
    if s.split("->")[1][:3] in amino_acids_3to1_map:
        return amino_acids_3to1_map[s.split("->")[1][:3]]
    elif s.split("[")[0][-3:] in amino_acids_3to1_map:
        return amino_acids_3to1_map[s.split("[")[0][-3:]]
    else:
        return None


def check_illegal(data):
    for i in range(1, data["small"]["num_res"] + 1):
        if contains_non_uppercase(data["small"][str(i)]["peptide"]):
            return True
    return False


def load_delete(file_path):

    with open(file_path, "r", encoding="utf-8") as file:
        # 使用列表推导式读取非空行
        lines = [line.strip() for line in file if line.strip()]
    return lines


def data_2_pep(data):
    peptide = data["small"]["1"]["peptide"]
    for mod in data["small"]["1"]["mods"]:
        if "->" in mod[1]:
            new_residue = get_aa(mod[1])
            if new_residue != None:
                # peptide[mod[0]] = new_residue
                # peptide = peptide[:mod[0]-1] + new_residue + peptide[mod[0]:]
                if mod[0] == 0:
                    mod = (mod[0] + 1, mod[1])
                    # mod[0] += 1
                if mod[0] > len(data["small"]["1"]["peptide"]):
                    mod = (mod[0] - 1, mod[1])
                peptide_list = list(peptide)
                try:
                    peptide_list[mod[0] - 1] = new_residue
                except:
                    assert 0, (peptide_list, mod)
                peptide = "".join(peptide_list)
            # print(data["small"]["1"]["peptide"], peptide, mod)
    return peptide


def data_2_str(data, modification_dict):
    inst = str(int(data["small"]["mgf"]["inst"]))
    nce = str(int(data["small"]["mgf"]["HCD"]))
    peptide = data_2_pep(data)

    mod_info = data["small"]["1"]["mods"]
    mod_str = "_"
    for mod in mod_info:
        mod_type = str(int(modification_dict[mod[1]][2]))
        mod_pos = str(int(mod[0]))
        if "->" not in mod[1] or mod[1].split("->")[1][:3] not in amino_acids_3to1_map:
            mod_str += mod_type + "_" + mod_pos + "_"

    return inst + "_" + nce + "_" + peptide + mod_str


def read_spectra_to_dict(file_path):
    result_dict = {}
    with open(file_path, "r", encoding="utf-8") as file:
        # 跳过第一行
        next(file)
        for line in file:
            # 分割行内容为列表
            columns = line.strip().split()
            # assert len(columns) == 19 or len(columns) == 20, columns
            if len(columns) >= 9:
                if "." not in columns[0]:
                    key = " ".join([columns[0], columns[1]])
                else:
                    key = columns[0]
                try:
                    if "." not in columns[0]:
                        value = float(columns[9])
                    else:
                        value = float(columns[8])
                except ValueError:
                    continue
                result_dict[key] = value
    return result_dict


def read_modification(
    mod_path=r"/mnt/vepfs/fs_users/zhaojiale//scripts/modification.ini",
):
    mod_meta = {}
    mod_list = []
    map1 = {"PEP_C": 1, "NORMAL": 2, "PEP_N": 3, "PRO_N": 4, "PRO_C": 5}
    set1 = set()

    with open(mod_path, "r") as file:
        content = file.readlines()
        mod_name = None
        mod_index = 1
        for line in content:
            if line.startswith("name"):
                mod_name = line.split(" ")[0].split("=")[1]
                mod_list.append(mod_name)
            else:
                try:
                    # if True:
                    mod_neutral_loss = 0
                    if line.split(" ")[4] != "0":
                        mod_neutral_loss = float(line.split(" ")[5])
                    mod_meta[mod_name] = torch.Tensor(
                        [
                            map1[line.split(" ")[1]],
                            float(line.split(" ")[2]),
                            mod_index,
                            float(mod_neutral_loss),
                        ]
                    )

                    """
                        mod position type
                        mod mass
                        index
                        mod loss mass
                    """
                    mod_index += 1
                except:
                    print(line)
    # print(mod_index)
    return mod_meta, mod_list


def get_meta(header_file):
    # get nce
    local_dict = {}
    df = pd.read_csv(header_file)
    for index, row in df.iterrows():
        scan = (
            os.path.basename(header_file).split("_omitted")[0]
            + "."
            + str(int(row["Scan"]))
            + "."
            + str(int(row["Scan"]))
        )
        hcd = round(float(row["Energy1"]))
        local_dict[scan] = hcd
    return local_dict


def custom_sort(strings):
    def sort_key(s):
        return s[0].split(".")[-4]

    sorted_strings = sorted(strings, key=sort_key)
    return sorted_strings


def read_header_dataset(
    dataset,
    header_pkl_path,
    header_path=r"/mnt/vepfs/fs_ckps/zhaojiale/dataset/mol_spec/dataset/raw_header",
):
    # datasets = ["prombe"]
    # datasets = ["Kuster_tr"]
    # for dataset in datasets:
    if True:
        print(f"processing {dataset}")
        head_folder = join(header_path, dataset)
        # for header_file in listdir(head_folder):
        header_meta = {}

        with Pool(116) as pool:
            for ret in tqdm(
                pool.imap(
                    get_meta,
                    [join(head_folder, _) for _ in listdir(head_folder)],
                    chunksize=10,
                ),
                total=len(listdir(head_folder)),
            ):
                header_meta.update(ret)

        with open(join(header_pkl_path, f"{dataset}_header.pkl"), "wb") as file:
            pickle.dump(header_meta, file)
        print(f"finished processing header for dataset {dataset}")
    return header_meta
