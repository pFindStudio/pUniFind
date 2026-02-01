# use summary to filter results
# filter special modification for syn pep
# reserve the best match for spectrum prediction

# filter those with bad results

import copy
import csv
import gzip
import linecache
import os
import pickle
import sys

# import torch
from multiprocessing import Pool
from os import listdir
from os.path import isdir, join

import numpy as np
import pandas as pd
from pyteomics.mgf import MGF
from tqdm import tqdm

from scripts.process_qryres_utils import *


def read_mgf(input):
    mgf_path = input
    spectra = MGF(str(mgf_path))
    ret = {}
    for spectrum in spectra:
        title = spectrum["params"]["title"].strip()
        ret[title] = {
            "mz_array": spectrum["m/z array"],
            "intensity": spectrum["intensity array"],
            "precursor_mz": float(spectrum["params"]["pepmass"][0]),
            "precursor_charge": int(spectrum["params"].get("charge", [0])[0]),
            "rtinseconds": spectrum["params"]["rtinseconds"],
        }
    return ret


def read_denovo_mgfs(mgf_path):
    all_specs = {}
    mgf_content = read_mgf(mgf_path)
    for title in mgf_content.keys():
        item = {
            "title": title,
            "mgf": mgf_content[title],
            "1": {
                "idx": 1,
                "peptide": "PEP",
                "mass": 0,
                "num_mod": 0,
                "mods": [],
                "fdr_value": 0,
                "bm25_score": 0,
                "pfind_score": 0,
                "is_target": True,
                "m_uCleave": 0,
                "proteins": ["1"],
            },
            "num_res": 1,
        }
        item["mgf"]["HCD"] = 30
        item["mgf"]["inst"] = ins2num[strQE]
        if item["mgf"]["precursor_charge"] > 0:
            all_specs[title] = item

    return all_specs


def read_one_results(inputs):
    res_file, mgf_dir = inputs
    all_specs = {}
    mgf_name = linecache.getline(res_file, 2).strip().split(".")[0]
    mgf_path = mgf_name + "_HCDFT.mgf"
    mgf_content = read_mgf(join(mgf_dir, mgf_path))

    with open(res_file, "r") as file:
        content = file.readlines()

        spec_content = None
        flag_read_title = False
        fdr_value = None
        flag_illegal = False
        legal_cnt = 0
        illegal_cnt = 0
        for line in content:
            if line[0] == "S" and len(line.split()) == 3:
                if spec_content is not None:
                    if str(1) in spec_content and spec_content["1"]["fdr_value"] <= 0.1:
                        try:
                            spec_content["mgf"] = mgf_content[spec_content["title"]]
                        except:
                            spec_content["mgf"] = mgf_content[
                                ".".join(spec_content["title"].split(".")[:4]) + ".dta"
                            ]
                        spec_content["mgf"]["HCD"] = 30
                        spec_content["mgf"]["inst"] = ins2num[strQE]

                        if not flag_illegal:
                            legal_cnt += 1
                            all_specs[spec_content["title"]] = spec_content
                        else:
                            illegal_cnt += 1
                        flag_illegal = False

                spec_content = {}
                try:
                    spec_content["precursor_mass"] = float(line.split()[1])
                except:
                    assert 0, line
                flag_read_title = True
            elif flag_read_title:
                flag_read_title = False
                spec_content["title"] = line.strip()
            else:
                items = line.strip().split()
                idx = int(items[0])
                peptide = items[1].strip()
                if (
                    "B" in peptide
                    or "J" in peptide
                    or "U" in peptide
                    or "X" in peptide
                    or "Z" in peptide
                    or "O" in peptide
                ):
                    flag_illegal = True
                mass = float(items[2].strip())
                bm25_score = float(items[3].strip())
                pfind_score = float(items[4].strip())
                fdr_value = float(items[5].strip())
                num_mod = int(items[6].strip())
                mods = []
                for i in range(num_mod):
                    mods.append((int(items[6 + i * 2 + 1]), items[6 + i * 2 + 2][:-2]))
                    assert items[6 + i * 2 + 2].endswith("#0"), items[6 + i * 2 + 2]
                """
                position
                name
                """
                if num_mod > 0:
                    bDecoy = items[6 + (i + 1) * 2 + 2]
                    m_uCleave = items[6 + (i + 1) * 2 + 1]
                else:
                    bDecoy = items[6 + 2]
                    m_uCleave = items[6 + 1]
                assert bDecoy in ["0", "1"], items
                proteins = []
                for item in items:
                    if ";" in item:
                        assert len(item.split(";")) == 5, (item, item.split(";"))
                        proteins.append(int(item.split(";")[0]))
                ret = {
                    "idx": idx,
                    "peptide": peptide,
                    "mass": mass,
                    "num_mod": num_mod,
                    "mods": mods,
                    "fdr_value": fdr_value,
                    "bm25_score": bm25_score,
                    "pfind_score": pfind_score,
                    "is_target": bDecoy == "0",
                    "m_uCleave": int(m_uCleave),
                    "proteins": proteins,
                }
                spec_content[str(idx)] = ret
                spec_content["num_res"] = idx

        if str(1) in spec_content:
            if not flag_illegal:
                spec_content["mgf"] = mgf_content[spec_content["title"]]
                split_string = spec_content["title"].split(".")
                spec_content["mgf"]["HCD"] = 25
                spec_content["mgf"]["inst"] = ins2num[strQE]
                all_specs[spec_content["title"]] = spec_content
    return all_specs
