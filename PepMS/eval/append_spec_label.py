import copy
import gzip
import os
import pickle

# from pyteomics.mgf import MGF
import random
import sys

# import linecache
import time
from multiprocessing import Pool
from os import listdir
from os.path import isdir, join

import lmdb
import numpy as np
from tqdm import tqdm

from scripts.process_qryres_utils import *

# add: ins
# track: ins, nce, peptide with mod -> max pfind score
# filter: 1.false Enzyme
#         2.false synic peptide
#         3.synic mod
#         4.fix replace
#         5.fix I L issue

# first load and clean original
# second merge new ones

# mean while track id and spectrum label,track true/false ratio of sync


def get(key):
    data_ziped = txn_read.get(key)
    return data_ziped


def process_new(inputs):
    """
    return: data, count_pt_all_d, count_pt_mod_d, coutt_pt_correct_d
    """
    spec_key = inputs
    key = best_id[spec_key][0]
    data_ziped = txn_read.get(key)
    data = pickle.loads(gzip.decompress(data_ziped))
    str_data = data_2_str(data, modification_meta_dict)

    return str_data, get_spec_label(data), data["small"]["1"]["pfind_score"]


def get_best_ids(inputs):
    key = inputs
    data_ziped = txn_read.get(key)
    data = pickle.loads(gzip.decompress(data_ziped))
    if not data["small"]["1"]["is_target"] or (data["small"]["1"]["fdr_value"] > 0.001):
        return None, None, None, None
    str_data = data_2_str(data, modification_meta_dict)

    return str_data, str_data, key, data["small"]["1"]["pfind_score"]
