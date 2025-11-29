# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import copy
import gzip
import os
import pickle
import random
from functools import lru_cache
from multiprocessing import Pool
from os.path import join

import lmdb
import numpy as np
import spectrum_utils.spectrum as sus
import torch
import torch.nn as nn
from scipy.spatial import distance_matrix
from torch.utils.data.dataloader import default_collate
from unicore.data import BaseWrapperDataset, UnicoreDataset
from unicore.distributed.utils import get_data_parallel_rank

from . import data_utils
from .mass_calculation import PeptideIonCalculator
from .plot_utils import (
    append_mgf,
    color_array,
    plot_spectrum,
    remove_precursor_peak,
    sqrt_and_norm,
    write_mgf,
)
from .utils import remove_precursor_peak, sqrt_and_norm

protein_letters_3to1 = {
    "00C": "C",
    "01W": "X",
    "02K": "A",
    "03Y": "C",
    "07O": "C",
    "08P": "C",
    "0A0": "D",
    "0A1": "Y",
    "0A2": "K",
    "0A8": "C",
    "0AA": "V",
    "0AB": "V",
    "0AC": "G",
    "0AD": "G",
    "0AF": "W",
    "0AG": "L",
    "0AH": "S",
    "0AK": "D",
    "0AM": "A",
    "0AP": "C",
    "0AU": "U",
    "0AV": "A",
    "0AZ": "P",
    "0BN": "F",
    "0C ": "C",
    "0CS": "A",
    "0DC": "C",
    "0DG": "G",
    "0DT": "T",
    "0FL": "A",
    "0G ": "G",
    "0NC": "A",
    "0SP": "A",
    "0U ": "U",
    "0YG": "YG",
    "10C": "C",
    "125": "U",
    "126": "U",
    "127": "U",
    "128": "N",
    "12A": "A",
    "143": "C",
    "175": "ASG",
    "193": "X",
    "1AP": "A",
    "1MA": "A",
    "1MG": "G",
    "1PA": "F",
    "1PI": "A",
    "1PR": "N",
    "1SC": "C",
    "1TQ": "W",
    "1TY": "Y",
    "1X6": "S",
    "200": "F",
    "23F": "F",
    "23S": "X",
    "26B": "T",
    "2AD": "X",
    "2AG": "A",
    "2AO": "X",
    "2AR": "A",
    "2AS": "X",
    "2AT": "T",
    "2AU": "U",
    "2BD": "I",
    "2BT": "T",
    "2BU": "A",
    "2CO": "C",
    "2DA": "A",
    "2DF": "N",
    "2DM": "N",
    "2DO": "X",
    "2DT": "T",
    "2EG": "G",
    "2FE": "N",
    "2FI": "N",
    "2FM": "M",
    "2GT": "T",
    "2HF": "H",
    "2LU": "L",
    "2MA": "A",
    "2MG": "G",
    "2ML": "L",
    "2MR": "R",
    "2MT": "P",
    "2MU": "U",
    "2NT": "T",
    "2OM": "U",
    "2OT": "T",
    "2PI": "X",
    "2PR": "G",
    "2SA": "N",
    "2SI": "X",
    "2ST": "T",
    "2TL": "T",
    "2TY": "Y",
    "2VA": "V",
    "2XA": "C",
    "32S": "X",
    "32T": "X",
    "3AH": "H",
    "3AR": "X",
    "3CF": "F",
    "3DA": "A",
    "3DR": "N",
    "3GA": "A",
    "3MD": "D",
    "3ME": "U",
    "3NF": "Y",
    "3QN": "K",
    "3TY": "X",
    "3XH": "G",
    "4AC": "N",
    "4BF": "Y",
    "4CF": "F",
    "4CY": "M",
    "4DP": "W",
    "4F3": "GYG",
    "4FB": "P",
    "4FW": "W",
    "4HT": "W",
    "4IN": "W",
    "4MF": "N",
    "4MM": "X",
    "4OC": "C",
    "4PC": "C",
    "4PD": "C",
    "4PE": "C",
    "4PH": "F",
    "4SC": "C",
    "4SU": "U",
    "4TA": "N",
    "4U7": "A",
    "56A": "H",
    "5AA": "A",
    "5AB": "A",
    "5AT": "T",
    "5BU": "U",
    "5CG": "G",
    "5CM": "C",
    "5CS": "C",
    "5FA": "A",
    "5FC": "C",
    "5FU": "U",
    "5HP": "E",
    "5HT": "T",
    "5HU": "U",
    "5IC": "C",
    "5IT": "T",
    "5IU": "U",
    "5MC": "C",
    "5MD": "N",
    "5MU": "U",
    "5NC": "C",
    "5PC": "C",
    "5PY": "T",
    "5SE": "U",
    "5ZA": "TWG",
    "64T": "T",
    "6CL": "K",
    "6CT": "T",
    "6CW": "W",
    "6HA": "A",
    "6HC": "C",
    "6HG": "G",
    "6HN": "K",
    "6HT": "T",
    "6IA": "A",
    "6MA": "A",
    "6MC": "A",
    "6MI": "N",
    "6MT": "A",
    "6MZ": "N",
    "6OG": "G",
    "70U": "U",
    "7DA": "A",
    "7GU": "G",
    "7JA": "I",
    "7MG": "G",
    "8AN": "A",
    "8FG": "G",
    "8MG": "G",
    "8OG": "G",
    "9NE": "E",
    "9NF": "F",
    "9NR": "R",
    "9NV": "V",
    "A  ": "A",
    "A1P": "N",
    "A23": "A",
    "A2L": "A",
    "A2M": "A",
    "A34": "A",
    "A35": "A",
    "A38": "A",
    "A39": "A",
    "A3A": "A",
    "A3P": "A",
    "A40": "A",
    "A43": "A",
    "A44": "A",
    "A47": "A",
    "A5L": "A",
    "A5M": "C",
    "A5N": "N",
    "A5O": "A",
    "A66": "X",
    "AA3": "A",
    "AA4": "A",
    "AAR": "R",
    "AB7": "X",
    "ABA": "A",
    "ABR": "A",
    "ABS": "A",
    "ABT": "N",
    "ACB": "D",
    "ACL": "R",
    "AD2": "A",
    "ADD": "X",
    "ADX": "N",
    "AEA": "X",
    "AEI": "D",
    "AET": "A",
    "AFA": "N",
    "AFF": "N",
    "AFG": "G",
    "AGM": "R",
    "AGT": "C",
    "AHB": "N",
    "AHH": "X",
    "AHO": "A",
    "AHP": "A",
    "AHS": "X",
    "AHT": "X",
    "AIB": "A",
    "AKL": "D",
    "AKZ": "D",
    "ALA": "A",
    "ALC": "A",
    "ALM": "A",
    "ALN": "A",
    "ALO": "T",
    "ALQ": "X",
    "ALS": "A",
    "ALT": "A",
    "ALV": "A",
    "ALY": "K",
    "AN8": "A",
    "AP7": "A",
    "APE": "X",
    "APH": "A",
    "API": "K",
    "APK": "K",
    "APM": "X",
    "APP": "X",
    "AR2": "R",
    "AR4": "E",
    "AR7": "R",
    "ARG": "R",
    "ARM": "R",
    "ARO": "R",
    "ARV": "X",
    "AS ": "A",
    "AS2": "D",
    "AS9": "X",
    "ASA": "D",
    "ASB": "D",
    "ASI": "D",
    "ASK": "D",
    "ASL": "D",
    "ASM": "X",
    "ASN": "N",
    "ASP": "D",
    "ASQ": "D",
    "ASU": "N",
    "ASX": "B",
    "ATD": "T",
    "ATL": "T",
    "ATM": "T",
    "AVC": "A",
    "AVN": "X",
    "AYA": "A",
    "AYG": "AYG",
    "AZK": "K",
    "AZS": "S",
    "AZY": "Y",
    "B1F": "F",
    "B1P": "N",
    "B2A": "A",
    "B2F": "F",
    "B2I": "I",
    "B2V": "V",
    "B3A": "A",
    "B3D": "D",
    "B3E": "E",
    "B3K": "K",
    "B3L": "X",
    "B3M": "X",
    "B3Q": "X",
    "B3S": "S",
    "B3T": "X",
    "B3U": "H",
    "B3X": "N",
    "B3Y": "Y",
    "BB6": "C",
    "BB7": "C",
    "BB8": "F",
    "BB9": "C",
    "BBC": "C",
    "BCS": "C",
    "BE2": "X",
    "BFD": "D",
    "BG1": "S",
    "BGM": "G",
    "BH2": "D",
    "BHD": "D",
    "BIF": "F",
    "BIL": "X",
    "BIU": "I",
    "BJH": "X",
    "BLE": "L",
    "BLY": "K",
    "BMP": "N",
    "BMT": "T",
    "BNN": "F",
    "BNO": "X",
    "BOE": "T",
    "BOR": "R",
    "BPE": "C",
    "BRU": "U",
    "BSE": "S",
    "BT5": "N",
    "BTA": "L",
    "BTC": "C",
    "BTR": "W",
    "BUC": "C",
    "BUG": "V",
    "BVP": "U",
    "BZG": "N",
    "C  ": "C",
    "C12": "TYG",
    "C1X": "K",
    "C25": "C",
    "C2L": "C",
    "C2S": "C",
    "C31": "C",
    "C32": "C",
    "C34": "C",
    "C36": "C",
    "C37": "C",
    "C38": "C",
    "C3Y": "C",
    "C42": "C",
    "C43": "C",
    "C45": "C",
    "C46": "C",
    "C49": "C",
    "C4R": "C",
    "C4S": "C",
    "C5C": "C",
    "C66": "X",
    "C6C": "C",
    "C99": "TFG",
    "CAF": "C",
    "CAL": "X",
    "CAR": "C",
    "CAS": "C",
    "CAV": "X",
    "CAY": "C",
    "CB2": "C",
    "CBR": "C",
    "CBV": "C",
    "CCC": "C",
    "CCL": "K",
    "CCS": "C",
    "CCY": "CYG",
    "CDE": "X",
    "CDV": "X",
    "CDW": "C",
    "CEA": "C",
    "CFL": "C",
    "CFY": "FCYG",
    "CG1": "G",
    "CGA": "E",
    "CGU": "E",
    "CH ": "C",
    "CH6": "MYG",
    "CH7": "KYG",
    "CHF": "X",
    "CHG": "X",
    "CHP": "G",
    "CHS": "X",
    "CIR": "R",
    "CJO": "GYG",
    "CLE": "L",
    "CLG": "K",
    "CLH": "K",
    "CLV": "AFG",
    "CM0": "N",
    "CME": "C",
    "CMH": "C",
    "CML": "C",
    "CMR": "C",
    "CMT": "C",
    "CNU": "U",
    "CP1": "C",
    "CPC": "X",
    "CPI": "X",
    "CQR": "GYG",
    "CR0": "TLG",
    "CR2": "GYG",
    "CR5": "G",
    "CR7": "KYG",
    "CR8": "HYG",
    "CRF": "TWG",
    "CRG": "THG",
    "CRK": "MYG",
    "CRO": "GYG",
    "CRQ": "QYG",
    "CRU": "EYG",
    "CRW": "ASG",
    "CRX": "ASG",
    "CS0": "C",
    "CS1": "C",
    "CS3": "C",
    "CS4": "C",
    "CS8": "N",
    "CSA": "C",
    "CSB": "C",
    "CSD": "C",
    "CSE": "C",
    "CSF": "C",
    "CSH": "SHG",
    "CSI": "G",
    "CSJ": "C",
    "CSL": "C",
    "CSO": "C",
    "CSP": "C",
    "CSR": "C",
    "CSS": "C",
    "CSU": "C",
    "CSW": "C",
    "CSX": "C",
    "CSY": "SYG",
    "CSZ": "C",
    "CTE": "W",
    "CTG": "T",
    "CTH": "T",
    "CUC": "X",
    "CWR": "S",
    "CXM": "M",
    "CY0": "C",
    "CY1": "C",
    "CY3": "C",
    "CY4": "C",
    "CYA": "C",
    "CYD": "C",
    "CYF": "C",
    "CYG": "C",
    "CYJ": "X",
    "CYM": "C",
    "CYQ": "C",
    "CYR": "C",
    "CYS": "C",
    "CZ2": "C",
    "CZO": "GYG",
    "CZZ": "C",
    "D11": "T",
    "D1P": "N",
    "D3 ": "N",
    "D33": "N",
    "D3P": "G",
    "D3T": "T",
    "D4M": "T",
    "D4P": "X",
    "DA ": "A",
    "DA2": "X",
    "DAB": "A",
    "DAH": "F",
    "DAL": "A",
    "DAR": "R",
    "DAS": "D",
    "DBB": "T",
    "DBM": "N",
    "DBS": "S",
    "DBU": "T",
    "DBY": "Y",
    "DBZ": "A",
    "DC ": "C",
    "DC2": "C",
    "DCG": "G",
    "DCI": "X",
    "DCL": "X",
    "DCT": "C",
    "DCY": "C",
    "DDE": "H",
    "DDG": "G",
    "DDN": "U",
    "DDX": "N",
    "DFC": "C",
    "DFG": "G",
    "DFI": "X",
    "DFO": "X",
    "DFT": "N",
    "DG ": "G",
    "DGH": "G",
    "DGI": "G",
    "DGL": "E",
    "DGN": "Q",
    "DHA": "S",
    "DHI": "H",
    "DHL": "X",
    "DHN": "V",
    "DHP": "X",
    "DHU": "U",
    "DHV": "V",
    "DI ": "I",
    "DIL": "I",
    "DIR": "R",
    "DIV": "V",
    "DLE": "L",
    "DLS": "K",
    "DLY": "K",
    "DM0": "K",
    "DMH": "N",
    "DMK": "D",
    "DMT": "X",
    "DN ": "N",
    "DNE": "L",
    "DNG": "L",
    "DNL": "K",
    "DNM": "L",
    "DNP": "A",
    "DNR": "C",
    "DNS": "K",
    "DOA": "X",
    "DOC": "C",
    "DOH": "D",
    "DON": "L",
    "DPB": "T",
    "DPH": "F",
    "DPL": "P",
    "DPP": "A",
    "DPQ": "Y",
    "DPR": "P",
    "DPY": "N",
    "DRM": "U",
    "DRP": "N",
    "DRT": "T",
    "DRZ": "N",
    "DSE": "S",
    "DSG": "N",
    "DSN": "S",
    "DSP": "D",
    "DT ": "T",
    "DTH": "T",
    "DTR": "W",
    "DTY": "Y",
    "DU ": "U",
    "DVA": "V",
    "DXD": "N",
    "DXN": "N",
    "DYG": "DYG",
    "DYS": "C",
    "DZM": "A",
    "E  ": "A",
    "E1X": "A",
    "ECC": "Q",
    "EDA": "A",
    "EFC": "C",
    "EHP": "F",
    "EIT": "T",
    "ENP": "N",
    "ESB": "Y",
    "ESC": "M",
    "EXB": "X",
    "EXY": "L",
    "EY5": "N",
    "EYS": "X",
    "F2F": "F",
    "FA2": "A",
    "FA5": "N",
    "FAG": "N",
    "FAI": "N",
    "FB5": "A",
    "FB6": "A",
    "FCL": "F",
    "FFD": "N",
    "FGA": "E",
    "FGL": "G",
    "FGP": "S",
    "FHL": "X",
    "FHO": "K",
    "FHU": "U",
    "FLA": "A",
    "FLE": "L",
    "FLT": "Y",
    "FME": "M",
    "FMG": "G",
    "FMU": "N",
    "FOE": "C",
    "FOX": "G",
    "FP9": "P",
    "FPA": "F",
    "FRD": "X",
    "FT6": "W",
    "FTR": "W",
    "FTY": "Y",
    "FVA": "V",
    "FZN": "K",
    "G  ": "G",
    "G25": "G",
    "G2L": "G",
    "G2S": "G",
    "G31": "G",
    "G32": "G",
    "G33": "G",
    "G36": "G",
    "G38": "G",
    "G42": "G",
    "G46": "G",
    "G47": "G",
    "G48": "G",
    "G49": "G",
    "G4P": "N",
    "G7M": "G",
    "GAO": "G",
    "GAU": "E",
    "GCK": "C",
    "GCM": "X",
    "GDP": "G",
    "GDR": "G",
    "GFL": "G",
    "GGL": "E",
    "GH3": "G",
    "GHG": "Q",
    "GHP": "G",
    "GL3": "G",
    "GLH": "Q",
    "GLJ": "E",
    "GLK": "E",
    "GLM": "X",
    "GLN": "Q",
    "GLQ": "E",
    "GLU": "E",
    "GLX": "Z",
    "GLY": "G",
    "GLZ": "G",
    "GMA": "E",
    "GMS": "G",
    "GMU": "U",
    "GN7": "G",
    "GND": "X",
    "GNE": "N",
    "GOM": "G",
    "GPL": "K",
    "GS ": "G",
    "GSC": "G",
    "GSR": "G",
    "GSS": "G",
    "GSU": "E",
    "GT9": "C",
    "GTP": "G",
    "GVL": "X",
    "GYC": "CYG",
    "GYS": "SYG",
    "H2U": "U",
    "H5M": "P",
    "HAC": "A",
    "HAR": "R",
    "HBN": "H",
    "HCS": "X",
    "HDP": "U",
    "HEU": "U",
    "HFA": "X",
    "HGL": "X",
    "HHI": "H",
    "HHK": "AK",
    "HIA": "H",
    "HIC": "H",
    "HIP": "H",
    "HIQ": "H",
    "HIS": "H",
    "HL2": "L",
    "HLU": "L",
    "HMR": "R",
    "HOL": "N",
    "HPC": "F",
    "HPE": "F",
    "HPH": "F",
    "HPQ": "F",
    "HQA": "A",
    "HRG": "R",
    "HRP": "W",
    "HS8": "H",
    "HS9": "H",
    "HSE": "S",
    "HSL": "S",
    "HSO": "H",
    "HTI": "C",
    "HTN": "N",
    "HTR": "W",
    "HV5": "A",
    "HVA": "V",
    "HY3": "P",
    "HYP": "P",
    "HZP": "P",
    "I  ": "I",
    "I2M": "I",
    "I58": "K",
    "I5C": "C",
    "IAM": "A",
    "IAR": "R",
    "IAS": "D",
    "IC ": "C",
    "IEL": "K",
    "IEY": "HYG",
    "IG ": "G",
    "IGL": "G",
    "IGU": "G",
    "IIC": "SHG",
    "IIL": "I",
    "ILE": "I",
    "ILG": "E",
    "ILX": "I",
    "IMC": "C",
    "IML": "I",
    "IOY": "F",
    "IPG": "G",
    "IPN": "N",
    "IRN": "N",
    "IT1": "K",
    "IU ": "U",
    "IYR": "Y",
    "IYT": "T",
    "IZO": "M",
    "JJJ": "C",
    "JJK": "C",
    "JJL": "C",
    "JW5": "N",
    "K1R": "C",
    "KAG": "G",
    "KCX": "K",
    "KGC": "K",
    "KNB": "A",
    "KOR": "M",
    "KPI": "K",
    "KST": "K",
    "KYQ": "K",
    "L2A": "X",
    "LA2": "K",
    "LAA": "D",
    "LAL": "A",
    "LBY": "K",
    "LC ": "C",
    "LCA": "A",
    "LCC": "N",
    "LCG": "G",
    "LCH": "N",
    "LCK": "K",
    "LCX": "K",
    "LDH": "K",
    "LED": "L",
    "LEF": "L",
    "LEH": "L",
    "LEI": "V",
    "LEM": "L",
    "LEN": "L",
    "LET": "X",
    "LEU": "L",
    "LEX": "L",
    "LG ": "G",
    "LGP": "G",
    "LHC": "X",
    "LHU": "U",
    "LKC": "N",
    "LLP": "K",
    "LLY": "K",
    "LME": "E",
    "LMF": "K",
    "LMQ": "Q",
    "LMS": "N",
    "LP6": "K",
    "LPD": "P",
    "LPG": "G",
    "LPL": "X",
    "LPS": "S",
    "LSO": "X",
    "LTA": "X",
    "LTR": "W",
    "LVG": "G",
    "LVN": "V",
    "LYF": "K",
    "LYK": "K",
    "LYM": "K",
    "LYN": "K",
    "LYR": "K",
    "LYS": "K",
    "LYX": "K",
    "LYZ": "K",
    "M0H": "C",
    "M1G": "G",
    "M2G": "G",
    "M2L": "K",
    "M2S": "M",
    "M30": "G",
    "M3L": "K",
    "M5M": "C",
    "MA ": "A",
    "MA6": "A",
    "MA7": "A",
    "MAA": "A",
    "MAD": "A",
    "MAI": "R",
    "MBQ": "Y",
    "MBZ": "N",
    "MC1": "S",
    "MCG": "X",
    "MCL": "K",
    "MCS": "C",
    "MCY": "C",
    "MD3": "C",
    "MD6": "G",
    "MDH": "X",
    "MDO": "ASG",
    "MDR": "N",
    "MEA": "F",
    "MED": "M",
    "MEG": "E",
    "MEN": "N",
    "MEP": "U",
    "MEQ": "Q",
    "MET": "M",
    "MEU": "G",
    "MF3": "X",
    "MFC": "GYG",
    "MG1": "G",
    "MGG": "R",
    "MGN": "Q",
    "MGQ": "A",
    "MGV": "G",
    "MGY": "G",
    "MHL": "L",
    "MHO": "M",
    "MHS": "H",
    "MIA": "A",
    "MIS": "S",
    "MK8": "L",
    "ML3": "K",
    "MLE": "L",
    "MLL": "L",
    "MLY": "K",
    "MLZ": "K",
    "MME": "M",
    "MMO": "R",
    "MMT": "T",
    "MND": "N",
    "MNL": "L",
    "MNU": "U",
    "MNV": "V",
    "MOD": "X",
    "MP8": "P",
    "MPH": "X",
    "MPJ": "X",
    "MPQ": "G",
    "MRG": "G",
    "MSA": "G",
    "MSE": "M",
    "MSL": "M",
    "MSO": "M",
    "MSP": "X",
    "MT2": "M",
    "MTR": "T",
    "MTU": "A",
    "MTY": "Y",
    "MVA": "V",
    "N  ": "N",
    "N10": "S",
    "N2C": "X",
    "N5I": "N",
    "N5M": "C",
    "N6G": "G",
    "N7P": "P",
    "NA8": "A",
    "NAL": "A",
    "NAM": "A",
    "NB8": "N",
    "NBQ": "Y",
    "NC1": "S",
    "NCB": "A",
    "NCX": "N",
    "NCY": "X",
    "NDF": "F",
    "NDN": "U",
    "NEM": "H",
    "NEP": "H",
    "NF2": "N",
    "NFA": "F",
    "NHL": "E",
    "NIT": "X",
    "NIY": "Y",
    "NLE": "L",
    "NLN": "L",
    "NLO": "L",
    "NLP": "L",
    "NLQ": "Q",
    "NMC": "G",
    "NMM": "R",
    "NMS": "T",
    "NMT": "T",
    "NNH": "R",
    "NP3": "N",
    "NPH": "C",
    "NPI": "A",
    "NRP": "LYG",
    "NRQ": "MYG",
    "NSK": "X",
    "NTY": "Y",
    "NVA": "V",
    "NYC": "TWG",
    "NYG": "NYG",
    "NYM": "N",
    "NYS": "C",
    "NZH": "H",
    "O12": "X",
    "O2C": "N",
    "O2G": "G",
    "OAD": "N",
    "OAS": "S",
    "OBF": "X",
    "OBS": "X",
    "OCS": "C",
    "OCY": "C",
    "ODP": "N",
    "OHI": "H",
    "OHS": "D",
    "OIC": "X",
    "OIP": "I",
    "OLE": "X",
    "OLT": "T",
    "OLZ": "S",
    "OMC": "C",
    "OMG": "G",
    "OMT": "M",
    "OMU": "U",
    "ONE": "U",
    "ONH": "A",
    "ONL": "X",
    "OPR": "R",
    "ORN": "A",
    "ORQ": "R",
    "OSE": "S",
    "OTB": "X",
    "OTH": "T",
    "OTY": "Y",
    "OXX": "D",
    "P  ": "G",
    "P1L": "C",
    "P1P": "N",
    "P2T": "T",
    "P2U": "U",
    "P2Y": "P",
    "P5P": "A",
    "PAQ": "Y",
    "PAS": "D",
    "PAT": "W",
    "PAU": "A",
    "PBB": "C",
    "PBF": "F",
    "PBT": "N",
    "PCA": "E",
    "PCC": "P",
    "PCE": "X",
    "PCS": "F",
    "PDL": "X",
    "PDU": "U",
    "PEC": "C",
    "PF5": "F",
    "PFF": "F",
    "PFX": "X",
    "PG1": "S",
    "PG7": "G",
    "PG9": "G",
    "PGL": "X",
    "PGN": "G",
    "PGP": "G",
    "PGY": "G",
    "PHA": "F",
    "PHD": "D",
    "PHE": "F",
    "PHI": "F",
    "PHL": "F",
    "PHM": "F",
    "PIA": "AYG",
    "PIV": "X",
    "PLE": "L",
    "PM3": "F",
    "PMT": "C",
    "POM": "P",
    "PPN": "F",
    "PPU": "A",
    "PPW": "G",
    "PQ1": "N",
    "PR3": "C",
    "PR5": "A",
    "PR9": "P",
    "PRN": "A",
    "PRO": "P",
    "PRS": "P",
    "PSA": "F",
    "PSH": "H",
    "PST": "T",
    "PSU": "U",
    "PSW": "C",
    "PTA": "X",
    "PTH": "Y",
    "PTM": "Y",
    "PTR": "Y",
    "PU ": "A",
    "PUY": "N",
    "PVH": "H",
    "PVL": "X",
    "PYA": "A",
    "PYO": "U",
    "PYX": "C",
    "PYY": "N",
    "QLG": "QLG",
    "QMM": "Q",
    "QPA": "C",
    "QPH": "F",
    "QUO": "G",
    "R  ": "A",
    "R1A": "C",
    "R4K": "W",
    "RC7": "HYG",
    "RE0": "W",
    "RE3": "W",
    "RIA": "A",
    "RMP": "A",
    "RON": "X",
    "RT ": "T",
    "RTP": "N",
    "S1H": "S",
    "S2C": "C",
    "S2D": "A",
    "S2M": "T",
    "S2P": "A",
    "S4A": "A",
    "S4C": "C",
    "S4G": "G",
    "S4U": "U",
    "S6G": "G",
    "SAC": "S",
    "SAH": "C",
    "SAR": "G",
    "SBL": "S",
    "SC ": "C",
    "SCH": "C",
    "SCS": "C",
    "SCY": "C",
    "SD2": "X",
    "SDG": "G",
    "SDP": "S",
    "SEB": "S",
    "SEC": "A",
    "SEG": "A",
    "SEL": "S",
    "SEM": "S",
    "SEN": "S",
    "SEP": "S",
    "SER": "S",
    "SET": "S",
    "SGB": "S",
    "SHC": "C",
    "SHP": "G",
    "SHR": "K",
    "SIB": "C",
    "SIC": "DC",
    "SLA": "P",
    "SLR": "P",
    "SLZ": "K",
    "SMC": "C",
    "SME": "M",
    "SMF": "F",
    "SMP": "A",
    "SMT": "T",
    "SNC": "C",
    "SNN": "N",
    "SOC": "C",
    "SOS": "N",
    "SOY": "S",
    "SPT": "T",
    "SRA": "A",
    "SSU": "U",
    "STY": "Y",
    "SUB": "X",
    "SUI": "DG",
    "SUN": "S",
    "SUR": "U",
    "SVA": "S",
    "SVV": "S",
    "SVW": "S",
    "SVX": "S",
    "SVY": "S",
    "SVZ": "X",
    "SWG": "SWG",
    "SYS": "C",
    "T  ": "T",
    "T11": "F",
    "T23": "T",
    "T2S": "T",
    "T2T": "N",
    "T31": "U",
    "T32": "T",
    "T36": "T",
    "T37": "T",
    "T38": "T",
    "T39": "T",
    "T3P": "T",
    "T41": "T",
    "T48": "T",
    "T49": "T",
    "T4S": "T",
    "T5O": "U",
    "T5S": "T",
    "T66": "X",
    "T6A": "A",
    "TA3": "T",
    "TA4": "X",
    "TAF": "T",
    "TAL": "N",
    "TAV": "D",
    "TBG": "V",
    "TBM": "T",
    "TC1": "C",
    "TCP": "T",
    "TCQ": "Y",
    "TCR": "W",
    "TCY": "A",
    "TDD": "L",
    "TDY": "T",
    "TFE": "T",
    "TFO": "A",
    "TFQ": "F",
    "TFT": "T",
    "TGP": "G",
    "TH6": "T",
    "THC": "T",
    "THO": "X",
    "THR": "T",
    "THX": "N",
    "THZ": "R",
    "TIH": "A",
    "TLB": "N",
    "TLC": "T",
    "TLN": "U",
    "TMB": "T",
    "TMD": "T",
    "TNB": "C",
    "TNR": "S",
    "TOX": "W",
    "TP1": "T",
    "TPC": "C",
    "TPG": "G",
    "TPH": "X",
    "TPL": "W",
    "TPO": "T",
    "TPQ": "Y",
    "TQI": "W",
    "TQQ": "W",
    "TRF": "W",
    "TRG": "K",
    "TRN": "W",
    "TRO": "W",
    "TRP": "W",
    "TRQ": "W",
    "TRW": "W",
    "TRX": "W",
    "TS ": "N",
    "TST": "X",
    "TT ": "N",
    "TTD": "T",
    "TTI": "U",
    "TTM": "T",
    "TTQ": "W",
    "TTS": "Y",
    "TY1": "Y",
    "TY2": "Y",
    "TY3": "Y",
    "TY5": "Y",
    "TYB": "Y",
    "TYI": "Y",
    "TYJ": "Y",
    "TYN": "Y",
    "TYO": "Y",
    "TYQ": "Y",
    "TYR": "Y",
    "TYS": "Y",
    "TYT": "Y",
    "TYU": "N",
    "TYW": "Y",
    "TYX": "X",
    "TYY": "Y",
    "TZB": "X",
    "TZO": "X",
    "U  ": "U",
    "U25": "U",
    "U2L": "U",
    "U2N": "U",
    "U2P": "U",
    "U31": "U",
    "U33": "U",
    "U34": "U",
    "U36": "U",
    "U37": "U",
    "U8U": "U",
    "UAR": "U",
    "UCL": "U",
    "UD5": "U",
    "UDP": "N",
    "UFP": "N",
    "UFR": "U",
    "UFT": "U",
    "UMA": "A",
    "UMP": "U",
    "UMS": "U",
    "UN1": "X",
    "UN2": "X",
    "UNK": "X",
    "UR3": "U",
    "URD": "U",
    "US1": "U",
    "US2": "U",
    "US3": "T",
    "US5": "U",
    "USM": "U",
    "VAD": "V",
    "VAF": "V",
    "VAL": "V",
    "VB1": "K",
    "VDL": "X",
    "VLL": "X",
    "VLM": "X",
    "VMS": "X",
    "VOL": "X",
    "WCR": "GYG",
    "X  ": "G",
    "X2W": "E",
    "X4A": "N",
    "X9Q": "AFG",
    "XAD": "A",
    "XAE": "N",
    "XAL": "A",
    "XAR": "N",
    "XCL": "C",
    "XCN": "C",
    "XCP": "X",
    "XCR": "C",
    "XCS": "N",
    "XCT": "C",
    "XCY": "C",
    "XGA": "N",
    "XGL": "G",
    "XGR": "G",
    "XGU": "G",
    "XPR": "P",
    "XSN": "N",
    "XTH": "T",
    "XTL": "T",
    "XTR": "T",
    "XTS": "G",
    "XTY": "N",
    "XUA": "A",
    "XUG": "G",
    "XX1": "K",
    "XXY": "THG",
    "XYG": "DYG",
    "Y  ": "A",
    "YCM": "C",
    "YG ": "G",
    "YOF": "Y",
    "YRR": "N",
    "YYG": "G",
    "Z  ": "C",
    "Z01": "A",
    "ZAD": "A",
    "ZAL": "A",
    "ZBC": "C",
    "ZBU": "U",
    "ZCL": "F",
    "ZCY": "C",
    "ZDU": "U",
    "ZFB": "X",
    "ZGU": "G",
    "ZHP": "N",
    "ZTH": "T",
    "ZU0": "T",
    "ZZJ": "A",
}


restypes = [
    "A",
    "R",
    "N",
    "D",
    "C",
    "Q",
    "E",
    "G",
    "H",
    "I",
    "L",
    "K",
    "M",
    "F",
    "P",
    "S",
    "T",
    "W",
    "Y",
    "V",
    "X",
]
restypes_with_x = restypes + ["X"]
restype_order_with_x = {restype: i for i, restype in enumerate(restypes_with_x)}


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

loss_dict_not_change_length = {
    "Deamidated[Q]": ("Q", "E"),
    "Deamidated[N]": ("N", "D"),
}

loss_dict_loss_length = {
    # "Lys-loss[ProteinC-termK]": ("K", ""),
    # "Arg-loss[AnyC-termR]": ("R", ""),
    # Met-loss+Acetyl[ProteinN-termM]
    # Gly-loss+Amide[AnyC-termG]
    # Met-loss[ProteinN-termM]
}

loss_dict_add_length = {
    # "Arg[AnyN-term]": ("", "R"),
}


denovo_not_pred_set = {
    # "Lys-loss[ProteinC-termK]",
    # "Arg-loss[AnyC-termR]",
    "Deamidated[Q]",
    "Deamidated[N]",
    # "Arg[AnyN-term]",
}

same_modification = {"GG[K]": "Ubiquitination[K]"}


def preprocess_modifications(data):

    return data


def get_aa(s):
    if s.split("->")[1][:3] in amino_acids_3to1_map:
        return amino_acids_3to1_map[s.split("->")[1][:3]]
    elif s.split("[")[0][-3:] in amino_acids_3to1_map:
        return amino_acids_3to1_map[s.split("[")[0][-3:]]
    else:
        return None


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


def check_fix_mod(data):
    cnt_c = 0
    for i in range(len(data["small"]["1"]["peptide"])):
        if data["small"]["1"]["peptide"][i] == "C":
            cnt_c += 1
    cnt_mod = 0
    for mod in data["small"]["1"]["mods"]:
        if mod[1] == "Carbamidomethyl[C]":
            cnt_mod += 1
    if cnt_c != cnt_mod:
        print(
            "not good",
            cnt_c,
            cnt_mod,
            data["small"]["1"]["peptide"],
            data["small"]["1"]["mods"],
        )
        mgf_file_path = (
            r"/mnt/vepfs/fs_users/zhaojiale/Contrast_MS_Pep/scripts/mgfs/mod.mgf"
        )
        # try:
        #     os.remove(mgf_file_path)
        # except:
        #     pass
        # for i in range(1, 11):
        #     print(data["small"][f"{i}"]["peptide"], data["small"][f"{i}"]["mods"], data["small"][f"{i}"]["pfind_score"])
        # assert 0, data
        try:
            mgf = data["small"]["mgf"]
        except:
            mgf = data["small"][list(data["small"].keys())[0]]
        modinfo = ""
        for mod in data["small"]["1"]["mods"]:
            modinfo += "_" + str(mod[0]) + mod[1]
        append_mgf(
            mgf_file_path,
            f'{data["small"]["1"]["peptide"]}{modinfo}',
            mgf["precursor_charge"],
            0,
            mgf["precursor_mz"],
            mgf["mz_array"],
            mgf["intensity"],
            pepseq=data["small"]["1"]["peptide"],
        )
    elif cnt_c > 0:
        print("good")


def pep_mod_2_pep(peptide, mods):
    for mod in mods:
        if "->" in mod[1]:
            new_residue = get_aa(mod[1])
            if new_residue != None:
                # peptide[mod[0]] = new_residue
                # peptide = peptide[:mod[0]-1] + new_residue + peptide[mod[0]:]
                if mod[0] == 0:
                    mod = (mod[0] + 1, mod[1])
                    # mod[0] += 1
                if mod[0] > len(peptide):
                    mod = (mod[0] - 1, mod[1])
                peptide_list = list(peptide)
                try:
                    peptide_list[mod[0] - 1] = new_residue
                except:
                    assert 0, (peptide_list, mod)
                peptide = "".join(peptide_list)
        if mod[1] in loss_dict_not_change_length:
            new_residue = loss_dict_not_change_length[mod[1]][1]
            if mod[0] == 0:
                mod = (mod[0] + 1, mod[1])
                # mod[0] += 1
            if mod[0] > len(peptide):
                mod = (mod[0] - 1, mod[1])
            peptide_list = list(peptide)
            try:
                # assert peptide_list[mod[0] - 1] == loss_dict_not_change_length[mod[1]][0]
                peptide_list[mod[0] - 1] = new_residue
            except:
                assert 0, (peptide_list, mod, mods)
            peptide = "".join(peptide_list)

        if mod[1] in loss_dict_loss_length:
            new_residue = loss_dict_loss_length[mod[1]][1]
            if mod[0] == 0:
                mod = (mod[0] + 1, mod[1])
                # mod[0] += 1
            if mod[0] > len(peptide):
                mod = (mod[0] - 1, mod[1])
            peptide_list = list(peptide)
            try:
                # assert peptide_list[mod[0] - 1] == loss_dict_loss_length[mod[1]][0]
                peptide_list[mod[0] - 1] = new_residue
            except:
                assert 0, (peptide_list, mod)
            peptide = "".join(peptide_list)

        if mod[1] in loss_dict_add_length:
            new_residue = loss_dict_add_length[mod[1]][1]
            if mod[0] == 0:
                mod = (mod[0] + 1, mod[1])
                # mod[0] += 1
            if mod[0] > len(peptide):
                mod = (mod[0] - 1, mod[1])
            peptide_list = list(peptide)
            assert mod[0] - 1 == 0, "assuming only C term Arg"
            # assert peptide_list[mod[0] - 1] == loss_dict_not_change_length[mod[1]][0]
            peptide_list.insert(0, new_residue)
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


class LMDB2Dataset:
    def __init__(self, db_path):
        self.db_path = db_path
        assert os.path.isfile(self.db_path), "{} not found".format(self.db_path)

        self.env = self.connect_db(self.db_path)
        with self.env.begin() as txn:
            self._keys = list(txn.cursor().iternext(values=False))

    def connect_db(self, lmdb_path=None):
        env = lmdb.open(
            lmdb_path,
            subdir=False,
            readonly=True,
            lock=False,
            readahead=True,
            meminit=False,
            max_readers=128,
        )
        return env

    def __len__(self):
        return len(self._keys)

    @lru_cache(maxsize=16)
    def __getitem__(self, idx):
        datapoint_pickled = self.env.begin().get(self._keys[idx])
        data = pickle.loads(datapoint_pickled)
        return data


def random_sample_from_dict(d):
    all_values = []
    keys_for_values = []
    for key, value in d.items():
        all_values.extend(value)
        keys_for_values.extend([key] * len(value))
    random_index = random.randint(0, len(all_values) - 1)
    return keys_for_values[random_index], all_values[random_index]


def random_sample_from_dict_weighted(d):
    """
    从字典中随机采样一个键值对。
    """
    all_values = []
    keys_for_values = []
    for key, value in d.items():
        all_values.extend(value)
        keys_for_values.extend([key] * len(value))
    if not all_values:
        return None, None  # 处理空情况
    random_index = random.randint(0, len(all_values) - 1)
    return keys_for_values[random_index], all_values[random_index]


class WeightedMultiZipLMDBDataset:
    def __init__(
        self,
        db_path,
        splits=["new1", "new2", "new3", "new4", "new5", "new6", "new61"],
        is_train=True,
        ratio=0.01,
        ft_tims=0,
    ):
        self.db_path = db_path
        self.env_dict = {}

        if is_train:
            if not ft_tims:
                self.key_dict = pickle.load(
                    open(os.path.join(db_path, "train_keys61.pkl"), "rb")
                )
            else:
                self.key_dict = pickle.load(
                    open(os.path.join(db_path, "train_keys61.pkl"), "rb")
                )
        else:
            if not ft_tims:
                self.key_dict = pickle.load(
                    open(os.path.join(db_path, "test_keys61.pkl"), "rb")
                )
            else:
                self.key_dict = pickle.load(
                    open(os.path.join(db_path, "test_keys9_tims.pkl"), "rb")
                )

        # 加载预处理后的扩展字典文件
        self.expanded_key_dict = pickle.load(
            open(os.path.join(db_path, "expanded_train_keys61.pkl"), "rb")
        )

        for split in splits:
            sub_path = os.path.join(self.db_path, split + ".lmdb")
            assert os.path.isfile(sub_path), f"{sub_path} not found"
            self.env_dict[split] = self.connect_db(sub_path)

        # with self.env.begin() as txn:
        # self._keys = list(txn.cursor().iternext(values=False))
        self._keys = list(self.expanded_key_dict.keys())

    def connect_db(self, lmdb_path=None):
        env = lmdb.open(
            lmdb_path,
            subdir=False,
            readonly=True,
            lock=False,
            readahead=True,
            meminit=False,
            max_readers=128,
        )
        return env

    def __len__(self):
        return len(self._keys)

    @lru_cache(maxsize=16)
    def __getitem__(self, idx):
        expanded_key = self._keys[idx]
        peptide = self.expanded_key_dict[expanded_key]
        # 从原始字典中获取对应的文件和LMDB键
        split, key = random_sample_from_dict(self.key_dict[peptide])
        datapoint_pickled = self.env_dict[split].begin().get(key)
        data = pickle.loads(gzip.decompress(datapoint_pickled))
        return data


class MultiZipLMDB2Dataset:
    def __init__(
        self,
        db_path,
        splits=["new1", "new2", "new3", "new4", "new5", "new6", "new61"],
        is_train=True,
        ratio=0.01,
        ft_tims=0,
    ):
        self.db_path = db_path
        self.env_dict = {}
        weighted = False
        self.weighted = weighted
        assert not weighted, "not ready"

        if self.weighted:
            if is_train:
                self.weighted_key_dict = pickle.load(
                    open(join(db_path, "weighted_train_keys.pkl"), "rb")
                )
            else:
                self.weighted_key_dict = pickle.load(
                    open(join(db_path, "weighted_test_keys.pkl"), "rb")
                )
            for split in splits:
                sub_path = join(self.db_path, split + ".lmdb")
                assert os.path.isfile(sub_path), f"{sub_path} not found"
                self.env_dict[split] = self.connect_db(sub_path)

            # 提取所有键对
            self.keys = []
            for peptide, file_dict in self.weighted_key_dict.items():
                for file, keys in file_dict.items():
                    self.keys.extend(keys)

            # 统一的键空间
            self.all_keys = list(self.keys)
        else:
            if is_train:
                if not ft_tims:
                    self.key_dict = pickle.load(
                        open(join(db_path, "train_keys61.pkl"), "rb")
                    )
                else:
                    self.key_dict = pickle.load(
                        open(join(db_path, "train_keys61.pkl"), "rb")
                    )
            else:
                if not ft_tims:
                    self.key_dict = pickle.load(
                        open(join(db_path, "test_keys61.pkl"), "rb")
                    )
                else:
                    self.key_dict = pickle.load(
                        open(join(db_path, "test_keys9_tims.pkl"), "rb")
                    )

            for split in splits:
                sub_path = join(self.db_path, split + ".lmdb")
                assert os.path.isfile(sub_path), "{} not found".format(sub_path)
                self.env_dict[split] = self.connect_db(sub_path)

            # with self.env.begin() as txn:
            #     self._keys = list(txn.cursor().iternext(values=False))
            self._keys = list(self.key_dict.keys())

    def connect_db(self, lmdb_path=None):
        env = lmdb.open(
            lmdb_path,
            subdir=False,
            readonly=True,
            lock=False,
            readahead=True,
            meminit=False,
            max_readers=128,
        )
        return env

    def __len__(self):
        return len(self._keys)

    @lru_cache(maxsize=16)
    def __getitem__(self, idx):
        if self.weighted:
            pass
            # random_peptide, random_file, sample_key = random_sample_from_weighted_dict(self.weighted_key_dict)
            # datapoint_pickled = self.env_dict[random_file].begin().get(sample_key)
            # data = pickle.loads(gzip.decompress(datapoint_pickled))
            # return data
        else:
            split, key = random_sample_from_dict(self.key_dict[self._keys[idx]])
            datapoint_pickled = self.env_dict[split].begin().get(key)
            data = pickle.loads(gzip.decompress(datapoint_pickled))
            return data


class ZipLMDB2Dataset:
    def __init__(
        self,
        db_path,
        split,
        sample_size=50,
        num=10000,
        bsz=128,
        massive=False,
        keys=None,
    ):
        self.db_path = db_path
        assert os.path.isfile(self.db_path), "{} not found".format(self.db_path)

        self.env = self.connect_db(self.db_path)
        if keys is None or isinstance(keys, str):
            with self.env.begin() as txn:
                self._keys = list(txn.cursor().iternext(values=False))
            if isinstance(keys, str):
                with open(keys, "wb") as f:
                    pickle.dump(self._keys, f)
                print(f"keys saved at {keys}")

            if "test" in split and sample_size > 1:
                self._keys = self._keys[:: int(sample_size)]

            if split == "train" and not massive:
                self._keys = pickle.load(
                    open(join(os.path.dirname(db_path), {"pkl"}, "{split}.pkl"), "rb")
                )
        else:
            self._keys = keys

    def connect_db(self, lmdb_path=None):
        env = lmdb.open(
            lmdb_path,
            subdir=False,
            readonly=True,
            lock=False,
            readahead=True,
            meminit=False,
            max_readers=128,
        )
        return env

    def __len__(self):
        return len(self._keys)

    @lru_cache(maxsize=16)
    def __getitem__(self, idx):
        datapoint_pickled = self.env.begin().get(self._keys[idx])
        data = pickle.loads(gzip.decompress(datapoint_pickled))
        return data


from tqdm import tqdm


class pFindIs2reDataset(BaseWrapperDataset):
    """A wrapper around a LMDB database that reads and returns items from it
    lazily."""

    def __init__(
        self, dataset, args, crop_rational=1 / 4, is_train=False, finetune=False
    ):
        super().__init__(dataset)
        self.dataset = dataset
        self.is_train = is_train
        self.args = args
        self.softmax = nn.Softmax(dim=-1)
        self.set_epoch(None)
        modification_meta_dict_path = (
            args.modification_meta_dict_path
        )  # r"/mnt/vepfs/fs_users/zhaojiale/Contrast_MS_Pep/consts/modification_meta_dict.pkl"
        self.modification_meta_dict = pickle.load(
            open(modification_meta_dict_path, "rb")
        )
        tokenize1_pkl_path = (
            args.tokenize1_pkl_path
        )  # r"/mnt/vepfs/fs_users/zhaojiale/Contrast_MS_Pep/consts/tokenize1.pkl"
        self.tokenize1_dict = pickle.load(open(tokenize1_pkl_path, "rb"))
        self.modification_meta_dict["Ammonia-loss"] = self.modification_meta_dict[
            "Ammonia-loss[AnyN-termC]"
        ]
        self.min_mz = 0.0
        self.max_mz = 6500.0
        self.remove_precursor_tol = 2
        self.n_peaks = self.args.cutoff_spectra
        self.min_intensity = 0.0
        # ION_TYPES = ['b', 'b-NH3', 'b-H2O', 'b-ModLoss', 'y', 'y-NH3', 'y-H2O', 'y-ModLoss']
        if not self.args.no_mod:
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
        else:
            self.ion_types = ["b", "b-NH3", "b-H2O", "y", "y-NH3", "y-H2O"]

        self.token_dict = {
            "A": 1,  # 71.03711
            "B": 2,  # 0.000
            "C": 3,  # 103.009180
            "D": 4,  # 115.02694
            "E": 5,  # 129.04259
            "F": 6,  # 147.06841
            "G": 7,  # 57.02146
            "H": 8,  # 137.05891
            "I": 9,  # 0.000
            "J": 10,  # 0.000
            "K": 11,  # 128.09496
            "L": 12,  # 113.08406
            "M": 13,  # 131.04049
            "N": 14,  # 114.04293
            "O": 15,  # 0.00000
            "P": 16,  # 97.05276
            "Q": 17,  # 128.05858
            "R": 18,  # 156.10111
            "S": 19,  # 87.03203
            "T": 20,  # 101.04768
            "U": 21,  # 0.00000
            "V": 22,  # 99.06841
            "W": 23,  # 186.07931
            "X": 24,  # 0.0000
            "Y": 25,  # 163.06333
            "Z": 26,  # 0.00000
        }

        try:
            # self.pept_to_spec_dict = pickle.load(open(join(self.args.data, "pept_to_spec.pkl"), 'rb'))
            if not finetune:
                pep2spec_path = join(self.args.data, "pept_to_spec61.lmdb")
            else:
                pep2spec_path = join(
                    r"/mnt/vepfs/fs_users/zhaojiale/dataset/lmdbs",
                    "pept_to_spec61.lmdb",
                )
                # pep2spec_path = join(self.args.data, "pept_to_spec3.lmdb")

            env_read = lmdb.open(
                pep2spec_path,
                subdir=False,
                readonly=True,
                lock=False,
                readahead=False,
                meminit=False,
                max_readers=1,
                map_size=int(1e9),
            )
            self.txn_read = env_read.begin()
            # keys = list(txn_read.cursor().iternext(values=False))
        except:
            print("warning: failed loading spec label")
            # if not self.args.inference:
            #     assert 0

    def __len__(self):
        return len(self.dataset)

    def set_epoch(self, epoch, **unused):
        super().set_epoch(epoch)
        self.epoch = epoch

    def __getitem__(self, idx: int):
        return self.__getitem_cached__(self.epoch, idx)

    @lru_cache(maxsize=16)
    def __getitem_cached__(self, epoch: int, idx):
        with data_utils.numpy_seed(self.args.seed, epoch, idx):

            spectrum = self.dataset[idx]
            try:
                spectrum = gzip.decompress(spectrum)
            except:
                pass
            try:
                spectrum = pickle.loads(spectrum)
            except:
                pass

            if self.args.only_decoy:
                # assert not self.args.inference
                spectrum = reorgnize(spectrum, self.args.start_id)

            try:
                if "mgf" in spectrum["small"].keys():
                    mgf = spectrum["small"]["mgf"]
                else:
                    mgf = spectrum["small"][list(spectrum["small"].keys())[0]]
            except:
                if "mgf" in spectrum["large"].keys():
                    mgf = spectrum["large"]["mgf"]
                else:
                    mgf = spectrum["large"][list(spectrum["large"].keys())[0]]
            rt = torch.tensor([float(mgf["rtinseconds"])])

            mz_array = torch.Tensor(mgf["mz_array"])
            intensity = torch.Tensor(mgf["intensity"])
            precursor_mz = mgf["precursor_mz"]
            precursor_charge = mgf["precursor_charge"]
            try:
                title = mgf["title"]
            except:
                title = spectrum["small"]["title"]

            modification_cls_label = torch.zeros(2610).long()

            flag_ITMS = False
            if self.is_train:
                start = self.args.start_id
            else:
                start = 2

            rets = []
            if "small" in spectrum.keys():
                num_sample = spectrum["small"]["num_res"]
                if (num_sample >= 2 or self.args.inference) and num_sample > 1:
                    for i in range(start, num_sample + 1):
                        # if not self.args.inference and not self.is_train:
                        #     assert spectrum["small"][str(i)]["is_target"]
                        rets.append(spectrum["small"][str(i)])

            residue_type = [spectrum["small"]["1"]["peptide"]]
            if "mods" not in spectrum["small"]["1"].keys():
                spectrum["small"]["1"]["mods"] = []
                spectrum["small"]["1"]["fdr_value"] = 0

            residue_type_after_mod_all = [
                pep_mod_2_pep(
                    spectrum["small"]["1"]["peptide"], spectrum["small"]["1"]["mods"]
                )
            ]
            if len(rets) > 0:
                residue_type.extend([_["peptide"] for _ in rets])
                residue_type_after_mod_all.extend(
                    [pep_mod_2_pep(_["peptide"], _["mods"]) for _ in rets]
                )

            residue_type_after_mod = [residue_type_after_mod_all[0]]
            modification = [[]]
            mod_label = torch.zeros(len(spectrum["small"]["1"]["peptide"]))
            mod_mass_label = torch.zeros(len(spectrum["small"]["1"]["peptide"]))

            assert len(spectrum["small"]["1"]["peptide"]) == len(residue_type[0]), (
                len(spectrum["small"]["1"]["peptide"]),
                len(residue_type[0]),
            )
            loss_residue = False
            add_residue = False
            for i in range(len(spectrum["small"]["1"]["mods"])):
                mod = spectrum["small"]["1"]["mods"][i]
                if mod[0] == 0:
                    mod = (mod[0] + 1, mod[1])
                    # mod[0] += 1
                if mod[0] > len(spectrum["small"]["1"]["peptide"]):
                    mod = (mod[0] - 1, mod[1])
                    # mod[0] -= 1
                try:
                    modification[0].append(
                        (mod[0], self.modification_meta_dict[mod[1]])
                    )
                except:
                    if mod[1] == "Methyl[KRHQNEDC]":
                        new_mod_str = (
                            f'Methyl[{spectrum["small"]["1"]["peptide"][mod[0]-1]}]'
                        )
                        if new_mod_str in self.modification_meta_dict.keys():
                            mod = (mod[0], new_mod_str)
                            modification[0].append(
                                (mod[0], self.modification_meta_dict[mod[1]])
                            )
                            spectrum["small"]["1"]["mods"][i] = mod
                        else:
                            spectrum["small"]["1"]["mods"].pop(i)
                            continue
                            # assert 0, (spectrum["small"]["1"]["mods"])
                try:
                    if "->" not in mod[1]:
                        modification_cls_label[
                            int(self.modification_meta_dict[mod[1]][2])
                        ] = 1
                except:
                    pass
                if mod[1] in loss_dict_loss_length.keys():
                    loss_residue = True

                if mod[1] in loss_dict_add_length.keys():
                    add_residue = True

                if (
                    "->" not in mod[1]
                    or mod[1].split("->")[1].split("[")[0]
                    not in amino_acids_3to1_map.keys()
                ):
                    if mod[1] not in denovo_not_pred_set:
                        if self.args.tokenize_mod == 0:
                            mod_label[mod[0] - 1] = (
                                self.modification_meta_dict[mod[1]][2].item() + 1
                            )
                        elif self.args.tokenize_mod == 1:
                            mod_label[mod[0] - 1] = self.tokenize1_dict[
                                self.modification_meta_dict[mod[1]][2].item()
                            ]["token_idx"]

                        mod_mass_label[mod[0] - 1] += self.modification_meta_dict[
                            mod[1]
                        ][1].item()
            if loss_residue:
                mod_label = mod_label[:-1]

            if add_residue:
                zero_tensor = torch.tensor([0.0])
                mod_label = torch.cat((zero_tensor, mod_label))

            for i in range(len(rets)):
                modification.append([])
                for mod in rets[i]["mods"]:
                    if mod[0] == 0:
                        mod = (mod[0] + 1, mod[1])  # 位置 meta信息
                    if mod[0] > len(rets[i]["peptide"]):
                        mod = (mod[0] - 1, mod[1])
                    try:
                        modification[i + 1].append(
                            (mod[0], self.modification_meta_dict[mod[1]])
                        )
                    except:
                        if mod[1] == "Methyl[KRHQNEDC]":
                            try:
                                mod = (
                                    mod[0],
                                    f'Methyl[{spectrum["small"]["1"]["peptide"][mod[0]-1]}]',
                                )
                                modification[i + 1].append(
                                    (mod[0], self.modification_meta_dict[mod[1]])
                                )
                            except:
                                pass

            assert len(modification) == len(residue_type), (
                len(modification),
                len(residue_type),
                len(residue_type_after_mod),
            )

            spectrum_ = sus.MsmsSpectrum(
                "",
                precursor_mz,
                precursor_charge,
                mz_array.numpy().astype(np.float64),
                intensity.numpy().astype(np.float32),
            )

            spectrum_.set_mz_range(self.min_mz, self.max_mz)
            if len(spectrum_.mz) == 0:
                raise ValueError
            spectrum_.remove_precursor_peak(20, "ppm")
            if len(spectrum_.mz) == 0:
                raise ValueError
            spectrum_.filter_intensity(self.min_intensity, self.n_peaks)
            if len(spectrum_.mz) == 0:
                raise ValueError
            spectrum_.scale_intensity("root", 1)
            intensity = spectrum_.intensity / np.linalg.norm(spectrum_.intensity)
            mz_array = spectrum_.mz
            mz_array, intensity = torch.Tensor(mz_array), torch.Tensor(intensity)

            (
                spec_label,
                ion_peak_flag,
                ion_peak_class,
                ion_res_num,
                res_idx,
                error_tol,
                ion_mz,
            ) = get_spectrum_prediction_label(
                spectrum["small"]["1"]["peptide"],
                mz_array,
                intensity,
                spectrum["small"]["1"]["mods"],
                self.ion_types,
                self.args.max_charges,
                flag_ITMS=flag_ITMS,
                shift=self.args.shift,
                return_mz=True,
            )
            str_pept = data_2_str(spectrum, self.modification_meta_dict)
            # tmp = spec_label.copy()
            try:
                spec_label, _ = pickle.loads(
                    gzip.decompress(self.txn_read.get(str_pept.encode("ascii")))
                )
                # print(f"success_loading {str_pept} spec")
            except:
                pass
                # print(f"error_loading {str_pept} spec")
            # print(spec_label)
            spec_label = torch.Tensor(spec_label)

            ion_comp_res_num = (ion_res_num > 0) * (
                len(spectrum["small"]["1"]["peptide"]) - ion_res_num
            )

            peak_indices = torch.nonzero(ion_peak_flag)
            peak_classes = ion_peak_class[peak_indices]
            res_idx_ = res_idx[peak_indices]

            if self.is_train and self.args.denoise_pred:
                assert 0
            else:
                denoise_label = torch.zeros(mz_array.shape[0])
                denoise_pred_mask = torch.zeros(len(mz_array))

            if self.is_train and self.args.intensity_pred:
                select_index_2 = torch.randint(
                    0,
                    len(peak_indices),
                    (int(len(peak_indices) * self.args.intensity_prob + 1),),
                    dtype=torch.int64,
                )
                intensity_label = torch.zeros(mz_array.shape[0]).type(intensity.dtype)
                intensity_label[select_index_2] = intensity[select_index_2]
                intensity[select_index_2] = 0
                intensity_pred_mask = torch.zeros(len(mz_array))
                intensity_pred_mask[select_index_2] += 1
            else:
                intensity_label = torch.zeros(mz_array.shape[0])
                intensity_pred_mask = torch.zeros(len(mz_array))

            if self.args.noise_peak_pred:
                noise_peak_label = torch.zeros(mz_array.shape[0])
                if not self.args.multi_class_noise_peak_pred:
                    noise_peak_label[peak_indices] += 1
                else:
                    noise_peak_label[peak_indices] += peak_classes
                # assert torch.all(noise_peak_label < 33)
                noise_peak_mask = torch.ones(mz_array.shape[0])
                if intensity_pred_mask.sum() > 0:
                    noise_peak_mask[select_index_2] -= 1
                if denoise_pred_mask.sum() > 0:
                    # noise_peak_mask[select_index] -= 1
                    pass
                noise_peak_mask = noise_peak_mask > 0
            else:
                noise_peak_label = torch.zeros(mz_array.shape[0])
                noise_peak_mask = torch.ones(mz_array.shape[0])

            try:
                instrument = torch.Tensor([mgf["inst"]]).long()
                nce = torch.Tensor([mgf["HCD"]]).long()
            except:
                instrument = torch.Tensor([0]).long()
                nce = torch.Tensor([0]).long()
                print("warning no ins nce")

            if self.args.tof:
                instrument = torch.Tensor([4]).long()

            if self.args.res_type_pred:
                pep_seq = self.tokenize(residue_type_after_mod[0])
                res_idx_ = res_idx_.long()
                # res_idx_ = res_idx[res_idx > 0]
                assert torch.all(res_idx >= 0) and torch.all(
                    res_idx < len(spectrum["small"]["1"]["peptide"])
                ), (len(spectrum["small"]["1"]["peptide"]), res_idx)
                res_type_label = torch.zeros(noise_peak_label.shape)

                for i in range(len(res_idx_)):
                    # print(residue_type_after_mod[0], peak_indices[i].item(), res_idx_[i])
                    res_type_label[peak_indices[i].item()] = pep_seq[res_idx_[i]]
                res_type_label = res_type_label * (noise_peak_label > 0)
            else:
                res_type_label = torch.Tensor([0])

            if self.args.use_rope:
                rope_embeding = precompute_freqs_cis(
                    self.args.node_dim,
                    torch.cat(
                        [
                            torch.Tensor([mgf["precursor_mz"]]),
                            torch.Tensor([spectrum["small"]["mgf"]["precursor_mz"]]),
                            mz_array,
                        ]
                    ),
                )
            else:
                rope_embeding = torch.Tensor([0])

            seq_len_label = torch.Tensor([len(residue_type_after_mod_all[0])]).long()

            label_index = torch.zeros(len(residue_type))
            label_index[0] = 1

            precursor_charge = min(self.args.max_prec_charge, precursor_charge)
            fdr = spectrum["small"]["1"]["fdr_value"]

            # try:
            #     is_target = spectrum["small"]["1"]["is_target"]
            is_target = [0]
            ret = {
                "mz_array": mz_array,
                "intensity": intensity,
                "precursor_mz": torch.Tensor(
                    [(precursor_mz - 1.007276) * precursor_charge]
                ),
                "precursor_charge": torch.Tensor([precursor_charge]).long(),
                "batch_index": torch.ones(len(residue_type)).long(),
                "residue_type": residue_type,
                "residue_type_after_mod": residue_type_after_mod,
                "residue_type_after_mod_all": residue_type_after_mod_all,
                "mod_label": mod_label.long(),
                "mod_mass_label": mod_mass_label,
                "modification": modification,
                "instrument": instrument,
                "spec_label": spec_label,
                "RT": rt,
                "nce": nce,
                "denoise_pred_mask": denoise_pred_mask.bool(),
                "intensity_pred_mask": intensity_pred_mask.bool(),
                "denoise_label": denoise_label,
                "intensity_label": intensity_label,
                "rope_embeding": rope_embeding,
                "noise_peak_label": noise_peak_label.long(),
                "res_type_label": res_type_label.long(),
                "noise_peak_mask": noise_peak_mask.long(),
                "modification_cls_label": modification_cls_label.long(),
                "ion_res_num": ion_res_num.long(),
                "ion_comp_res_num": ion_comp_res_num.long(),
                "seq_len_label": seq_len_label,
                "label_index": label_index.bool(),
                "idx": torch.Tensor([idx]),
                "title": [title],
                "fdr": torch.Tensor([fdr]),
                "is_target": torch.Tensor(is_target),
                "ion_mz": torch.Tensor(ion_mz),
                "repeat_num": torch.Tensor([len(residue_type)]),
            }
            if self.args.inference:
                ret["ion_mz"] = torch.Tensor(ion_mz)
            # assert 0, ret
            return ret

    def tokenize(self, input):
        ret = torch.zeros(len(input))
        for i in range(len(input)):
            ret[i] = self.token_dict[input[i]]
        return ret.long()


class InferenceZipLMDB2Dataset:
    def __init__(
        self, db_path, split, check=False, fdr_thread=0.1, pkl_path="", spec_pred=False
    ):
        self.db_path = db_path
        self.fdr_thread = fdr_thread
        self.pkl_path = pkl_path
        self.split = split
        self.check = check

        assert os.path.isfile(self.db_path), "{} not found".format(self.db_path)

        self.env = self.connect_db(self.db_path)
        # with self.env.begin() as txn:
        #     self._keys = list(txn.cursor().iternext(values=False)) # 9Aips_FDR0.001_keys
        if fdr_thread <= 1:
            # if not spec_pred:
            self._keys = pickle.load(
                open(join(pkl_path, f"{split}_FDR{fdr_thread}_keys.pkl"), "rb")
            )
        # else:
        # self._keys = pickle.load(open(join(pkl_path, f"{split}_FDR{fdr_thread}_keys_spec_pred.pkl"), 'rb'))

        else:
            with self.env.begin() as txn:
                self._keys = list(
                    txn.cursor().iternext(values=False)
                )  # 9Aips_FDR0.001_keys
        # if not self.check:
        #     try:
        #         self.filtered_keys = pickle.load(open(join(pkl_path, f"{split}_{fdr_thread}.pkl"), 'rb'))
        #     except:
        #         print("no pkl, start generating ... ")
        #         self.reset_with_fdr_thread()
        # else:
        #     self.filtered_keys = self._keys

        # print(f"finished loading: # original {len(self._keys)}, # filtered {len(self.filtered_keys)}")
        print(f"finished loading: # original {len(self._keys)}")
        # self._keys = self._keys[::num]
        # num_sample = len(self._keys) // bsz
        # self._keys = self._keys[:(num_sample * bsz)]

    def connect_db(self, lmdb_path=None):
        env = lmdb.open(
            lmdb_path,
            subdir=False,
            readonly=True,
            lock=False,
            readahead=True,
            meminit=False,
            max_readers=128,
        )
        return env

    def __len__(self):
        return len(self._keys)

    @lru_cache(maxsize=16)
    def __getitem__(self, idx):
        datapoint_pickled = self.env.begin().get(self._keys[idx])
        data = pickle.loads(gzip.decompress(datapoint_pickled))
        return data

    def reset_with_fdr_thread(self, thread=16):

        def if_thread(key):
            datapoint_pickled = self.env.begin().get(key)
            data = pickle.loads(gzip.decompress(datapoint_pickled))
            if data["small"]["1"]["fdr_value"] > self.fdr_thread:
                return None
            return key

        self.filtered_keys = []

        def input_files():
            for fn in self._keys:
                yield fn, self.env, self.fdr_thread

        for key in tqdm(self._keys, total=len(self._keys)):
            ret = if_thread(key)
            if ret is not None:
                self.filtered_keys.append(ret)

        # with Pool(56) as pool:
        #     for ret in tqdm(
        #         pool.imap(if_thread, input_files(), chunksize=10), total=len(self._keys)
        #     ):
        #         if ret is not None:
        #             self.filtered_keys.append(ret)
        pickle.dump(
            self.filtered_keys,
            open(join(self.pkl_path, f"{self.split}_{self.fdr_thread}.pkl"), "wb"),
        )


class SampleZipLMDB2Dataset:
    def __init__(self, db_path, num=1000, bsz=128):
        self.db_path = db_path
        assert os.path.isfile(self.db_path), "{} not found".format(self.db_path)

        self.env = self.connect_db(self.db_path)
        with self.env.begin() as txn:
            self._keys = list(txn.cursor().iternext(values=False))
        # sample_size = 1000
        # # 计算采样间隔
        # interval = len(self._keys) // sample_size
        # # 从列表中采样元素
        # self._keys = self._keys[::interval]

        # self._keys = self._keys[::num]
        # num_sample = len(self._keys) // bsz
        # self._keys = self._keys[:(num_sample * bsz)]

    def connect_db(self, lmdb_path=None):
        env = lmdb.open(
            lmdb_path,
            subdir=False,
            readonly=True,
            lock=False,
            readahead=True,
            meminit=False,
            max_readers=128,
        )
        return env

    def __len__(self):
        return len(self._keys)

    @lru_cache(maxsize=16)
    def __getitem__(self, idx):
        datapoint_pickled = self.env.begin().get(self._keys[idx])
        data = pickle.loads(gzip.decompress(datapoint_pickled))
        return data


class ClusteredDataset:
    def __init__(self, db_path):
        self.db_path = db_path
        assert os.path.isfile(self.db_path), "{} not found".format(self.db_path)

        self.env = self.connect_db(self.db_path)
        with self.env.begin() as txn:
            self._keys = list(txn.cursor().iternext(values=False))

    def connect_db(self, lmdb_path=None):
        env = lmdb.open(
            lmdb_path,
            subdir=False,
            readonly=True,
            lock=False,
            readahead=True,
            meminit=False,
            max_readers=128,
        )
        return env

    def __len__(self):
        return len(self._keys)

    @lru_cache(maxsize=16)
    def __getitem__(self, idx):
        datapoint_pickled = self.env.begin().get(self._keys[idx])
        data = pickle.loads(gzip.decompress(datapoint_pickled))
        ret = data[np.random.randint(len(data))]
        return ret


class ScoreInferenceDataset(BaseWrapperDataset):
    """A wrapper around a LMDB database that reads and returns items from it
    lazily."""

    def __init__(self, dataset, args, crop_rational=1 / 4, is_train=False):
        super().__init__(dataset)
        self.dataset = dataset
        self.is_train = is_train
        self.args = args
        self.softmax = nn.Softmax(dim=-1)
        self.set_epoch(None)
        modification_meta_dict_path = args.modification_meta_dict_path
        self.modification_meta_dict = pickle.load(
            open(modification_meta_dict_path, "rb")
        )
        tokenize1_pkl_path = args.tokenize1_pkl_path
        self.tokenize1_dict = pickle.load(open(tokenize1_pkl_path, "rb"))
        self.modification_meta_dict["Ammonia-loss"] = self.modification_meta_dict[
            "Ammonia-loss[AnyN-termC]"
        ]
        self.min_mz = 1
        self.max_mz = 6500.0
        self.remove_precursor_tol = 2
        self.n_peaks = self.args.cutoff_spectra
        self.min_intensity = 0.0
        # ION_TYPES = ['b', 'b-NH3', 'b-H2O', 'b-ModLoss', 'y', 'y-NH3', 'y-H2O', 'y-ModLoss']
        if not self.args.no_mod:
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
        else:
            self.ion_types = ["b", "b-NH3", "b-H2O", "y", "y-NH3", "y-H2O"]

        self.token_dict = {
            "A": 1,  # 71.03711
            "B": 2,  # 0.000
            "C": 3,  # 103.009180
            "D": 4,  # 115.02694
            "E": 5,  # 129.04259
            "F": 6,  # 147.06841
            "G": 7,  # 57.02146
            "H": 8,  # 137.05891
            "I": 9,  # 0.000
            "J": 10,  # 0.000
            "K": 11,  # 128.09496
            "L": 12,  # 113.08406
            "M": 13,  # 131.04049
            "N": 14,  # 114.04293
            "O": 15,  # 0.00000
            "P": 16,  # 97.05276
            "Q": 17,  # 128.05858
            "R": 18,  # 156.10111
            "S": 19,  # 87.03203
            "T": 20,  # 101.04768
            "U": 21,  # 0.00000
            "V": 22,  # 99.06841
            "W": 23,  # 186.07931
            "X": 24,  # 0.0000
            "Y": 25,  # 163.06333
            "Z": 26,  # 0.00000
        }

        try:
            # self.pept_to_spec_dict = pickle.load(open(join(self.args.data, "pept_to_spec.pkl"), 'rb'))
            env_read = lmdb.open(
                join(self.args.data, "pept_to_spec2.lmdb"),
                subdir=False,
                readonly=True,
                lock=False,
                readahead=False,
                meminit=False,
                max_readers=1,
                map_size=int(1e9),
            )
            self.txn_read = env_read.begin()
            # keys = list(txn_read.cursor().iternext(values=False))
        except:
            print("warning: failed loading spec label")

    def __len__(self):
        return len(self.dataset)

    def set_epoch(self, epoch, **unused):
        super().set_epoch(epoch)
        self.epoch = epoch

    def __getitem__(self, idx: int):
        return self.__getitem_cached__(self.epoch, idx)

    @lru_cache(maxsize=16)
    def __getitem_cached__(self, epoch: int, idx):
        with data_utils.numpy_seed(self.args.seed, epoch, idx):
            spectrum = self.dataset[idx]
            try:
                spectrum = gzip.decompress(spectrum)
            except:
                pass
            try:
                spectrum = pickle.loads(spectrum)
            except:
                pass
            if "mgf" in spectrum["small"].keys():
                mgf = spectrum["small"]["mgf"]
            else:
                mgf = spectrum["small"][list(spectrum["small"].keys())[0]]

            mz_array = torch.Tensor(mgf["mz_array"])
            intensity = torch.Tensor(mgf["intensity"])
            precursor_mz = mgf["precursor_mz"]
            precursor_charge = mgf["precursor_charge"]
            try:
                title = mgf["title"]
            except:
                title = spectrum["small"]["title"]

            modification_cls_label = torch.zeros(2610).long()

            flag_ITMS = False

            rets = []
            if "small" in spectrum.keys():
                num_sample = spectrum["small"]["num_res"]
                for i in range(2, num_sample + 1):
                    rets.append(spectrum["small"][str(i)])
            # if "large" in spectrum.keys():
            #     num_sample = spectrum["small"]["num_res"]
            #     for i in range(2, num_sample + 1):
            #         if spectrum["small"][str(i)]["peptide"] != spectrum["small"]["1"]["peptide"]:
            #             rets.append(spectrum["small"][str(i)])

            residue_type = [spectrum["small"]["1"]["peptide"]]
            residue_type_after_mod_all = [
                pep_mod_2_pep(
                    spectrum["small"]["1"]["peptide"], spectrum["small"]["1"]["mods"]
                )
            ]
            if len(rets) > 0:
                residue_type.extend([_["peptide"] for _ in rets])
                residue_type_after_mod_all.extend(
                    [pep_mod_2_pep(_["peptide"], _["mods"]) for _ in rets]
                )

            residue_type_after_mod = [
                pep_mod_2_pep(
                    spectrum["small"]["1"]["peptide"], spectrum["small"]["1"]["mods"]
                )
            ]
            modification = [[]]
            mod_label = torch.zeros(len(spectrum["small"]["1"]["peptide"]))
            mod_mass_label = torch.zeros(len(spectrum["small"]["1"]["peptide"]))

            assert len(spectrum["small"]["1"]["peptide"]) == len(residue_type[0]), (
                len(spectrum["small"]["1"]["peptide"]),
                len(residue_type[0]),
            )

            for mod in spectrum["small"]["1"]["mods"]:
                if mod[0] == 0:
                    mod = (mod[0] + 1, mod[1])
                    # mod[0] += 1
                if mod[0] > len(spectrum["small"]["1"]["peptide"]):
                    mod = (mod[0] - 1, mod[1])
                    # mod[0] -= 1
                modification[0].append((mod[0], self.modification_meta_dict[mod[1]]))
                if "->" not in mod[1]:
                    modification_cls_label[
                        int(self.modification_meta_dict[mod[1]][2])
                    ] = 1

                if (
                    "->" not in mod[1]
                    or mod[1].split("->")[1].split("[")[0]
                    not in amino_acids_3to1_map.keys()
                ):
                    if self.args.tokenize_mod == 0:
                        mod_label[mod[0] - 1] = (
                            self.modification_meta_dict[mod[1]][2].item() + 1
                        )
                    elif self.args.tokenize_mod == 1:
                        mod_label[mod[0] - 1] = self.tokenize1_dict[
                            self.modification_meta_dict[mod[1]][2].item()
                        ]["token_idx"]

                    mod_mass_label[mod[0] - 1] += self.modification_meta_dict[mod[1]][
                        1
                    ].item()
                    # if "->" in mod[1]:
                    #     print(mod[1])
                    # residue_type_after_mod[0][mod[0]] = protein_letters_3to1[mod[1].split("->")[1].split("-")[1].split("[")[0].upper()]
            for i in range(len(rets)):
                modification.append([])
                for mod in rets[i]["mods"]:
                    if mod[0] == 0:
                        mod = (mod[0] + 1, mod[1])  # 位置 meta信息
                    if mod[0] > len(rets[i]["peptide"]):
                        mod = (mod[0] - 1, mod[1])
                    modification[i + 1].append(
                        (mod[0], self.modification_meta_dict[mod[1]])
                    )

            assert len(modification) == len(residue_type), (
                len(modification),
                len(residue_type),
                len(residue_type_after_mod),
            )

            spectrum_ = sus.MsmsSpectrum(
                "",
                precursor_mz,
                precursor_charge,
                mz_array.numpy().astype(np.float64),
                intensity.numpy().astype(np.float32),
            )

            spectrum_.set_mz_range(50.5, 4500.0)
            if len(spectrum_.mz) == 0:
                raise ValueError
            spectrum_.remove_precursor_peak(2.0, "Da")
            if len(spectrum_.mz) == 0:
                raise ValueError
            spectrum_.filter_intensity(self.min_intensity, self.n_peaks)
            if len(spectrum_.mz) == 0:
                raise ValueError
            spectrum_.scale_intensity("root", 1)
            intensity = spectrum_.intensity / np.linalg.norm(spectrum_.intensity)
            mz_array = spectrum_.mz
            mz_array, intensity = torch.Tensor(mz_array), torch.Tensor(intensity)

            # mz_array, intensity = sqrt_and_norm(mz_array, intensity, precursor_mz, precursor_charge)
            # mz_array, intensity = remove_precursor_peak(mz_array, intensity, precursor_mz)

            # top_values, top_indices = torch.topk(intensity.view(-1), k=min(self.args.cutoff_spectra, mz_array.shape[0]))

            # print(spectrum["small"]["title"], spectrum["small"]["1"]["peptide"])
            (
                spec_label,
                ion_peak_flag,
                ion_peak_class,
                ion_res_num,
                res_idx,
                error_tol,
                ion_mz,
            ) = get_spectrum_prediction_label(
                spectrum["small"]["1"]["peptide"],
                mz_array,
                intensity,
                spectrum["small"]["1"]["mods"],
                self.ion_types,
                self.args.max_charges,
                flag_ITMS=flag_ITMS,
                shift=self.args.shift,
                return_mz=True,
            )
            str_pept = data_2_str(spectrum, self.modification_meta_dict)
            # tmp = spec_label.copy()
            try:
                spec_label, _ = pickle.loads(
                    gzip.decompress(self.txn_read.get(str_pept.encode("ascii")))
                )
            except:
                # print(f"error_loading {str_pept} spec")
                pass
            # print(spec_label)
            spec_label = torch.Tensor(spec_label)

            # from PepMS.plot.plot_spectrum import plot_two_spectrum
            # plot_two_spectrum(ion_mz.reshape(-1), tmp.reshape(-1), ion_mz.reshape(-1), spec_label.reshape(-1), save_name=str_pept)

            ion_comp_res_num = (ion_res_num > 0) * (
                len(spectrum["small"]["1"]["peptide"]) - ion_res_num
            )
            # ion_peak_flag_strict = ion_peak_flag_strict[top_indices]

            peak_indices = torch.nonzero(ion_peak_flag)
            peak_classes = ion_peak_class[peak_indices]
            res_idx_ = res_idx[peak_indices]

            if self.is_train and self.args.denoise_pred:
                assert 0
            else:
                denoise_label = torch.zeros(mz_array.shape[0])
                denoise_pred_mask = torch.zeros(len(mz_array))

            if self.is_train and self.args.intensity_pred:
                select_index_2 = torch.randint(
                    0,
                    len(peak_indices),
                    (int(len(peak_indices) * self.args.intensity_prob + 1),),
                    dtype=torch.int64,
                )
                intensity_label = torch.zeros(mz_array.shape[0]).type(intensity.dtype)
                intensity_label[select_index_2] = intensity[select_index_2]
                intensity[select_index_2] = 0
                intensity_pred_mask = torch.zeros(len(mz_array))
                intensity_pred_mask[select_index_2] += 1
            else:
                intensity_label = torch.zeros(mz_array.shape[0])
                intensity_pred_mask = torch.zeros(len(mz_array))

            if self.args.noise_peak_pred:
                noise_peak_label = torch.zeros(mz_array.shape[0])
                if not self.args.multi_class_noise_peak_pred:
                    noise_peak_label[peak_indices] += 1
                else:
                    noise_peak_label[peak_indices] += peak_classes
                # assert torch.all(noise_peak_label < 33)
                noise_peak_mask = torch.ones(mz_array.shape[0])
                if intensity_pred_mask.sum() > 0:
                    noise_peak_mask[select_index_2] -= 1
                if denoise_pred_mask.sum() > 0:
                    # noise_peak_mask[select_index] -= 1
                    pass
                noise_peak_mask = noise_peak_mask > 0
            else:
                noise_peak_label = torch.zeros(mz_array.shape[0])
                noise_peak_mask = torch.ones(mz_array.shape[0])

            try:
                instrument = torch.Tensor([mgf["inst"]]).long()
                nce = torch.Tensor([mgf["HCD"]]).long()
            except:
                instrument = torch.Tensor([0]).long()
                nce = torch.Tensor([0]).long()
                print("warning no ins nce")

            if self.args.res_type_pred:
                pep_seq = self.tokenize(residue_type_after_mod[0])
                # print(pep_seq)
                # ion_res_num
                # ion_comp_res_num
                res_idx_ = res_idx_.long()
                # res_idx_ = res_idx[res_idx > 0]
                assert torch.all(res_idx >= 0) and torch.all(
                    res_idx < len(spectrum["small"]["1"]["peptide"])
                ), (len(spectrum["small"]["1"]["peptide"]), res_idx)
                res_type_label = torch.zeros(noise_peak_label.shape)

                for i in range(len(res_idx_)):
                    # print(residue_type_after_mod[0], peak_indices[i].item(), res_idx_[i])
                    res_type_label[peak_indices[i].item()] = pep_seq[res_idx_[i]]
                # assert 0, (res_type_label, res_idx, ion_peak_class, residue_type_after_mod[0], peak_indices, peak_classes, len(peak_classes), noise_peak_label)
                res_type_label = res_type_label * (noise_peak_label > 0)
                # print(res_type_label.shape, res_idx.shape,  pep_seq.shape, noise_peak_label.shape, mz_array.shape, peak_classes.shape)
            else:
                res_type_label = torch.Tensor([0])

            # assert mod_label.shape[0] == len(residue_type_after_mod[0]) == len(residue_type[0]), (mod_label.shape[0], len(residue_type_after_mod), len(residue_type[0]))
            if self.args.use_rope:
                rope_embeding = precompute_freqs_cis(
                    self.args.node_dim,
                    torch.cat(
                        [
                            torch.Tensor([mgf["precursor_mz"]]),
                            torch.Tensor([spectrum["small"]["mgf"]["precursor_mz"]]),
                            mz_array,
                        ]
                    ),
                )
            else:
                rope_embeding = torch.Tensor([0])

            seq_len_label = torch.Tensor(
                [len(spectrum["small"]["1"]["peptide"])]
            ).long()

            label_index = torch.zeros(len(residue_type))
            label_index[0] = 1
            # if len(residue_type) < 5:
            #     print(len(residue_type))
            # print(label_index.shape)

            #  print(rope_embeding.shape)
            precursor_charge = min(self.args.max_prec_charge, precursor_charge)
            fdr = spectrum["small"]["1"]["fdr_value"]
            return {
                "mz_array": mz_array,
                "intensity": intensity,
                "precursor_mz": torch.Tensor(
                    [(precursor_mz - 1.007276) * precursor_charge]
                ),
                "precursor_charge": torch.Tensor([precursor_charge]).long(),
                "batch_index": torch.ones(len(residue_type)).long(),
                "residue_type": residue_type,
                "residue_type_after_mod": residue_type_after_mod,
                "residue_type_after_mod_all": residue_type_after_mod_all,
                "mod_label": mod_label.long(),
                "mod_mass_label": mod_mass_label,
                "modification": modification,
                "instrument": instrument,
                "spec_label": spec_label,
                "nce": nce,
                "denoise_pred_mask": denoise_pred_mask.bool(),
                "intensity_pred_mask": intensity_pred_mask.bool(),
                "denoise_label": denoise_label,
                "intensity_label": intensity_label,
                "rope_embeding": rope_embeding,
                "noise_peak_label": noise_peak_label.long(),
                "res_type_label": res_type_label.long(),
                "noise_peak_mask": noise_peak_mask.long(),
                "modification_cls_label": modification_cls_label.long(),
                "ion_res_num": ion_res_num.long(),
                "ion_comp_res_num": ion_comp_res_num.long(),
                "seq_len_label": seq_len_label,
                "label_index": label_index.bool(),
                "index": torch.Tensor([idx]).long(),
                "title": [title],
                "fdr": torch.Tensor([fdr]),
                # "is_target": torch.Tensor(is_target),
                "ion_mz": torch.Tensor(ion_mz),
            }

    def tokenize(self, input):
        ret = torch.zeros(len(input))
        for i in range(len(input)):
            ret[i] = self.token_dict[input[i]]
        return ret.long()


mass_calculator = PeptideIonCalculator()

MAX_CHARGES = 5
MIN_FDR = 0
MAX_FDR = 0.01
# ION_TYPES = ['y']

PPM = 1 / 1000000
# PPM_THRESHOLD=20


def sum_intensity_in_range(A, B, interval):
    min_val, max_val = interval
    # 找到落在区间内的索引
    indices = torch.where((A >= min_val) & (A <= max_val))
    ret_flag = (A >= min_val) & (A <= max_val)
    # 提取落在区间内的强度值，并求和
    # print(B[indices])
    if len(B[indices]) == 0:
        return 0, ret_flag
    # print(B[indices], len(B[indices]))
    sum_intensity = torch.sum(B[indices])
    return sum_intensity, ret_flag


def get_spectrum_prediction_label(
    peptide,
    mz_array,
    intensity,
    mods,
    ion_types,
    max_charges=2,
    PPM_THRESHOLD=20,
    flag_ITMS=False,
    shift=True,
    return_mz=False,
):
    peptide_mass, ions = mass_calculator.calc_pepmass_and_ions_from_iontypes(
        peptide, mods, ion_types, max_charges, shift=shift
    )
    mask = ions < 2
    ions[mask] = 0
    # assert 0, (ions, peptide)
    error_tol = PPM_THRESHOLD * PPM
    # assert 0, (ions, error_tol)

    intens = np.zeros(ions.shape)
    # print(len(peptide), ions)
    ion_peak_flag = torch.zeros(
        mz_array.shape[0],
    )
    ion_peak_class = torch.zeros(
        mz_array.shape[0],
    )
    ion_res_num = torch.zeros(
        mz_array.shape[0],
    )
    res_idx = torch.zeros(
        mz_array.shape[0],
    )
    # print(ions.shape, len(ion_types))
    for i in range(ions.shape[0]):
        for j in range(ions.shape[1]):
            if ions[i, j] <= 0:
                pass
            else:
                inten, indices = sum_intensity_in_range(
                    mz_array,
                    intensity,
                    (
                        ions[i, j] - error_tol * ions[i, j],
                        ions[i, j] + error_tol * ions[i, j],
                    ),
                )
                # print(indices)
                intens[i, j] = inten
                # print(ion_peak_flag.shape, indices.reshape(ion_peak_flag.shape).shape)
                ion_peak_flag = ion_peak_flag + indices
                ion_peak_class = (
                    ion_peak_class + (indices * (j + 1)) * ~ion_peak_class.bool()
                )
                # assert torch.all(ion_peak_class < 33)
                if j < (len(ion_types) * max_charges) / 2:
                    ion_res_num += indices * (i + 1) * ~ion_res_num.bool()
                    # if torch.any(indices > 0):
                    #     print(">?????", i, j)
                    # print(">>>>", ion_res_num)
                    if torch.any(ion_res_num >= 100):
                        print(">>", ion_res_num)
                    res_idx += indices * i * ~res_idx.bool()
                else:

                    if shift:
                        ion_res_num += (
                            indices * (ions.shape[0] - i) * ~ion_res_num.bool()
                        )
                        if torch.any(ion_res_num >= 100):
                            print("<<", ion_res_num)
                        res_idx += indices * i * ~res_idx.bool()
                    elif i > 0:
                        ion_res_num += (
                            indices * (ions.shape[0] - i - 1) * ~ion_res_num.bool()
                        )
                        if torch.any(ion_res_num >= 100):
                            print("<<", ion_res_num)
                        res_idx += indices * (i + 1) * ~res_idx.bool()

                assert torch.all(ion_res_num < 110), (peptide, ion_res_num)
                # if np.any(ion_peak_flag > 3):
                #     print("warning: too thick")
    # print(intens)
    # assert 0, (intens, ions, peptide, mods)
    if not return_mz:
        return (
            intens,
            ion_peak_flag > 0,
            ion_peak_class,
            ion_res_num,
            res_idx,
            error_tol,
        )
    else:
        return (
            intens,
            ion_peak_flag > 0,
            ion_peak_class,
            ion_res_num,
            res_idx,
            error_tol,
            ions,
        )


def precompute_freqs_cis(dim: int, t: int, theta: float = 10000.0):
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
    # t = torch.arange(end, device=freqs.device)  # type: ignore
    freqs = torch.outer(t, freqs).float()  # type: ignore
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)  # complex64
    return freqs_cis


class SpecPredDataset(BaseWrapperDataset):
    """A wrapper around a LMDB database that reads and returns items from it
    lazily."""

    def __init__(self, dataset, args, crop_rational=1 / 4, is_train=False):
        super().__init__(dataset)
        self.dataset = dataset
        self.is_train = is_train
        self.args = args
        self.softmax = nn.Softmax(dim=-1)
        self.set_epoch(None)
        modification_meta_dict_path = args.modification_meta_dict_path
        self.modification_meta_dict = pickle.load(
            open(modification_meta_dict_path, "rb")
        )
        tokenize1_pkl_path = (
            args.tokenize1_pkl_path
        )  # r"/mnt/vepfs/fs_users/zhaojiale/Contrast_MS_Pep/consts/tokenize1.pkl"
        self.tokenize1_dict = pickle.load(open(tokenize1_pkl_path, "rb"))
        self.modification_meta_dict["Ammonia-loss"] = self.modification_meta_dict[
            "Ammonia-loss[AnyN-termC]"
        ]
        self.min_mz = 1
        self.max_mz = 6500.0
        self.remove_precursor_tol = 2
        self.n_peaks = self.args.cutoff_spectra
        self.min_intensity = 0.0
        # ION_TYPES = ['b', 'b-NH3', 'b-H2O', 'b-ModLoss', 'y', 'y-NH3', 'y-H2O', 'y-ModLoss']
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

        self._aa2idx = {
            "A": 1,  # 71.03711
            "B": 2,  # 0.000
            "C": 3,  # 103.009180
            "D": 4,  # 115.02694
            "E": 5,  # 129.04259
            "F": 6,  # 147.06841
            "G": 7,  # 57.02146
            "H": 8,  # 137.05891
            "I": 9,  # 0.000
            "J": 10,  # 0.000
            "K": 11,  # 128.09496
            "L": 12,  # 113.08406
            "M": 13,  # 131.04049
            "N": 14,  # 114.04293
            "O": 15,  # 0.00000
            "P": 16,  # 97.05276
            "Q": 17,  # 128.05858
            "R": 18,  # 156.10111
            "S": 19,  # 87.03203
            "T": 20,  # 101.04768
            "U": 21,  # 0.00000
            "V": 22,  # 99.06841
            "W": 23,  # 186.07931
            "X": 24,  # 0.0000
            "Y": 25,  # 163.06333
            "Z": 26,  # 0.00000
        }

        try:
            # self.pept_to_spec_dict = pickle.load(open(join(self.args.data, "pept_to_spec.pkl"), 'rb'))
            pep2spec_path = join(self.args.data, "pept_to_spec2.lmdb")

            env_read = lmdb.open(
                pep2spec_path,
                subdir=False,
                readonly=True,
                lock=False,
                readahead=False,
                meminit=False,
                max_readers=1,
                map_size=int(1e9),
            )
            self.txn_read = env_read.begin()
            # keys = list(txn_read.cursor().iternext(values=False))
        except:
            assert 0
            print("warning: failed loading spec label")

    def __len__(self):
        return len(self.dataset)

    def set_epoch(self, epoch, **unused):
        super().set_epoch(epoch)
        self.epoch = epoch

    def __getitem__(self, idx: int):
        return self.__getitem_cached__(self.epoch, idx)

    @lru_cache(maxsize=16)
    def __getitem_cached__(self, epoch: int, idx):
        with data_utils.numpy_seed(self.args.seed, epoch, idx):
            spectrum = self.dataset[idx]
            try:
                spectrum = gzip.decompress(spectrum)
            except:
                pass
            try:
                spectrum = pickle.loads(spectrum)
            except:
                pass
            try:
                if "mgf" in spectrum["small"].keys():
                    mgf = spectrum["small"]["mgf"]
                else:
                    mgf = spectrum["small"][list(spectrum["small"].keys())[0]]
            except:
                if "mgf" in spectrum["large"].keys():
                    mgf = spectrum["large"]["mgf"]
                else:
                    mgf = spectrum["large"][list(spectrum["large"].keys())[0]]

            mz_array = torch.Tensor(mgf["mz_array"])
            intensity = torch.Tensor(mgf["intensity"])
            precursor_mz = mgf["precursor_mz"]
            precursor_charge = mgf["precursor_charge"]
            try:
                title = mgf["title"]
            except:
                title = spectrum["small"]["title"]

            residue_type = [spectrum["small"]["1"]["peptide"]]
            residue_type_after_mod_all = [
                pep_mod_2_pep(
                    spectrum["small"]["1"]["peptide"], spectrum["small"]["1"]["mods"]
                )
            ]

            residue_type_after_mod = [
                pep_mod_2_pep(
                    spectrum["small"]["1"]["peptide"], spectrum["small"]["1"]["mods"]
                )
            ]
            modification = [[]]
            mod_label = torch.zeros(len(spectrum["small"]["1"]["peptide"]))
            for mod in spectrum["small"]["1"]["mods"]:
                if mod[0] == 0:
                    mod = (mod[0] + 1, mod[1])
                    # mod[0] += 1
                if mod[0] > len(spectrum["small"]["1"]["peptide"]):
                    mod = (mod[0] - 1, mod[1])
                    # mod[0] -= 1
                modification[0].append((mod[0], self.modification_meta_dict[mod[1]]))

            spectrum_ = sus.MsmsSpectrum(
                "",
                precursor_mz,
                precursor_charge,
                mz_array.numpy().astype(np.float64),
                intensity.numpy().astype(np.float32),
            )

            spectrum_.set_mz_range(50.5, 4500.0)
            if len(spectrum_.mz) == 0:
                raise ValueError
            spectrum_.remove_precursor_peak(2.0, "Da")
            if len(spectrum_.mz) == 0:
                raise ValueError
            spectrum_.filter_intensity(self.min_intensity, self.n_peaks)
            if len(spectrum_.mz) == 0:
                raise ValueError
            spectrum_.scale_intensity("root", 1)
            intensity = spectrum_.intensity / np.linalg.norm(spectrum_.intensity)
            mz_array = spectrum_.mz
            mz_array, intensity = torch.Tensor(mz_array), torch.Tensor(intensity)

            (
                spec_label,
                ion_peak_flag,
                ion_peak_class,
                ion_res_num,
                res_idx,
                error_tol,
                ion_mz,
            ) = get_spectrum_prediction_label(
                spectrum["small"]["1"]["peptide"],
                mz_array,
                intensity,
                spectrum["small"]["1"]["mods"],
                self.ion_types,
                self.args.max_charges,
                flag_ITMS=False,
                shift=True,
                return_mz=True,
            )
            str_pept = data_2_str(spectrum, self.modification_meta_dict)
            try:
                spec_label, _ = pickle.loads(
                    gzip.decompress(self.txn_read.get(str_pept.encode("ascii")))
                )
            except:
                print(f"error_loading {str_pept} spec")
            spec_label = torch.Tensor(spec_label)

            try:
                instrument = torch.Tensor([spectrum["small"]["mgf"]["inst"]]).long()
                nce = torch.Tensor([spectrum["small"]["mgf"]["HCD"]]).long()
            except:
                instrument = torch.Tensor([0]).long()
                nce = torch.Tensor([0]).long()
                print("warning no ins nce")

            residue_type = torch.zeros(len(residue_type_after_mod[0]))
            for i in range(len(residue_type)):
                residue_type[i] = self._aa2idx[residue_type_after_mod[0][i]]
            assert len(spectrum["small"]["1"]["peptide"]) == len(
                residue_type_after_mod[0]
            ), (spectrum["small"]["1"]["peptide"], residue_type_after_mod[0])
            assert residue_type.shape[0] == spec_label.shape[0], (
                residue_type_after_mod[0],
                residue_type.shape,
                spec_label.shape,
            )
            return {
                "precursor_charge": torch.Tensor([precursor_charge]).long(),
                "residue_type": residue_type.long(),
                "modification": modification,
                "instrument": instrument.long(),
                "spec_label": spec_label,
                "nce": nce.long(),
            }


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


import re


class DeNovoIs2reDataset(BaseWrapperDataset):
    """A wrapper around a LMDB database that reads and returns items from it
    lazily."""

    def __init__(
        self, dataset, args, crop_rational=1 / 4, is_train=False, finetune=False
    ):
        super().__init__(dataset)
        self.dataset = dataset
        self.is_train = is_train
        self.args = args
        self.softmax = nn.Softmax(dim=-1)
        self.set_epoch(None)
        modification_meta_dict_path = args.modification_meta_dict_path
        self.modification_meta_dict = pickle.load(
            open(modification_meta_dict_path, "rb")
        )
        tokenize1_pkl_path = args.tokenize1_pkl_path
        self.tokenize1_dict = pickle.load(open(tokenize1_pkl_path, "rb"))
        self.modification_meta_dict["Ammonia-loss"] = self.modification_meta_dict[
            "Ammonia-loss[AnyN-termC]"
        ]
        self.min_mz = 1
        self.max_mz = 6500.0
        self.remove_precursor_tol = 2
        self.n_peaks = self.args.cutoff_spectra
        self.min_intensity = 0.0
        # ION_TYPES = ['b', 'b-NH3', 'b-H2O', 'b-ModLoss', 'y', 'y-NH3', 'y-H2O', 'y-ModLoss']
        if not self.args.no_mod:
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
        else:
            self.ion_types = ["b", "b-NH3", "b-H2O", "y", "y-NH3", "y-H2O"]

        self.token_dict = {
            "A": 1,  # 71.03711
            "B": 2,  # 0.000
            "C": 3,  # 103.009180
            "D": 4,  # 115.02694
            "E": 5,  # 129.04259
            "F": 6,  # 147.06841
            "G": 7,  # 57.02146
            "H": 8,  # 137.05891
            "I": 9,  # 0.000
            "J": 10,  # 0.000
            "K": 11,  # 128.09496
            "L": 12,  # 113.08406
            "M": 13,  # 131.04049
            "N": 14,  # 114.04293
            "O": 15,  # 0.00000
            "P": 16,  # 97.05276
            "Q": 17,  # 128.05858
            "R": 18,  # 156.10111
            "S": 19,  # 87.03203
            "T": 20,  # 101.04768
            "U": 21,  # 0.00000
            "V": 22,  # 99.06841
            "W": 23,  # 186.07931
            "X": 24,  # 0.0000
            "Y": 25,  # 163.06333
            "Z": 26,  # 0.00000
        }

        # try:
        #     # self.pept_to_spec_dict = pickle.load(open(join(self.args.data, "pept_to_spec.pkl"), 'rb'))
        #     if not finetune:
        #         pep2spec_path = join(self.args.data, "pept_to_spec2.lmdb")
        #     else:
        #         pep2spec_path = join(self.args.data, "labels", "pept_to_spec_ft_9.lmdb")

        #     env_read = lmdb.open(
        #         pep2spec_path,
        #         subdir=False,
        #         readonly=True,
        #         lock=False,
        #         readahead=False,
        #         meminit=False,
        #         max_readers=1,
        #         map_size=int(1e9),
        #     )
        #     self.txn_read = env_read.begin()
        #     # keys = list(txn_read.cursor().iternext(values=False))
        # except:
        #     print("warning: failed loading spec label")
        #     if not self.args.inference:
        #         assert 0

    def __len__(self):
        return len(self.dataset)

    def set_epoch(self, epoch, **unused):
        super().set_epoch(epoch)
        self.epoch = epoch

    def __getitem__(self, idx: int):
        return self.__getitem_cached__(self.epoch, idx)

    @lru_cache(maxsize=16)
    def __getitem_cached__(self, epoch: int, idx):
        with data_utils.numpy_seed(self.args.seed, epoch, idx):

            # input modification: all modification
            # input sequence: original to calcluate mass, after mod to embeding
            # pred label: delete(-> and cannot tell the exact residue type)

            spectrum = self.dataset[idx]
            try:
                spectrum = gzip.decompress(spectrum)
            except:
                pass
            try:
                spectrum = pickle.loads(spectrum)
            except:
                pass
            try:
                if "mgf" in spectrum["small"].keys():
                    mgf = spectrum["small"]["mgf"]
                else:
                    mgf = spectrum["small"][list(spectrum["small"].keys())[0]]
            except:
                if "mgf" in spectrum["large"].keys():
                    mgf = spectrum["large"]["mgf"]
                else:
                    mgf = spectrum["large"][list(spectrum["large"].keys())[0]]

            mz_array = torch.Tensor(mgf["mz_array"])
            intensity = torch.Tensor(mgf["intensity"])
            rt = torch.tensor([float(mgf["rtinseconds"])])
            precursor_mz = mgf["precursor_mz"]
            precursor_charge = mgf["precursor_charge"]
            try:
                title = mgf["title"]
            except:
                title = spectrum["small"]["title"]
            loss_residue = False
            add_residue = False
            # if self.args.fdr_thread > 1:
            #     if "C" in spectrum["small"]["1"]["peptide"]:
            #         if "C[" not in spectrum["small"]["1"]["peptide"]:
            #             print("not fix mod", spectrum["small"]["1"]["peptide"])
            #         else:
            #             print("fix mod", spectrum["small"]["1"]["peptide"])
            spectrum["small"]["1"]["peptide"] = re.sub(
                r"\[.*?\]", "", spectrum["small"]["1"]["peptide"]
            )

            residue_type = [spectrum["small"]["1"]["peptide"]]
            residue_type_after_mod_all = [
                pep_mod_2_pep(
                    spectrum["small"]["1"]["peptide"], spectrum["small"]["1"]["mods"]
                )
            ]

            residue_type_after_mod = [
                pep_mod_2_pep(
                    spectrum["small"]["1"]["peptide"], spectrum["small"]["1"]["mods"]
                )
            ]
            modification = [[]]
            mod_label = torch.zeros(len(spectrum["small"]["1"]["peptide"]))
            mod_mass_label = torch.zeros(len(spectrum["small"]["1"]["peptide"]))
            assert len(spectrum["small"]["1"]["peptide"]) == len(residue_type[0]), (
                len(spectrum["small"]["1"]["peptide"]),
                len(residue_type[0]),
            )

            # for mod in spectrum["small"]["1"]["mods"]:
            #     if mod[0] == 0:
            #         mod = (mod[0] + 1, mod[1])
            #         # mod[0] += 1
            #     if mod[0] > len(spectrum["small"]["1"]["peptide"]):
            #         mod = (mod[0] - 1, mod[1])
            #     if "->" not in mod[1] or mod[1].split("->")[1].split("[")[0] not in amino_acids_3to1_map.keys():
            #         modification[0].append((mod[0], self.modification_meta_dict[mod[1]]))
            #     token1 = get_token1(mod[1])
            #     residue_type_after_mod_all[0] += f"_{mod[0]}_{token1}"

            for i in range(len(spectrum["small"]["1"]["mods"])):
                mod = spectrum["small"]["1"]["mods"][i]
                if mod[0] == 0:
                    mod = (mod[0] + 1, mod[1])
                    # mod[0] += 1
                if mod[0] > len(spectrum["small"]["1"]["peptide"]):
                    mod = (mod[0] - 1, mod[1])
                    # mod[0] -= 1
                try:
                    modification[0].append(
                        (mod[0], self.modification_meta_dict[mod[1]])
                    )
                except:
                    if mod[1] == "Methyl[KRHQNEDC]":
                        new_mod_str = (
                            f'Methyl[{spectrum["small"]["1"]["peptide"][mod[0]-1]}]'
                        )
                        if new_mod_str in self.modification_meta_dict.keys():
                            mod = (mod[0], new_mod_str)
                            modification[0].append(
                                (mod[0], self.modification_meta_dict[mod[1]])
                            )
                            spectrum["small"]["1"]["mods"][i] = mod
                        else:
                            spectrum["small"]["1"]["mods"].pop(i)
                            continue
                            # assert 0, (spectrum["small"]["1"]["mods"])

                if mod[1] in loss_dict_loss_length.keys():
                    loss_residue = True

                if mod[1] in loss_dict_add_length.keys():
                    add_residue = True

                if (
                    "->" not in mod[1]
                    or mod[1].split("->")[1].split("[")[0]
                    not in amino_acids_3to1_map.keys()
                ):
                    if mod[1] not in denovo_not_pred_set:
                        if self.args.tokenize_mod == 0:
                            mod_label[mod[0] - 1] = (
                                self.modification_meta_dict[mod[1]][2].item() + 1
                            )
                        elif self.args.tokenize_mod == 1:
                            mod_label[mod[0] - 1] = self.tokenize1_dict[
                                self.modification_meta_dict[mod[1]][2].item()
                            ]["token_idx"]

                        mod_mass_label[mod[0] - 1] += self.modification_meta_dict[
                            mod[1]
                        ][1].item()
            if loss_residue:
                mod_label = mod_label[:-1]

            if add_residue:
                zero_tensor = torch.tensor([0.0])
                mod_label = torch.cat((zero_tensor, mod_label))

            assert len(modification) == len(residue_type), (
                len(modification),
                len(residue_type),
                len(residue_type_after_mod),
            )

            assert len(modification) == len(residue_type), (
                len(modification),
                len(residue_type),
                len(residue_type_after_mod),
            )

            spectrum_ = sus.MsmsSpectrum(
                "",
                precursor_mz,
                precursor_charge,
                mz_array.numpy().astype(np.float64),
                intensity.numpy().astype(np.float32),
            )

            spectrum_.set_mz_range(50.5, 4500.0)
            if len(spectrum_.mz) == 0:
                raise ValueError
            spectrum_.remove_precursor_peak(20, "ppm")
            if len(spectrum_.mz) == 0:
                raise ValueError
            spectrum_.filter_intensity(self.min_intensity, self.n_peaks)
            if len(spectrum_.mz) == 0:
                raise ValueError
            spectrum_.scale_intensity("root", 1)
            intensity = spectrum_.intensity / np.linalg.norm(spectrum_.intensity)
            mz_array = spectrum_.mz
            mz_array, intensity = torch.Tensor(mz_array), torch.Tensor(intensity)

            try:
                instrument = torch.Tensor([mgf["inst"]]).long()
                nce = torch.Tensor([mgf["HCD"]]).long()
            except:
                instrument = torch.Tensor([0]).long()
                nce = torch.Tensor([0]).long()
                print("warning no ins nce")
            if self.args.spectrum_pred:
                spec_label = [0]
                # spec_label, ion_peak_flag, ion_peak_class, ion_res_num, res_idx, error_tol, ion_mz = get_spectrum_prediction_label(spectrum["small"]["1"]["peptide"], mz_array, intensity, spectrum["small"]["1"]["mods"], self.ion_types, self.args.max_charges, flag_ITMS=False, shift=True, return_mz=True)
            else:
                spec_label = [0]
            seq_len_label = torch.Tensor(
                [len(spectrum["small"]["1"]["peptide"])]
            ).long()

            precursor_charge = min(self.args.max_prec_charge, precursor_charge)
            try:
                fdr = spectrum["small"]["1"]["fdr_value"]
            except:
                fdr = 0
            ret = {
                "mz_array": mz_array,
                "intensity": intensity,
                "precursor_mz": torch.Tensor(
                    [(precursor_mz - 1.007276) * precursor_charge]
                ),
                "precursor_charge": torch.Tensor([precursor_charge]).long(),
                "batch_index": torch.ones(self.args.range_pred).long(),
                "residue_type": residue_type,
                "residue_type_after_mod": residue_type_after_mod,
                "residue_type_after_mod_all": residue_type_after_mod_all,
                "modification": modification,
                "instrument": instrument,
                "nce": nce,
                "RT": rt,
                "seq_len_label": seq_len_label,
                "idx": torch.Tensor([idx]),
                "title": [title],
                "fdr": torch.Tensor([fdr]),
                "repeat_num": torch.Tensor([self.args.range_pred]),
                "spec_label": torch.Tensor(spec_label),
            }
            return ret

    def tokenize(self, input):
        ret = torch.zeros(len(input))
        for i in range(len(input)):
            ret[i] = self.token_dict[input[i]]
        return ret.long()


def reorgnize(data, start_id=2):
    num = data["small"]["num_res"]
    true_num = 0
    if num == 1:
        return data
    for i in range(1, num + 1):
        if data["small"][str(i)]["is_target"] or i < start_id:
            true_num += 1
            data["small"][str(true_num)] = data["small"][str(i)]
    data["small"]["num_res"] = true_num
    # print(data["small"]["num_res"])
    return data


class ContrastIs2reDataset(BaseWrapperDataset):
    """A wrapper around a LMDB database that reads and returns items from it
    lazily."""

    def __init__(
        self, dataset, args, crop_rational=1 / 4, is_train=False, finetune=False
    ):
        super().__init__(dataset)
        self.dataset = dataset
        self.is_train = is_train
        self.args = args
        self.softmax = nn.Softmax(dim=-1)
        self.set_epoch(None)
        modification_meta_dict_path = args.modification_meta_dict_path
        self.modification_meta_dict = pickle.load(
            open(modification_meta_dict_path, "rb")
        )
        tokenize1_pkl_path = args.tokenize1_pkl_path
        self.tokenize1_dict = pickle.load(open(tokenize1_pkl_path, "rb"))
        self.modification_meta_dict["Ammonia-loss"] = self.modification_meta_dict[
            "Ammonia-loss[AnyN-termC]"
        ]
        self.min_mz = 1
        self.max_mz = 6500.0
        self.remove_precursor_tol = 2
        self.n_peaks = self.args.cutoff_spectra
        self.min_intensity = 0.0
        # ION_TYPES = ['b', 'b-NH3', 'b-H2O', 'b-ModLoss', 'y', 'y-NH3', 'y-H2O', 'y-ModLoss']
        if not self.args.no_mod:
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
        else:
            self.ion_types = ["b", "b-NH3", "b-H2O", "y", "y-NH3", "y-H2O"]

        self.token_dict = {
            "A": 1,  # 71.03711
            "B": 2,  # 0.000
            "C": 3,  # 103.009180
            "D": 4,  # 115.02694
            "E": 5,  # 129.04259
            "F": 6,  # 147.06841
            "G": 7,  # 57.02146
            "H": 8,  # 137.05891
            "I": 9,  # 0.000
            "J": 10,  # 0.000
            "K": 11,  # 128.09496
            "L": 12,  # 113.08406
            "M": 13,  # 131.04049
            "N": 14,  # 114.04293
            "O": 15,  # 0.00000
            "P": 16,  # 97.05276
            "Q": 17,  # 128.05858
            "R": 18,  # 156.10111
            "S": 19,  # 87.03203
            "T": 20,  # 101.04768
            "U": 21,  # 0.00000
            "V": 22,  # 99.06841
            "W": 23,  # 186.07931
            "X": 24,  # 0.0000
            "Y": 25,  # 163.06333
            "Z": 26,  # 0.00000
        }

        # self.change_dict = {}
        # top1_path = self.args.fix_top1
        # pkls = [join(top1_path, _) for _ in os.listdir(top1_path)]
        # for pkl in pkls:
        #     print(f"loaded{pkl}")
        #     self.change_dict.update(pickle.load(open(pkl, 'rb')))

        try:
            # self.pept_to_spec_dict = pickle.load(open(join(self.args.data, "pept_to_spec.pkl"), 'rb'))
            env_read = lmdb.open(
                join(self.args.data, "pept_to_spec61.lmdb"),
                subdir=False,
                readonly=True,
                lock=False,
                readahead=False,
                meminit=False,
                max_readers=1,
                map_size=int(1e9),
            )
            self.txn_read = env_read.begin()
            # keys = list(txn_read.cursor().iternext(values=False))
        except:
            print("warning: failed loading spec label")

    def __len__(self):
        return len(self.dataset)

    def set_epoch(self, epoch, **unused):
        super().set_epoch(epoch)
        self.epoch = epoch

    def __getitem__(self, idx: int):
        return self.__getitem_cached__(self.epoch, idx)

    @lru_cache(maxsize=16)
    def __getitem_cached__(self, epoch: int, idx):
        with data_utils.numpy_seed(self.args.seed, epoch, idx):

            # input modification: all modification
            # input sequence: original to calcluate mass, after mod to embeding
            # pred label: delete(-> and cannot tell the exact residue type)

            spectrum = self.dataset[idx]
            try:
                spectrum = gzip.decompress(spectrum)
            except:
                pass
            try:
                spectrum = pickle.loads(spectrum)
            except:
                pass

            if "mgf" in spectrum["small"].keys():
                mgf = spectrum["small"]["mgf"]
            else:
                mgf = spectrum["small"][list(spectrum["small"].keys())[0]]

            try:
                title = mgf["title"]
            except:
                title = spectrum["small"]["title"]

            # if title in self.change_dict.keys():
            #     spectrum["small"]["1"] = pickle.loads(gzip.decompress(self.change_dict[title]))["1"]

            mz_array = torch.Tensor(mgf["mz_array"])
            intensity = torch.Tensor(mgf["intensity"])
            precursor_mz = mgf["precursor_mz"]
            precursor_charge = mgf["precursor_charge"]
            spectrum["small"]["1"]["peptide"] = spectrum["small"]["1"][
                "peptide"
            ].replace("I", "L")

            modification_cls_label = torch.zeros(2610).long()

            flag_ITMS = False

            # check_fix_mod(spectrum)
            ret = spectrum["small"]["1"]

            residue_type = [spectrum["small"]["1"]["peptide"]]
            residue_type_after_mod_all = [
                pep_mod_2_pep(
                    spectrum["small"]["1"]["peptide"], spectrum["small"]["1"]["mods"]
                )
            ]
            residue_type_after_mod = [
                pep_mod_2_pep(
                    spectrum["small"]["1"]["peptide"], spectrum["small"]["1"]["mods"]
                )
            ]
            modification = [[]]
            mod_label = torch.zeros(len(spectrum["small"]["1"]["peptide"]))
            mod_mass_label = torch.zeros(len(spectrum["small"]["1"]["peptide"]))
            loss_residue = False
            add_residue = False
            spectrum["small"]["1"]["mods"] = [
                (mod[0], mod[1]) for mod in spectrum["small"]["1"]["mods"]
            ]
            for mod in spectrum["small"]["1"]["mods"]:
                if mod[0] == 0:
                    mod = (mod[0] + 1, mod[1])
                    # mod[0] += 1
                if mod[0] > len(spectrum["small"]["1"]["peptide"]):
                    mod = (mod[0] - 1, mod[1])
                    # mod[0] -= 1
                modification[0].append((mod[0], self.modification_meta_dict[mod[1]]))
                if "->" not in mod[1]:
                    modification_cls_label[
                        int(self.modification_meta_dict[mod[1]][2])
                    ] = 1
                if (
                    "->" not in mod[1]
                    or mod[1].split("->")[1].split("[")[0]
                    not in amino_acids_3to1_map.keys()
                ):
                    if self.args.tokenize_mod == 0:
                        mod_label[mod[0] - 1] = (
                            self.modification_meta_dict[mod[1]][2].item() + 1
                        )
                    elif self.args.tokenize_mod == 1:
                        mod_label[mod[0] - 1] = self.tokenize1_dict[
                            self.modification_meta_dict[mod[1]][2].item()
                        ]["token_idx"]

                    mod_mass_label[mod[0] - 1] += self.modification_meta_dict[mod[1]][
                        1
                    ].item()
                    # print(mod_mass_label, mod_label, spectrum["small"]["1"]["peptide"])
                    # residue_type_after_mod[0][mod[0]] = protein_letters_3to1[mod[1].split("->")[1].split("-")[1].split("[")[0].upper()]
                if mod[1] in loss_dict_loss_length.keys():
                    loss_residue = True

                if mod[1] in loss_dict_add_length.keys():
                    add_residue = True

            if loss_residue:
                mod_label = mod_label[:-1]

            if add_residue:
                zero_tensor = torch.tensor([0.0])
                mod_label = torch.cat((zero_tensor, mod_label))

            spectrum_ = sus.MsmsSpectrum(
                "",
                precursor_mz,
                precursor_charge,
                mz_array.numpy().astype(np.float64),
                intensity.numpy().astype(np.float32),
            )

            spectrum_.set_mz_range(50.5, 4500.0)
            if len(spectrum_.mz) == 0:
                raise ValueError
            spectrum_.remove_precursor_peak(20, "ppm")
            if len(spectrum_.mz) == 0:
                raise ValueError
            spectrum_.filter_intensity(self.min_intensity, self.n_peaks)
            if len(spectrum_.mz) == 0:
                raise ValueError
            spectrum_.scale_intensity("root", 1)
            intensity = spectrum_.intensity / np.linalg.norm(spectrum_.intensity)
            mz_array = spectrum_.mz
            mz_array, intensity = torch.Tensor(mz_array), torch.Tensor(intensity)

            (
                spec_label,
                ion_peak_flag,
                ion_peak_class,
                ion_res_num,
                res_idx,
                error_tol,
            ) = get_spectrum_prediction_label(
                spectrum["small"]["1"]["peptide"],
                mz_array,
                intensity,
                spectrum["small"]["1"]["mods"],
                self.ion_types,
                self.args.max_charges,
                flag_ITMS=flag_ITMS,
                shift=self.args.shift,
            )
            # assert (ion_peak_flag > 0).sum() == (ion_res_num > 0).sum() == (ion_peak_class > 0).sum(), (ion_peak_flag.shape, ion_res_num.shape, noise_peak_label.shape, (ion_peak_flag > 0).sum(), (ion_res_num > 0).sum(), (noise_peak_label > 0).sum())

            # spec_label_strict, ion_peak_flag_strict, ion_peak_class_strict, error_tol_strict = get_spectrum_prediction_label(spectrum["small"]["1"]["peptide"], mz_array, intensity, spectrum["small"]["1"]["mods"], self.args.max_charges, PPM_THRESHOLD=5)
            str_pept = data_2_str(spectrum, self.modification_meta_dict)
            # tmp = spec_label.copy()
            try:
                spec_label, _ = pickle.loads(
                    gzip.decompress(self.txn_read.get(str_pept.encode("ascii")))
                )
            except:
                # print(f"error_loading {str_pept} spec")
                pass
            spec_label = torch.Tensor(spec_label)

            ion_comp_res_num = (ion_res_num > 0) * (
                len(spectrum["small"]["1"]["peptide"]) - ion_res_num
            )
            # ion_peak_flag_strict = ion_peak_flag_strict[top_indices]

            peak_indices = torch.nonzero(ion_peak_flag)
            peak_classes = ion_peak_class[peak_indices]

            res_idx_ = res_idx[peak_indices]

            if self.is_train and self.args.denoise_pred:
                assert 0
            else:
                denoise_label = torch.zeros(mz_array.shape[0])
                denoise_pred_mask = torch.zeros(len(mz_array))

            if self.is_train and self.args.intensity_pred:
                select_index_2 = torch.randint(
                    0,
                    len(peak_indices),
                    (int(len(peak_indices) * self.args.intensity_prob + 1),),
                    dtype=torch.int64,
                )
                intensity_label = torch.zeros(mz_array.shape[0]).type(intensity.dtype)
                intensity_label[select_index_2] = intensity[select_index_2]
                intensity[select_index_2] = 0
                intensity_pred_mask = torch.zeros(len(mz_array))
                intensity_pred_mask[select_index_2] += 1
            else:
                intensity_label = torch.zeros(mz_array.shape[0])
                intensity_pred_mask = torch.zeros(len(mz_array))

            if self.args.noise_peak_pred:
                noise_peak_label = torch.zeros(mz_array.shape[0])
                if not self.args.multi_class_noise_peak_pred:
                    noise_peak_label[peak_indices] += 1
                else:
                    noise_peak_label[peak_indices] += peak_classes
                # assert torch.all(noise_peak_label < 33)
                noise_peak_mask = torch.ones(mz_array.shape[0])
                if intensity_pred_mask.sum() > 0:
                    noise_peak_mask[select_index_2] -= 1
                if denoise_pred_mask.sum() > 0:
                    # noise_peak_mask[select_index] -= 1
                    pass
                noise_peak_mask = noise_peak_mask > 0
            else:
                noise_peak_label = torch.zeros(mz_array.shape[0])
                noise_peak_mask = torch.ones(mz_array.shape[0])

            try:
                instrument = torch.Tensor([mgf["inst"]]).long()
                nce = torch.Tensor([mgf["HCD"]]).long()
            except:
                instrument = torch.Tensor([0]).long()
                nce = torch.Tensor([0]).long()
                print("warning no ins nce")

            if self.args.res_type_pred:
                pep_seq = self.tokenize(residue_type_after_mod[0])
                # print(pep_seq)
                # ion_res_num
                # ion_comp_res_num
                res_idx_ = res_idx_.long()
                # res_idx_ = res_idx[res_idx > 0]
                assert torch.all(res_idx >= 0) and torch.all(
                    res_idx < len(spectrum["small"]["1"]["peptide"])
                ), (len(spectrum["small"]["1"]["peptide"]), res_idx)
                res_type_label = torch.zeros(noise_peak_label.shape)

                for i in range(len(res_idx_)):
                    # print(residue_type_after_mod[0], peak_indices[i].item(), res_idx_[i])
                    res_type_label[peak_indices[i].item()] = pep_seq[res_idx_[i]]
                # assert 0, (res_type_label, res_idx, ion_peak_class, residue_type_after_mod[0], peak_indices, peak_classes, len(peak_classes), noise_peak_label)
                # assert len(peak_indices) == len(res_idx_), (len(peak_indices), len(res_idx_))
                res_type_label = res_type_label * (noise_peak_label > 0)
                # print(res_type_label.shape, res_idx.shape,  pep_seq.shape, noise_peak_label.shape, mz_array.shape, peak_classes.shape)
            else:
                res_type_label = torch.Tensor([0])

            # assert mod_label.shape[0] == len(residue_type_after_mod[0]) == len(residue_type[0]), (mod_label.shape[0], len(residue_type_after_mod), len(residue_type[0]))
            if self.args.use_rope:
                rope_embeding = precompute_freqs_cis(
                    self.args.node_dim,
                    torch.cat(
                        [
                            torch.Tensor([mgf["precursor_mz"]]),
                            torch.Tensor([spectrum["small"]["mgf"]["precursor_mz"]]),
                            mz_array,
                        ]
                    ),
                )
            else:
                rope_embeding = torch.Tensor([0])

            # seq_len_label = torch.Tensor([len(spectrum["small"]["1"]["peptide"])]).long()
            seq_len_label = torch.Tensor([len(residue_type_after_mod_all[0])]).long()

            #  print(rope_embeding.shape)
            precursor_charge = min(self.args.max_prec_charge, precursor_charge)

            label_index = torch.zeros(1)
            label_index[0] = 1
            return {
                "mz_array": mz_array,
                "intensity": intensity,
                "precursor_mz": torch.Tensor(
                    [(precursor_mz - 1.007276) * precursor_charge]
                ),
                "precursor_charge": torch.Tensor([precursor_charge]).long(),
                "batch_index": torch.ones(1).long(),
                "residue_type": residue_type,
                "residue_type_after_mod": residue_type_after_mod,
                "residue_type_after_mod_all": residue_type_after_mod_all,
                "mod_label": mod_label.long(),
                "mod_mass_label": mod_mass_label,
                "modification": modification,
                "instrument": instrument,
                "spec_label": spec_label,
                "nce": nce,
                "denoise_pred_mask": denoise_pred_mask.bool(),
                "intensity_pred_mask": intensity_pred_mask.bool(),
                "denoise_label": denoise_label,
                "intensity_label": intensity_label,
                "rope_embeding": rope_embeding,
                "noise_peak_label": noise_peak_label.long(),
                "res_type_label": res_type_label.long(),
                "noise_peak_mask": noise_peak_mask.long(),
                "modification_cls_label": modification_cls_label.long(),
                "ion_res_num": ion_res_num.long(),
                "ion_comp_res_num": ion_comp_res_num.long(),
                "seq_len_label": seq_len_label,
                "idx": torch.Tensor([idx]).long(),
                "label_index": label_index.bool(),
            }

    def tokenize(self, input):
        ret = torch.zeros(len(input))
        for i in range(len(input)):
            ret[i] = self.token_dict[input[i]]
        return ret.long()


# class DenovoIs2reDataset(BaseWrapperDataset):
#     """A wrapper around a LMDB database that reads and returns items from it
#     lazily."""

#     def __init__(self, dataset, args, crop_rational=1/4, is_train=False):
#         super().__init__(dataset)
#         self.dataset = dataset
#         self.is_train = is_train
#         self.args = args
#         self.softmax = nn.Softmax(dim=-1)
#         self.set_epoch(None)
#         modification_meta_dict_path = args.modification_meta_dict_path
#         self.modification_meta_dict = pickle.load(open(modification_meta_dict_path, 'rb'))
#         self.modification_meta_dict["Ammonia-loss"] =  self.modification_meta_dict["Ammonia-loss[AnyN-termC]"]
#         self.min_mz = 1
#         self.max_mz = 6500.0
#         self.remove_precursor_tol = 2
#         self.n_peaks = self.args.cutoff_spectra
#         self.min_intensity = 0.0
#         # ION_TYPES = ['b', 'b-NH3', 'b-H2O', 'b-ModLoss', 'y', 'y-NH3', 'y-H2O', 'y-ModLoss']
#         if not self.args.no_mod:
#             self.ion_types = ['b', 'b-NH3', 'b-H2O', 'b-ModLoss', 'y', 'y-NH3', 'y-H2O', 'y-ModLoss']
#         else:
#             self.ion_types = ['b', 'b-NH3', 'b-H2O', 'y', 'y-NH3', 'y-H2O']

#         self.token_dict = {
#             "A": 1,# 71.03711
#             "B": 2,# 0.000
#             "C": 3,# 103.009180
#             "D": 4,# 115.02694
#             "E": 5,# 129.04259
#             "F": 6,# 147.06841
#             "G": 7,# 57.02146
#             "H": 8,# 137.05891
#             "I": 9,# 0.000
#             "J": 10,# 0.000
#             "K": 11,# 128.09496
#             "L": 9,# 113.08406
#             "M": 12,# 131.04049
#             "N": 13,# 114.04293
#             "O": 14,# 0.00000
#             "P": 15,# 97.05276
#             "Q": 16,# 128.05858
#             "R": 17,# 156.10111
#             "S": 18,# 87.03203
#             "T": 19,# 101.04768
#             "U": 20,# 0.00000
#             "V": 21,# 99.06841
#             "W": 22,# 186.07931
#             "X": 23,# 0.0000
#             "Y": 24,# 163.06333
#             "Z": 25,# 0.00000
#         }

#     def __len__(self):
#         return len(self.dataset)

#     def set_epoch(self, epoch, **unused):
#         super().set_epoch(epoch)
#         self.epoch = epoch

#     def __getitem__(self, idx: int):
#         return self.__getitem_cached__(self.epoch, idx)
#     @lru_cache(maxsize=16)
#     def __getitem_cached__(self, epoch: int, idx):
#         with data_utils.numpy_seed(self.args.seed, epoch, idx):
#             spectrum = self.dataset[idx]
#             try:
#                 spectrum = gzip.decompress(spectrum)
#             except:
#                 pass
#             try:
#                 spectrum = pickle.loads(spectrum)
#             except:
#                 pass
#             if "mgf" in spectrum["small"].keys():
#                 mgf = spectrum["small"]["mgf"]
#             else:
#                 mgf = spectrum["small"][list(spectrum["small"].keys())[0]]

#             mz_array = torch.Tensor(mgf["mz_array"])

#             intensity = torch.Tensor(mgf["intensity"])
#             precursor_mz = mgf["precursor_mz"]
#             precursor_charge = mgf["precursor_charge"]
#             modification_cls_label = torch.zeros(2610).long()
#             # try:
#             #     title = mgf["title"]
#             #     if "xIT_" in title:
#             #         flag_ITMS = True
#             #     else:
#             #         flag_ITMS = False
#             # except:
#             #     flag_ITMS = False
#             flag_ITMS = False

#             # if "large" in spectrum.keys():
#             #     if not spectrum["different"]:
#             #         rand = np.random.rand()
#             #         # if spectrum["small"]["num_res"] == 1 and spectrum["large"]["num_res"] == 1:
#             #         #     assert 0, idx
#             #         if (rand < 0.5 or spectrum["large"]["num_res"] < 2) and spectrum["small"]["num_res"] >= 2:
#             #             if spectrum["small"]["num_res"] == 2:
#             #                 index = 2
#             #             else:
#             #                 index = np.random.randint(2, spectrum["small"]["num_res"])

#             #             ret = spectrum["small"][str(index)]
#             #         else:
#             #             if spectrum["large"]["num_res"] == 2:
#             #                 index = 2
#             #             else:
#             #                 index = np.random.randint(2, spectrum["large"]["num_res"])
#             #                 # while spectrum["large"][str(idx)]["peptide"] == spectrum["small"]["1"]["peptide"] and \
#             #                 #     spectrum["large"][str(idx)]["mods"] == spectrum["small"]["1"]["mods"]:
#             #                 #     print("warning: same not good, change")
#             #                 #     idx = np.random.randint(2, spectrum["large"]["num_res"])

#             #             ret = spectrum["large"][str(index)]
#             #     else:
#             #         ret = spectrum["large"]["1"]
#             # else:
#             #     ret = spectrum["small"]["1"]

#             residue_type = [spectrum["small"]["1"]["peptide"]]
#             residue_type_after_mod = [spectrum["small"]["1"]["peptide"]]
#             modification = [[]]
#             mod_label = torch.zeros(len(spectrum["small"]["1"]["peptide"]))
#             for mod in spectrum["small"]["1"]["mods"]:
#                 if mod[0] == 0:
#                     mod = (mod[0] + 1, mod[1])
#                     # mod[0] += 1
#                 if mod[0] > len(spectrum["small"]["1"]["peptide"]):
#                     mod = (mod[0] - 1, mod[1])
#                     # mod[0] -= 1
#                 modification[0].append((mod[0], self.modification_meta_dict[mod[1]]))
#                 modification_cls_label[int(self.modification_meta_dict[mod[1]][2])] = 1
#                 if "->" in mod[1]:
#                     # print(mod[1])
#                     modto = mod[1].split("->")[1]
#                     try:
#                         if "-" not in modto:
#                             new_residue = protein_letters_3to1[modto[:3].upper()]
#                             # new_residue = protein_letters_3to1[modto.split("[")[0].upper()]
#                         else:
#                             new_residue = protein_letters_3to1[modto.split("-")[1].split("[")[0].upper()]
#                         residue_type_after_mod[0] = residue_type_after_mod[0][:mod[0]-1] + new_residue + residue_type_after_mod[0][mod[0]:]
#                     except:
#                         # print(">>>>", mod[1])
#                         pass
#                 else:
#                     # print(self.modification_meta_dict[mod[1]], mod_label.shape)
#                     # pass
#                     # print(spectrum)
#                     mod_label[mod[0] - 1] = self.modification_meta_dict[mod[1]][2].item() + 1
#                     # residue_type_after_mod[0][mod[0]] = protein_letters_3to1[mod[1].split("->")[1].split("-")[1].split("[")[0].upper()]
#             # for mod in ret["mods"]:
#             #     if mod[0] == 0:
#             #         mod = (mod[0] + 1, mod[1]) # 位置 meta信息
#             #         # mod[0] += 1
#             #     if mod[0] > len(ret["peptide"]):
#             #         mod = (mod[0] - 1, mod[1])
#             #         # mod[0] -= 1
#             #     modification[1].append((mod[0], self.modification_meta_dict[mod[1]]))

#             spectrum_ = sus.MsmsSpectrum(
#                 "",
#                 precursor_mz,
#                 precursor_charge,
#                 mz_array.numpy().astype(np.float64),
#                 intensity.numpy().astype(np.float32),
#             )

#             spectrum_.set_mz_range(self.min_mz, self.max_mz)
#             if len(spectrum_.mz) == 0:
#                 raise ValueError
#             spectrum_.remove_precursor_peak(self.remove_precursor_tol, "Da")
#             if len(spectrum_.mz) == 0:
#                 raise ValueError
#             # spectrum_.filter_intensity(self.min_intensity, self.n_peaks)
#             if len(spectrum_.mz) == 0:
#                 raise ValueError
#             spectrum_.scale_intensity("root", 1)
#             intensity = spectrum_.intensity / np.linalg.norm(
#                 spectrum_.intensity
#             )
#             mz_array = spectrum_.mz
#             mz_array, intensity = torch.Tensor(mz_array), torch.Tensor(intensity)

#             # mz_array, intensity = sqrt_and_norm(mz_array, intensity, precursor_mz, precursor_charge)
#             # mz_array, intensity = remove_precursor_peak(mz_array, intensity, precursor_mz)

#             # top_values, top_indices = torch.topk(intensity.view(-1), k=min(self.args.cutoff_spectra, mz_array.shape[0]))

#             # print(spectrum["small"]["title"], spectrum["small"]["1"]["peptide"])
#             # spec_label, ion_peak_flag, ion_peak_class, ion_res_num, res_idx, error_tol = get_spectrum_prediction_label(spectrum["small"]["1"]["peptide"], mz_array, intensity, spectrum["small"]["1"]["mods"], self.ion_types, self.args.max_charges, flag_ITMS=flag_ITMS, shift=self.args.shift)
#             # spec_label_strict, ion_peak_flag_strict, ion_peak_class_strict, error_tol_strict = get_spectrum_prediction_label(spectrum["small"]["1"]["peptide"], mz_array, intensity, spectrum["small"]["1"]["mods"], self.args.max_charges, PPM_THRESHOLD=5)
#             # spec_label = torch.from_numpy(spec_label)

#             # assert 0, (torch.sum(torch.from_numpy(spec_label)), torch.sum(intensity), len(residue_type[0]), spec_label, spec_label.shape)

#             # mz_array = mz_array[top_indices]
#             # intensity = intensity[top_indices]
#             # ion_peak_flag = ion_peak_flag[top_indices]
#             # ion_peak_class = ion_peak_class[top_indices]
#             # ion_res_num = ion_res_num[top_indices]
#             # res_idx = res_idx[top_indices]
#             # ion_comp_res_num = (ion_res_num > 0) * (len(spectrum["small"]["1"]["peptide"]) - ion_res_num)
#             # ion_peak_flag_strict = ion_peak_flag_strict[top_indices]

#             # peak_indices = torch.nonzero(ion_peak_flag)
#             # peak_classes = ion_peak_class[peak_indices]

#             # assert torch.all(peak_classes > 0)
#             # print("peak num", len(peak_indices), len(spectrum["small"]["1"]["peptide"]))
#             # if len(peak_indices) == 0:
#                 # print("too less peaks:", idx, spectrum["small"]["1"]["peptide"], len(peak_indices))
#             # peak_indices_strict = torch.nonzero(ion_peak_flag_strict)


#             # if self.is_train and self.args.denoise_pred:
#             #     assert 0
#             #     # peak_classes_strict = ion_peak_class_strict[peak_indices_strict]

#             #     # select_index = torch.randint(0, len(peak_indices_strict), (int(len(peak_indices_strict) * self.args.denoise_prob + 1),), dtype=torch.int64)
#             #     # select_index = peak_indices_strict[select_index]
#             #     # denoise_label = torch.zeros(mz_array.shape[0])
#             #     # noise = torch.from_numpy(np.random.randn(len(select_index))).float() * error_tol * 5
#             #     # # print(torch.sign(noise) * torch.log(torch.abs(noise) + 0.00001))
#             #     # # print(denoise_label[select_index].shape, noise.shape)
#             #     # noise = noise.unsqueeze(1)
#             #     # denoise_label[select_index] += torch.sign(noise) * torch.log(torch.abs(noise) + 0.00001)
#             #     # mz_array[select_index] += noise
#             #     # denoise_pred_mask = torch.zeros(len(mz_array))
#             #     # denoise_pred_mask[select_index] += 1
#             # else:
#             #     denoise_label = torch.zeros(mz_array.shape[0])
#             #     denoise_pred_mask = torch.zeros(len(mz_array))


#             # if self.is_train and self.args.intensity_pred:
#             #     select_index_2 = torch.randint(0, len(peak_indices), (int(len(peak_indices) * self.args.intensity_prob + 1),), dtype=torch.int64)
#             #     intensity_label = torch.zeros(mz_array.shape[0]).type(intensity.dtype)
#             #     intensity_label[select_index_2] = intensity[select_index_2]
#             #     intensity[select_index_2] = 0
#             #     intensity_pred_mask = torch.zeros(len(mz_array))
#             #     intensity_pred_mask[select_index_2] += 1
#             # else:
#             #     intensity_label = torch.zeros(mz_array.shape[0])
#             #     intensity_pred_mask = torch.zeros(len(mz_array))

#             # if self.args.noise_peak_pred:
#             #     noise_peak_label = torch.zeros(mz_array.shape[0])
#             #     if not self.args.multi_class_noise_peak_pred:
#             #         noise_peak_label[peak_indices] += 1
#             #     else:
#             #         noise_peak_label[peak_indices] += peak_classes
#             #     assert torch.all(noise_peak_label < 33)
#             #     noise_peak_mask = torch.ones(mz_array.shape[0])
#             #     if intensity_pred_mask.sum() > 0:
#             #         noise_peak_mask[select_index_2] -= 1
#             #     if denoise_pred_mask.sum() > 0:
#             #         # noise_peak_mask[select_index] -= 1
#             #         pass
#             #     noise_peak_mask = noise_peak_mask > 0
#             # else:
#             #     noise_peak_label = torch.zeros(mz_array.shape[0])
#             #     noise_peak_mask = torch.ones(mz_array.shape[0])

#             try:
#                 instrument = torch.Tensor([mgf["inst"]]).long()
#                 nce = torch.Tensor([mgf["HCD"]]).long()
#             except:
#                 instrument = torch.Tensor([0]).long()
#                 nce = torch.Tensor([0]).long()
#                 print("warning no ins nce")

#             # if self.args.res_type_pred:
#             #     pep_seq = self.tokenize(spectrum["small"]["1"]["peptide"])
#             #     # ion_res_num
#             #     # ion_comp_res_num
#             #     assert torch.all(res_idx > 0) and torch.all(res_idx < len(spectrum["small"]["1"]["peptide"]))
#             #     res_type_label = pep_seq[res_idx] * (peak_classes > 0)
#             # else:
#             #     res_type_label = torch.Tensor([0])


#             # assert mod_label.shape[0] == len(residue_type_after_mod[0]) == len(residue_type[0]), (mod_label.shape[0], len(residue_type_after_mod), len(residue_type[0]))
#             # if self.args.use_rope:
#             #     rope_embeding = precompute_freqs_cis(self.args.node_dim, torch.cat([torch.Tensor([mgf["precursor_mz"]]), torch.Tensor([spectrum["small"]["mgf"]["precursor_mz"]]), mz_array]))
#             # else:
#             #     rope_embeding = torch.Tensor([0])

#             #  print(rope_embeding.shape)
#             return {
#                 "mz_array": mz_array,
#                 "intensity": intensity,
#                 "precursor_mz": torch.Tensor([(precursor_mz - 1.007276) * precursor_charge]),
#                 "precursor_charge": torch.Tensor([precursor_charge]).long(),
#                 "batch_index": torch.ones(2).long(),
#                 "residue_type": residue_type,
#                 "residue_type_after_mod": residue_type_after_mod,
#                 # "mod_label": torch.flip(mod_label.long()),
#                 "modification": modification,
#                 "instrument": instrument,
#                 # "spec_label": spec_label,
#                 "nce": nce,
#                 "index": torch.Tensor([idx]).long(),
#                 # "denoise_pred_mask": denoise_pred_mask.bool(),
#                 # "intensity_pred_mask": intensity_pred_mask.bool(),
#                 # "denoise_label": denoise_label,
#                 # "intensity_label": intensity_label,
#                 # "rope_embeding": rope_embeding,
#                 # "noise_peak_label": noise_peak_label.long(),
#                 # "res_type_label": res_type_label.long(),
#                 # "noise_peak_mask": noise_peak_mask.long(),
#                 # "modification_cls_label": modification_cls_label.long(),
#                 # "ion_res_num": ion_res_num.long(),
#                 # "ion_comp_res_num": ion_comp_res_num.long()
#             }
#     def tokenize(self, input):
#         ret = torch.zeros(len(input))
#         for i in range(len(input)):
#             ret[i] = self.token_dict[input[i]]
#         return ret.long()


# class Is2reDataset(BaseWrapperDataset):
#     """A wrapper around a LMDB database that reads and returns items from it
#     lazily."""

#     def __init__(self, dataset, args, crop_rational=1/4, is_train=False):
#         super().__init__(dataset)
#         self.dataset = dataset
#         self.is_train = is_train
#         self.args = args
#         self.softmax = nn.Softmax(dim=-1)
#         self.set_epoch(None)


#     def __len__(self):
#         return len(self.dataset)

#     def set_epoch(self, epoch, **unused):
#         super().set_epoch(epoch)
#         self.epoch = epoch

#     def __getitem__(self, idx: int):
#         return self.__getitem_cached__(self.epoch, idx)
#     @lru_cache(maxsize=16)
#     def __getitem_cached__(self, epoch: int, idx):
#         with data_utils.numpy_seed(self.args.seed, epoch, idx):
#             spectrum = self.dataset[idx]
#             try:
#                 spectrum = gzip.decompress(spectrum)
#             except:
#                 pass
#             try:
#                 spectrum = pickle.loads(spectrum)
#             except:
#                 pass
#             mz_array = torch.Tensor(spectrum["mz_array"])
#             intensity = torch.Tensor(spectrum["intensity"])
#             precursor_mz = spectrum["precursor_mz"]
#             precursor_charge = spectrum["precursor_charge"]
#             rt_normalized = spectrum["rt_normalized"]
#             nce = spectrum["nce"] + 1
#             ins = spectrum["ins"] + 1
#             if not self.args.use_nce and not self.args.pred_nce:
#                 nce = 0
#             if not self.args.use_ins and not self.args.pred_nce:
#                 ins = 0
#             peptide_seq = spectrum["peptide_seq"]
#             list_aa = []
#             for aa in peptide_seq:
#                 try:
#                     list_aa.append(restype_order_with_x[aa])
#                 except:
#                     list_aa.append(21)

#             residue_type = torch.Tensor(list_aa) + 1

#             mz_array, intensity = sqrt_and_norm(mz_array, intensity, precursor_mz, precursor_charge)
#             mz_array, intensity = remove_precursor_peak(mz_array, intensity, precursor_mz)

#             top_values, top_indices = torch.topk(intensity.view(-1), k=min(self.args.cutoff, mz_array.shape[0]))

#             mz_array = mz_array[top_indices]
#             intensity = intensity[top_indices]

#             return {
#                 "mz_array": mz_array,
#                 "intensity": intensity,
#                 "precursor_mz": torch.Tensor([(precursor_mz - 1.007276) * precursor_charge]),
#                 "precursor_charge": torch.Tensor([precursor_charge]).long(),
#                 "nce": torch.Tensor([nce]).long(),
#                 "ins": torch.Tensor([ins]).long(),
#                 "rt_normalized": torch.Tensor([rt_normalized]).long(),
#                 "residue_type": residue_type,
#             }


class SeqStrDataset(BaseWrapperDataset):
    def __init__(
        self,
        dataset: torch.utils.data.Dataset,
        key="",
        padding_value=0,
    ):
        self.dataset = dataset
        self.key = key
        self.padding_value = padding_value
        self.set_epoch(None)

    def set_epoch(self, epoch, **unused):
        super().set_epoch(epoch)
        self.epoch = epoch

    def __getitem__(self, idx: int):
        return self.__getitem_cached__(self.epoch, idx)

    @lru_cache(maxsize=16)
    def __getitem_cached__(self, epoch: int, index: int):
        return self.dataset[index][self.key]

    def __len__(self):
        return len(self.dataset)

    def collater(self, samples):
        list_samples = [x for sublist in samples for x in sublist]
        return list_samples


class SeqDataset(BaseWrapperDataset):
    def __init__(
        self,
        dataset: torch.utils.data.Dataset,
        key="",
        padding_value=0,
    ):
        self.dataset = dataset
        self.key = key
        self.padding_value = padding_value
        self.set_epoch(None)

    def set_epoch(self, epoch, **unused):
        super().set_epoch(epoch)
        self.epoch = epoch

    def __getitem__(self, idx: int):
        return self.__getitem_cached__(self.epoch, idx)

    @lru_cache(maxsize=16)
    def __getitem_cached__(self, epoch: int, index: int):
        return self.dataset[index][self.key]

    def __len__(self):
        return len(self.dataset)

    def collater(self, samples):
        return torch.nn.utils.rnn.pad_sequence(
            samples, batch_first=True, padding_value=self.padding_value
        )


class ScalerDataset(BaseWrapperDataset):
    def __init__(
        self,
        dataset: torch.utils.data.Dataset,
        key="",
    ):
        self.dataset = dataset
        self.key = key
        self.set_epoch(None)

    def set_epoch(self, epoch, **unused):
        super().set_epoch(epoch)
        self.epoch = epoch

    def __getitem__(self, idx: int):
        return self.__getitem_cached__(self.epoch, idx)

    @lru_cache(maxsize=16)
    def __getitem_cached__(self, epoch: int, index: int):
        return self.dataset[index][self.key]

    def __len__(self):
        return len(self.dataset)

    def collater(self, samples):
        return torch.cat(samples, dim=0)


class BatchIndexDataset(BaseWrapperDataset):
    def __init__(
        self,
        dataset: torch.utils.data.Dataset,
        key="batch_index",
    ):
        self.dataset = dataset
        self.key = key
        self.set_epoch(None)

    def set_epoch(self, epoch, **unused):
        super().set_epoch(epoch)
        self.epoch = epoch

    def __getitem__(self, idx: int):
        return self.__getitem_cached__(self.epoch, idx)

    @lru_cache(maxsize=16)
    def __getitem_cached__(self, epoch: int, index: int):
        return self.dataset[index][self.key]

    def __len__(self):
        return len(self.dataset)

    def collater(self, samples):
        idx = 0
        a = torch.cat(samples, dim=0)
        if torch.all(a == 1):
            for i in range(len(samples)):
                assert samples[i].shape[0] > 0
                if (samples[i] == 1).all():
                    b = samples[i]
                    b = b * idx
                    samples[i] = b
                idx += 1
            return torch.cat(samples, dim=0)

        return torch.cat(samples, dim=0).long()
