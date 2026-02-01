import gzip
import logging
import os
import pickle
import sys
from multiprocessing import Pool
from os.path import join

import numpy as np
import torch
from tqdm import tqdm

# import matplotlib.pyplot as plt

png_path = r"/mnt/vepfs/fs_users/zhaojiale/Contrast_MS_Pep/results"


def calculate_cumulative_count_peptide(tuple_list, x_range, trap_stat_thread=0.01):
    counts = [0] * len(x_range)
    trap_counts = [0] * len(x_range)
    decoy_counts = [0] * len(x_range)
    trap_stat = 0
    decoy_stat = 0

    prefix_sum = 0
    peptide_set = set()
    trap_num = []
    not_trap_decoy_num = []
    trap_num_sum = 0
    not_trap_decoy_num_sum = 0

    target_num = []
    decoy_num = []
    trap_num_forward, target_num_forward = [], []
    target_sum = 0
    decoy_sum = 0
    sequence_set = set()

    for tup in tuple_list:
        if not isinstance(tup, dict):
            # if isinstance(tup, tuple):
            first_value = tup[0]
            is_target = tup[1]
            peptide_str = tup[2]
            is_trap = tup[3]
            is_trap_decoy = tup[4]
        elif isinstance(tup, dict):
            first_value = tup["pscore_fdr"]
            is_target = tup["pscore_is_target"]
            peptide_str = tup["peptide_str"]
            is_trap = tup["is_trap"]
            is_trap_decoy = tup["is_trap_decoy"]
        else:
            assert 0, "not implemented"
        if peptide_str in peptide_set:
            continue
        peptide_set.add(peptide_str)
        if first_value <= x_range[-1] and is_target:
            index = int(
                first_value / x_range[-1] * (len(x_range) - 1)
            )  # 计算当前值在范围中的索引
            counts[index] += 1
            prefix_sum += 1

        if first_value <= x_range[-1] and is_trap:
            index = int(
                first_value / x_range[-1] * (len(x_range) - 1)
            )  # 计算当前值在范围中的索引
            trap_counts[index] += 1
            if first_value < trap_stat_thread:
                trap_stat += 1
                trap_num_forward.append(trap_stat)
                target_num_forward.append(decoy_stat)
                # trap_num_forward.append(trap_stat)
                # target_num_forward.append(decoy_stat)

        if first_value <= x_range[-1] and not is_target:
            index = int(
                first_value / x_range[-1] * (len(x_range) - 1)
            )  # 计算当前值在范围中的索引
            decoy_counts[index] += 1
            if first_value < trap_stat_thread:
                # decoy_stat += 1
                pass
        if is_target and first_value < trap_stat_thread:
            decoy_stat += 1  # now stat all target
            sequence_set.add(peptide_str.split("_")[0])

    peptide_set = set()
    for tup in reversed(tuple_list):
        if not isinstance(tup, dict):
            # if isinstance(tup, tuple):
            first_value = tup[0]
            is_target = tup[1]
            peptide_str = tup[2]
            is_trap = tup[3]
            is_trap_decoy = tup[4]
        elif isinstance(tup, dict):
            first_value = tup["pscore_fdr"]
            is_target = tup["pscore_is_target"]
            peptide_str = tup["peptide_str"]
            is_trap = tup["is_trap"]
            is_trap_decoy = tup["is_trap_decoy"]
        else:
            assert 0, "not implemented"
        if peptide_str in peptide_set:
            continue
        if first_value > 0.2:
            continue
        # if not_trap_decoy_num_sum > 20000:
        #     continue
        peptide_set.add(peptide_str)
        if is_target:
            target_sum += 1
        else:
            decoy_sum += 1
            target_num.append(target_sum)
            decoy_num.append(decoy_sum)

        # if not is_target: # all decoy
        if not is_target and is_trap_decoy:  # decoy trap
            not_trap_decoy_num_sum += 1

        # if is_target: # all_target
        if is_target and is_trap:  # target trap
            trap_num_sum += 1
            # trap_num.append(first_value)
            # not_trap_decoy_num.append((trap_num_sum+0.0001) / (not_trap_decoy_num_sum+0.0001))
            trap_num.append(trap_num_sum)
            not_trap_decoy_num.append(not_trap_decoy_num_sum)
    # print(not_trap_decoy_num_sum)

    cumulative_counts = [sum(counts[: i + 1]) for i in range(len(counts))]  # 计算前缀和

    trap_decoy_ratio = [
        trap_counts[i] / (decoy_counts[i] + 0.0001) for i in range(len(trap_counts))
    ]
    # print(trap_stat, decoy_stat, trap_stat / decoy_stat)
    return (
        counts,
        cumulative_counts,
        (trap_num_forward, target_num_forward),
        trap_stat,
        decoy_stat,
        trap_num,
        not_trap_decoy_num,
        sequence_set,
    )


def calculate_cumulative_count(tuple_list, x_range, trap_stat_thread=0.01):
    counts = [0] * len(x_range)
    # print(tuple_list[0])
    prefix_sum = 0
    trap_stat = 0
    decoy_stat = 0

    target_num = []
    decoy_num = []
    target_sum = 0
    decoy_sum = 0

    for tup in tuple_list:
        if not isinstance(tup, dict):
            # if isinstance(tup, tuple):
            first_value = tup[0]
            is_target = tup[1]
            is_trap = tup[3]
        elif isinstance(tup, dict):
            first_value = tup["pscore_fdr"]
            is_target = tup["pscore_is_target"]
            is_trap = tup["is_trap"]
        else:
            assert 0, "not implemented"
        if first_value <= trap_stat_thread and is_target:
            index = int(
                first_value / x_range[-1] * (len(x_range) - 1)
            )  # 计算当前值在范围中的索引
            counts[index] += 1
            prefix_sum += 1
            decoy_stat += 1

        if first_value <= x_range[-1] and is_trap:
            # index = int(first_value / x_range[-1] * (len(x_range) - 1))  # 计算当前值在范围中的索引
            # trap_counts[index] += 1
            if first_value < trap_stat_thread:
                trap_stat += 1

        # if first_value <= x_range[-1] and not is_target:
        #     # index = int(first_value / x_range[-1] * (len(x_range) - 1))  # 计算当前值在范围中的索引
        #     # decoy_counts[index] += 1
        #     if first_value < trap_stat_thread:
        #         decoy_stat += 1

    peptide_set = set()
    for tup in reversed(tuple_list):
        if not isinstance(tup, dict):
            # if isinstance(tup, tuple):
            first_value = tup[0]
            is_target = tup[1]
            peptide_str = tup[2]
            is_trap = tup[3]
            is_trap_decoy = tup[4]
        elif isinstance(tup, dict):
            first_value = tup["pscore_fdr"]
            is_target = tup["pscore_is_target"]
            peptide_str = tup["peptide_str"]
            is_trap = tup["is_trap"]
            is_trap_decoy = tup["is_trap_decoy"]
        else:
            assert 0, "not implemented"
        if is_target:
            target_sum += 1
        else:
            decoy_sum += 1
            target_num.append(target_sum)
            decoy_num.append(decoy_sum)

    cumulative_counts = [sum(counts[: i + 1]) for i in range(len(counts))]  # 计算前缀和

    return counts, cumulative_counts, trap_stat, decoy_stat, (decoy_num, target_num)


def draw(sorted_tuple_list, dataset_name, prefix=""):
    filtered_data = [
        (qvalue, is_target) for qvalue, _, is_target in sorted_tuple_list if qvalue <= 1
    ]
    # 计算每个qvalue对应的小于等于该qvalue的target元素数量
    qvalues = [tup[0] for tup in filtered_data]
    target_counts = []
    count = 0
    for qvalue, is_target in filtered_data:
        if is_target:
            count += 1
        target_counts.append(count)
    # 绘制图形
    plt.plot(qvalues, target_counts, marker="o")
    plt.xlabel("qvalue")
    plt.ylabel("Number of target elements with qvalue <= x")
    plt.title("Target Elements vs qvalue (0-0.1)")
    plt.grid(True)
    plt.savefig(join(png_path, f"{dataset_name}_pfind.png"))


def draw2(data1, data2, raw_name, fdr_thread):
    qvalues1 = [tup[0] for tup in data1]
    x_range = [i / 1000 for i in range(1001)]
    cnt_t = 0
    cnt_d = 0
    for data in data1:
        if data[1]:
            cnt_t += 1
        else:
            cnt_d += 1
    print(f"pfind # of target and decoy: {cnt_t} {cnt_d}")

    cnt_t = 0
    cnt_d = 0
    for data in data2:
        if data[1]:
            cnt_t += 1
        else:
            cnt_d += 1
    print(f"pscore # of target and decoy: {cnt_t} {cnt_d}")
    # target_counts1 = [sum(1 for q, is_target in data1 if q <= x and is_target) for x in qvalues1]
    _, target_counts1 = calculate_cumulative_count(data1, x_range)
    # 第二个数据集
    qvalues2 = [tup[0] for tup in data2]
    # target_counts2 = [sum(1 for q, is_target in data2 if q <= x and is_target) for x in qvalues2]
    _, target_counts2 = calculate_cumulative_count(data2, x_range)

    # 绘制第一个曲线
    plt.plot(x_range, target_counts1, label="pfind", marker="o")
    # 绘制第二个曲线
    plt.plot(x_range, target_counts2, label="pscore", marker="s")
    plt.xlabel("qvalue")
    plt.ylabel("Number of target elements with qvalue <= x")
    plt.title("Target Elements vs qvalue")
    plt.grid(True)
    plt.legend()
    # 保存当前图并清空画布
    plt.savefig(join(png_path, f"{raw_name}_cmp.png"))
    plt.clf()  # 清空画布


def draw_multiple_psm(raw_name, list_data, png_path_root=None):
    # assert 0, list_data[:10]
    target_decoy_stats = []
    for tuple in list_data:
        try:
            data, name = tuple
        except:
            # print(tuple)
            assert 0
            pass
        x_range = [i / 1000 for i in range(101)]
        _, target_counts, trap_stat, decoy_stat, target_decoy_stat = (
            calculate_cumulative_count(data, x_range)
        )
        target_decoy_stats.append((target_decoy_stat, name))

        plt.plot(x_range, target_counts, label=name)
        # print(f"finished {name}")
        print(
            raw_name,
            name,
            "_psm_level_check:",
            trap_stat,
            decoy_stat,
            trap_stat / (decoy_stat + 0.001),
        )

    plt.xlabel("qvalue")
    plt.ylabel("Number of target psm with qvalue <= x")
    plt.title("Target Elements vs qvalue")
    plt.grid(True)
    plt.legend()
    # 保存当前图并清空画布
    if png_path_root is None:
        plt.savefig(join(png_path, f"{raw_name}_cmp.png"))
    else:
        plt.savefig(join(png_path_root, f"{raw_name}_cmp.png"))
    plt.clf()  # 清空画布

    # for item in target_decoy_stats:
    #     target_decoy_stat, name = item
    #     plt.scatter(target_decoy_stat[0], target_decoy_stat[1], label=name, s=0.1)
    # plt.xlabel('# decoy')
    # plt.ylabel('# target')
    # plt.title('target vs decoy')
    # plt.grid(True)
    # plt.legend()
    # plt.savefig(join(png_path_root, f'{raw_name}_psm_target_decoy.png'))
    # plt.clf()  # 清空画布


def draw_multiple_peptide(raw_name, list_data, png_path_root=None):
    trap_decoy_ratios = []
    target_decoy_stats = []
    for tuple in list_data:
        try:
            data, name = tuple
        except:
            # print(tuple)
            assert 0
            pass
        x_range = [i / 1000 for i in range(101)]
        (
            _,
            target_counts,
            target_decoy_stat,
            trap_stat,
            decoy_stat,
            trap_num,
            not_trap_decoy_num,
            sequence_set,
        ) = calculate_cumulative_count_peptide(data, x_range)
        trap_decoy_ratios.append((trap_num, not_trap_decoy_num, name))
        target_decoy_stats.append((target_decoy_stat, name))
        plt.plot(x_range, target_counts, label=name)
        print(
            raw_name,
            name,
            "_peptide_level_check:",
            trap_stat,
            decoy_stat,
            trap_stat / (decoy_stat + 0.001),
            len(sequence_set),
        )
        # print(f"finished {name}")

    plt.xlabel("qvalue")
    plt.ylabel("Number of target peptide with qvalue <= x")
    plt.title("Target Elements vs qvalue")
    plt.grid(True)
    plt.legend()
    # 保存当前图并清空画布
    if png_path_root is None:
        plt.savefig(join(png_path, f"{raw_name}_cmp.png"))
    else:
        plt.savefig(join(png_path_root, f"{raw_name}_cmp.png"))
    plt.clf()  # 清空画布

    if sum(trap_decoy_ratios[0][0]) > 0:
        for item in trap_decoy_ratios:
            trap_num, not_trap_decoy_num, name = item
            plt.scatter(trap_num, not_trap_decoy_num, label=name, s=0.1)
        plt.xlabel("# trap")
        plt.ylabel("# not trap decoy")
        plt.title("not trap decoy")
        plt.grid(True)
        plt.legend()
        plt.savefig(join(png_path_root, f"{raw_name}_not_trap_decoy_trap.png"))
        plt.clf()  # 清空画布

        for item in target_decoy_stats:
            target_decoy_stat, name = item
            plt.scatter(target_decoy_stat[0], target_decoy_stat[1], label=name, s=0.1)
        plt.xlabel("# trap")
        plt.ylabel("# target")
        plt.title("target vs decoy")
        plt.grid(True)
        plt.legend()
        plt.savefig(join(png_path_root, f"{raw_name}_peptide_trap_target.png"))
        plt.clf()  # 清空画布


# def draw_multiple_peptide_stat(raw_name, list_data, png_path_root=None):
#     trap_decoy_ratios = []
#     for tuple in list_data:
#         try:
#             data, name = tuple
#         except:
#             # print(tuple)
#             assert 0
#             pass
#         x_range = [i / 1000 for i in range(101)]
#         _, target_counts, trap_decoy_ratio, trap_stat, decoy_stat, trap_num, not_trap_decoy_num = calculate_cumulative_count_peptide(data, x_range)
#         trap_decoy_ratios.append((trap_num, not_trap_decoy_num, name))
#         plt.plot(x_range, target_counts, label=name)
#         print(raw_name, name, "_peptide_level_check:", trap_stat, decoy_stat, trap_stat / (decoy_stat + 0.001))
#         # print(f"finished {name}")

#     plt.xlabel('qvalue')
#     plt.ylabel('Number of target peptide with qvalue <= x')
#     plt.title('Target Elements vs qvalue')
#     plt.grid(True)
#     plt.legend()
#     # 保存当前图并清空画布
#     if png_path_root is None:
#         plt.savefig(join(png_path, f'{raw_name}_cmp.png'))
#     else:
#         plt.savefig(join(png_path_root, f'{raw_name}_cmp.png'))
#     plt.clf()  # 清空画布

#     if sum(trap_decoy_ratios[0][0]) > 0:
#         for item in trap_decoy_ratios:
#             trap_num, not_trap_decoy_num, name = item
#             plt.scatter(trap_num, not_trap_decoy_num, label=name)
#         plt.xlabel('# trap')
#         plt.ylabel('# not trap decoy')
#         plt.title('not trap decoy')
#         plt.grid(True)
#         plt.legend()
#         plt.savefig(join(png_path_root, f'{raw_name}_not_trap_decoy_trap.png'))
#         plt.clf()  # 清空画布
