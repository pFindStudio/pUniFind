import csv
import gzip
import logging
import os
import pickle
import sys
from collections import Counter
from datetime import timedelta
from multiprocessing import Pool
from os import listdir
from os.path import isdir, join

import lmdb
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from tqdm import tqdm
from unicore import checkpoint_utils, distributed_utils, options, tasks, utils
from unicore.logging import progress_bar

from scripts.process_full_qryresv4_with_decoy import read_denovo_mgfs

logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=os.environ.get("LOGLEVEL", "INFO").upper(),
    stream=sys.stdout,
)
logger = logging.getLogger("pUniFind.inference_de_novo_sequencing")


def write_fasta(protein_sequences, output_file):
    """
    将一个蛋白质序列的集合写入一个Fasta文件。

    :param protein_sequences: 包含蛋白质序列的集合，每个序列视为一个字符串
    :param output_file: 输出的Fasta文件路径
    """
    with open(output_file, "w") as file:
        # for index, sequence in enumerate(protein_sequences):
        header = f">sp|PROT|Protein \n"
        file.write(header)
        str_all = ""
        for item in protein_sequences:
            str_all += item + "RRRR"
        file.write(str_all + "\n")


def preprocess_data(args):
    mgf_path = args.mgf_path
    lmdb_path = args.tmp_data_path
    mgf_paths = [join(mgf_path, _) for _ in os.listdir(mgf_path)]

    try:
        os.remove(lmdb_path)
    except:
        pass
    env_new_train = lmdb.open(
        lmdb_path,
        subdir=False,
        readonly=False,
        lock=False,
        readahead=False,
        meminit=False,
        max_readers=1,
        map_size=int(1000e9),
    )

    txn_write_train = env_new_train.begin(write=True)
    keys = []
    with Pool(args.num_proc) as pool:
        i = 0
        for results in tqdm(
            pool.imap_unordered(read_denovo_mgfs, mgf_paths, chunksize=1),
            total=len(mgf_paths),
        ):
            small_specs = set(results.keys())

            difference1 = small_specs

            for spec_name in difference1:
                ret = {}
                ret["small"] = results[spec_name]
                i += 1
                txn_write_train.put(
                    f"{i}".encode("ascii"), gzip.compress(pickle.dumps(ret))
                )
                keys.append(f"{i}".encode("ascii"))
                if i % 100 == 0:
                    txn_write_train.commit()
                    txn_write_train = env_new_train.begin(write=True)

    txn_write_train.commit()
    env_new_train.close()

    with open(
        join(os.path.dirname(lmdb_path), f"{args.project_name}_FDR0.1_keys.pkl"), "wb"
    ) as file:
        pickle.dump(keys, file)

    print("{} process {} ms/ms".format(lmdb_path, i))


def filter_unreliable(input_file, output_file, chunksize=10000):
    df = pd.read_csv(input_file, header=None, dtype={2: float, 3: float})

    # filtered_df = df[(df.iloc[:, 2] > -4) & (df.iloc[:, 3] > 0.8)]
    filtered_df = df[df.iloc[:, 1] > -30]
    # filtered_df = df[(df.iloc[:, 2] > -4) & (df.iloc[:, 3] > 0.8)]

    filtered_df["group_key"] = filtered_df.iloc[:, 0].apply(
        lambda x: ".".join(x.split(".")[:-2]) if len(x.split(".")) >= 2 else x
    )

    # filtered_df = filtered_df[
    #     filtered_df.iloc[:, 9].str.len().between(7+2, 14+2, inclusive='both')
    # ]

    filtered_df = filtered_df.loc[filtered_df.groupby("group_key")[2].idxmax()]

    filtered_df = filtered_df.drop(columns=["group_key"])

    m3, s3 = filtered_df[1].mean(), filtered_df[1].std()
    m4, s4 = filtered_df[2].mean(), filtered_df[2].std()

    threshold_3 = m3 - 2 * s3
    threshold_4 = m4 - s4
    threshold_4 = max(threshold_4, 0.8)
    final_condition = (filtered_df[1] > threshold_3) & (filtered_df[2] > threshold_4)
    print(f"score thread: {threshold_3}, cos sim thread: {threshold_4}")
    filtered_df = filtered_df[final_condition]

    filtered_df = filtered_df[
        filtered_df[4].str.count("_").fillna(0) <= 1  # 处理空值并统计数量
    ]

    filtered_df.to_csv(output_file, index=False, header=False)
    filtered_df.columns = [
        "title",
        "score",
        "cos_similarity",
        "mz",
        "missing_site",
        "delta_mass",
        "peptide_list",
        "peptide_sequence",
        "modification_info",
    ]

    peptide_set = set(filtered_df["peptide_list"].str[2:-2])

    return peptide_set


def main(args):
    assert (
        args.batch_size is not None
    ), "Must specify batch size either with --batch-size"

    use_cuda = torch.cuda.is_available() and not args.cpu

    if use_cuda:
        torch.cuda.set_device(args.device_id)

    if args.distributed_world_size > 1:
        data_parallel_world_size = distributed_utils.get_data_parallel_world_size()
        data_parallel_rank = distributed_utils.get_data_parallel_rank()
    else:
        data_parallel_world_size = 1
        data_parallel_rank = 0

    # Load model
    logger.info("loading model(s) from {}".format(args.weight_path))

    state = checkpoint_utils.load_checkpoint_to_cpu(args.weight_path)
    task = tasks.setup_task(args)
    model = task.build_model(args)
    # print(state.keys())
    missing_keys, unexpected_keys = model.load_state_dict(state["model"], strict=False)

    # print(f"missing keys: {missing_keys}")
    # print(f"unexpected keys: {unexpected_keys}")

    # Move models to GPU
    # if use_fp16:
    #     model.half()
    if use_cuda:
        model.cuda()

    model.eval()

    # Print args
    # logger.info(args)

    # Build loss
    loss = task.build_loss(args)
    loss.eval()
    if data_parallel_world_size > 1:
        tmp = distributed_utils.all_gather_list(
            [torch.tensor(0)],
            max_size=10000,
            group=distributed_utils.get_data_parallel_group(),
        )

    fieldnames = [
        "title",
        "score",
        "cos_sim",
        "RT_true",
        "miss_cleav",
        "mass_shift",
        "peptides_pred",
        "peptide_mod_pred",
        "mod_pred",
    ]
    # fieldnames = ["title", "peptide_same", "score", "cos_sim", "RT_true", "miss_cleav", "len_diff", "mass_shift", "peptides_true", "peptides_pred", "peptide_mod_true", "peptide_mod_pred", "mod_pred"]

    try:
        os.mkdir(join(args.results_path, args.run_name))
    except:
        pass

    for subset in args.valid_subset.split(","):
        csv_list = [
            join(args.result_path, f"{subset}_FDR{args.fdr_thread}_{idx}.csv")
            for idx in range(data_parallel_world_size)
        ]

        try:
            task.load_dataset(subset, combine=False, epoch=1, force_valid=True)
            dataset = task.dataset(subset)
        except KeyError:
            raise Exception("Cannot find dataset: " + subset)

        # Initialize data iterator
        itr = task.get_batch_iterator(
            dataset=dataset,
            batch_size=args.batch_size,
            ignore_invalid_inputs=True,
            required_batch_size_multiple=args.required_batch_size_multiple,
            seed=args.seed,
            num_shards=data_parallel_world_size,
            shard_id=data_parallel_rank,
            num_workers=args.num_workers,
            data_buffer_size=args.data_buffer_size,
        ).next_epoch_itr(shuffle=False)

        progress = progress_bar.progress_bar(
            itr,
            log_format=args.log_format,
            log_interval=args.log_interval,
            prefix=f"Inferencing on '{subset}' subset",
            default_log_format=("tqdm" if not args.no_progress_bar else "simple"),
        )
        all_aa_recall = 0
        all_pep_recall = 0
        all_bsz = 0
        # set_all_peptides = set()
        mod_stat_all = {}

        try:
            os.remove(
                join(
                    args.results_path,
                    args.run_name,
                    f"{subset}_{args.range_pred}_{data_parallel_rank}.csv",
                )
            )
        except:
            pass

        for i, sample in enumerate(progress):
            if i % 100 == 0:
                print(f"processing {i}")
            sample = utils.move_to_cuda(sample) if use_cuda else sample
            # if len(sample) == 0:
            #     continue
            # print(len(sample))
            with torch.no_grad():
                if "net_input" not in sample.keys():
                    print(sample)
                    continue
                ret_info, peptide_set, mod_stat = model.forward_faster_denovo(
                    **sample["net_input"],
                    min_length=args.min_length,
                    max_length=args.max_length,
                )
            if ret_info is None:
                continue
            for k in mod_stat.keys():
                if k in mod_stat_all.keys():
                    mod_stat_all[k] += mod_stat[k]
                else:
                    mod_stat_all[k] = mod_stat[k]

            bsz = len(ret_info["peptide_mod_pred"])
            all_aa_recall += ret_info["aa_recall"] * bsz
            all_pep_recall += ret_info["pep_recall"] * bsz
            all_bsz += bsz

            with open(
                join(
                    args.results_path,
                    f"{subset}_{args.range_pred}_{data_parallel_rank}.csv",
                ),
                "a",
                newline="",
            ) as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                for id in range(len(ret_info["peptide_mod_pred"])):
                    writer.writerow(
                        {
                            "title": ret_info["title"][id],
                            # "peptide_same": ret_info["peptides_true"][id] == ret_info["peptides_pred"][id],
                            "score": ret_info["score"][id].item(),
                            "cos_sim": ret_info["cos_sim"][id].item(),
                            "RT_true": ret_info["RT"][id].item(),
                            "miss_cleav": ret_info["miss_cleav"][id],
                            # "len_diff": len(ret_info["peptides_true"][id][0]) - len(ret_info["peptides_pred"][id][0]),
                            "mass_shift": ret_info["mass_shift"][id].item(),
                            # "peptides_true": ret_info["peptides_true"][id],
                            "peptides_pred": ret_info["peptides_pred"][id],
                            # "peptide_mod_true": ret_info["peptide_mod_true"][id],
                            "peptide_mod_pred": ret_info["peptide_mod_pred"][id],
                            "mod_pred": ret_info["mod_pred"][id],
                        }
                    )

            # print(">>>>>>>", ret_info["aa_recall"], ret_info["pep_recall"], bsz, all_aa_recall / all_bsz, all_pep_recall / all_bsz, all_bsz)
            progress.log({}, step=i)
        # print("Finished {} subset, rank {}".format(subset, data_parallel_rank))
        # print(">>>>", all_aa_recall / all_bsz, all_pep_recall / all_bsz, all_bsz)
        mod_stat_all = Counter(mod_stat_all)
        with open(
            join(args.results_path, subset + f"_rank{data_parallel_rank}_mod.txt"), "w"
        ) as f:
            for key, value in mod_stat_all.most_common():
                f.write(key + " " + str(value) + "\n")

        if data_parallel_world_size > 1:
            tmp = distributed_utils.all_gather_list(
                [torch.tensor(0)],
                max_size=10000,
                group=distributed_utils.get_data_parallel_group(),
            )
        if data_parallel_rank == 0:
            input_files = [
                join(args.results_path, f"{subset}_{args.range_pred}_{i}.csv")
                for i in range(data_parallel_world_size)
            ]  # 替换为实际文件路径
            output_file = join(
                args.results_path, f"{subset}_{args.range_pred}_merged.csv"
            )
            filtered_file = join(
                args.results_path, f"{subset}_{args.range_pred}_filtered.csv"
            )

            # 确保输出文件为空
            if os.path.exists(output_file):
                os.remove(output_file)

            # 合并文件
            with open(output_file, "w", newline="", encoding="utf-8") as f_out:
                writer = csv.writer(f_out)
                for file in input_files:
                    with open(file, "r", encoding="utf-8") as f_in:
                        reader = csv.reader(f_in)
                        writer.writerows(reader)

            print(f"Merged done! Total {len(input_files)} files → {output_file}")
            peptide_set = filter_unreliable(output_file, filtered_file)
            print(f"Filter done! Total {len(output_file)} files → {filtered_file}")
            write_fasta(list(peptide_set), join(args.results_path, subset + ".fasta"))

    return None


def cli_main():
    parser = options.get_validation_parser()
    parser.add_argument(
        "--qry-res-path",
        type=str,
    )
    parser.add_argument(
        "--mgf-path",
        type=str,
    )
    parser.add_argument("--result-path", type=str, default="")
    parser.add_argument("--weight-path", type=str, default="")
    parser.add_argument("--prefix", type=str, default="")
    parser.add_argument("--num-proc", type=int, default=16)
    parser.add_argument(
        "--tmp-data-path",
        type=str,
    )
    parser.add_argument(
        "--project-name",
        type=str,
    )
    parser.add_argument(
        "--max-length",
        type=int,
    )
    parser.add_argument(
        "--min-length",
        type=int,
    )

    options.add_model_args(parser)
    args = options.parse_args_and_arch(parser)

    torch.distributed.init_process_group(
        backend="nccl", timeout=timedelta(seconds=1800)  # 30分钟超时
    )

    local_rank = os.environ["LOCAL_RANK"]  # 节点内的本地进程ID
    if local_rank == "0":
        logger.info("Start preprocessing data to lmdb.")
        preprocess_data(args)
        logger.info("Finished preprocessing data.")
    else:
        while not os.path.exists(
            join(
                os.path.dirname(args.tmp_data_path),
                f"{args.project_name}_FDR0.1_keys.pkl",
            )
        ):
            pass

    logger.info("Start inferencing data from lmdb.")
    distributed_utils.call_main(args, main)
    logger.info("Finished inferencing data.")


if __name__ == "__main__":
    cli_main()
