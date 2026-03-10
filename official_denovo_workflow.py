import csv
import gzip
import logging
import os
import pickle
import sys
import types
from collections import Counter
from datetime import timedelta
from multiprocessing import Pool
from os import listdir
from os.path import isdir, join

import numpy as np
import pandas as pd
import torch

# Workaround: torch.optim.Adam lazily imports torch._dynamo, which crashes
# on this environment due to transformers/PyTorch version incompatibility.
# Pre-register a minimal dummy module so the lazy import succeeds silently.
if 'torch._dynamo' not in sys.modules:
    try:
        import torch._dynamo  # noqa
    except (ImportError, AttributeError):
        _dynamo_mod = types.ModuleType('torch._dynamo')
        _dynamo_mod.config = types.SimpleNamespace(suppress_errors=True, disable=True)
        _dynamo_mod.disable = lambda fn=None, recursive=True: fn if fn else (lambda f: f)
        _dynamo_mod.is_compiling = lambda: False
        sys.modules['torch._dynamo'] = _dynamo_mod
import torch.nn.functional as F
from tqdm import tqdm

# Cross-platform compatible imports (unicore removed, using compat module)
from compat import checkpoint_utils, distributed_utils, options, tasks, utils
from compat import progress_bar as progress_bar_module
from compat.parquet_storage import ParquetWriter, get_storage_path, check_parquet_up_to_date

# Import PepMS.tasks to register task classes
import PepMS.tasks

from scripts.process_full_qryresv4_with_decoy import read_denovo_mgfs

# Pickle protocol for pre-serialization in worker processes
_PICKLE_PROTOCOL = pickle.HIGHEST_PROTOCOL


def _read_and_pickle_denovo(mgf_path):
    """Read MGF and pre-pickle each spectrum in worker process.
    Returns list of (title, pickled_data) tuples for deduplication."""
    results = read_denovo_mgfs(mgf_path)
    pickled = []
    for spec_name in results:
        ret = {"small": results[spec_name]}
        pickled.append((spec_name, pickle.dumps(ret, protocol=_PICKLE_PROTOCOL)))
    return pickled

logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=os.environ.get("LOGLEVEL", "INFO").upper(),
    stream=sys.stdout,
)
logger = logging.getLogger("pUniFind.inference_de_novo_sequencing")


def get_distributed_backend():
    """Get the appropriate distributed backend based on platform."""
    if sys.platform == "win32":
        return "gloo"  # Windows does not support NCCL
    else:
        return "nccl" if torch.cuda.is_available() else "gloo"


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
    storage_path = get_storage_path(args.tmp_data_path)
    keys_path = join(os.path.dirname(storage_path), f"{args.project_name}_FDR0.1_keys.pkl")
    mgf_paths = [join(mgf_path, _) for _ in os.listdir(mgf_path)]

    # Check if parquet is already up-to-date
    if check_parquet_up_to_date(storage_path, keys_path, mgf_paths):
        logger.info(f"Parquet file {storage_path} is up-to-date, skipping preprocessing.")
        return

    # Sort by file size descending so large files start processing first,
    # preventing progress stalls when all workers are stuck on big files
    mgf_paths.sort(key=lambda p: os.path.getsize(p), reverse=True)

    # Write to temp file first, rename on success to avoid corrupted files from Ctrl+C
    tmp_storage_path = storage_path + ".tmp"
    tmp_keys_path = keys_path + ".tmp"

    # Clean up any leftover temp files from previous interrupted runs
    for tmp in (tmp_storage_path, tmp_keys_path):
        try:
            os.remove(tmp)
        except OSError:
            pass

    writer = None
    pool = None
    try:
        writer = ParquetWriter(tmp_storage_path)
        keys = []
        pool = Pool(args.num_proc)
        i = 0
        seen_titles = set()
        dup_count = 0
        for pickled_list in tqdm(
            pool.imap_unordered(_read_and_pickle_denovo, mgf_paths, chunksize=1),
            total=len(mgf_paths),
        ):
            for title, pickled_data in pickled_list:
                if title in seen_titles:
                    dup_count += 1
                    continue
                seen_titles.add(title)
                i += 1
                key = f"{i}".encode("ascii")
                writer.put(key, pickled_data)
                keys.append(key)
        if dup_count > 0:
            logger.warning(
                f"Skipped {dup_count} duplicate spectra (same title in multiple mgf files). "
                f"This is usually caused by bloated mgf files where file names like "
                f"'raw_1' incorrectly include spectra from 'raw_10'-'raw_19'."
            )

        pool.close()
        pool.join()
        pool = None
        writer.close()
        writer = None

        # Save keys to temp file
        with open(tmp_keys_path, "wb") as file:
            pickle.dump(keys, file)

        # Atomic rename: only after both files are fully written
        # Remove old files first
        for f in (storage_path, keys_path):
            try:
                os.remove(f)
            except OSError:
                pass
        os.rename(tmp_storage_path, storage_path)
        os.rename(tmp_keys_path, keys_path)

        print("{} process {} ms/ms".format(storage_path, i))

    except (KeyboardInterrupt, Exception):
        logger.warning("Preprocessing interrupted, cleaning up temp files...")
        # Ensure writer is closed to release file handles
        if writer is not None:
            try:
                writer.close()
            except Exception:
                pass
        # Terminate pool workers immediately
        if pool is not None:
            try:
                pool.terminate()
                pool.join()
            except Exception:
                pass
        # Remove incomplete temp files
        for tmp in (tmp_storage_path, tmp_keys_path):
            try:
                os.remove(tmp)
            except OSError:
                pass
        raise


def filter_unreliable(input_file, output_file, chunksize=10000):
    # Check if file exists and is not empty
    if not os.path.exists(input_file) or os.path.getsize(input_file) == 0:
        print(f"Warning: Input file {input_file} is empty or does not exist. Skipping filter.")
        # Create empty output file
        open(output_file, 'w').close()
        return set()

    df = pd.read_csv(input_file, header=None, dtype={2: float, 3: float})

    # Check if dataframe is empty
    if df.empty:
        print(f"Warning: No data in {input_file}. Skipping filter.")
        open(output_file, 'w').close()
        return set()

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
        # Enable cudnn benchmark for faster convolution kernel selection
        torch.backends.cudnn.benchmark = True
    else:
        logger.warning("CUDA not available, using CPU. Performance will be slower.")

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
            skip_batches=args.skip_batches,
        ).next_epoch_itr(shuffle=False)

        if args.skip_batches > 0:
            logger.info(
                "Skipping first %d batch(es) for subset '%s' on rank %d.",
                args.skip_batches,
                subset,
                data_parallel_rank,
            )

        progress = progress_bar_module(
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

        # Open CSV file once and keep it open for the whole loop (avoid per-batch open/close)
        csv_path = join(
            args.results_path,
            f"{subset}_{args.range_pred}_{data_parallel_rank}.csv",
        )
        csv_buffer = []  # Buffer rows, flush periodically
        CSV_FLUSH_INTERVAL = 50  # Flush every N batches

        for i, sample in enumerate(progress):
            sample = utils.move_to_cuda(sample) if use_cuda else sample
            with torch.no_grad():
                if not sample or "net_input" not in sample.keys():
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

            # Buffer CSV rows instead of writing per-batch
            for id in range(len(ret_info["peptide_mod_pred"])):
                csv_buffer.append({
                    "title": ret_info["title"][id],
                    "score": ret_info["score"][id].item(),
                    "cos_sim": ret_info["cos_sim"][id].item(),
                    "RT_true": ret_info["RT"][id].item(),
                    "miss_cleav": ret_info["miss_cleav"][id],
                    "mass_shift": ret_info["mass_shift"][id].item(),
                    "peptides_pred": ret_info["peptides_pred"][id],
                    "peptide_mod_pred": ret_info["peptide_mod_pred"][id],
                    "mod_pred": ret_info["mod_pred"][id],
                })

            # Flush buffer periodically
            if i % CSV_FLUSH_INTERVAL == 0 and csv_buffer:
                with open(csv_path, "a", newline="") as csvfile:
                    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                    writer.writerows(csv_buffer)
                csv_buffer = []

            progress.log({}, step=i)

        # Flush remaining buffer
        if csv_buffer:
            with open(csv_path, "a", newline="") as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writerows(csv_buffer)
            csv_buffer = []
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
                    if os.path.exists(file) and os.path.getsize(file) > 0:
                        with open(file, "r", encoding="utf-8") as f_in:
                            reader = csv.reader(f_in)
                            writer.writerows(reader)
                    else:
                        print(f"Warning: File {file} is empty or does not exist, skipping.")

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
    parser.add_argument(
        "--skip-batches",
        type=int,
        default=0,
        help="Skip the first N batches on each rank before running inference.",
    )

    options.add_model_args(parser)
    args = options.parse_args_and_arch(parser)

    # Determine backend based on platform
    backend = get_distributed_backend()
    logger.info(f"Using distributed backend: {backend}")

    torch.distributed.init_process_group(
        backend=backend, timeout=timedelta(seconds=1800)  # 30分钟超时
    )

    # Get local rank and global rank from environment (set by torchrun)
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    global_rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))

    # Set device_id and distributed settings in args
    args.device_id = local_rank
    args.distributed_rank = global_rank
    args.distributed_world_size = world_size

    # Set CUDA device for this process
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
        logger.info(f"Process rank {global_rank}/{world_size}, using GPU {local_rank}")

    storage_path = get_storage_path(args.tmp_data_path)
    if local_rank == 0:
        logger.info("Start preprocessing data to parquet.")
        preprocess_data(args)
        logger.info("Finished preprocessing data.")

    # Use barrier to synchronize all processes - wait for rank 0 to finish writing
    torch.distributed.barrier()

    logger.info("Start inferencing data from parquet.")
    distributed_utils.call_main(args, main)
    logger.info("Finished inferencing data.")


if __name__ == "__main__":
    cli_main()
