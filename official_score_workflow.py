import gzip
import logging
import os
import pickle
import sys
import types
from datetime import timedelta
from multiprocessing import Pool
from os import listdir
from os.path import isdir, join

import numpy as np
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
from compat import progress_bar
from compat.parquet_storage import ParquetWriter, get_storage_path, check_parquet_up_to_date

from PepMS.eval.eval_draw import draw_multiple_peptide, draw_multiple_psm
from PepMS.eval.percolator import Percolator, PercolatorConfig
from PepMS.eval.trans_to_pfind import write_spectra_file
from scripts.process_full_qryresv4_with_decoy import custom_sort, read_one_results

# Pickle protocol for pre-serialization in worker processes
_PICKLE_PROTOCOL = pickle.HIGHEST_PROTOCOL


def _read_and_pickle_results(inputs):
    """Read qry.res and pre-pickle each spectrum in worker process.
    Returns list of (title, pickled_data) tuples for deduplication."""
    results = read_one_results(inputs)
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
logger = logging.getLogger("pUniFind.inference_database_search")


def get_distributed_backend():
    """Get the appropriate distributed backend based on platform."""
    if sys.platform == "win32":
        return "gloo"  # Windows does not support NCCL
    else:
        return "nccl" if torch.cuda.is_available() else "gloo"


def preprocess_data(args):
    mgf_path = args.mgf_path
    qry_res_path = args.qry_res_path
    storage_path = get_storage_path(args.tmp_data_path)
    keys_path = join(os.path.dirname(storage_path), f"{args.project_name}_FDR0.1_keys.pkl")

    qryress = custom_sort(
        [
            (join(qry_res_path, _), mgf_path)
            for _ in listdir(qry_res_path)
            if _.endswith("qry.res")
        ]
    )

    # Collect all source files for freshness check (qry.res files + mgf files)
    source_files = [q[0] for q in qryress]
    mgf_files = [join(mgf_path, f) for f in os.listdir(mgf_path) if f.endswith(".mgf")]
    source_files.extend(mgf_files)

    # Check if parquet is already up-to-date
    if check_parquet_up_to_date(storage_path, keys_path, source_files):
        logger.info(f"Parquet file {storage_path} is up-to-date, skipping preprocessing.")
        return

    # Sort by qry.res file size descending so large files start processing first,
    # preventing progress stalls when all workers are stuck on big files
    qryress.sort(key=lambda x: os.path.getsize(x[0]), reverse=True)

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
            pool.imap_unordered(_read_and_pickle_results, qryress, chunksize=1),
            total=len(qryress),
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
                f"Skipped {dup_count} duplicate spectra (same title in multiple qry.res files). "
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
        if writer is not None:
            try:
                writer.close()
            except Exception:
                pass
        if pool is not None:
            try:
                pool.terminate()
                pool.join()
            except Exception:
                pass
        for tmp in (tmp_storage_path, tmp_keys_path):
            try:
                os.remove(tmp)
            except OSError:
                pass
        raise


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
    # print(f"fdr: {args.fdr_thread}")
    subset = args.valid_subset
    all_result = {}
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
    progress = progress_bar(
        itr,
        log_format=args.log_format,
        log_interval=args.log_interval,
        prefix=f"Inferencing on '{subset}' subset",
        default_log_format=("tqdm" if not args.no_progress_bar else "simple"),
    )
    # assert data_parallel_world_size == 1
    ret = {}
    count_res = 0
    # Pending results from previous batch - process while GPU computes current batch
    pending_cpu_work = None

    for i, sample in enumerate(progress):
        sample = utils.move_to_cuda(sample) if use_cuda else sample
        if len(sample) == 0:
            continue
        if "net_input" not in sample.keys():
            continue
        with torch.no_grad():
            return_dict = model.forward_score(**sample["net_input"])

        # Process PREVIOUS batch's results while GPU may still be finishing
        # (overlaps CPU post-processing with GPU kernel launch/execution)
        if pending_cpu_work is not None:
            p_scores, p_best_rank, p_batch_idx, p_index, p_titles = pending_cpu_work
            for s in range(len(p_titles)):
                mask = p_batch_idx == s
                sample_idx = int(p_index[s])
                ret[sample_idx] = {
                    "index": sample_idx,
                    "title": p_titles[s],
                    "best_rank": (
                        int(p_best_rank[s]) if p_best_rank is not None else -1
                    ),
                    "joint_scores": p_scores[mask],
                }
                count_res += 1

        # GPU→CPU transfer (sync point, but previous batch already processed)
        joint_scores_cpu = return_dict["scores"].cpu().numpy()
        best_rank_val = return_dict["top_indices"]
        best_rank_cpu = best_rank_val.cpu().numpy() if best_rank_val is not None else None
        batch_index_cpu = sample["net_input"]["batch_index"].long().cpu().numpy()
        index_cpu = sample["net_input"]["index"].long().cpu().numpy()
        titles = sample["net_input"]["title"]

        # Defer processing to next iteration (overlap with next forward pass)
        pending_cpu_work = (joint_scores_cpu, best_rank_cpu, batch_index_cpu, index_cpu, titles)
        progress.log({}, step=i)

    # Process last batch
    if pending_cpu_work is not None:
        p_scores, p_best_rank, p_batch_idx, p_index, p_titles = pending_cpu_work
        for s in range(len(p_titles)):
            mask = p_batch_idx == s
            sample_idx = int(p_index[s])
            ret[sample_idx] = {
                "index": sample_idx,
                "title": p_titles[s],
                "best_rank": (
                    int(p_best_rank[s]) if p_best_rank is not None else -1
                ),
                "joint_scores": p_scores[mask],
            }
            count_res += 1

    all_result.update(ret)

    print("Finished {} subset, rank {}".format(subset, data_parallel_rank))

    if data_parallel_world_size > 1:
        tmp = distributed_utils.all_gather_list(
            [torch.tensor(0)],
            max_size=10000,
            group=distributed_utils.get_data_parallel_group(),
        )

    pickle.dump(
        all_result,
        open(
            os.path.join(
                args.results_path,
                subset + "_{}_{}.pkl".format(subset, data_parallel_rank),
            ),
            "wb",
        ),
    )

    return None


def evaluate_database_search(args, all_result=None):
    print(
        os.path.join(
            args.results_path,
            args.valid_subset + "_{}_{}.pkl".format(args.valid_subset, args.fdr_thread),
        )
    )
    # if all_result is None:
    #     all_result = pickle.load(open(os.path.join(args.results_path, args.valid_subset + "_{}_{}.pkl".format(args.valid_subset, args.fdr_thread)),"rb",))

    cfg = PercolatorConfig(
        run_name="rankloss",
        dataset_name=args.valid_subset,
        fdr_threashold=0.1,
        prefix="",
        processes=16,
        use_rank=False,
        use_joint_scores=True,
        use_pred_spec=False,
        reset_pfind=False,
        reset_pscore=True,
        mgf_path_root=args.mgf_path,
        res_path_base=args.results_path,
        key_pkl_path=os.path.dirname(args.tmp_data_path),
        # lmdb_path_base = r"/mnt/vepfs/fs_ckps/zhaojiale/dataset/mol_spec/dataset/lmdbs_full",
        lmdb_path_base=os.path.dirname(args.tmp_data_path),
        middle_path=os.path.dirname(args.tmp_data_path),
        res_path=args.results_path,
    )

    percolator = Percolator(config=cfg)
    (
        pfind_results,
        pscore_results,
        pscore_results_detail,
        pscore_per_results,
        all_pfind_results,
        all_pscore_results_detail,
    ) = percolator.run()
    print(f"# of raw {len(pscore_results.keys())}")
    base_path = os.path.dirname(args.qry_res_path)
    pac_name = [_ for _ in os.listdir(base_path) if _.endswith(".pac")]
    if len(pac_name) == 1:
        write_spectra_file(
            result_detail=all_pscore_results_detail,
            pac_path=join(base_path, pac_name[0]),
            output_path=join(
                args.results_path, args.valid_subset + "fdr0.01" + "_pUniFind.spectra"
            ),
        )
    else:
        print(
            f"pac files:{pac_name} There should be one and only one .pac file in pFind task path.\n (pac file is generated by open-pFind at the same path as fasta file recording proteins)"
        )


def check_inference_pkl_complete(results_path, subset, keys_path):
    """
    Check if inference pkl files already exist and contain complete results.

    Validates:
    1. pkl files exist and are loadable
    2. Each pkl file is non-empty
    3. Total key count matches the expected count from keys pkl

    Returns:
        True if pkl files exist and results are complete, False otherwise.
    """
    if not os.path.isdir(results_path):
        return False

    # Find all pkl files matching the inference output pattern: {subset}_{subset}_{rank}.pkl
    pkl_files = [
        f for f in os.listdir(results_path)
        if f.startswith(f"{subset}_{subset}_") and f.endswith(".pkl")
    ]
    if not pkl_files:
        return False

    # Load expected key count from keys pkl
    expected_keys = 0
    if os.path.isfile(keys_path):
        try:
            with open(keys_path, "rb") as fp:
                expected_keys = len(pickle.load(fp))
        except Exception as e:
            logger.info(f"Failed to load keys pkl {keys_path}: {e}, will re-run inference.")
            return False
    else:
        logger.info(f"Keys pkl not found: {keys_path}, will re-run inference.")
        return False

    # Check that each pkl file is loadable and non-empty, count total keys
    total_keys = 0
    for f in pkl_files:
        fpath = os.path.join(results_path, f)
        try:
            with open(fpath, "rb") as fp:
                data = pickle.load(fp)
            if not isinstance(data, dict) or len(data) == 0:
                logger.info(f"Inference pkl {f} is empty or invalid, will re-run inference.")
                return False
            total_keys += len(data)
        except Exception as e:
            logger.info(f"Failed to load inference pkl {f}: {e}, will re-run inference.")
            return False

    # Verify completeness: total keys must match expected count
    if total_keys != expected_keys:
        logger.info(
            f"Inference pkl incomplete: {total_keys} keys found, "
            f"expected {expected_keys}. Will re-run inference."
        )
        return False

    logger.info(
        f"Found {len(pkl_files)} complete inference pkl files with {total_keys} total results. "
        f"Skipping inference."
    )
    return True


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
        "--skip-inference",
        action="store_true",
        default=False,
        help="Skip inference and run evaluation only. "
             "Automatically enabled if inference pkl files already exist.",
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
    keys_path = join(os.path.dirname(storage_path), f"{args.project_name}_FDR0.1_keys.pkl")

    # Check if inference results already exist (checkpoint)
    skip_inference = args.skip_inference
    if not skip_inference and global_rank == 0:
        skip_inference = check_inference_pkl_complete(
            args.results_path, args.valid_subset, keys_path
        )

    if skip_inference:
        if global_rank == 0:
            logger.info("Inference pkl checkpoint found, skipping inference.")
            # Still need preprocessing (parquet) for evaluation
            preprocess_data(args)
            logger.info("Start evaluating data from parquet.")
            evaluate_database_search(args)
            logger.info("Finished rescoring!")
        # Other ranks just wait
        torch.distributed.barrier()
    else:
        if global_rank == 0:
            logger.info("Start preprocessing data to parquet.")
            preprocess_data(args)
            logger.info("Finished preprocessing data.")
        torch.distributed.barrier()

        logger.info("Start inferencing data from parquet.")
        distributed_utils.call_main(args, main)
        logger.info("Finished inferencing data.")

        # Ensure all ranks have finished writing pkl files before evaluation
        torch.distributed.barrier()

        # Evaluation is CPU-only post-processing, only run on rank 0
        if global_rank == 0:
            logger.info("Start evaluating data from parquet.")
            evaluate_database_search(args)
            logger.info("Finished rescoring!")

        # Wait for rank 0 to finish evaluation before all processes exit
        torch.distributed.barrier()


if __name__ == "__main__":
    cli_main()
