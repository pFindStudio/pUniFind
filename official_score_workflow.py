import gzip
import logging
import os
import pickle
import sys
from datetime import timedelta
from multiprocessing import Pool
from os import listdir
from os.path import isdir, join

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

# Cross-platform compatible imports (unicore removed, using compat module)
from compat import checkpoint_utils, distributed_utils, options, tasks, utils
from compat import progress_bar
from compat.parquet_storage import ParquetWriter, get_storage_path

from PepMS.eval.eval_draw import draw_multiple_peptide, draw_multiple_psm
from PepMS.eval.percolator import Percolator, PercolatorConfig
from PepMS.eval.trans_to_pfind import write_spectra_file
from scripts.process_full_qryresv4_with_decoy import custom_sort, read_one_results

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

    qryress = custom_sort(
        [
            (join(qry_res_path, _), mgf_path)
            for _ in listdir(qry_res_path)
            if _.endswith("qry.res")
        ]
    )

    # Remove existing storage file
    try:
        os.remove(storage_path)
    except:
        pass

    # Use Parquet-based storage (cross-platform compatible)
    writer = ParquetWriter(storage_path, batch_size=100)

    keys = []
    with Pool(args.num_proc) as pool:
        i = 0
        for results in tqdm(
            pool.imap_unordered(read_one_results, qryress, chunksize=1),
            total=len(qryress),
        ):
            small_specs = set(results.keys())

            difference1 = small_specs

            for spec_name in difference1:
                ret = {}
                ret["small"] = results[spec_name]
                i += 1
                key = f"{i}".encode("ascii")
                writer.put(key, gzip.compress(pickle.dumps(ret)))
                keys.append(key)

    writer.close()

    with open(
        join(os.path.dirname(storage_path), f"{args.project_name}_FDR0.1_keys.pkl"), "wb"
    ) as file:
        pickle.dump(keys, file)

    print("{} process {} ms/ms".format(storage_path, i))


def main(args):

    assert (
        args.batch_size is not None
    ), "Must specify batch size either with --batch-size"

    use_cuda = torch.cuda.is_available() and not args.cpu

    if use_cuda:
        torch.cuda.set_device(args.device_id)
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
    for i, sample in enumerate(progress):
        sample = utils.move_to_cuda(sample) if use_cuda else sample
        if len(sample) == 0:
            continue
        if "net_input" not in sample.keys():
            continue
        with torch.no_grad():
            return_dict = model.forward_score(**sample["net_input"])

        joint_scores = return_dict["scores"]
        best_rank = return_dict["top_indices"]

        mz_array = sample["net_input"]["mz_array"]
        batch_index = sample["net_input"]["batch_index"].long()
        index = sample["net_input"]["index"].long().cpu()
        titles = sample["net_input"]["title"]
        fdr = sample["net_input"]["fdr"]
        assert len(titles) == len(fdr) == len(mz_array)
        for s in range(len(titles)):
            joint_scores_ = joint_scores[batch_index == s].cpu().numpy()
            ret[titles[s]] = gzip.compress(
                pickle.dumps(
                    {
                        "index": index[s].item(),
                        "best_rank": (
                            best_rank[s].cpu().item() if best_rank is not None else -1
                        ),
                        "joint_scores": joint_scores_,
                    }
                )
            )
            count_res += 1
        progress.log({}, step=i)
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
    torch.distributed.barrier()

    logger.info("Start inferencing data from parquet.")
    distributed_utils.call_main(args, main)
    logger.info("Finished inferencing data.")

    logger.info("Start evaluating data from parquet.")
    evaluate_database_search(args)
    logger.info("Finished rescoring!")


if __name__ == "__main__":
    cli_main()
