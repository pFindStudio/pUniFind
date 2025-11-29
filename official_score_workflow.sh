echo "==================================pUniFind rescore==========================================="

projects_base=./projects
project_name=$1
batch_size=$2
[ -z "${project_name}" ] && project_name=debug
# project_name=pfind3Cere
# name of folder name in projects_base
# should contain:
# 1. result folder: pFind result file
# 2. mgfs folder: mgf files produced by pFind (or pXtract and pParse)
mgf_path=$projects_base/$project_name/mgfs
qry_res_path=$projects_base/$project_name/result
results_path=$projects_base/$project_name/pUniFind_result/
tokenize1_pkl_path=./consts/tokenize1.pkl
modification_meta_dict_path=./consts/modification_meta_dict.pkl

export OMPI_COMM_WORLD_SIZE=$MLP_WORKER_NUM
export OMPI_COMM_WORLD_RANK=$MLP_ROLE_INDEX

[ -z "${num_proc}" ] && num_proc=32
mkdir -p $results_path
[ -z "${fdr_thread}" ] && fdr_thread=0.1
[ -z "${data_path}" ] && data_path=$projects_base/$project_name/pUniFind_data_tmp/
mkdir -p $data_path
mkdir -p $data_path/pkls
[ -z "${pkl_path}" ] && pkl_path=$data_path
[ -z "${weight_path}" ] && weight_path=./ckpts/checkpoint_rank.pt
[ -z "${valid_subset}" ] && valid_subset=$project_name

[ -z "${MASTER_PORT}" ] && MASTER_PORT=10078
[ -z "${MASTER_IP}" ] && MASTER_IP=127.0.0.1
[ -z "${n_gpu}" ] && n_gpu=$(nvidia-smi -L | wc -l)
[ -z "${OMPI_COMM_WORLD_SIZE}" ] && OMPI_COMM_WORLD_SIZE=1
[ -z "${OMPI_COMM_WORLD_RANK}" ] && OMPI_COMM_WORLD_RANK=0

[ -z "${lr}" ] && lr=1e-3
[ -z "${end_lr}" ] && end_lr=1e-9
[ -z "${warmup_steps}" ] && warmup_steps=50000
[ -z "${total_steps}" ] && total_steps=500000

[ -z "${node_dim}" ] && node_dim=512
[ -z "${num_layers}" ] && num_layers=9
[ -z "${num_pept_layers}" ] && num_pept_layers=9
[ -z "${hidden_size}" ] && hidden_size=512
[ -z "${ffn_size}" ] && ffn_size=512
[ -z "${num_head}" ] && num_head=8
[ -z "${cutoff}" ] && cutoff=200
[ -z "${batch_size}" ] && batch_size=256
[ -z "${batch_size_valid}" ] && batch_size_valid=$batch_size

[ -z "${update_freq}" ] && update_freq=1

[ -z "${seed}" ] && seed=1
[ -z "${data_seed}" ] && data_seed=$seed
[ -z "${clip_norm}" ] && clip_norm=5

[ -z "${pair_dropout}" ] && pair_dropout=0.25
[ -z "${ema_decay}" ] && ema_decay=0
[ -z "${validate_interval_updates}" ] && validate_interval_updates=2000

[ -z "${use_nce}" ] && use_nce=1
[ -z "${use_ins}" ] && use_ins=1
[ -z "${pred_nce}" ] && pred_nce=0
[ -z "${pred_ins}" ] && pred_ins=0
[ -z "${pred_rt}" ] && pred_rt=0
[ -z "${denovo_pred}" ] && denovo_pred=0
[ -z "${spectrum_pred}" ] && spectrum_pred=0
[ -z "${use_clip}" ] && use_clip=1
[ -z "${use_hard_clip}" ] && use_hard_clip=0

[ -z "${triplet_loss_weight}" ] && triplet_loss_weight=0.2
[ -z "${denovo_loss_weight}" ] && denovo_loss_weight=0.4
[ -z "${spectrum_loss_weight}" ] && spectrum_loss_weight=0.2
[ -z "${peak_loss_weight}" ] && peak_loss_weight=0.1
[ -z "${modification_loss_weight}" ] && modification_loss_weight=0.2
[ -z "${clip_loss_weight}" ] && clip_loss_weight=1
[ -z "${num_res_loss_weight}" ] && num_res_loss_weight=0.2

[ -z "${intensity_pred}" ] && intensity_pred=0
[ -z "${denoise_pred}" ] && denoise_pred=0
[ -z "${noise_peak_pred}" ] && noise_peak_pred=0
[ -z "${res_num_pred}" ] && res_num_pred=0
[ -z "${use_rope}" ] && use_rope=0
[ -z "${dataset_name}" ] && dataset_name="_"
[ -z "${max_charges}" ] && max_charges=4
[ -z "${multi_class_noise_peak_pred}" ] && multi_class_noise_peak_pred=1
[ -z "${res_seq_len_pred}" ] && res_seq_len_pred=0
[ -z "${half_spectrum_pred}" ] && half_spectrum_pred=0
[ -z "${res_type_pred}" ] && res_type_pred=0
[ -z "${norm_first}" ] && norm_first=0
[ -z "${modification_pred}" ] && modification_pred=0
[ -z "${modification_cls_pred}" ] && modification_cls_pred=0
[ -z "${cutoff_spectra}" ] && cutoff_spectra=300
[ -z "${train_pfind}" ] && train_pfind=1
[ -z "${comp_res_num_pred}" ] && comp_res_num_pred=0
[ -z "${shift}" ] && shift=1
[ -z "${hard_neg}" ] && hard_neg=1
[ -z "${head_clip}" ] && head_clip=1
[ -z "${joint_encoding_layers}" ] && joint_encoding_layers=4
[ -z "${is_TIMS}" ] && is_TIMS=0

[ -z "${tof}" ] && tof=0

# 创建唯一临时目录名（避免冲突）
UNIQUE_NAME="PepMS_$(date +%s)"  # 例如 PepMS_1620000000

# 创建符号链接（指向原始 PepMS 目录）
ln -sf "./PepMS" "./${UNIQUE_NAME}"

torchrun --nproc_per_node=$n_gpu --master_port $MASTER_PORT --nnodes=$OMPI_COMM_WORLD_SIZE --node_rank=$OMPI_COMM_WORLD_RANK --master_addr=$MASTER_IP \
       ./official_score_workflow.py --user-dir ${UNIQUE_NAME} $data_path --valid-subset $valid_subset \
       --results-path $results_path \
       --tmp-data-path $data_path/$project_name.lmdb \
       --project-name $project_name \
       --weight-path $weight_path \
       --qry-res-path $qry_res_path \
       --mgf-path $mgf_path \
       --num-proc $num_proc \
       --data-buffer-size 32 \
       --num-workers 0 --ddp-backend=c10d --batch-size $batch_size \
       --task score_inference --loss cross_entropy --arch pep_ms \
       --node-dim $node_dim \
       --num-layers $num_layers \
       --num-head $num_head \
       --use-nce $use_nce \
       --use-ins $use_ins \
       --use-clip $use_clip \
       --use-hard-clip $use_hard_clip \
       --denovo-pred $denovo_pred \
       --spectrum-pred $spectrum_pred \
       --denoise-pred $denoise_pred \
       --intensity-pred $intensity_pred \
       --dataset-name $dataset_name \
       --max-charges $max_charges \
       --use-rope $use_rope \
       --noise-peak-pred $noise_peak_pred \
       --half-spectrum-pred $half_spectrum_pred \
       --from-scratch 1 \
       --norm-first $norm_first \
       --modification-pred $modification_pred \
       --modification-cls-pred $modification_cls_pred \
       --multi-class-noise-peak-pred $multi_class_noise_peak_pred \
       --res-num-pred $res_num_pred \
       --res-type-pred $res_type_pred \
       --res-seq-len-pred $res_seq_len_pred \
       --cutoff-spectra $cutoff_spectra \
       --comp-res-num-pred $comp_res_num_pred \
       --train-pfind $train_pfind \
       --hard-neg $hard_neg \
       --shift $shift \
       --head-clip $head_clip \
       --fdr-thread $fdr_thread \
       --pkl-path $pkl_path \
       --joint-encoding-layers $joint_encoding_layers \
       --num-pept-layers $num_pept_layers \
       --inference 1 \
       --check 0 \
       --joint-encoding 1 \
       --tof $tof \
       --tokenize1-pkl-path $tokenize1_pkl_path \
       --modification-meta-dict-path $modification_meta_dict_path \

rm -f "./${UNIQUE_NAME}"
