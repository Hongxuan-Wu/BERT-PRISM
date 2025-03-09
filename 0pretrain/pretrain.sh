# export DATA_PATH='/hy-tmp/prokaryotes/tokens/'  # e.g., ./sample_data
export DATA_PATH='/hy-tmp/prokaryotes/tokens_new/'
# export DATA_PATH='/hy-tmp/prokaryotes/tokens_new_promoters/'
export MAX_LENGTH=100                       # Please set the number as 0.25 * your sequence length. 
											# e.g., set it as 250 if your DNA sequences have 1000 nucleotide bases
											# This is because the tokenized will reduce the sequence length by about 5 times
                                            # sequence length: 380*0.25=95, 400*0.25=100
# export LR=3e-5
export LR=5e-4
# export LR=1e-4

# python pretrain.py \
#     --model_name_or_path /root/pretrained/DNABERT-2-117M \
#     --data_path  ${DATA_PATH} \
#     --kmer -1 \
#     --run_name DNABERT2_${DATA_PATH} \
#     --model_max_length ${MAX_LENGTH} \
#     --per_device_train_batch_size 320 \
#     --per_device_eval_batch_size 320 \
#     --gradient_accumulation_steps 1 \
#     --learning_rate ${LR} \
#     --num_train_epochs 200 \
#     --save_steps 120000 \
#     --output_dir output/bf16 \
#     --evaluation_strategy steps \
#     --eval_steps 2000 \
#     --warmup_steps 50 \
#     --logging_steps 1000 \
#     --overwrite_output_dir True \
#     --log_level info \
#     --find_unused_parameters False
#     --bf16

# Using RTX 3090 or 4000 series doesn't support faster communication broadband via P2P or IB.
export NCCL_IB_DISABLE=1
export NCCL_P2P_DISABLE=1

# --model_name_or_path /root/pretrained/DNABERT-2-117M \
# --tokenizer_path /root/projects/DNABERT_Promotor/0pretrain/tokenizer/20240603_093134_tokenizer4096_multiprocess/ \
# --model_name_or_path /root/pretrained/DNABERT-2-117M_merge \
# --tokenizer_path /root/projects/DNABERT_Promotor/0pretrain/tokenizer/20240605_020547_tokenizer7737_merge_dnabert2/ \

# Training use DistributedDataParallel (more efficient)
export num_gpu=8 # please change the value based on your setup
torchrun --nproc_per_node=${num_gpu} pretrain.py \
    --model_name_or_path /root/pretrained/DNABERT-2-117M \
    --tokenizer_path /root/projects/DNABERT_Promotor/0pretrain/tokenizer/20240603_093134_tokenizer4096_multiprocess/ \
    --data_path  ${DATA_PATH} \
    --kmer -1 \
    --run_name DNABERT2_${DATA_PATH} \
    --model_max_length ${MAX_LENGTH} \
    --per_device_train_batch_size 320 \
    --per_device_eval_batch_size 320 \
    --gradient_accumulation_steps 1 \
    --learning_rate ${LR} \
    --num_train_epochs 300 \
    --save_steps 120000 \
    --output_dir output/bf16 \
    --evaluation_strategy steps \
    --eval_steps 2000 \
    --warmup_steps 50 \
    --logging_steps 1000 \
    --overwrite_output_dir True \
    --log_level info \
    --find_unused_parameters False \
    --bf16
    