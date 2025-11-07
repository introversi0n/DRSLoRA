#!/bin/bash
export WANDB_MODE=offline
export HF_ENDPOINT=https://hf-mirror.com
export HF_HOME=~/autodl-tmp # autoDL
export PYTHONPATH="/root/autodl-fs/MSPLoRA/SRLoRA:$PYTHONPATH" # SRLoRA
# export HF_HOME=./data # yours
gpu=0

declare -A epochs=(["mrpc"]=30 ["qnli"]=25 ["rte"]=80 ["sst2"]=60 ["stsb"]=40 ["cola"]=80)
declare -A bs=(["mrpc"]=64 ["qnli"]=64 ["rte"]=64 ["sst2"]=64 ["stsb"]=64 ["cola"]=64)
declare -A ml=(["mrpc"]=256 ["qnli"]=256 ["rte"]=512 ["sst2"]=256 ["stsb"]=256 ["cola"]=256)
declare -A lr=(["mrpc"]="4e-4" ["qnli"]="4e-4" ["rte"]="4e-4" ["sst2"]="5e-4" ["stsb"]="4e-4" ["cola"]="4e-4")
declare -A metrics=(["mrpc"]="accuracy" ["qnli"]="accuracy" ["rte"]="accuracy" ["sst2"]="accuracy" ["stsb"]="pearson" ["cola"]="matthews_correlation")
declare -A target=(["all"]="query value key attention.output.dense intermediate.dense output.dense" ["qv"]="query value")
declare -A target_length=(["all"]=6 ["qv"]=2)  # 添加target长度映射

export WANDB_MODE=offline

run(){
  task_name=$1
  learning_rate=${lr[$1]}
  num_train_epochs=${epochs[$1]}
  per_device_train_batch_size=${bs[$1]}
  gradient_accumulation_steps=1
  rank=$2
  l_num=$3
  seed=$6
  use_sr=$7
  softnorm=$8
  lora_alpha="16"
  target_modules=${target[$4]}
  target_name=$4
  target_len=${target_length[$4]}
  mode=$5
  lora_dropout=0.05
  lora_bias=none
  lora_task_type=SEQ_CLS
  wandb_project=msplora-reproduce
  share=false
  is_train_in_stage=false
  lora_budget_rank=$((12 * target_len * rank))
  
  wandb_run_name=roberta-lora-${mode}-${target_name}-${task_name}-stage-${is_train_in_stage}-r-${rank}-n-${l_num}-sr-${use_sr}-softmax-${softnorm}-budget-${lora_budget_rank}-alpha-16-seed-${seed}-bs-${per_device_train_batch_size}-lr-${learning_rate}-epochs-${num_train_epochs}
  
  exp_dir=./roberta_glue_reproduce_${mode}/${wandb_run_name}
  mkdir -p $exp_dir
  
  START_TIME=$(date +%s)
  echo "===== ${mode} Train started at: $(date) =====" > "${exp_dir}/time.txt"
  
  CUDA_VISIBLE_DEVICES=0 python ./src/run_glue_lora.py \
  --model_name_or_path FacebookAI/roberta-base  \
  --task_name ${task_name} \
  --do_train --do_eval \
  --max_seq_length ${ml[$1]} \
  --per_device_train_batch_size ${per_device_train_batch_size} \
  --per_device_eval_batch_size ${per_device_train_batch_size} \
  --load_best_model_at_end True --metric_for_best_model ${metrics[$1]} \
  --learning_rate ${learning_rate} \
  --num_train_epochs ${num_train_epochs} \
  --evaluation_strategy epoch \
  --save_strategy epoch \
  --weight_decay 0.1 \
  --warmup_ratio 0.06 \
  --logging_steps 10 \
  --seed ${seed} --wandb_project ${wandb_project} \
  --lora_alpha ${lora_alpha} --lora_dropout ${lora_dropout} --lora_bias ${lora_bias} \
  --lora_task_type ${lora_task_type} --target_modules ${target_modules} --rank ${rank} \
  --l_num ${l_num} --mode ${mode} \
  --is_train_in_stage ${is_train_in_stage} \
  --output_dir ${exp_dir}/model \
  --logging_dir ${exp_dir}/log \
  --run_name ${wandb_run_name} \
  --overwrite_output_dir \
  --use_sr_rank_allocation ${use_sr} \
  --softnorm ${softnorm} \
  --lora_budget_rank ${lora_budget_rank} \
  --calib_batch_size ${per_device_train_batch_size}

  END_TIME=$(date +%s)
  ELAPSED_TIME=$((END_TIME - START_TIME))
  echo "===== ${mode} Train completed at: $(date) =====" >> "${exp_dir}/time.txt"
  echo "===== Elapsed time: ${ELAPSED_TIME} seconds =====" >> "${exp_dir}/time.txt"
}
# task_base=('mrpc' 'qnli' 'rte' 'sst2' 'stsb' 'cola')
task_base=('stsb' 'mrpc' 'cola')
seeds=(40 41 42)
ranks=(8)
modes=('base')
use_sr_list=(true)  # 新增：use_sr_rank_allocation 参数列表
softmaxs=(false)

for task in "${task_base[@]}"; do
  for seed in "${seeds[@]}"; do
    for rank in "${ranks[@]}"; do
      for mode in "${modes[@]}"; do
        for use_sr in "${use_sr_list[@]}"; do  # 新增循环
            for softnorm in "${softmaxs[@]}"; do  # 新增循环
              echo "Running with task: $task, seed: $seed, rank: $rank, mode: $mode, use_sr: $use_sr, softmax: $softnorm"
              run "$task" "$rank" "3" "qv" "$mode" "$seed" "$use_sr" "$softnorm"
            done
        done
      done
    done
  done
done