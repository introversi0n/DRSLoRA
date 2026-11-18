#!/bin/bash
# ==============================================
# GLUE LoRA微调批量实验脚本
# 支持 base/ERLora/DRS 模式，正交正则消融
# ==============================================

# --------------------------
# 1. 全局环境配置
# --------------------------
export WANDB_MODE=offline
export HF_ENDPOINT=https://hf-mirror.com
export HF_HOME=~/autodl-tmp
export PYTHONPATH="/root/autodl-fs/MSPLoRA/SRLoRA:$PYTHONPATH"
gpu=0

# --------------------------
# 2. 消融实验可配置参数（改这里就行）
# --------------------------
# 正交正则消融列表（仅DRS模式生效）
orth_reg_list=(0.5)

# 批量实验维度
task_base=('cola')
# seeds=(41 42 43)
seeds=(42)
ranks=(8)
modes=('drs')
use_sr_list=(false)
softmaxs=(false)

# --------------------------
# 3. 任务级固定超参
# --------------------------
# 基础训练参数
declare -A epochs=(["mrpc"]=30 ["qnli"]=25 ["rte"]=80 ["sst2"]=60 ["stsb"]=40 ["cola"]=80)
declare -A bs=(["mrpc"]=64 ["qnli"]=64 ["rte"]=64 ["sst2"]=64 ["stsb"]=64 ["cola"]=64)
declare -A ml=(["mrpc"]=256 ["qnli"]=256 ["rte"]=512 ["sst2"]=256 ["stsb"]=256 ["cola"]=256)
declare -A lr=(["mrpc"]="4e-4" ["qnli"]="4e-4" ["rte"]="4e-4" ["sst2"]="5e-4" ["stsb"]="4e-4" ["cola"]="4e-4")
declare -A metrics=(["mrpc"]="accuracy" ["qnli"]="accuracy" ["rte"]="accuracy" ["sst2"]="accuracy" ["stsb"]="pearson" ["cola"]="matthews_correlation")

# LoRA目标模块配置
declare -A target=(["all"]="query value key attention.output.dense intermediate.dense output.dense" ["qv"]="query value")
declare -A target_length=(["all"]=6 ["qv"]=2)

# DRS专属调度参数
declare -A orth_reg_default=(["mrpc"]=0.5 ["qnli"]=0.5 ["rte"]=0.5 ["sst2"]=0.5 ["stsb"]=0.5 ["cola"]=0.5)
declare -A drs_tinit=(["mrpc"]=300 ["qnli"]=7000 ["rte"]=500 ["sst2"]=7500 ["stsb"]=640 ["cola"]=1280)
declare -A drs_tfinal=(["mrpc"]=900 ["qnli"]=20000 ["rte"]=1500 ["sst2"]=27500 ["stsb"]=1600 ["cola"]=5600)

# 固定公共参数
lora_alpha="16"
lora_dropout=0.05
lora_bias=none
lora_task_type=SEQ_CLS
wandb_project=msplora-reproduce
is_train_in_stage=false

# --------------------------
# 4. 核心训练函数
# --------------------------
run() {
    # 入参：$1=任务 $2=秩 $3=l_num $4=目标模块 $5=模式 $6=seed $7=use_sr $8=softnorm $9=orth覆盖值
    local task_name=$1
    local rank=$2
    local l_num=$3
    local target_name=$4
    local mode=$5
    local seed=$6
    local use_sr=$7
    local softnorm=$8
    local orth_override=${9:-}

    # 任务级参数读取
    local learning_rate=${lr[$task_name]}
    local num_train_epochs=${epochs[$task_name]}
    local per_device_train_batch_size=${bs[$task_name]}
    local target_modules=${target[$target_name]}
    local target_len=${target_length[$target_name]}
    local drs_tinit=${drs_tinit[$task_name]}
    local drs_tfinal=${drs_tfinal[$task_name]}
    local lora_budget_rank=$((12 * target_len * rank))

    # 正交正则：传入值优先，否则取任务默认
    if [ -n "$orth_override" ]; then
        local orth_reg_weight=$orth_override
    else
        local orth_reg_weight=${orth_reg_default[$task_name]}
    fi

    # ---------- 路径命名逻辑 ----------
    local dir_mode
    if [ "$mode" = "base" ]; then
        [ "$use_sr" = "true" ] && dir_mode=erlora || dir_mode=lora
    else
        dir_mode=$mode
    fi

    local run_name="roberta-lora-${dir_mode}-${target_name}-${task_name}"
    [ "$dir_mode" = "drs" ] && run_name+="-orth${orth_reg_weight}-tinit${drs_tinit}-tfinal${drs_tfinal}"
    # [ "$dir_mode" = "drs" ] && run_name+="-orth${orth_reg_weight}-tinit${drs_tinit}-tfinal${drs_tfinal}-RS0-L20"
    
    local suffix="stage-${is_train_in_stage}-r-${rank}-n-${l_num}-sr-${use_sr}-budget-${lora_budget_rank}-alpha-16-seed-${seed}-bs-${per_device_train_batch_size}-lr-${learning_rate}-epochs-${num_train_epochs}"
    
    local wandb_run_name
    if [ "$use_sr" = "true" ]; then
        wandb_run_name="${run_name}-softmax-${softnorm}-${suffix}"
    else
        wandb_run_name="${run_name}-${suffix}"
    fi

    # ---------- 目录与日志初始化 ----------
    local exp_dir=./roberta_glue_reproduce_${mode}/${wandb_run_name}
    mkdir -p $exp_dir
    local time_log="${exp_dir}/time.txt"

    echo "===== ${mode} Train machine info is: =====" >> "$time_log"
    nvidia-smi -q > "$time_log"
    echo "orth_reg_weight = ${orth_reg_weight}" >> "$time_log"
    local START_TIME=$(date +%s)
    echo "===== ${mode} Train started at: $(date) =====" >> "$time_log"

    # ---------- 执行训练 ----------
    CUDA_VISIBLE_DEVICES=0 python ./src/run_glue_lora.py \
    --model_name_or_path FacebookAI/roberta-base \
    --task_name ${task_name} \
    --do_train --do_eval \
    --max_seq_length ${ml[$task_name]} \
    --per_device_train_batch_size ${per_device_train_batch_size} \
    --per_device_eval_batch_size ${per_device_train_batch_size} \
    --load_best_model_at_end True --metric_for_best_model ${metrics[$task_name]} \
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
    --calib_batch_size ${per_device_train_batch_size} \
    --orth_reg_weight ${orth_reg_weight} \
    --drs_tinit ${drs_tinit} \
    --drs_tfinal ${drs_tfinal}

    # ---------- 收尾计时 ----------
    local END_TIME=$(date +%s)
    local ELAPSED_TIME=$((END_TIME - START_TIME))
    echo "===== ${mode} Train completed at: $(date) =====" >> "$time_log"
    echo "===== Elapsed time: ${ELAPSED_TIME} seconds =====" >> "$time_log"
}

# --------------------------
# 5. 批量实验循环调度
# --------------------------
for task in "${task_base[@]}"; do
    # 遍历随机种子
    for seed in "${seeds[@]}"; do
        # 遍历LoRA秩
        for rank in "${ranks[@]}"; do
            # 遍历实验模式
            for mode in "${modes[@]}"; do
                # 仅DRS模式遍历正交正则列表，其他模式仅跑1次基线
                if [ "$mode" = "drs" ]; then
                    orth_values=("${orth_reg_list[@]}")
                else
                    orth_values=("")
                fi

                for orth_val in "${orth_values[@]}"; do
                    # 遍历是否开启稳定秩分配
                    for use_sr in "${use_sr_list[@]}"; do
                        if [ "$use_sr" = "true" ]; then
                            for softnorm in "${softmaxs[@]}"; do
                                echo "Running: task=$task, seed=$seed, mode=$mode, orth=${orth_val:-default}, use_sr=$use_sr"
                                run "$task" "$rank" "3" "all" "$mode" "$seed" "$use_sr" "$softnorm" "$orth_val"
                            done
                        else
                            echo "Running: task=$task, seed=$seed, mode=$mode, orth=${orth_val:-default}, use_sr=$use_sr"
                            run "$task" "$rank" "3" "all" "$mode" "$seed" "$use_sr" "false" "$orth_val"
                        fi
                    done
                done
            done
        done
    done
done