#!/bin/bash
# ==============================================
# MobileBERT + DRSLoRA 冒烟测试 (cola, qv, seed42)
# 关键改动: budget 系数 12->24 (MobileBERT 24层)
# ==============================================
export WANDB_MODE=offline
export HF_ENDPOINT=https://hf-mirror.com
export HF_HOME=~/autodl-tmp
export PYTHONPATH="/root/autodl-fs/MSPLoRA/SRLoRA:$PYTHONPATH"

cd /root/autodl-fs/ERANK\(baseMSPLoRA\)

# ---- 冒烟测试超参 ----
model=google/mobilebert-uncased
task=cola
rank=8
seed=42
mode=drs
target_modules="query value"
target_len=2
epochs=5                 # ~665 steps (133 steps/epoch)
bs=64
ml=256
lr=4e-4
metric=matthews_correlation
lora_alpha=16
lora_dropout=0.05
orth_reg_weight=0.5
drs_tinit=100            # 搜索起始 step
drs_tfinal=500           # 剪枝 step
l_num=3
# budget = 24 layers * target_len * rank  (MobileBERT 24层, 非RoBERTa的12层)
lora_budget_rank=$((24 * target_len * rank))

run_name="mobilebert-lora-${mode}-${task}-qv-budget${lora_budget_rank}-seed${seed}-smoke"
exp_dir=./mobilebert_glue_smoke/${run_name}
mkdir -p $exp_dir
time_log="${exp_dir}/time.txt"

echo "===== MobileBERT DRSLoRA smoke test =====" | tee "$time_log"
echo "budget=${lora_budget_rank} tinit=${drs_tinit} tfinal=${drs_tfinal} epochs=${epochs}" | tee -a "$time_log"
echo "started at: $(date)" | tee -a "$time_log"
nvidia-smi -q > "$time_log" 2>&1

CUDA_VISIBLE_DEVICES=0 python ./src/run_glue_lora.py \
--model_name_or_path ${model} \
--task_name ${task} \
--do_train --do_eval \
--max_seq_length ${ml} \
--per_device_train_batch_size ${bs} \
--per_device_eval_batch_size ${bs} \
--load_best_model_at_end True --metric_for_best_model ${metric} \
--learning_rate ${lr} \
--num_train_epochs ${epochs} \
--evaluation_strategy epoch \
--save_strategy epoch \
--weight_decay 0.1 \
--warmup_ratio 0.06 \
--logging_steps 10 \
--seed ${seed} --wandb_project mobilebert-smoke \
--lora_alpha ${lora_alpha} --lora_dropout ${lora_dropout} --lora_bias none \
--lora_task_type SEQ_CLS --target_modules ${target_modules} --rank ${rank} \
--l_num ${l_num} --mode ${mode} \
--is_train_in_stage false \
--output_dir ${exp_dir}/model \
--logging_dir ${exp_dir}/log \
--run_name ${run_name} \
--overwrite_output_dir \
--use_sr_rank_allocation false \
--softnorm false \
--lora_budget_rank ${lora_budget_rank} \
--calib_batch_size ${bs} \
--orth_reg_weight ${orth_reg_weight} \
--drs_tinit ${drs_tinit} \
--drs_tfinal ${drs_tfinal} 2>&1 | tee ${exp_dir}/train.log

echo "===== completed at: $(date) =====" | tee -a "$time_log"
