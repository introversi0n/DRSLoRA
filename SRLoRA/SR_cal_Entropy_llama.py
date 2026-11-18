import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
import time
import os
import json

def linear_func(composite_scores, mode='log', min_rank=1, max_rank=None):
    budget_rank=192
    n_layers = len(composite_scores)
    sorted_indices = np.argsort(composite_scores)[::-1]
    
    if mode == 'exp':
        x = np.arange(1, n_layers + 1)
        base_curve = np.exp(n_layers + 1 - x)
    else:
        x = np.arange(n_layers)-1
        base_curve = n_layers - x -1

    base_curve_normalized = base_curve / np.sum(base_curve)
    priority_curve = np.zeros(n_layers)
    cola_alloc = [1, 1, 1, 1, 1, 2, 2, 2, 3, 3, 4, 4, 7, 8, 9, 10, 11, 11, 13, 13, 14, 15, 23, 33][::-1]
    allocated_ranks = {}
    for i, orig_idx in enumerate(sorted_indices):
        if orig_idx%2==0:
            layer_name = f"roberta.encoder.layer.{int(orig_idx/2)}.attention.self.query.weight"
        else:
            layer_name = f"roberta.encoder.layer.{int(orig_idx/2)}.attention.self.value.weight"       
        allocated_ranks[layer_name] = cola_alloc[i]
    return allocated_ranks

    for i, orig_idx in enumerate(sorted_indices):
        priority_curve[orig_idx] = base_curve_normalized[i]
    
    initial_ranks = priority_curve * budget_rank
    rounded_ranks = np.round(initial_ranks).astype(int)
    
    if max_rank is not None:
        rounded_ranks = np.clip(rounded_ranks, min_rank, max_rank)
    else:
        rounded_ranks = np.clip(rounded_ranks, min_rank, None)
    
    total_allocated = np.sum(rounded_ranks)
    diff = budget_rank - total_allocated
    if diff != 0:
        if diff > 0:
            for i in range(min(diff, n_layers)):
                rounded_ranks[sorted_indices[i % n_layers]] += 1
        else:
            diff = abs(diff)
            for i in range(min(diff, n_layers)):
                idx = sorted_indices[-(i % n_layers) - 1]
                if rounded_ranks[idx] > min_rank:
                    rounded_ranks[idx] -= 1
    
    allocated_ranks = {f"layer_{i}": rank for i, rank in enumerate(rounded_ranks)}
    return allocated_ranks

def SVD_fig_print(S, file_path_name ,batch_i, layer_name):
    os.makedirs(file_path_name, exist_ok=True)    
    plt.hist(S.cpu().numpy(), bins=50, alpha=0.7, edgecolor='black')
    plt.xlabel('Singular Values')
    plt.ylabel('Frequency')
    plt.title(f'Batch {batch_i} - {layer_name} Singular Values Distribution')
    plt.savefig(os.path.join(file_path_name, f"batch_{batch_i}_{layer_name}_sv_dist.png"), dpi=300)
    plt.close()    

# ====================== 🔥 终极版：自动保存/恢复梯度冻结（适配int8训练） ======================
def calibrate_lora_ranks_Entropy_llama(
    model, train_dataset, data_collator, tokenizer, logger, 
    calib_size, budget_rank, seed, data_args, softnorm, 
    num_batches=None,
    model_type='llama'
):
    logger.info(f"*** LoRA秩校准 | 模型类型: {model_type.upper()} | 自动恢复梯度冻结 ***")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(dtype=torch.float16, device=device)
    # model = model.to(device)
    
    # ====================== 核心：1. 保存模型原始状态（冻结/训练模式） ======================
    original_training = model.training
    # 保存所有参数的原始requires_grad（prepare_model_for_int8_training的冻结状态）
    original_grad_states = {name: param.requires_grad for name, param in model.named_parameters()}
    
    # 2. 临时开启所有参数梯度（仅用于校准计算）
    model.train()
    for param in model.parameters():
        param.requires_grad_(True)
    model.zero_grad()

    # ====================== LLaMA数据预处理（指令模板+分词） ======================
    if model_type == 'llama':
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "right"
        MAX_SEQ_LEN = 512

        def format_alpaca_prompt(sample):
            instruction = sample["instruction"]
            input_text = sample.get("input", "")
            if input_text.strip():
                return f"Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n### Instruction:\n{instruction}\n\n### Input:\n{input_text}\n\n### Response:"
            return f"Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n### Instruction:\n{instruction}\n\n### Response:"

        def tokenize_function(sample):
            tokenized = tokenizer(
                format_alpaca_prompt(sample),
                truncation=True, padding="max_length", max_length=MAX_SEQ_LEN, return_tensors="pt"
            )
            tokenized = {k: v.squeeze(0) for k, v in tokenized.items()}
            tokenized["labels"] = tokenized["input_ids"].clone()
            return tokenized

        processed_dataset = train_dataset.map(tokenize_function, remove_columns=train_dataset.column_names, desc="Tokenizing")
    else:
        processed_dataset = train_dataset

    # 数据加载器
    if num_batches:
        processed_dataset = processed_dataset.select(range(min(calib_size * num_batches, len(processed_dataset))))
    calib_dataloader = DataLoader(processed_dataset, batch_size=calib_size, collate_fn=data_collator, shuffle=False)

    # ====================== 校准逻辑 ======================
    all_batch_metrics = []
    for i, batch in enumerate(tqdm(calib_dataloader, desc="校准批次")):
        batch = {k: v.to(device) for k, v in batch.items()}
        model.zero_grad()
        outputs = model(**batch)
        loss = outputs.loss
        loss.backward()

        batch_layer_metrics = {}
        max_fro = 0.0
        # import pdb;pdb.set_trace()
        # for name, param in model.named_parameters():
            # if param.grad is not None and 'weight' in name and param.grad.dim() == 2 and ('self_attn.q_proj.weight' in name or 'self_attn.q_proj.weight' in name):
            #     max_fro = max(max_fro, torch.norm(param.grad, p='fro').item())

        for name, param in model.named_parameters():
            if param.grad is not None and 'weight' in name and param.grad.dim() == 2 and ('self_attn.q_proj.weight' in name or 'self_attn.v_proj.weight' in name):
                G = param.grad.data.float()  # 转 FP32
                with torch.no_grad():
                    fro_norm = torch.norm(G, p='fro')
                    _, S_all, _ = torch.svd_lowrank(G, q=min(G.shape)//10, niter=3)
                    spec_norm = S_all[0] if len(S_all) else torch.norm(G, p=2)

                    
                    sr = (fro_norm ** 2) / (spec_norm ** 2 + 1e-12)
                    S_norm = torch.abs(S_all) / torch.sum(torch.abs(S_all))
                    entropy = -torch.sum(S_norm * torch.log(S_norm + 1e-20))
                    composite_score = torch.exp(entropy).item()

                batch_layer_metrics[name] = {
                    'spec_norm': spec_norm.item(), 'fro_norm': fro_norm.item(),
                    'stable_rank': sr.item(), 'entropy': entropy.item(), 'composite_score': composite_score
                }

        all_batch_metrics.append(batch_layer_metrics)
        torch.cuda.empty_cache()

    # ====================== 🔥 核心：3. 100%恢复模型原始冻结状态 ======================
    for name, param in model.named_parameters():
        param.requires_grad_(original_grad_states[name])
    model.train(original_training)
    model.zero_grad()
    logger.info("✅ 校准完成，模型已恢复原始梯度冻结状态！")

    # 秩分配计算
    aggregated = {k: np.mean([m[k]['composite_score'] for m in all_batch_metrics], axis=0) for k in all_batch_metrics[0].keys()}
    scores = np.array([v for v in aggregated.values()])
    norm = np.exp(scores)/np.sum(np.exp(scores)) if softnorm else scores/np.sum(scores)
    allocated = {k: int(round(v * budget_rank)) for k, v in zip(aggregated.keys(), norm)}

    # 构造层秩配置
    import pdb;pdb.set_trace()
    num_layers = len(allocated)//2
    lora_r_per_layer = [{"q_proj":8,"v_proj":8} for _ in range(num_layers)]
    for k, r in allocated.items():
        idx = int(k.split('.')[2])
        if 'q_proj.weight' in k: lora_r_per_layer[idx]["q_proj"] = r
        if 'v_proj.weight' in k: lora_r_per_layer[idx]["v_proj"] = r

    # 保存结果
    os.makedirs("./rank_alloc", exist_ok=True)
    task = data_args.task_name if data_args else "alpaca"
    with open(f"./rank_alloc/lora_rank_{seed}_{task}_{budget_rank}.json", "w") as f:
        json.dump(allocated, f, indent=2)

    return lora_r_per_layer