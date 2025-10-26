# 在文件开头添加导入
import torch
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
import time
def calibrate_lora_ranks(model, train_dataset, data_collator, training_args, logger, calib_size, budget_rank, seed, num_batches=None):
    """
    使用稳定秩校准每层的LoRA秩

    Args:
        model: 待校准的模型
        train_dataset: 训练数据集
        data_collator: 数据收集器
        training_args: 训练参数
        logger: 日志器
        calib_size: 校准批次大小
        budget_rank: 总秩预算
    """
    logger.info("*** Calibrating LoRA ranks using Stable Rank ***")

    # 设置模型为训练模式
    model.train()
    original_training = model.training

    # 创建校准数据加载器
    if num_batches is None:
        # 直接使用完整数据集
        calib_dataset = train_dataset
    else:
        # 使用指定批次数
        calib_dataset = train_dataset.select(range(min(calib_size * num_batches, len(train_dataset))))

    # 创建数据加载器
    calib_dataloader = DataLoader(
        calib_dataset,
        batch_size=calib_size,
        collate_fn=data_collator,
        shuffle=False  # 打乱数据
    )

    # 初始化统计容器和计时器
    all_batch_metrics = []
    batch_times = []

    # 遍历多个批次
    for i, batch in enumerate(tqdm(calib_dataloader, desc="Processing batches", total=None)):
        # if i >= num_batches:
        #     break

        # 循环迭代次数为
        total_steps = len(calib_dataloader)


        start_time = time.time()  # 记录批次开始时间

        # batch = next(iter(calib_dataloader))
        batch.pop('idx', None)  # 安全移除idx字段
        batch = {k: v.to(training_args.device) if isinstance(v, torch.Tensor) else v
                 for k, v in batch.items()}

        # 前向 + 反向
        model.zero_grad()

        outputs = model(**batch)
        loss = outputs.loss
        loss.backward()

        # 收集当前批次的稳定秩和Frobenius范数
        batch_layer_metrics = {}
        for name, param in model.named_parameters():
            # 条件1：必须是权重矩阵（排除bias/LayerNorm）
            # 条件2：是2D矩阵
            # 条件3：是query或value投影层（根据模型结构，只有这些层有LoRA）
            if (param.grad is not None and
                    'weight' in name and
                    param.grad.dim() == 2 and
                    ('attention.self.query.weight' in name or
                     'attention.self.value.weight' in name)):

                G = param.grad.data

                # 使用更高效的计算方式
                with torch.no_grad():
                    # 计算Frobenius范数

                    fro_norm = torch.norm(G, p='fro')
                    # 使用SVD计算最大奇异值（谱范数）
                    try:
                        U, S, V = torch.svd_lowrank(G, q=1, niter=10)
                        spec_norm = S[0]
                    except:
                        # 备用方法：直接计算矩阵范数
                        spec_norm = torch.norm(G, p=2)

                    # 计算稳定秩
                    sr = (fro_norm ** 2) / (spec_norm ** 2 + 1e-12)

                    # 构建复合分数：稳定秩 × Frobenius范数
                    composite_score = sr.item() * fro_norm.item()

                batch_layer_metrics[name] = {
                    'spec_norm': spec_norm.item(),
                    'fro_norm': fro_norm.item(),
                    'stable_rank': sr.item(),
                    'composite_score': composite_score
                }

        all_batch_metrics.append(batch_layer_metrics)
        batch_times.append(time.time() - start_time)  # 记录批次耗时
        logger.info(f"Processed batch {i+1}/{total_steps}")

        # 及时清理显存
        torch.cuda.empty_cache()

    # 计算平均用时
    avg_batch_time = np.mean(batch_times) if batch_times else 0

    # 聚合多批次结果（取中位数减少异常值影响）
    aggregated_metrics = {}
    layer_names = list(all_batch_metrics[0].keys()) if all_batch_metrics else []

    for name in layer_names:
        fro_norms = [bm[name]['fro_norm'] for bm in all_batch_metrics if name in bm]
        spec_norms = [bm[name]['spec_norm'] for bm in all_batch_metrics if name in bm]
        stable_ranks = [bm[name]['stable_rank'] for bm in all_batch_metrics if name in bm]
        composite_scores = [bm[name]['composite_score'] for bm in all_batch_metrics if name in bm]

        aggregated_metrics[name] = {
            # 中位数更鲁棒（jk：要考虑所有数据点，均值对最后微调后的性能最有帮助）
            # 'spec_norm': np.median(spec_norms),
            # 'fro_norm': np.median(fro_norms),
            # 'stable_rank': np.median(stable_ranks),
            # 'composite_score': np.median(composite_scores)
            # 均值
            'spec_norm': np.mean(spec_norms),  # 改为均值
            'fro_norm': np.mean(fro_norms),  # 改为均值
            'stable_rank': np.mean(stable_ranks),  # 改为均值
            'composite_score': np.mean(composite_scores)  # 改为均值
        }

    if not aggregated_metrics:
        logger.warning("No gradient information found for LoRA calibration")
        return

    # 计算每层的归一化分数
    composite_scores = np.array([metrics['composite_score'] for metrics in aggregated_metrics.values()])
    softmax_scores = np.exp(composite_scores) / np.sum(np.exp(composite_scores))

    # 分配LoRA秩
    allocated_ranks = {}
    for i, layer_name in enumerate(aggregated_metrics.keys()):
        rank = int(round(softmax_scores[i] * budget_rank))  # 添加round()实现四舍五入
        allocated_ranks[layer_name] = rank

    # 初始化lora_r_per_layer为一个包含空字典的列表
    num_layers = int(len(allocated_ranks)/2) # 这里所有的q、v层数除2即layer层数
    lora_r_per_layer = [{"query": 8, "value": 8} for _ in range(num_layers)]  # 使用默认值8初始化

    # 根据allocated_ranks更新每层的query和value的LoRA秩
    for layer_name, rank in allocated_ranks.items():
        if 'attention.self.query.weight' in layer_name:
            layer_idx = int(layer_name.split('.')[3])  # 提取层索引
            lora_r_per_layer[layer_idx]["query"] = rank
        elif 'attention.self.value.weight' in layer_name:
            layer_idx = int(layer_name.split('.')[3])  # 提取层索引
            lora_r_per_layer[layer_idx]["value"] = rank

    # 保存校准结果到JSON文件
    import os
    import json

    # 确保目录存在并保存文件
    os.makedirs("./rank_alloc", exist_ok=True)
    with open(f"./rank_alloc/lora_rank_calibration_results_seed{seed}.json", "w") as f:
        json.dump({layer_name: {**metrics, "rank": allocated_ranks.get(layer_name, 1)}
                   for layer_name, metrics in aggregated_metrics.items()}, f, indent=2)

    # 打印校准结果
    logger.info("=== LoRA Rank Calibration Results ===")
    for layer_name, metrics in aggregated_metrics.items():
        rank = allocated_ranks.get(layer_name, 1)
        logger.info(
            f"{layer_name}: "
            f"Fro={metrics['fro_norm']:.3f}, "
            f"spec={metrics['spec_norm']:.3f}, "
            f"SR={metrics['stable_rank']:.3f}, "
            f"Score={metrics['composite_score']:.3f}, "
            f"Rank={rank}"
        )

    # 这里可以添加代码来动态修改模型的LoRA配置
    # 例如：model.update_lora_ranks(allocated_ranks)
    logger.info(f"Total budget rank: {budget_rank}")
    logger.info(f"Actual allocated: {sum(allocated_ranks.values())}")
    logger.info(f"Average time per batch: {avg_batch_time:.2f} seconds")  # 打印平均用时

    model.train(original_training)
    model.zero_grad()

    return lora_r_per_layer