# 在文件开头添加导入
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
import time
def linear_func(composite_scores, mode='log', min_rank=1, max_rank=None):
    """
    根据复合分数分配LoRA秩，满足优先级和总秩约束
    
    Args:
        composite_scores: 各层的复合分数（优先级指标），数值越大优先级越高
        budget_rank: 总秩预算，默认为192
        mode: 分配模式，'log'（对数函数）或'linear'（线性函数）
        min_rank: 最小分配的秩，默认为1
        max_rank: 最大分配的秩，如为None则自动计算
    
    Returns:
        allocated_ranks: 各层分配的秩字典，键为层名，值为分配的秩
    """
    budget_rank=192
    n_layers = len(composite_scores) # 24
    
    # 1. 按复合分数排序确定优先级（从高到低）
    sorted_indices = np.argsort(composite_scores)[::-1]  # 降序排列
    priority_order = list(range(1, n_layers + 1))  # 优先级编号，1为最高
    
    # 2. 生成基础分配曲线（对数或线性）
    if mode == 'exp':
        # 对数函数：r_i = a - b * ln(i)
        # 确保i=1时ln(1)=0不会除零
        x = np.arange(1, n_layers + 1)
        # 基础对数曲线（递减）
        base_curve = np.exp(n_layers + 1 - x)  # 使用n_layers+1-i避免ln(0)
    else:  # linear
        # 线性函数：r_i = a - b*(i-1)
        x = np.arange(n_layers)-1
        base_curve = n_layers - x -1 # 正比例函数递减

        
    # 3. 归一化基础曲线，使其和为1（表示分配比例）
    base_curve_normalized = base_curve / np.sum(base_curve)
    
    # 4. 按优先级重新排列曲线（高优先级对应大的分配比例）
    priority_curve = np.zeros(n_layers)
    # cola_alloc = [7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 8, 8,  8,  8,  8,  8,  9,  9,  11, 18][::-1] # cola_softmax_41 
    cola_alloc = [1, 1, 1, 1, 1, 2, 2, 2, 3, 3, 4, 4, 7, 8, 9, 10, 11, 11, 13, 13, 14, 15, 23, 33][::-1] # mrpc_L1_42
    allocated_ranks = {}
    for i, orig_idx in enumerate(sorted_indices):
        if orig_idx%2==0:
            layer_name = f"roberta.encoder.layer.{int(orig_idx/2)}.attention.self.query.weight"  # 根据实际层名修改
        else:
            layer_name = f"roberta.encoder.layer.{int(orig_idx/2)}.attention.self.value.weight"  # 根据实际层名修改        
        allocated_ranks[layer_name] = cola_alloc[i]
    return allocated_ranks
    
    for i, orig_idx in enumerate(sorted_indices):
        priority_curve[orig_idx] = base_curve_normalized[i]
    
    # 5. 计算初步分配的秩（保持优先级顺序）
    initial_ranks = priority_curve * budget_rank
    
    # 6. 整数化处理（四舍五入后调整总和）
    rounded_ranks = np.round(initial_ranks).astype(int)
    
    # 应用秩的上下限约束
    if max_rank is not None:
        rounded_ranks = np.clip(rounded_ranks, min_rank, max_rank)
    else:
        rounded_ranks = np.clip(rounded_ranks, min_rank, None)
    
    # 7. 调整总和至精确等于budget_rank
    total_allocated = np.sum(rounded_ranks)
    diff = budget_rank - total_allocated

    if diff != 0:
        # 按优先级调整：缺秩时优先给高优先级层加，多秩时优先从低优先级层减
        if diff > 0:
            # 需要增加秩，按优先级从高到低增加
            adjustments = min(diff, n_layers)  # 每次调整最多n_layers个层
            for i in range(adjustments):
                layer_idx = sorted_indices[i % n_layers]  # 循环处理高优先级层
                rounded_ranks[layer_idx] += 1
                diff -= 1
                if diff == 0:
                    break
        else:
            # 需要减少秩，按优先级从低到高减少（保证不低于min_rank）
            diff = abs(diff)
            adjustments = min(diff, n_layers)
            for i in range(adjustments):
                layer_idx = sorted_indices[-(i % n_layers) - 1]  # 从最低优先级开始
                if rounded_ranks[layer_idx] > min_rank:
                    rounded_ranks[layer_idx] -= 1
                    diff -= 1
                    if diff == 0:
                        break
    
    # 8. 最终验证
    total_allocated = np.sum(rounded_ranks)
    assert total_allocated == budget_rank, f"秩分配错误：总和{total_allocated} != 预算{budget_rank}"
    
    # 转换为层名到秩的字典（这里假设composite_scores的索引对应层名）
    allocated_ranks = {}
    for i, rank in enumerate(rounded_ranks):
        layer_name = f"layer_{i}"  # 根据实际层名修改
        allocated_ranks[layer_name] = rank
    
    return allocated_ranks

def SVD_fig_print(S, file_path_name ,batch_i, layer_name):
    import os
    # 创建保存目录
    os.makedirs(file_path_name, exist_ok=True)    
    plt.hist(S.cpu().numpy(), bins=50, alpha=0.7, edgecolor='black')
    plt.xlabel('Singular Values')
    plt.ylabel('Frequency')
    plt.title(f'Batch {batch_i} - {layer_name} Singular Values Distribution')
    filename = f"batch_{batch_i}_{layer_name}_sv_dist.png"
    filepath = os.path.join(file_path_name, filename)  # 保存到子文件夹
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close()  # 关闭图形释放内存    


def calibrate_lora_ranks_Entropy(model, train_dataset, data_collator, logger, calib_size, budget_rank, seed, data_args, softnorm, num_batches=None):
    """
    使用稳定秩校准每层的LoRA秩

    Args:
        model: 待校准的模型
        train_dataset: 训练数据集
        data_collator: 数据收集器
        logger: 日志器
        calib_size: 校准批次大小
        budget_rank: 总秩预算
    """
    logger.info("*** Calibrating LoRA ranks using Stable Rank ***")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
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
        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                 for k, v in batch.items()}

        # 前向 + 反向
        model.zero_grad()

        outputs = model(**batch)
        loss = outputs.loss
        loss.backward()

        # 收集当前批次的稳定秩和Frobenius范数
        batch_layer_metrics = {}
        max_fro = 0.0
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
                        U, S, V = torch.svd_lowrank(G, q=1, niter=3)
                        spec_norm = S[0]
                    except:
                        # 备用方法：直接计算矩阵范数
                        spec_norm = torch.norm(G, p=2)

                max_fro = max(fro_norm.item(), max_fro)
                
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
                        # U, S, V = torch.svd_lowrank(G, q=1, niter=10)
                        _, S_all, _ = torch.svd_lowrank(G, q=min(G.shape[0], G.shape[1]), niter=3)
                        spec_norm = S[0]
                    except:
                        # 备用方法：直接计算矩阵范数
                        spec_norm = torch.norm(G, p=2)

                    # 计算稳定秩
                    sr = (fro_norm ** 2) / (spec_norm ** 2 + 1e-12)

                    # # 构建复合分数：稳定秩 × Frobenius范数
                    # composite_score = sr.item() * fro_norm.item()

                    # # 奇异值分布信息收集
                    # SVD_fig_print(S,'svd_fig',i,name)

                    # 1.21 添加信息熵在复合分数里替代稳定秩
                    #这里添加公式
                    # SR_H = 
                    # composite_score = SR_H * fro_norm.item()
                    
                    # # 11.27 修改了构建复合分数：稳定秩 × Frobenius范数（归一化到1以内后的）
                    # composite_score = sr.item() * ( fro_norm.item()/max_fro )
                    
                    # # 不构建复合分数，直接使用稳定秩
                    # composite_score = sr.item()
                      # 1.21 添加信息熵在复合分数里替代稳定秩
                    # composite_score = entropy.item() * ( fro_norm.item()/max_fro )
                    # composite_score = entropy.item() * fro_norm.item()
    
                    # # 2.4 添加仅信息熵，效果比复合分数要好一些
                    # composite_score = entropy.item()
    
                    # 3.2 参考NEAR中的有效秩，信息熵添加exp函数
                
                    S_normalized = torch.abs(S_all) / (torch.abs(S_all)).sum() # L1（从softmax和L1里选的
                    
                    entropy = -torch.sum(S_normalized * torch.log(S_normalized + 1e-20))
                    
                    composite_score = torch.exp(entropy).item()                

                batch_layer_metrics[name] = {
                    'spec_norm': spec_norm.item(),
                    'fro_norm': fro_norm.item(),
                    'stable_rank': sr.item(),
                    'entropy': entropy.item(),
                    'composite_score': composite_score
                }

        all_batch_metrics.append(batch_layer_metrics)
        batch_times.append(time.time() - start_time)  # 记录批次耗时
        logger.info(f"Processed batch {i+1}/{total_steps}")

        # 及时清理显存
        torch.cuda.empty_cache()

    # 计算平均用时
    avg_batch_time = np.mean(batch_times) if batch_times else 0

    # 聚合多批次结果（取中位数（其实是均值）减少异常值影响）
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
    
    if softnorm:
        normalized = np.exp(composite_scores) / np.sum(np.exp(composite_scores)) # softmax norm
    else:
        normalized = composite_scores / np.sum(composite_scores) # L1 norm

    # 分配LoRA秩
    allocated_ranks = {}
    for i, layer_name in enumerate(aggregated_metrics.keys()):
        rank = int(round(normalized[i] * budget_rank))  # 添加round()实现四舍五入
        allocated_ranks[layer_name] = rank

    # # 根据分数排序计算秩分配结果，解除注释则覆盖掉上面原有的计算逻辑
    # allocated_ranks = linear_func(composite_scores)
    
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
    if data_args==None:
        with open(f"./rank_alloc/lora_rank_calibration_results_seed_{seed}_task_alphca_softmax_{softnorm}_Fronorm.json", "w") as f: # 11.27增加了FroNorm
            json.dump({layer_name: {**metrics, "rank": allocated_ranks.get(layer_name, 1)}
                       for layer_name, metrics in aggregated_metrics.items()}, f, indent=2)
    else:
        with open(f"./rank_alloc/lora_rank_calibration_results_seed_{seed}_task_{data_args.task_name}_softmax_{softnorm}_Fronorm.json", "w") as f: # 11.27增加了FroNorm
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