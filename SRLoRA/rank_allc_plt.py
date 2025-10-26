import matplotlib.pyplot as plt
import numpy as np

# 以下是中位数情况
# # int取整情况
# ranks_1_batch = [7, 7, 7, 8, 7,   8, 7, 8, 7, 7,   7, 7, 7, 8, 7,   8, 7, 8, 7, 9,   7, 10, 7, 10]
# ranks_8_batch = [7, 7, 7, 7, 7,   8, 7, 8, 7, 7,   7, 8, 7, 8, 7,   8, 7, 8, 7, 9,   7, 11, 7, 11]
# ranks_100_batch = [7, 7, 7, 7, 7,   7, 7, 8, 7, 7,   7, 8, 7, 8, 7,   8, 7, 8, 7, 9,   7, 10, 7, 11]

# # 四舍五入情况
# ranks_1_batch = [7, 8, 8, 8, 8,   8, 8, 8, 8, 8,   8, 8, 8, 8, 8,   8, 8, 8, 8, 9,   8, 9, 8, 10]
# ranks_8_batch = [7, 8, 7, 8, 7,   8, 7, 8, 7, 8,   7, 8, 7, 8, 7,   8, 7, 9, 7, 9,   7, 11, 7, 12]
# ranks_100_batch = [7, 8, 7, 8, 7,   8, 7, 8, 7, 8,   7, 8, 7, 8, 7,   8, 7, 9, 7, 9,   7, 10, 8, 12]
# ranks_230_batch = [7, 7, 7, 8, 7,   8, 7, 8, 7, 8,   7, 8, 7, 8, 7,   8, 7, 9, 7, 9,   7, 11, 8, 12]

# 以下是均值情况
# 四舍五入情况
ranks_1_batch = [7, 8, 8, 8, 8,   8, 8, 8, 8, 8,   8, 8, 8, 8, 8,   8, 8, 8, 8, 9,   8, 9, 8, 10]
ranks_8_batch = [7, 10, 7, 14, 7,   14, 7, 12, 6, 8,   6, 8, 6, 7, 6,   7, 6, 8, 6, 8,   6, 9, 6, 10]
ranks_100_batch = [7, 8, 7, 8, 7,   9, 7, 8, 7, 8,   7, 8, 7, 8, 7,   8, 7, 9, 7, 9,   7, 10, 7, 12]
ranks_230_batch = [7, 8, 7, 8, 7,   8, 7, 8, 7, 8,   7, 8, 7, 8, 7,   8, 7, 9, 7, 9,   7, 11, 7, 13]

# 计算各折线的Rank总和
total_1_batch = sum(ranks_1_batch)
total_8_batch = sum(ranks_8_batch)
total_100_batch = sum(ranks_100_batch)

# 生成横轴数字（1到24）
layer_indices = np.arange(1, len(ranks_1_batch) + 1)

# 绘制折线（仅修改颜色和样式）
plt.figure(figsize=(12, 6))
plt.plot(
    layer_indices, ranks_1_batch,
    marker='o', linestyle='-',
    color='#8da0cb',  # 浅蓝色
    label=f'1 Batch (Total Rank={total_1_batch})'
)
plt.plot(
    layer_indices, ranks_8_batch,
    marker='s', linestyle='--',
    color='#377eb8',  # 中等蓝色
    label=f'8 Batches (Total Rank={total_8_batch})'
)
plt.plot(
    layer_indices, ranks_100_batch,
    marker='^', linestyle='-.',
    color='#084594',  # 深蓝色
    linewidth=2,      # 加粗线条
    label=f'100 Batches (Total Rank={total_100_batch})'
)
plt.title("Rank Allocation (int Truncation)", fontsize=14, pad=20)
plt.xlabel("Layer Index (1-24)")
plt.ylabel("Rank")
plt.xticks(layer_indices)
# 设置纵轴刻度以1为精度单位
y_min = min(min(ranks_1_batch), min(ranks_8_batch), min(ranks_100_batch))
y_max = max(max(ranks_1_batch), max(ranks_8_batch), max(ranks_100_batch))
plt.yticks(np.arange(y_min, y_max + 1, 1))
plt.grid(True)
plt.legend()

# 保存图片（新增）
plt.savefig('lora_rank_allocation_int.png',
           dpi=300,
           bbox_inches='tight',
           facecolor='white')  # 设置白色背景

plt.show()