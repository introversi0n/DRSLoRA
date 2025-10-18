from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import os

# 加载基础模型（需提前下载或指定本地路径）
base_model_path = "/root/autodl-tmp/hub/models--meta-llama--Llama-2-7b-hf/snapshots/01c7f73d771dfac7d292323805ebc428287df4f9"
# 你的 LoRA 适配器目录
lora_path = "/root/autodl-fs/MSPLoRA/llama-lora-drs/llama-msplora-qv-drs-stage-false-r-96-n-1-alpha-16-seed-42-bs-128-lr-3e-4-len-256-epochs-3/model"  
# 指定合并后的模型保存路径
output_path = os.path.join(lora_path, "merge_model")  # 在lora_path下创建merge_model子目录

# 确保输出目录存在
os.makedirs(output_path, exist_ok=True)

# 加载基础模型和 tokenizer
base_model = AutoModelForCausalLM.from_pretrained(base_model_path)
tokenizer = AutoTokenizer.from_pretrained(base_model_path)

# 注入 LoRA 权重并合并
model = PeftModel.from_pretrained(base_model, lora_path)
merged_model = model.merge_and_unload()  # 合并权重并移除 LoRA 结构

# 保存完整模型到指定路径
merged_model.save_pretrained(output_path)
tokenizer.save_pretrained(output_path)

print(f"合并后的模型已保存到：{output_path}")

# 检查保存的文件
print("保存的文件列表：")
print(os.listdir(output_path))