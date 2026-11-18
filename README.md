# ERANKLoRA

ERANKLoRA by Yuhai Yang.

## 项目定位

本项目是 **ERANKLoRA** 的主项目。除 ERANKLoRA 外，仓库中同样包含 **MSPLoRA**、**LoRA**、**DRSLoRA**、**AdaLoRA** 等方法的实现代码，便于对比实验：

- **ERANK 的核心库位于 [`SRLoRA/`](SRLoRA)** 目录（并随 `peft-0.5.0/` 一同封装）；
- **DRSLoRA 与 AdaLoRA 的实现主要在 [`src/Trainers.py`](src/Trainers.py)**；
- LoRA / MSPLoRA 等基线可通过 `src/glue_finetune.sh` 中的模式切换运行。

## Method Overview

## Quickstart

1. Install dependencies

   ```bash
   conda create -n ERANKLoRA python=3.10
   conda activate ERANKLoRA
   pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 -i https://pypi.tuna.tsinghua.edu.cn/simple/
   pip install /root/autodl-fs/pip_pkg/py310/*.whl
   pip install bitsandbytes==0.41.1
   pip install numpy==1.24.3 --force-reinstall
   pip install -r requirements.txt
   ```

   ```bash
   cd peft-0.5.0
   pip install -e .
   ```
2. Run experiments

fill in the `--model_name_or_path` `--wandb_project` and `--output_dir` in `llama_finetune.sh` and `glue_finetune.sh` with the path to the model and the output directory.

### Instruction Tuning

```bash
bash llama_finetune.sh
```

### NLU

```bash
bash glue_finetune.sh
```

### TEST On INSTRUCTION FOLLOWING

修改 `src/merge_model/merge_model.py` 中模型的 `lora_path` 路径，即可生成合并 LoRA 权重后的完整模型：

```bash
cd /root/autodl-fs/ERANK\(baseMSPLoRA\)/
python src/merge_model/merge_model.py
```

修改 `/root/autodl-fs/instruct-eval/run_ERANK.sh` 中的 `MODEL_PATH`：

```bash
cd /root/autodl-fs/instruct-eval/
bash run_ERANK.sh
```

## Thanks

Code is largely based on [AGI-Edgerunners/LLM-Adapters](https://github.com/AGI-Edgerunners/LLM-Adapters), [huggingface/peft](https://github.com/huggingface/peft), [huggingface/transformers](https://github.com/huggingface/transformers), [
Oblivioniss/MSPLoRA](https://github.com/Oblivioniss/MSPLoRA)

# SRLoRA

所有的核心方法均封装在 peft 的 `SRLoRA` 中，并统一继承自 peft。
使用方法参见 `src/run_glue_lora.py` 脚本。

* `SRLoRA/` 目录为核心算法实现；
* 核心方法作为 `peft_config` 的成员参数，在 peft 中切换时自动识别并更新 LoRA 在各模块中的配置。
