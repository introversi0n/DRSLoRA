from transformers import (
    TrainerCallback,
)
class LoRAFreezeCallback(TrainerCallback):
    def __init__(self, model, adapter_name, num_epoches, l_num, scale=None):
        self.model = model
        self.schedule = []
        self.current_adapter = None
        self.adapter_name = adapter_name
        if scale is not None:
            self.scale = scale
        else:
            self.scale = [1.0 / l_num for i in range(0, l_num)]
        for i in range(0, l_num):
            if not i:
                start = 0
            end = start + self.scale[i] * num_epoches
            self.schedule.append((start, end, adapter_name + f"_{i}"))
            start = end
        print("------------Train Schedule Init Successfully!!!------------")
        for msp_start, msp_end, msp_adapter_name in self.schedule:
            print(f"------------Start:{msp_start}! End:{msp_end}! Adaptername:{msp_adapter_name}!------------")

    def on_epoch_begin(self, args, state, control, **kwargs):
        epoch = state.epoch if state.epoch is not None else 0
        adapter_to_unfreeze = None
        for start, end, adapter_name in self.schedule:
            if start <= epoch < end:
                adapter_to_unfreeze = adapter_name
                break

        if adapter_to_unfreeze is None:
            return

        if adapter_to_unfreeze == self.current_adapter:
            return

        self.current_adapter = adapter_to_unfreeze
        print(f"Epoch {epoch}: Unfreezing LoRA parameters with adapter '{adapter_to_unfreeze}' and freezing others.")

        for name, param in self.model.named_parameters():
             if "lora_" in name:
                if self.current_adapter not in name:
                    param.requires_grad = False
                else:
                    param.requires_grad = True
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                # print(f"Parameter {name} is not frozen.")
                pass

# use AdaLoRA rankalloctor to AdaTrainer
# 1. on_train_begin use set_total_step
# 2. on_step_end use update_and_allocate (before model.zero_grad())
#TODO Done: 2 add some AdaLoRA param like init_warmup, end_warmup, beta1, beta2, delaT, orth, init_r, target_r
class AdaLoRACallback(TrainerCallback):
    def __init__(self, model):
        self.model = model
    def on_train_begin(self, args, state, control, **kwargs):
        self.model.rankallocator.set_total_step(state.max_steps)
    def on_step_end(self, args, state, control, **kwargs):
        self.model.update_and_allocate(state.global_step)

# 1. based on AdaLoRACallback
# 2. rankalloctor function different of Ada,
#TODO Done: 3 DRSLoRA need to create a DRSLoRACallback to Trainer or NewTrainer
class DRSLoRACallback(AdaLoRACallback):
    """DRSLoRA的子类，回调功能与AdaLoRACallback完全相同，但函数实现不同，具体区别见peft/src/tuner/drslora"""
    pass