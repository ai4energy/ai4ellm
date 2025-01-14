from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import torch

# 1) 基座模型路径（本地或HF仓库）
base_model_path = "/home/yancong/Qwen2.5-7B"

# 2) LoRA 适配器路径
lora_model_path = "/home/yancong/LLaMA-Factory/saves/Qwen2.5-7B/lora/train_2025-01-07-19-13-41/"

# 3) 加载原始模型
model = AutoModelForCausalLM.from_pretrained(
    base_model_path,
    torch_dtype=torch.float16,      # 视自己情况选择精度
    device_map="auto"              # 多卡推理/训练时常用
)

# 4) 基于PeftModel加载LoRA适配器
model = PeftModel.from_pretrained(
    model,
    lora_model_path
)

# 5) 将LoRA增量权重合并到基座模型
#    此操作会将LoRA矩阵覆盖到原始权重中，并卸载LoRA结构
model = model.merge_and_unload()

# 6) （可选）将合并后的完整模型保存到一个新路径
save_path = "/home/yancong/sft1/"
model.save_pretrained(save_path)

# 7) 同样需要保存tokenizer（通常和基座模型相同）
tokenizer = AutoTokenizer.from_pretrained(base_model_path, use_fast=False)
tokenizer.save_pretrained(save_path)

print("合并完成！现在 /path/to/merged_model 中的模型已不再需要LoRA权重，即可独立使用。")
