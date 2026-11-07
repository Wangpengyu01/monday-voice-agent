---
name: lora-train
description: 在 A100 训练服务器上给本地大模型训练 LoRA（用每晚整理的训练集），查看训练进度，取回结果
risk: ask
tags: lora, 训练, 微调, 模型, gpu, a100, 显卡, 训练集
---
状态：流水线还在搭建（基座模型 Qwen3.5-4B 转 RK1828 格式的导出脚本还缺），目前能做的是准备数据、检查服务器、
查看训练进度；不要自己发明训练命令。

服务器：别名 gpu（gpu@192.168.100.4，NVIDIA DRIVE PG199 = A100 32G）。工作目录 ~/monday-lora：
- venv/：训练环境（PyTorch 2.6 + CUDA 12.4，transformers、peft）
- rkvenv/：RK1828 转换工具 rknn3-toolkit 1.0.4
- models/Qwen3.5-4B/：基座权重
- data/：训练集（从板子同步）

常用步骤：
1. 看服务器和显卡：ssh gpu 'nvidia-smi --query-gpu=name,memory.used,utilization.gpu --format=csv'
2. 训练集大小：wc -l /userdata/agent/run/learning/lora_curated.jsonl（少于几百条训练意义不大，告诉主人还要攒多久）
3. 同步训练集：ssh gpu 'mkdir -p ~/monday-lora/data'，再用 scp 把 lora_curated.jsonl 传到 gpu:~/monday-lora/data/
4. 训练进度（等流水线搭好后）：ssh gpu 'tail -5 ~/monday-lora/train.log'

注意：训练会长时间占用显卡；开始训练、替换板子上的模型都必须主人确认，替换前先存备份节点。
