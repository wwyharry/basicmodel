# 基础大语言模型训练框架

这是一个简单的大语言模型训练框架，使用HuggingFace的数据集和模型进行训练，使模型具有正常对话的能力。

## 项目结构

- `model.py`: 模型加载和定义
- `data_processor.py`: 数据处理工具
- `train.py`: 训练脚本
- `inference.py`: 推理和对话脚本
- `requirements.txt`: 项目依赖

## 安装

1. 克隆此仓库
2. 安装依赖:

```bash
pip install -r requirements.txt
```

## 使用方法

### 训练模型

基本训练命令：

```bash
python train.py --model_name_or_path gpt2 --dataset_name wiktext/wikitext-2-raw-v1 --output_dir ./outputs
```

可用参数：

- `--model_name_or_path`: 预训练模型名称或路径，如果没有则从头训练（默认: "gpt2"）
- `--dataset_name`: HuggingFace数据集名称（默认: "wiktext"）
- `--output_dir`: 保存模型输出的目录（默认: "./outputs"）
- `--max_seq_length`: 最大序列长度（默认: 512）
- `--train_batch_size`: 训练批次大小（默认: 4）
- `--eval_batch_size`: 验证批次大小（默认: 4）
- `--learning_rate`: 学习率（默认: 5e-5）
- `--weight_decay`: 权重衰减（默认: 0.01）
- `--max_grad_norm`: 梯度裁剪最大范数（默认: 1.0）
- `--num_train_epochs`: 训练轮数（默认: 3）
- `--warmup_proportion`: 学习率预热比例（默认: 0.1）
- `--save_steps`: 每多少步保存一次模型（默认: 1000）
- `--tensorboard`: 是否使用TensorBoard记录训练状态（标志参数）
- `--no_cuda`: 不使用CUDA，即使可用（标志参数）

### 与模型对话

训练完成后，可以使用inference.py进行交互式对话：

```bash
python inference.py --model_path ./outputs/best_model
```

可用参数：

- `--model_path`: 模型路径（必需）
- `--device`: 运行设备，cuda或cpu（默认: "cuda"）

在对话过程中可以使用以下特殊命令：

- 输入"退出"、"exit"或"quit"：结束对话
- 输入"重置"：重置对话历史
- 输入以"系统提示:"开头的文本：设置新的系统提示词

## 推荐数据集

以下是一些可以用来训练对话模型的推荐数据集：

- `Dahoas/rm-static`: 高质量人类指令和回答对
- `tatsu-lab/alpaca`: 斯坦福Alpaca指令数据集
- `yizhongw/self_instruct`: Self-instruct自生成指令数据集
- `stanfordnlp/SHP`: 斯坦福人类偏好数据集
- `databricks/databricks-dolly-15k`: Databricks Dolly开源指令数据集
- `OpenAssistant/oasst1`: 开放助手对话数据集

## 自定义训练

如需使用自定义数据集，可以修改`data_processor.py`中的数据处理逻辑。

## 模型选择

建议使用以下小型模型开始训练：

- `gpt2`：OpenAI的GPT-2模型（小型版本）
- `EleutherAI/pythia-70m`：EleutherAI的小型Pythia模型
- `facebook/opt-125m`：Meta的小型OPT模型

## 注意事项

- 训练大语言模型需要大量计算资源，请根据您的硬件配置调整参数
- 对于大型模型，建议使用量化技术（如GPTQ、AWQ等）减少内存需求
- 如果训练过程中遇到内存不足，可以减小batch size或模型大小 