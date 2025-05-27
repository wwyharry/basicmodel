from datasets import load_dataset
from torch.utils.data import Dataset
import torch
import logging

# 设置日志
logger = logging.getLogger(__name__)

class ConversationDataset(Dataset):
    """
    对话数据集类，用于处理和准备训练数据
    """
    def __init__(self, dataset, tokenizer, max_length=512):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.inputs = []
        self.masks = []
        self.labels = []
        
        # 处理数据集
        self._prepare_dataset(dataset)
    
    def _prepare_dataset(self, dataset):
        """处理数据集，将文本转换为模型输入格式"""
        for item in dataset:
            # 确保有text字段
            if "text" in item:
                text = item["text"]
            elif "content" in item:
                text = item["content"]
            elif "conversation" in item:
                text = item["conversation"]
            else:
                # 尝试获取第一个可能是文本的字段
                for key, value in item.items():
                    if isinstance(value, str) and len(value) > 10:
                        text = value
                        break
                else:
                    continue  # 跳过无法处理的样本
            
            # 编码文本
            encodings = self.tokenizer(
                text,
                truncation=True,
                max_length=self.max_length,
                padding="max_length",
                return_tensors="pt"
            )
            
            # 存储编码后的结果
            self.inputs.append(encodings.input_ids[0])
            self.masks.append(encodings.attention_mask[0])
            # 标签与输入相同（用于自回归学习）
            self.labels.append(encodings.input_ids[0].clone())
            
            # 将填充位置的标签设置为-100，这样在计算损失时会被忽略
            self.labels[-1][encodings.attention_mask[0] == 0] = -100
    
    def __len__(self):
        return len(self.inputs)
    
    def __getitem__(self, idx):
        return {
            "input_ids": self.inputs[idx],
            "attention_mask": self.masks[idx],
            "labels": self.labels[idx]
        }

def load_conversation_dataset(dataset_name, tokenizer, split="train", max_length=512):
    """
    加载HuggingFace对话数据集
    
    Args:
        dataset_name: 数据集名称
        tokenizer: 分词器
        split: 数据集分割，如'train', 'validation'等
        max_length: 最大序列长度
        
    Returns:
        处理后的数据集
    """
    # 尝试加载数据集，如果失败则尝试备选方案
    try:
        logger.info(f"尝试加载数据集: {dataset_name}, split={split}")
        raw_dataset = load_dataset(dataset_name, split=split)
        logger.info(f"成功加载数据集: {dataset_name}")
    except Exception as e:
        logger.warning(f"无法加载数据集 {dataset_name}: {e}")
        
        # 尝试其他可能的数据集名称格式
        alternative_names = [
            "openwebtext",
            "EleutherAI/pile",
            "allenai/c4",
            "wikimedia/wikipedia",
            "tatsu-lab/alpaca",
            "databricks/databricks-dolly-15k"
        ]
        
        for alt_name in alternative_names:
            if alt_name != dataset_name:
                try:
                    logger.info(f"尝试备选数据集: {alt_name}")
                    raw_dataset = load_dataset(alt_name, split=split)
                    logger.info(f"成功加载备选数据集: {alt_name}")
                    break
                except Exception:
                    continue
        else:
            # 如果所有备选数据集都失败，创建一个简单的测试数据集
            logger.warning("所有备选数据集加载失败，创建简单测试数据集")
            raw_dataset = create_dummy_dataset()
    
    # 创建处理后的数据集
    processed_dataset = ConversationDataset(
        dataset=raw_dataset,
        tokenizer=tokenizer,
        max_length=max_length
    )
    
    return processed_dataset

def create_dummy_dataset():
    """创建一个简单的测试数据集，当所有远程数据集加载失败时使用"""
    from datasets import Dataset
    import pandas as pd
    
    # 创建一些简单的训练样本
    texts = [
        "人工智能(AI)是研究如何使计算机模拟人类智能的一门科学。",
        "机器学习是人工智能的一个分支，它使计算机能够从数据中学习。",
        "深度学习是机器学习的一种，它使用神经网络进行学习。",
        "大语言模型是基于Transformer架构的神经网络，用于处理和生成自然语言。",
        "Python是一种广泛用于人工智能和机器学习的编程语言。",
        "用户：你好！\n助手：您好！有什么可以帮您的吗？",
        "用户：今天天气如何？\n助手：我无法获取实时天气信息，建议您查看天气应用或网站获取准确信息。",
        "用户：解释一下量子计算。\n助手：量子计算是利用量子力学原理进行信息处理的计算方式。传统计算机使用比特，而量子计算机使用量子比特。",
        "用户：写一个简单的Python函数。\n助手：这是一个简单的Python函数：\ndef hello_world():\n    print('Hello, World!')\n\n# 调用函数\nhello_world()",
        "用户：如何学习编程？\n助手：学习编程的步骤：1. 选择一门编程语言，如Python；2. 学习基础语法；3. 解决简单问题；4. 参与项目实践；5. 持续学习和提高。"
    ]
    
    # 创建数据集
    data = {"text": texts}
    df = pd.DataFrame(data)
    dummy_dataset = Dataset.from_pandas(df)
    
    logger.info(f"创建简单测试数据集，包含{len(dummy_dataset)}个样本")
    return dummy_dataset 