import argparse
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import logging
import os
import sys
import threading
import psutil

# 设置日志
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

# 打印系统信息
print(f"Python版本: {sys.version}")
print(f"PyTorch版本: {torch.__version__}")
print(f"操作系统: {os.name}, {sys.platform}")

# 获取并打印CPU信息
print("\nCPU信息:")
try:
    cpu_count = psutil.cpu_count(logical=False)
    cpu_count_logical = psutil.cpu_count(logical=True)
    print(f"物理CPU核心数: {cpu_count}")
    print(f"逻辑CPU核心数: {cpu_count_logical}")
    
    # 设置PyTorch的线程数以优化性能
    if cpu_count_logical > 1:
        # 使用所有可用的逻辑核心，但保留一个核心给系统
        num_threads = max(1, cpu_count_logical - 1)
        torch.set_num_threads(num_threads)
        print(f"PyTorch线程数设置为: {num_threads}")
except Exception as e:
    print(f"获取CPU信息失败: {e}")

# 检查可用内存
try:
    total_ram = psutil.virtual_memory().total / (1024 ** 3)
    available_ram = psutil.virtual_memory().available / (1024 ** 3)
    print(f"系统总内存: {total_ram:.2f}GB, 可用内存: {available_ram:.2f}GB")
except Exception as e:
    print(f"获取内存信息失败: {e}")


class ChatBot:
    """聊天机器人类，用于与模型进行对话"""
    
    def __init__(self, model_path, device="cpu"):
        """
        初始化聊天机器人
        
        Args:
            model_path: 模型路径
            device: 运行设备，使用CPU
        """
        # 使用CPU
        self.device = torch.device("cpu")
        logger.info(f"使用CPU运行模型")
        
        # 加载模型和tokenizer
        logger.info(f"正在加载模型: {model_path}")
        try:
            logger.info("开始加载分词器...")
            self.tokenizer = AutoTokenizer.from_pretrained(model_path)
            logger.info("分词器加载成功，开始加载模型...")
            
            # 使用8位量化加载模型以节省内存
            try:
                from transformers import BitsAndBytesConfig
                
                # 尝试使用8位量化
                quantization_config = BitsAndBytesConfig(
                    load_in_8bit=True,
                    llm_int8_threshold=6.0
                )
                
                self.model = AutoModelForCausalLM.from_pretrained(
                    model_path,
                    quantization_config=quantization_config
                )
                logger.info("模型成功加载(8位量化)")
            except Exception as e:
                logger.warning(f"量化模型加载失败: {e}，尝试正常加载")
                # 使用常规方式加载模型
                self.model = AutoModelForCausalLM.from_pretrained(
                    model_path
                )
                logger.info("模型成功加载到CPU")
                
        except Exception as e:
            logger.error(f"模型加载失败: {e}")
            raise e
            
        # 确保pad_token存在
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
        # 设置对话历史
        self.conversation_history = ""
        self.system_prompt = "你是一个友好、有帮助的AI助手。"
    
    def reset_conversation(self):
        """重置对话历史"""
        self.conversation_history = ""
        logger.info("对话历史已重置")
        
    def set_system_prompt(self, prompt):
        """设置系统提示词"""
        self.system_prompt = prompt
        logger.info(f"系统提示词已设置为: {prompt}")
        
    def generate_response(self, user_input, max_new_tokens=50, temperature=0.7, top_p=0.9):
        """
        生成回复
        
        Args:
            user_input: 用户输入
            max_new_tokens: 生成的最大token数，减小以提高速度
            temperature: 温度参数，控制生成的随机性
            top_p: 核采样参数
            
        Returns:
            生成的回复
        """
        # 添加系统提示词（如果对话历史为空）
        if not self.conversation_history:
            self.conversation_history = f"系统: {self.system_prompt}\n"
            
        # 添加用户输入到对话历史
        self.conversation_history += f"用户: {user_input}\n助手: "
        
        # 编码对话历史
        inputs = self.tokenizer(self.conversation_history, return_tensors="pt")
        
        # 生成回复
        with torch.no_grad():
            # 使用更节省内存的设置
            outputs = self.model.generate(
                inputs.input_ids,
                max_new_tokens=max_new_tokens,  # 减小以加快速度
                temperature=temperature,
                top_p=top_p,
                do_sample=temperature > 0,
                pad_token_id=self.tokenizer.pad_token_id,
                attention_mask=inputs.attention_mask,
                use_cache=True  # 使用KV缓存提高效率
            )
        
        # 解码生成的回复
        generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # 提取新生成的回复
        response = generated_text[len(self.conversation_history):]
        
        # 更新对话历史
        self.conversation_history += response + "\n"
        
        return response

def interactive_chat(model_path, device="cpu"):
    """
    交互式聊天
    
    Args:
        model_path: 模型路径
        device: 运行设备
    """
    print("开始初始化聊天机器人...")
    chatbot = ChatBot(model_path, device)
    print(f"模型已加载，开始聊天！输入'退出'可以结束对话。")
    print(f"提示: 输入'重置'可清除对话历史，输入'系统提示:xxx'可设置新的系统提示")
    
    while True:
        user_input = input("用户: ")
        
        if user_input.lower() in ["退出", "exit", "quit"]:
            print("谢谢使用，再见！")
            break
            
        if user_input.lower() == "重置":
            chatbot.reset_conversation()
            print("对话已重置。")
            continue
            
        if user_input.lower().startswith("系统提示:"):
            system_prompt = user_input[6:].strip()
            chatbot.set_system_prompt(system_prompt)
            chatbot.reset_conversation()
            print(f"系统提示已设置为: {system_prompt}")
            continue
            
        try:
            print("助手正在思考...")
            response = chatbot.generate_response(user_input)
            print(f"助手: {response}")
        except Exception as e:
            print(f"生成回复时出错: {e}")

def main():
    parser = argparse.ArgumentParser(description="交互式聊天机器人")
    parser.add_argument("--model_path", type=str, required=True, help="模型路径")
    parser.add_argument("--device", type=str, default="cpu", help="运行设备（目前仅支持cpu）")
    parser.add_argument("--max_tokens", type=int, default=50, help="生成的最大token数")
    
    args = parser.parse_args()
    interactive_chat(args.model_path, args.device)

if __name__ == "__main__":
    main() 