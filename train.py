import argparse
import os
import torch
from torch.utils.data import DataLoader
from transformers import get_linear_schedule_with_warmup
from tqdm import tqdm
import logging
from tensorboard import program
import threading
import time

from model import load_model
from data_processor import load_conversation_dataset

# 设置日志
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

def train(args):
    """
    训练模型
    
    Args:
        args: 命令行参数
    """
    # 确保输出目录存在
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 记录训练参数
    logger.info(f"训练参数: {args}")
    
    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    logger.info(f"使用设备: {device}")
    
    # 加载模型和tokenizer
    try:
        model, tokenizer = load_model(args.model_name_or_path, device=device)
        logger.info(f"成功加载模型: {args.model_name_or_path}")
    except Exception as e:
        logger.error(f"加载模型失败: {e}")
        logger.info("尝试加载备选模型...")
        try:
            model, tokenizer = load_model("gpt2", device=device)
            logger.info("成功加载备选模型: gpt2")
        except Exception as e:
            logger.error(f"加载备选模型失败: {e}")
            logger.error("无法继续训练，退出程序")
            return
    
    # 设置重试策略
    max_retries = 3
    retry_delay = 5
    
    # 加载训练数据
    for attempt in range(max_retries):
        try:
            logger.info(f"尝试加载训练数据，第{attempt+1}次尝试...")
            train_dataset = load_conversation_dataset(
                args.dataset_name,
                tokenizer,
                split="train",
                max_length=args.max_seq_length
            )
            logger.info(f"训练数据样本数: {len(train_dataset)}")
            break
        except Exception as e:
            logger.error(f"第{attempt+1}次加载训练数据失败: {e}")
            if attempt < max_retries - 1:
                logger.info(f"等待{retry_delay}秒后重试...")
                time.sleep(retry_delay)
            else:
                logger.error("已达最大重试次数，无法加载训练数据")
                return
    
    # 尝试加载验证数据
    do_eval = False
    if args.do_eval:
        try:
            eval_dataset = load_conversation_dataset(
                args.dataset_name,
                tokenizer,
                split="validation",
                max_length=args.max_seq_length
            )
            logger.info(f"验证数据样本数: {len(eval_dataset)}")
            do_eval = True
        except Exception as e:
            logger.warning(f"无法加载验证数据: {e}")
            logger.warning("继续训练，但不进行验证")
    
    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.train_batch_size,
        shuffle=True
    )
    
    if do_eval:
        eval_loader = DataLoader(
            eval_dataset,
            batch_size=args.eval_batch_size,
            shuffle=False
        )
    
    # 设置优化器
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay
    )
    
    # 计算训练步数
    num_train_steps = len(train_loader) * args.num_train_epochs
    num_warmup_steps = int(num_train_steps * args.warmup_proportion)
    
    # 设置学习率调度器
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_train_steps
    )
    
    # 启动TensorBoard
    if args.tensorboard:
        try:
            tb = program.TensorBoard()
            tb.configure(argv=[None, '--logdir', os.path.join(args.output_dir, "logs")])
            url = tb.launch()
            logger.info(f"TensorBoard 运行在: {url}")
            
            # 创建TensorBoard写入器
            from torch.utils.tensorboard import SummaryWriter
            writer = SummaryWriter(os.path.join(args.output_dir, "logs"))
        except Exception as e:
            logger.warning(f"启动TensorBoard失败: {e}")
            args.tensorboard = False
    
    # 记录训练开始
    logger.info("***** 开始训练 *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Batch size = {args.train_batch_size}")
    logger.info(f"  Num epochs = {args.num_train_epochs}")
    
    # 训练循环
    global_step = 0
    best_eval_loss = float('inf')
    
    try:
        for epoch in range(args.num_train_epochs):
            model.train()
            epoch_loss = 0
            epoch_iterator = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.num_train_epochs}")
            
            for step, batch in enumerate(epoch_iterator):
                try:
                    # 将数据移动到设备
                    batch = {k: v.to(device) for k, v in batch.items()}
                    
                    # 前向传播
                    outputs = model(**batch)
                    loss = outputs.loss
                    
                    # 反向传播
                    loss.backward()
                    
                    # 梯度裁剪
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                    
                    # 更新参数
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()
                    
                    # 记录loss
                    epoch_loss += loss.item()
                    epoch_iterator.set_postfix({"loss": loss.item()})
                    
                    # TensorBoard记录
                    if args.tensorboard:
                        writer.add_scalar("train/loss", loss.item(), global_step)
                        writer.add_scalar("train/lr", scheduler.get_last_lr()[0], global_step)
                    
                    global_step += 1
                    
                    # 保存检查点
                    if global_step % args.save_steps == 0:
                        output_dir = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                        os.makedirs(output_dir, exist_ok=True)
                        
                        # 保存模型
                        try:
                            model.save_pretrained(output_dir)
                            tokenizer.save_pretrained(output_dir)
                            logger.info(f"保存模型检查点到 {output_dir}")
                        except Exception as e:
                            logger.warning(f"保存检查点失败: {e}")
                
                except Exception as e:
                    logger.warning(f"处理批次时出错: {e}")
                    continue
            
            # 计算epoch平均loss
            avg_train_loss = epoch_loss / len(train_loader)
            logger.info(f"Epoch {epoch+1} 训练完成，平均loss: {avg_train_loss:.4f}")
            
            # 验证
            if do_eval:
                logger.info("***** 运行验证 *****")
                model.eval()
                eval_loss = 0
                eval_iterator = tqdm(eval_loader, desc="Validation")
                
                with torch.no_grad():
                    for batch in eval_iterator:
                        try:
                            batch = {k: v.to(device) for k, v in batch.items()}
                            outputs = model(**batch)
                            loss = outputs.loss
                            eval_loss += loss.item()
                        except Exception as e:
                            logger.warning(f"验证批次处理出错: {e}")
                            continue
                
                avg_eval_loss = eval_loss / len(eval_loader)
                logger.info(f"验证loss: {avg_eval_loss:.4f}")
                
                # TensorBoard记录
                if args.tensorboard:
                    writer.add_scalar("eval/loss", avg_eval_loss, global_step)
                
                # 保存最佳模型
                if avg_eval_loss < best_eval_loss:
                    best_eval_loss = avg_eval_loss
                    best_model_dir = os.path.join(args.output_dir, "best_model")
                    os.makedirs(best_model_dir, exist_ok=True)
                    try:
                        model.save_pretrained(best_model_dir)
                        tokenizer.save_pretrained(best_model_dir)
                        logger.info(f"保存最佳模型到 {best_model_dir}")
                    except Exception as e:
                        logger.warning(f"保存最佳模型失败: {e}")
    except KeyboardInterrupt:
        logger.info("接收到中断信号，提前结束训练")
    except Exception as e:
        logger.error(f"训练过程中出错: {e}")
    finally:
        # 保存最终模型
        final_model_dir = os.path.join(args.output_dir, "final_model")
        os.makedirs(final_model_dir, exist_ok=True)
        try:
            model.save_pretrained(final_model_dir)
            tokenizer.save_pretrained(final_model_dir)
            logger.info(f"保存最终模型到 {final_model_dir}")
        except Exception as e:
            logger.warning(f"保存最终模型失败: {e}")
        
        # 关闭TensorBoard写入器
        if args.tensorboard and 'writer' in locals():
            writer.close()
        
        logger.info("***** 训练完成 *****")

def main():
    parser = argparse.ArgumentParser()
    
    # 模型和数据参数
    parser.add_argument("--model_name_or_path", default="gpt2", type=str,
                        help="预训练模型名称或路径，如果没有则从头训练")
    parser.add_argument("--dataset_name", default="wikitext/wikitext-2-v1", type=str,
                        help="HuggingFace数据集名称")
    parser.add_argument("--output_dir", default="./outputs", type=str,
                        help="保存模型输出的目录")
    
    # 训练参数
    parser.add_argument("--max_seq_length", default=512, type=int,
                        help="最大序列长度")
    parser.add_argument("--train_batch_size", default=4, type=int,
                        help="训练批次大小")
    parser.add_argument("--eval_batch_size", default=4, type=int,
                        help="验证批次大小")
    parser.add_argument("--learning_rate", default=5e-5, type=float,
                        help="学习率")
    parser.add_argument("--weight_decay", default=0.01, type=float,
                        help="权重衰减")
    parser.add_argument("--max_grad_norm", default=1.0, type=float,
                        help="梯度裁剪最大范数")
    parser.add_argument("--num_train_epochs", default=3, type=int,
                        help="训练轮数")
    parser.add_argument("--warmup_proportion", default=0.1, type=float,
                        help="学习率预热比例")
    parser.add_argument("--save_steps", default=1000, type=int,
                        help="每多少步保存一次模型")
    parser.add_argument("--tensorboard", action="store_true",
                        help="是否使用TensorBoard记录训练状态")
    parser.add_argument("--no_cuda", action="store_true",
                        help="不使用CUDA，即使可用")
    parser.add_argument("--do_eval", action="store_true", default=True,
                        help="是否进行验证")
    parser.add_argument("--local_rank", type=int, default=-1,
                        help="分布式训练的本地排名")
    
    args = parser.parse_args()
    train(args)

if __name__ == "__main__":
    main() 