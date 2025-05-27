from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig

def load_model(model_name_or_path, device="cuda"):
    """
    加载预训练模型和分词器
    
    Args:
        model_name_or_path: 模型名称或路径
        device: 运行设备，默认为cuda
        
    Returns:
        model: 加载的模型
        tokenizer: 分词器
    """
    # 加载配置
    try:
        config = AutoConfig.from_pretrained(model_name_or_path)
    except:
        # 如果无法加载配置，使用默认小型GPT2配置
        from transformers import GPT2Config
        config = GPT2Config(
            vocab_size=50257,
            n_positions=1024,
            n_ctx=1024,
            n_embd=768,
            n_layer=6,
            n_head=12
        )
        print(f"无法加载配置，使用默认GPT2配置")
    
    # 加载tokenizer
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        # 确保tokenizer有pad_token
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
    except:
        # 如果无法加载tokenizer，使用默认的GPT2 tokenizer
        from transformers import GPT2Tokenizer
        tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        tokenizer.pad_token = tokenizer.eos_token
        print(f"无法加载tokenizer，使用默认GPT2 tokenizer")
    
    # 加载模型
    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path,
            config=config,
        )
    except:
        # 如果无法加载模型，从头开始训练一个GPT2模型
        from transformers import GPT2LMHeadModel
        model = GPT2LMHeadModel(config)
        print(f"无法加载模型，创建一个新的GPT2模型")
    
    model = model.to(device)
    return model, tokenizer 