import torch
import sys

print("Python版本:", sys.version)
print("PyTorch版本:", torch.__version__)
print("CUDA可用:", torch.cuda.is_available())

if torch.cuda.is_available():
    print("CUDA版本:", torch.version.cuda)
    print("当前CUDA设备:", torch.cuda.current_device())
    print("设备数量:", torch.cuda.device_count())
    print("设备名称:", torch.cuda.get_device_name(0))
    print("设备属性:", torch.cuda.get_device_properties(0))
else:
    print("CUDA不可用，PyTorch将使用CPU运行")
    
print("\n尝试其他方法检测GPU...")
try:
    import subprocess
    if sys.platform == 'win32':
        gpu_info = subprocess.check_output('nvidia-smi', shell=True).decode('utf-8')
        print("NVIDIA-SMI输出:")
        print(gpu_info)
    else:
        gpu_info = subprocess.check_output(['nvidia-smi'], stderr=subprocess.STDOUT).decode('utf-8')
        print("NVIDIA-SMI输出:")
        print(gpu_info)
except Exception as e:
    print(f"获取GPU信息时出错: {e}") 