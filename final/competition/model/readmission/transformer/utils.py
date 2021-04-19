import torch

def get_cuda():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"available device: {device}")
    print(f"number of GPUS available: {torch.cuda.device_count()}")

    return device