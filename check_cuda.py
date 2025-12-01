import torch

def check_cuda():
    """Check if CUDA is available and print GPU information."""
    if torch.cuda.is_available():
        print("CUDA is available!")
        print(f"Number of GPUs: {torch.cuda.device_count()}")
        print(f"Current GPU: {torch.cuda.get_device_name(0)}")
        print(f"PyTorch version: {torch.__version__}")
    else:
        print("CUDA is not available. Running on CPU.")
        print(f"PyTorch version: {torch.__version__}")

if __name__ == "__main__":
    check_cuda()