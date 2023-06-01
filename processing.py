import torch

# Checking if CUDA GPU is available, use it. If not, use CPU for computations
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
