import torch
import os

print(torch.__version__)

if torch.cuda.is_available():
    print("gpu ready")
else:
    print("cuda not available")
