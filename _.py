import torch

# print("PyTorch version:", torch.__version__)
# print("CUDA version:", torch.version.cuda)
# print("CUDA is available:", torch.cuda.is_available())

print(torch.cuda.get_device_properties(0).major)
print(torch.cuda.get_device_properties(0).minor)
