import torch
print("GPU available:", torch.cuda.is_available())
torch.save(torch.tensor([1, 2, 3]), "test.pth")