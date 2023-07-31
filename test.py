import torch
print("hello there")
if(torch.cuda.is_available()):
    print("cuda is available")
a = torch.tensor([1.0, 2.0], device="cuda" if torch.cuda.is_available() else "cpu")
print(a)
