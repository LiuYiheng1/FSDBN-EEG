import torch

path = r"D:\pycharmproject\Uncertainty-aware-Blur-Prior-main\data\things-meg\Preprocessed_data\sub-01\train.pt"
data = torch.load(path, weights_only=False)

print("keys:", data.keys())
for k, v in data.items():
    if hasattr(v, 'shape'):
        print(k, v.shape)
    else:
        print(k, type(v))
