import torch
import clip

if torch.cuda.is_available():
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
else:
    device = torch.device("cpu")

clip_model, preprocess = clip.load('ViT-B/32', device, jit=False)




