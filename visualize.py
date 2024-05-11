import torch
from torch import nn
import torchvision
import matplotlib.pyplot as plt
import torch.nn.functional as F
import numpy as np
# from vit import VisionTransformer, VisionTransformerBlock, MultiHeadAttention, PositionalEmbedding
import vit

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
ViT = vit.VisionTransformer().to(device)
ViT.load_state_dict(torch.load('./checkpoints/model.pth'))       

pos_embedding = ViT.pe.pos_emb # [1, 65, 128]

cos = torch.nn.CosineSimilarity(dim=1, eps=1e-6)
fig = plt.figure(figsize=(8, 8))
for i in range(1, pos_embedding.shape[1]):
    sim = F.cosine_similarity(pos_embedding[0, i:i+1], pos_embedding[0, 1:], dim=1)
    sim = sim.reshape((8, 8)).detach().cpu().numpy()
    ax = fig.add_subplot(8, 8, i)
    ax.axes.get_xaxis().set_visible(False)
    ax.axes.get_yaxis().set_visible(False)
    ax.imshow(sim)