import torch
from torch import nn
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageOps
import torchvision
import torchvision.transforms as transforms
import os
import torch.optim as optim
from tqdm import tqdm
import torchinfo
import argparse

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

# 0. Patch Embedding Variables
p = 4 # patch
w = 32 # width
h = 32 # height
c = 3 # channel
b = 128 # batch
d = 128 # Dim of patched embeddings
cls = 10 # Class token size
L = 8 # Transformer block size
head_num = 8 # attention heads
drop_rate = 0.1
n = w//p
mlp_hidden_dim = int(d/2)
epochs = 500

class PositionalEmbedding(nn.Module):
    def __init__(self):
        super(PositionalEmbedding, self).__init__()
        self.projection = nn.Linear(p*p*c, d) # These image patch vectors are now encoded using a linear transformation. Fixed size `d`
        self.dropout = nn.Dropout(drop_rate)
        self.linear_proj = nn.Linear(p*p*c, d)
        self.x_class = nn.Parameter(torch.randn(b,1,d))
        self.pos_emb = nn.Parameter(torch.randn(1,n*n+1,d)) # Shape가 이해가 잘 안된다

        self.cls_token = nn.Parameter(torch.randn(1, d))
        self.pos_embedding = nn.Parameter(torch.randn(1, n*n+1, d))
    def patchify(self,img):
        # Divide to patch
        patched_img = img.view(b,c,h//p,p,w//p,p) # 이미지 1개당 N*N개 패치가 나오고, 패치 하나의 이미지는 P*P*C
        patched_img = patched_img.transpose(3,4)
        patched_img = patched_img.transpose(1,3)
        patched_img = patched_img.transpose(1,2)
        patched_img = patched_img.reshape(b,n*n,p*p*c)
        return patched_img
    def class_emb(self, patch):
        with_class = torch.cat((self.x_class, patch), dim = 1)
        return with_class

    def position_emb(self, class_patch):
        with_class_pos = class_patch + self.pos_emb
        return with_class_pos

    def forward(self, x):
        patched_ = self.patchify(x)
        patched_ = self.projection(patched_)
        patched_ = self.class_emb(patched_)
        patched_ = self.position_emb(patched_)
        patched_ = self.dropout(patched_)
        return patched_ 
# Transformer

class MultiHeadAttention(nn.Module):
    def __init__(self):
        super(MultiHeadAttention, self).__init__()
        # batch, d, n*n+1
        self.w_q = nn.Linear(d, d)
        self.w_k = nn.Linear(d, d)
        self.w_v = nn.Linear(d, d)
        self.dropout = nn.Dropout(drop_rate)
    def forward(self, x):
        head_dim = int(d/head_num)

        # W_q,W_k,W_v 를 정의
        q = self.w_q(x)
        k = self.w_k(x)
        v = self.w_v(x)
        # Split to Multi Head
        q = q.view(b, -1, head_num, head_dim).permute(0,2,1,3)
        k = k.view(b, -1, head_num, head_dim).permute(0,2,3,1) # k.t
        v = v.view(b, -1, head_num, head_dim).permute(0,2,1,3)

        qk_T = q @ k

        d_h = torch.sqrt(d/head_num*torch.ones(1)).to(device)
        qk_T = qk_T / d_h
        soft_ = nn.Softmax(dim = 0)
        attention_ = soft_(qk_T)

        ret = self.dropout(attention_) @ v
        ret = ret.permute(0,2,1,3).reshape(b, -1, d) # TODO : Check what is this
        return ret

class VisionTransformerBlock(nn.Module):
    def __init__(self):
        super(VisionTransformerBlock, self).__init__()
        self.bn1 = nn.LayerNorm(d)
        self.bn2 = nn.LayerNorm(d)
        self.msa = MultiHeadAttention()
        self.mlp = nn.Sequential(nn.Linear(d, mlp_hidden_dim),
                                    nn.GELU(), nn.Dropout(drop_rate),
                                    nn.Linear(mlp_hidden_dim, d),
                                    nn.Dropout(drop_rate))
        self.dropout = nn.Dropout(drop_rate)
    def forward(self, x):
        # Batch Norm 1d
        z = self.bn1(x)
        # Multi-head Attention (Done)
        z = self.msa(z)
        # print(x_attn.shape, x.shape)
        # Residual connections
        z = self.dropout(z)
        x = x + z
        # Norm
        z = self.bn2(x)
        # MLP
        z = self.mlp(x)
        # Concat
        x = x + z
        # print(out.shape)
        return x

class VisionTransformer(nn.Module):
    def __init__(self):
        super(VisionTransformer, self).__init__()
        self.pe = PositionalEmbedding()
        self.vit = nn.ModuleList([VisionTransformerBlock()
                                for _ in range(L)])
        # Full Connected Layer
        self.mlp = nn.Sequential(
                nn.LayerNorm(d),
                nn.Linear(d, cls)
            )

    def forward(self,x):
        x = self.pe(x)
        for layer in self.vit:
            x = layer(x)
        x = self.mlp(x[:,0])
        return x.to(device)
  
def accuracy(dataloader, model):
    correct = 0
    total = 0
    running_loss = 0
    n = len(dataloader)
    criterion = nn.CrossEntropyLoss()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    with torch.no_grad():
        model.eval()
        for data in dataloader:
            images, labels = data[0].to(device), data[1].to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            running_loss += loss.item()

        loss_result = running_loss / n

    acc = 100 * correct / total
    model.train()
    return acc, loss_result


if __name__=='__main__':
  # Import Dataset
    transform = transforms.Compose(
        [transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    batch_size = b

    train_set = torchvision.datasets.CIFAR10(root='./data', train=True,
                                          download=True, transform=transform)
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size,
                                            shuffle=True, num_workers=2, drop_last=True)

    test_set = torchvision.datasets.CIFAR10(root='./data', train=False,
                                        download=True, transform=transform)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size,
                                          shuffle=False, num_workers=2,drop_last=True)

    classes = ('plane', 'car', 'bird', 'cat',
            'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    ViT = VisionTransformer()
    ViT.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(ViT.parameters(),lr=2e-3, weight_decay=2e-4)

    ViT.train()
    torchinfo.summary(ViT)
    num_train_loader = len(train_loader)
    for epoch in range(epochs):
        running_loss = 0
        for img, label in tqdm(train_loader):
            optimizer.zero_grad()
            img = img.to(device)
            label = label.to(device)
            out = ViT(img)
            loss = criterion(out, label)

            # loss
            loss.backward() #retain_graph=True
            optimizer.step()
            running_loss += loss.item()
        train_loss = running_loss / num_train_loader
        val_acc, val_loss = accuracy(test_loader, ViT)
        print('[%d] train loss: %.3f, validation loss: %.3f, validation acc %.2f %%' % (epoch, train_loss, val_loss, val_acc))
    torch.save(ViT, f'checkpoints/model_{epoch}.pth')