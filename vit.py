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
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

# 1. Patch Embedding
# 0. Patch Embedding Variables
p = 4 # patch
w = 32 # width
h = 32 # height
c = 3 # channel
b = 128 # batch
d = 128 # Dim of patched embeddings
cls = 10 # Class token size
L = 8 # Transformer block size
drop_rate = 0.1
n = w//p
mlp_hidden_dim = int(d/2)

# Trainable Linear Projection이 필요
# nn.Module로 구성
class PositionalEmbedding(nn.Module):
    def __init__(self):
        super(PositionalEmbedding, self).__init__()
        self.projection = nn.Linear(p*p*c, d) # These image patch vectors are now encoded using a linear transformation. Fixed size `d`
        self.dropout = nn.Dropout(drop_rate)
    def patchify(self,img):
      # Divide to patch
      patched_img = img.view(b,c,h//p,p,w//p,p) # 이미지 1개당 N*N개 패치가 나오고, 패치 하나의 이미지는 P*P*C
      patched_img = patched_img.transpose(3,4)
      patched_img = patched_img.transpose(1,3)
      patched_img = patched_img.transpose(1,2)
      patched_img = patched_img.reshape(b,n*n,p*p*c)
      return patched_img
    def class_emb(self, patch):
      x_class = nn.Parameter(torch.randn(b,1,d)).to(device)
      with_class = torch.cat((x_class, patch), dim = 1)
      # print("with class embedding : ", with_class.shape)
      return with_class

    def position_emb(self, class_patch):
      pos_emb = nn.Parameter(torch.randn(b,n*n+1,d)).to(device)
      with_class_pos = class_patch + pos_emb # 이게 맞나? 그냥 더하는게?
      # print("with class & positional embedding : ", with_class_pos.shape)
      return with_class_pos

    def forward(self, x):
        patched_ = self.patchify(x)
        patched_ = self.projection(patched_)
        patched_ = self.class_emb(patched_)
        patched_ = self.position_emb(patched_)
        patched_ = self.dropout(patched_)
        return patched_

# Transformer

head_num = 8 # attention heads
# class Attention(nn.Module):
#   def __init__(self):
#     super(Attention, self).__init__()
#     self.w_q = nn.Parameter(torch.randn(d, n*n+1))
#     self.w_k = nn.Parameter(torch.randn(d, n*n+1))
#     self.w_v = nn.Parameter(torch.randn(d, n*n+1))
#     self.dropout = nn.Dropout(drop_rate)
#   def forward(self, x):
#     # W_q,W_k,W_v 를 정의
    
#     q = x @ self.w_q
#     k = x @ self.w_k
#     v = x @ self.w_v
#     # QK^T를 만들기
#     qk_T = q @ k.T
#     # k의 차원 : D (Latent vector)
#     qk_T = qk_T / d
#     soft_ = nn.Softmax(dim = 0)
#     attention_ = soft_(qk_T)
#     # print(attention_.shape, v.shape)
#     ret = self.dropout(attention_) @ v
#     return ret

# class MultiHeadAttention(nn.Module):
#   def __init__(self):
#     super(MultiHeadAttention, self).__init__()
#     self.attn = Attention()
#     self.w_o = nn.Parameter(torch.randn(head_num*(n*n+1), d))
#   def forward(self,x):
#     # Head의 Concat이 필요
#     head_list = []
#     for h in range(head_num):

#       x_h = self.attn(x)
#       head_list.append(x_h)
#     ret = torch.cat(head_list, dim =1)
#     ret = ret @ self.w_o
#     return ret

## TODO : Find Difference
class MultiheadedSelfAttention(nn.Module):
    def __init__(self):
        super().__init__()
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.num_heads = head_num
        self.latent_vec_dim = d
        self.head_dim = int(d / head_num)
        self.query = nn.Linear(d, d)
        self.key = nn.Linear(d, d)
        self.value = nn.Linear(d, d)
        self.scale = torch.sqrt(self.head_dim*torch.ones(1)).to(device)
        self.dropout = nn.Dropout(drop_rate)

    def forward(self, x):
        batch_size = x.size(0)
        q = self.query(x)
        k = self.key(x)
        v = self.value(x)
        q = q.view(batch_size, -1, self.num_heads, self.head_dim).permute(0,2,1,3)
        k = k.view(batch_size, -1, self.num_heads, self.head_dim).permute(0,2,3,1) # k.t
        v = v.view(batch_size, -1, self.num_heads, self.head_dim).permute(0,2,1,3)
        attention = torch.softmax(q @ k / self.scale, dim=-1)
        x = self.dropout(attention) @ v
        x = x.permute(0,2,1,3).reshape(batch_size, -1, self.latent_vec_dim) # TODO : Check what is this

        return x
class VisionTransformerBlock(nn.Module):
  def __init__(self):
    super(VisionTransformerBlock, self).__init__()
    self.bn1 = nn.LayerNorm(d) # Size of BatchNorm1d is the input's size
    self.bn2 = nn.LayerNorm(d) # Size of BatchNorm1d is the input's size
    self.msa = MultiheadedSelfAttention()
    self.mlp = nn.Sequential(nn.Linear(d, mlp_hidden_dim),
                                nn.GELU(), nn.Dropout(drop_rate),
                                nn.Linear(mlp_hidden_dim, d),
                                nn.Dropout(drop_rate))
    self.dropout = nn.Dropout(drop_rate)
  def forward(self, x):
    # Batch Norm 1d
    x = self.bn1(x)
    # Multi-head Attention (Done)
    x_attn = self.msa(x)
    # print(x_attn.shape, x.shape)
    # Residual connections
    x_attn = self.dropout(x_attn)
    x_attn = x_attn + x
    # Norm
    out = self.bn2(x_attn)
    # MLP
    out = self.mlp(x_attn)
    # Concat
    out = out + x_attn
    # print(out.shape)
    return out

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
    # Seqeuence L 반복
    # ViT가 계속 업데이트 되야되는데 ..
    # outputs = []
    for layer in self.vit:
       x = layer(x)
    # for d in pe_out:
    #   # print(d.shape)
    #   for layer in self.vit:
    #     d = layer(d)
    #   # print(d.shape)
    #   outputs.append(d)
    #   # 각 이미지에 대한 output을 의미해야되는데
    #   # label이 0,1이 아니라 1~10으로 구성이 되어 있다.
    # outputs = torch.stack(outputs,dim = 0).to(device)
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
optimizer = optim.SGD(ViT.parameters(),lr=2e-3, weight_decay=2e-4)
# 지금은 patch를 1D로 만들고, cls, pos 를 붙임
# patch , cls, pos를 붙인 다음에

ViT.train()
torchinfo.summary(ViT)
num_train_loader = len(train_loader)
for epoch in range(10):
  running_loss = 0
  for img, label in tqdm(train_loader):
    optimizer.zero_grad()
    img = img.to(device)
    label = label.to(device)
    out = ViT(img)
    label_f32 = label.type('torch.LongTensor').to(device)
    # print(out.dtype, label_f32.dtype)
    loss = criterion(out, label_f32)

    # loss
    optimizer.zero_grad()
    loss.backward() #retain_graph=True
    optimizer.step()
  train_loss = running_loss / num_train_loader
  val_acc, val_loss = accuracy(test_loader, ViT)
  # if epoch % 5 == 0:
  print('[%d] train loss: %.3f, validation loss: %.3f, validation acc %.2f %%' % (epoch, train_loss, val_loss, val_acc))
  torch.save(ViT, f'checkpoints/model_{epoch}.pth')
#   ViT.eval()
#   test_loss = 0.0
#   correct = 0

#   # 13
#   with torch.no_grad():
#       for images, labels in test_loader:
#           images = images.to(device)
#           labels = labels.to(device)

#           # 14
#           outputs = ViT(images)
#           predicted = torch.max(outputs, 1)[1]
#           loss = criterion(outputs, labels)

#           # 15
#           test_loss += loss.item()
#           correct += (labels == predicted).sum()
#   # 16
#   print(
#       f"epoch {epoch+1} - test loss: {test_loss / len(test_loader):.4f}"
#   )