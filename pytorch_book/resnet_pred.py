# model pred

import torch
import torch.nn as nn

import tqdm

from torchvision.datasets.cifar import CIFAR10
from torchvision.transforms import Compose, ToTensor, Resize, RandomHorizontalFlip, RandomCrop, Normalize
from torch.utils.data.dataloader import DataLoader

from torch.optim.adam import Adam

transforms = Compose([
    RandomCrop((32, 32), padding=4), # 랜더 크롭핑 . 이미지 랜덤하게 자름
    RandomHorizontalFlip(p=0.5), # y축으로 좌우대칭
    ToTensor(), # 텐서 변환
    Normalize(mean=(0.4914, 0.4822, 0.4465), std=(0.247, 0.243, 0.261)) # 정규화
])

test_data = CIFAR10(root="./", train=False, download=True, transform=transforms)

# 데이터로더 정의
test_loader = DataLoader(test_data, batch_size=32, shuffle=False)

device = "cuda" if torch.cuda.is_available() else "cpu"

from resnet import ResNet

# model = ResNet(num_classes=10)
# model.to(device)
model = ResNet(num_classes=10)
model.load_state_dict(torch.load("ResNet.pth", map_location=device))



num_corr = 0

with torch.no_grad():
    for data, label in test_loader:
        
        output = model(data.to(device))
        preds = output.data.max(1)[1]
        corr = preds.eq(label.to(device).data).sum().item()
        num_corr += corr

    print(f"Accuracy : {num_corr/len(test_data)}")