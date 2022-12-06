# 손글씨 다중분류 모델 성능 평가하기

import torch
import torch.nn as nn
from torchvision.datasets.mnist import MNIST
from torchvision.transforms import ToTensor
from torch.utils.data.dataloader import DataLoader

device = "cuda" if torch.cuda.is_available() else "cpu"
model = nn.Sequential(
    nn.Linear(784, 64),
    nn.ReLU(),
    nn.Linear(64, 64),
    nn.ReLU(),
    nn.Linear(64, 10)
)

model.load_state_dict(torch.load("MNIST.pth", map_location=device))
test_data = MNIST(root="./", train=False, download=True, transform=ToTensor())
test_loader = DataLoader(test_data, batch_size=32, shuffle=False)

num_corr = 0 # 분류에 성공한 전체 개수

with torch.no_grad(): # 기울기를 계산하지 않음
    for data, label in test_loader:
        data = torch.reshape(data, (-1, 784)).to(device)

        output = model(data.to(device))
        preds = output.data.max(1)[1] # 모델의 예측값 계산
        
        # 올바르게 분류한 개수
        corr = preds.eq(label.to(device).data).sum().item()
        num_corr += corr
    
    print(f"Accuracy:{num_corr/len(test_data)}") # 분류 정확도 출력