# 손글씨 분류하기 : 다중분류

# 데이터 mnist 손글씨 데이터셋 사용

import matplotlib.pyplot as plt

from torchvision.datasets.mnist import MNIST
from torchvision.transforms import ToTensor


# 학습용 데이터셋 평가용 데이터셋 분류
training_data = MNIST(root="./", train=True, download=True, transform=ToTensor())
test_data = MNIST(root="./", train=False, download=True, transform=ToTensor())

print(len(training_data)) # 60000
print(len(test_data)) # 10000

for i in range(9): # 샘플 이미지 9개 출력
    plt.subplot(3, 3, i+1)
    plt.imshow(training_data.data[i])

plt.show()


# 데이터 불러오기
from torch.utils.data.dataloader import DataLoader
# DataLoader : 데이터셋 A를 원하는 배치 크기 나누어 반환한다. batch_size가 배치 크기를 결정하고, shuffle은 데이터를 섞을지에 대한 여부를 결정한다. 
train_loader = DataLoader(training_data, batch_size=32, shuffle=True)

# 평가용은 데이터 섞을 필요 없다.
test_loader = DataLoader(test_data, batch_size=32, shuffle=False)


# GPU 이용 및 모델 정의
import torch
import torch.nn as nn

from torch.optim.adam import Adam

device = "cuda" if torch.cuda.is_available() else "cpu"

model = nn.Sequential(
    nn.Linear(784, 64),
    nn.ReLU(),
    nn.Linear(64, 64),
    nn.ReLU(),
    nn.Linear(64, 10)
)

model.to(device) # 모델의 파라미터를 GPU로 보냄

lr = 1e-3
optim = Adam(model.parameters(), lr=lr)

for epoch in range(20):
    for data, label in train_loader:
        optim.zero_grad()
        # 입력 데이터 모양을 모델의 입력에 맞게 변환
        data = torch.reshape(data, (-1, 784)).to(device)
        preds = model(data)

        loss = nn.CrossEntropyLoss()(preds, label.to(device))
        loss.backward() # 오차 역전파 
        optim.step() # 최적화 진행

    print(f"epoch{epoch+1} loss{loss.item()}")

# 모델을 MNIST.pth라는 이름으로 저장
torch.save(model.state_dict(), "MNIST.pth")
