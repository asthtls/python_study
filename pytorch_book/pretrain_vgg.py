# 전이학습을 이용한 VGG16.py


# VGG는 단순한 구조를 가진 만큼 데이터가 무난한 성능을 발휘한다.
# 층이 깊어질수록 기울기 소실 문제가 발생

# ImageNet 데이터로 사전 학습된 VGG 모델로 CIFAR-10학습


# 사전 학습 모델 준비
import torch
import torch.nn as nn

from torchvision.models.vgg import vgg16

device = "cuda" if torch.cuda.is_available() else "cpu"

model = vgg16(pretrained=True)

fc = nn.Sequential( # 분류층 정의
    nn.Linear(512*7*7, 4096),
    nn.ReLU(),
    nn.Dropout(), # 드롭아웃 층 정의 default 
    nn.Linear(4096, 4096),
    nn.ReLU(),
    nn.Dropout(),
    nn.Linear(4096, 10),
)

model.classifier = fc # VGG의 classifier를 덮어씀
model.to(device)

# 데이터 전처리와 증강
import tqdm

from torchvision.datasets.cifar import CIFAR10
from torchvision.transforms import Compose, ToTensor, Resize, RandomHorizontalFlip, RandomCrop, Normalize
from torch.utils.data.dataloader import DataLoader

from torch.optim.adam import Adam

transforms = Compose([
    Resize(224),
    RandomCrop((224, 224), padding=4),
    RandomHorizontalFlip(p=0.5),
    ToTensor(),
    Normalize(mean=(0.4914, 0.4822, 0.4465), std=(0.247, 0.243, 0.261))
])

# 데이터로더 정의

training_data = CIFAR10(root="./", train=True, download=True, transform=transforms)
test_data = CIFAR10(root="./", train=False, download=True, transform=transforms)

train_loader = DataLoader(training_data, batch_size=32, shuffle=True)
test_loader = DataLoader(test_data, batch_size=32, shuffle=False)

 #학습 루프 정의
lr = 1e-4
optim = Adam(model.parameters(), lr=lr)

for epoch in range(30):
    iterator = tqdm.tqdm(train_loader) # 학습 로그 출력
    for data, label in iterator:
        optim.zero_grad()

        preds = model(data.to(device))

        loss = nn.CrossEntropyLoss()(preds, label.to(device))
        loss.backward()
        optim.step()

        # tqdm이 출력할 
        iterator.set_description(f"epoch:{epoch+1} loss:{loss.item()}")

#torch.save(model.state_dict(), "CIFAR10_pretrained.pth")


# 모델 성능 평가
# model.load_state_dict(torch.load("CIFAR10_pretrained.pth"), map_location=device)

num_corr = 0

with torch.no_grad():
    for data, label in test_loader:
        output = model(data.to(device))
        preds = output.data.max(1)[1]
        corr = preds.eq(label.to(device).data).sum().item()
        num_corr += corr

    print(f"Accuracy : {num_corr/len(test_data)}")

