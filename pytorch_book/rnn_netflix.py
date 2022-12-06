# netflix 주가 예측 rnn

import numpy as np
import pandas as pd
from torch.utils.data.dataset import Dataset

import torch
import torch.nn as  nn

data = pd.read_csv('./netflix_stock_data/train.csv')

class Netflix(Dataset):
    def __init__(self):
        self.csv = pd.read_csv('./netflix_stock_data/train.csv')

        # 입력 데이터 정규화
        self.data = self.csv.iloc[:,1:4].values # 종가를 제외한 데이터
        self.data = self.data / np.max(self.data) # 0과 1사이로 정규화
        
        # 종가 데이터 정규화
        self.label = data["Close"].values
        self.label = self.label /np.max(self.label)

    def __len__(self):
        return len(self.data) - 30 # 사용 가능한 배치 개수 30일 기준
    
    def __getitem__(self, index):
        data = self.data[index:index+30] # 입력 데이터 30일치 읽기
        label = self.label[index+30] # 종가 데이터 30일치 읽기

        return data, label


class RNN(nn.Module):
    def __init__(self):
        super(RNN, self).__init__()

        # RNN층 정의
        self.rnn = nn.RNN(input_size=3, hidden_size=8, num_layers=5, batch_first=True)

        self.fc1 = nn.Linear(in_features=240, out_features=64)
        self.fc2 = nn.Linear(in_features=64, out_features=1)

        self.relu = nn.ReLU()
    
    def forward(self,x, h0):
        x, hn = self.rnn(x, h0) # 
        
        # MLP 층 모양 변경
        x = torch.reshape(x, (x.shape[0], -1))

        # MLP층 사용해 종가 예측
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)

        # 예측한 종가를 1차원 벡터로 표현
        x = torch.flatten(x)

        return x

# 모델 데이터셋 정의
import tqdm

from torch.optim.adam import Adam
from torch.utils.data.dataloader import DataLoader

device = "cuda" if torch.cuda.is_available() else "cpu"

model = RNN().to(device)
dataset = Netflix()

loader = DataLoader(dataset, batch_size=32)
optim = Adam(params=model.parameters(), lr=0.001) 

# train 
for epoch in range(200):
    iterator = tqdm.tqdm(loader)
    for data, label in iterator:
        optim.zero_grad()

        # 초기 은닉 상태
        h0 = torch.zeros(5, data.shape[0],8).to(device)

        # 모델의 예측값
        preds = model(data.type(torch.FloatTensor).to(device), h0)

        # 손실의 계산
        loss = nn.MSELoss()(preds, label.type(torch.FloatTensor).to(device))
        loss.backward()
        optim.step()

        iterator.set_description(f"epoch{epoch} loss:{loss.item()}")

torch.save(model.state_dict(), "./rnn_netflix.pth")


# 모델 성능 평가

loader = DataLoader(dataset, batch_size=1)

preds = []
total_loss = 0

with torch.no_grad():
    # model.load_state_dict(torch.load("./rnn_netflix.pth", map_location=device))

    for data, label in loader:
        h0 = torch.zeros(5, data.shape[0], 8).to(device)

        # 모델 예측값 출력
        pred = model(data.type(torch.FloatTensor).to(device), h0)
        preds.append(pred.item()) # 예측값 리스트 추가
        
        # 손실 계산

        loss = nn.MSELoss()(pred, label.type(torch.FloatTensor).to(device))

        # 평균치 계산
        total_loss += loss/len(loader)
    print(total_loss.item())


