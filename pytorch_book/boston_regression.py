# 보스턴 집값 예측하기 : 회귀분석

from sklearn.datasets import load_boston
import pandas as pd

dataset = load_boston() # 데이터셋 load
# print(dataset.keys()) # 데이터셋의 키를 출력
# dict_keys(['data', 'target', 'feature_names', 'DESCR', 'filename', 'data_module'])

dataFrame = pd.DataFrame(dataset["data"]) # 
dataFrame.columns = dataset["feature_names"] # 특징의 이름 불러오기
dataFrame["target"] = dataset["target"]  # 데이터프레임이 정답 추가

# print(dataFrame.head()) # 데이터프레임 요약 출력

# 선형회귀 MLP 모델
import torch
import torch.nn as nn
from torch.optim.adam import Adam

model = nn.Sequential(
    nn.Linear(13, 100),
    nn.ReLU(),
    nn.Linear(100, 1)
)

X = dataFrame.iloc[:,:13].values # 정답을 제외한 13개 특징 X에 입력
Y = dataFrame["target"].values # 정답 데이터 Y에 입력

batch_size = 100
learning_rate = 0.001

# 가중치를 수정하는 최적화 함수 정의
optim = Adam(model.parameters(), lr = learning_rate)

# 에포크 반복
for epoch in range(200):
    
    # 배치 반복
    for i in range(len(X) // batch_size):
        start = i * batch_size 
        end = start + batch_size # 배치 크기에 맞게 index저장

        # 파이토치 실수형 텐서로 변환
        x = torch.FloatTensor(X[start:end])
        y = torch.FloatTensor(Y[start:end])
        
        optim.zero_grad() # 가중치 기울기 0으로 초기회
        preds = model(x) # 모델 예측값 계산
        loss = nn.MSELoss()(preds, y) # MSE 손실 계산
        loss.backward() # 오차 역전파
        optim.step()

    if epoch % 20 == 0:
        print(f"epoch{epoch} loss:{loss.item()}")

# 모델 성능 평가
prediction = model(torch.FloatTensor(X[0, :13]))
real = Y[0]
print(f"prediction:{prediction.item()} real:{real}")