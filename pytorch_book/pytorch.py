# # pytorch 기본 구성 요소

# import torch as nn

# class Net(nn.Module):
#     def __init__(self):
#         super().__init__()
#         # 신경망의 구성요소 정의

#     def forward(self, input):
#         # 신경망의 동작 과정
#         pass
#         return output


# # 데이터 호출
# class Dataset():
#     def __init__(self):
#         # 필요한 데이터 불러오기

#     def __len__(self):
#         # 데이터 개수 반환
#         return len(data)
    
#     def __getitem__(self, i):
#         # i번쨰 입력 데이터와 
#         # i번째 정답을 ㅂ나환

#         return data[i], label[i]
    

# # 입력 데이터와 정답 호출

# # 데이터로더로부터 data, label 받아오기

# for data, label in DataLoader():
#     # 모델 예측값 계산
#     prediction = model(data)

#     # 손실 함수 이용해 오차 계산
#     loss = LossFunction(prediction, label)

#     # 오차 역전파
#     loss.backward()

#     # 신경망 가중치 수정
#     optimizer.step()


# 인공 뉴런(퍼셉트론) : 입력값과 가중치, 편향을 이용해 출력값을 내는 수학적 모델
# 단층 인공 신경망 : 퍼셉트론을 하나만 사용하는 인공 신경망
# 다층 인공 신경망 : 퍼셉트론을 여러 개 사용하는 인공 신경망
# 가중치 : 입력의 중요도를 나타냄
# 편향 : 활성화의 경계가 원점으로붑터 얼마나 이동할지를 결정한다.
# 활성화 함수 : 해당 뉴런의 출력을 다음 뉴런으로 넘길지를 결정한다.
# 손실 함수 : 정답과 신경망의 예측의 차이를 나타내는 함수
# 경사 하강법 : 손실을 가중치에 대해 미분한 다음, 기울기의 반대 방향으로 학습률만큼 이동시키는 알고리즘
# 오차 역전파 : 올바른 가중치를 찾기 위해 오차를 출력층으로부터 입력층까지 전파하는 방식