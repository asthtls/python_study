# 합성곱 : 작은 필터를 이용해 이미지로부터 특징을 뽑아내는 알고리즘
# CNN : 합성곱층을 반복적으로 쌓아서 만든 인공 신경망
# 특징 맵 : 합성곱층의 결과를 말한다. 합성곱층이 특징을 추출한 뒤의 이미지
# 데이터 증강과 전처리는 더 원할한 학습을 위해 데이터 수정하는 기법
    # - 데이터 증강은 이미지를 회전시키거나 잘라내는 등, 데이터 하나로 여러 가지 형태의 다른 데이터를 만들어 개수를 늘리는 기법이다.
    # - 데이터 전처리는 학습에 이용되기 이전에 처리하는 모든 기법을 의미한다. 
    # 데이터 증강또한 데이터 전처리의 일종이다.
# 이미지 정규화는 픽셀 간 편향을 제거하는 데 사용한다. 각 채널의 분포가 도일해지므로 학습이 원할하게 진행된다.
# 패딩은 이미지 외곽을 0으로 채우는 기법, 합성곱 전후 이미지 크기를 가텍 만든다.
# 크롭핑은 이미지의 일부분을 잘라내는 것을 의미한다.
# 최대 풀링은 이미지 크기를 줄이는 데 사용하는 기법으로 커널에서 가장 큰 값을 이용한다.
# 전이 학습은 사전 학습된 모델의 파라미터를 수정해 자신의 데이터셋에 최적화시키는 방법이다. 학습에 걸리는 시간을 단축할 수 있다.

# 데이터 전처리
import matplotlib.pyplot as plt
import torchvision.transforms as T

from torchvision.datasets.cifar import CIFAR10
from torchvision.transforms import Compose, RandomHorizontalFlip, RandomCrop,Normalize
# CIFAR-10 데이터셋 불러오기

# 데이터 전처리 및 정규화 
transforms = Compose([
    T.ToPILImage(),
    RandomCrop((32, 32), padding=4),  # 랜덤으로 이미지 일부 제거 후 패딩
    RandomHorizontalFlip(p=0.5), # y축으로 기준으로 대칭
    T.ToTensor(),

    Normalize(mean=(0.4914, 0.4822, 0.4465), std=(0.247, 0.243, 0.261)),
    T.ToPILImage()
])

training_data = CIFAR10(
    root="./",
    train=True,
    download=True,
    transform=transforms # 데이터 변환 함수 
)

test_data = CIFAR10(
    root="./",
    train=False,
    download=True,
    transform=transforms
)

for i in range(9):
    plt.subplot(3,3, i+1)
    plt.imshow(training_data.data[i])
plt.show()

