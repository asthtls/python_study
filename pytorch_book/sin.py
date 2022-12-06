# 랜덤하게 가중치를 적용해 사인곡선 그리기

import math
import torch
import matplotlib.pyplot as plt

# -pi부터 pi사이에서 점을 1,000개 추출
x = torch.linspace(-math.pi, math.pi, 1000) # -pi부터 pi까지 1,000개의 점 추출. 이때 모든 데이터의 간격은 같다.

# 실제 사인곡선에서 추출한 값으로 y만들기
y = torch.sin(x)

# 예측 사인곡선에서 사용할 임의의 가중치(계수)를 뽑아 y만들기
a = torch.randn(())
b = torch.randn(())
c = torch.randn(())
d = torch.randn(())

# 사인 함수를 근사할 3차 다항식 정의
y_random = a * x**3 + b * x**2 + c * x + d

# 실제 사인곡선 실제 y값으로 만들기
plt.subplot(2, 1, 1)
plt.title("y true")
plt.plot(x,y)

# 예측 사인곡선을 임의의 가중치로 만든 y값 만들기
plt.subplot(2,1,2)
plt.title("y pred")
plt.plot(x, y_random)

# 실제와 예측 사인곡선 출력
plt.show()
