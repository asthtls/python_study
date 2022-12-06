# 학습 전후 비교
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
learning_rate = 1e-6 

for epoch in range(2000):
    y_pred = a * x**3 +  b * x**2 + c * x + d
    
    loss =  (y_pred - y).pow(2).sum().item() # 손실 정의 # pow(2)는 제곱, sum()은 합, item()은 실수값으로 반환하라는 뜻
    if epoch % 100 == 0:
        print(f"epoch{epoch+1} loss:{loss}")
    
    grad_y_pred = 2.0 * (y_pred - y) # 기울기 미분값 # 가중치를 업데이트하는 데 사용되는 손실값을 미분한다.
    grad_a = (grad_y_pred * x**3).sum()
    grad_b = (grad_y_pred * x**2).sum()
    grad_c = (grad_y_pred * x).sum()
    grad_d = grad_y_pred.sum()

    a -= learning_rate * grad_a # 가중치 업데이트 # 가중치는 기울기의 반대 방향으로 이동한다.
    b -= learning_rate * grad_b 
    c -= learning_rate * grad_c
    d -= learning_rate * grad_d


# 실제 사인 곡선 그리기
plt.subplot(3,1,1)
plt.title("y true")
plt.plot(x,y)

# 예측한 가중치의 사인 곡선 그리기
plt.subplot(3,1,2)
plt.title("y pred")
plt.plot(x, y_pred)

# 랜덤한 가중치의 사인 곡선 그리기
plt.subplot(3,1,3)
plt.title("y random")
plt.plot(y_random)

plt.show()
