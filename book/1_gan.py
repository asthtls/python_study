# 1장_ 생성적 적대 신경망이란?


class Generator:
    
    def __init__(self):
        self.initVariable = 1

    def lossFunction(self): # 훈련 시에 쓸 사용자 정의 손실 함수를 정의(특정 구현에 필요한 경우)
        return
    
    def buildModel(self): # 주어진 신경망의 실제 모델을 구성한다.
        return

    def trainModel(self): # 훈련
        return


class Discriminator:
    
    def __init__(self):
        self.initVariable = 1
    
    def lossFunction(self):
        return
    
    def buildModel(self):
        return
    
    def trainModel(self, inputX, inputY):
        return


class Loss: # 손실 함수가 무엇이냐에 따라 선택적으로 구현되는 손실 함수의 클래스 탬플릿
    
    def __init__(self):
        self.initVariable = 1

    def lossBaseFunction1(self):
        return
    
    def lossBaseFunction2(self):
            return
    
    def lossBaseFunction3(self):
        return
    