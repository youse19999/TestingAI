from keras.models import Sequential
from keras.layers import Activation,Dense
import numpy as np
import random
xarray = np.array([[1,0],[0,1],[1,1],[0,0]])
yarray = np.array([0,0,1,1])

#create
def main():
  #create Sequential in here
  model = Sequential()
  #model add dense with input_dim 2
  model.add(Dense(3,input_dim=2))
  #model add Sigmoid
  model.add(Activation("sigmoid"))
  #デンスを追加
  model.add(Dense(1))
  #ここにも活性化関数
  model.add(Activation("sigmoid"))
  #判別機はとりあえずアダム、負の値？はとりあえずmseでだす。評価関数はaccuracy
  model.compile(
      optimizer= "adam",
      loss="mse",
      metrics=["accuracy"]
  )
  #fitする
  model.fit(xarray,yarray,epochs=30000)
  #outを一応作る
  out = model.predict(xarray)
  #outする
  print(out)
  for x in range(50):
    #ganarrayという、4つの配列の中に、2つのランダムな0~1の値をここに入れる。
    ganarray = np.array([[random.randint(0,1),random.randint(0,1)],[random.randint(0,1),random.randint(0,1)],[random.randint(0,1),random.randint(0,1)],[random.randint(0,1),random.randint(0,1)]])
    for v in range(4):
        #ganarrayを評価。
        out2 = model.predict(ganarray)
        #ganarrayの1~4の値の予測値を37行から見つけたので、それが、x > 0.9だったら表示する。
        if out2[v] > 0.9:
            print(str("no." + str(v) + "●") + str(ganarray[v]))
if __name__ == "__main__":
  main()
