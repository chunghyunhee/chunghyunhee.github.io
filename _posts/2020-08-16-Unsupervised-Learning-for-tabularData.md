- unsupervised learning을 통해 tabular data에서 noise에 강한 model을 만들 수 있습니다. 
- Porto Seguro's Safe Driver Prediction competition에서 이 방법을 사용하여 public score 0.28을 달성할 수 있습니다. 
- 총 6개의 모델을 사용하되, 5개의 neural net, 1개의 light gradient boosting model을 사용합니다. 
- neural net으로는 DAE(Denoiseing AutoEncoder)를 사용하여 원래 feature = target으로 생산하되, `inputSwapNoise`를 사용하여 noise가 낀 feature을 생산하게 합니다.
- 이렇게 5번의 과정을 거치면 마치 data augmentation을 한 효과를 볼 수 있으며, 마지막에 light gbm을 활용하여 supervised learning을 할 때 data representation이 훨씬 더 용이해집니다.
- 전반적인 model construction은 아래와 같습니다. 
![image](https://user-images.githubusercontent.com/49298791/90319425-fcdf9b80-df72-11ea-8c23-cc7dea357707.png)
