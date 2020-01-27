---
title : "Feature engineering method"
toc : True
---

## 1. feature selection
* wrapper method : feature의 조합을 정해 기계학습을 돌리고 성능을 평가. 이렇게 조합을 바꿔가면서 계속 돌려 성능이 가장 좋은 조합을 찾는다. 하지만 overfitting의 위험이 있다.
(stepwise regression,, forward selection, backward selection)
* Filter Method : 통계적인 방법으로 subset을 추출한 후에 모델을 돌리는 과정을 의미한다. 대표적으로 종속변수와 독립변수의 피어슨상관계수를 이용하는 방법이 있다. 
혹은 feature rank를 주어 각 feature가 얼만큼의 영향력을 주는가를 알려준다. 
* Embededed method : 모델 자체에 feature selection이 있는 경우이다. 즉 과적합을 줄이기 위해 내부적으로 penalty를 주는 경우를 의미한다. 
(LASSO, RIDGE)
* else.
information gain, Genetic algorithms (supervised)
PCA loading (unsupervised)

## 2. feature extraction
* partial least squares(PLS)
* Principal component analysis (PCA), Wavelets, transforms, Autoencoder (unsupervised)

## reference
https://mchoi07.github.io/