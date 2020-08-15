

NAS는 다음의 세가지 component로 이루어진다. 각 component의 하위 항목은 method에 해당한다. 

**1. search space**
- 각 아키텍쳐 내에서 연결될(사용될) opertion(conv layer, pooling layer, unit 등)들의 집합을 의미
	(1) sequential layer wise operation
	(2) cell-based representation(Normal cell, Reduction cell)
	(3) hierarchical structure
	(4) memory-bank representation

**2. search algorithm**
- 하위 네트워크 모집단을 샘플링 하기 위한 알고리즘이다.
- reward로 상위 모델의 성능측정항목을 받고 고성능 아키텍쳐 후보를 생성하는 방법을 학습하는 구조로 이루어진다. 
	(1) random search
	(2) reinforcement learning
	(3) evaluation algorithm ; GA기반 -> Goldmine
	(4) progressive decision process
	(5) gradient descent
 
**3. evaluation strategy** 
- search algorithm을 최적화시키기 위한 피드백을 얻기 위해 성능을 측정, 추정 혹은 예측하는 것을 의미한다. 
	(1) train from scratch	
	(2) proxy task performance
	(3) parameter sharing
	(4) prediction based

**+) one-shot approach : search + evaluation**
- 많은 수의 하위 모델을 독립적으로 검색하고 평가하는 것은 비용이 너무 비쌈
- 단일 학습 과정을 통해 검색 공간을 확인한다. (결국 단일 학습모델 구축)




**flow**
![image](https://user-images.githubusercontent.com/49298791/90207300-df9ab800-de20-11ea-8b78-35d7b0733c69.png)

**refer**
paper:
https://arxiv.org/abs/1808.05377
http://nn.cs.utexas.edu/downloads/papers/stanley.ec02.pdf

blog:
https://lilianweng.github.io/lil-log/2020/08/06/neural-architecture-search.html